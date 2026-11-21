"""Kaggriculture scripted-opponent PPO komut satırı aracı."""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
import webbrowser
from dataclasses import asdict, replace
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch

from curriculum import CurriculumConfig, resolve_episode_curriculum
from entropy_schedule import (
    ENTROPY_SCHEDULE_VERSION,
    EntropyScheduleConfig,
    episode_in_stage,
)
from lifecycle_credit import (
    CAUSAL_CHAIN_SCHEME_VERSION,
    LifecycleCreditConfig,
    LifecycleCreditMode,
)
from lagged_self_play import (
    SELF_PLAY_OPPONENT_NAME,
    LaggedSelfPlayConfig,
    LaggedSelfPlayOpponent,
)
from model import KaggricultureActorCritic
from ppo import PPOConfig, validate_terminal_actor_safety
from scripted_training import (
    LOGGER,
    SCRIPTED_AGENTS,
    append_jsonl,
    checkpoint_payload,
    collect_scripted_episode,
    configure_logging,
    default_run_name,
    environment_reward_config,
    load_checkpoint,
    render_replay_to_html,
    resolve_device,
    save_checkpoint,
    write_json,
)
from training import ppo_update
from span_credit import (
    ECONOMIC_SPAN_SCHEME_VERSION,
    RunningRoleRMS,
)


ROOT = Path(__file__).resolve().parent
TRAINING_SEMANTICS_VERSION = 5


def _add_environment_arguments(
    parser: argparse.ArgumentParser,
    *,
    allow_self_play: bool = False,
) -> None:
    opponent_choices = (
        (*SCRIPTED_AGENTS, SELF_PLAY_OPPONENT_NAME)
        if allow_self_play
        else SCRIPTED_AGENTS
    )
    parser.add_argument("--opponent", choices=opponent_choices, default="starter")
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument("--turns-per-day", type=int, default=24)
    parser.add_argument("--starting-money", type=float, default=3000.0)
    parser.add_argument("--learner-side", choices=("alternate", "0", "1"), default="alternate")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--environment-debug", action="store_true")


def _add_artifact_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--save-replay-every",
        type=int,
        default=1,
        help="0 kapatır; N her N episode'da Kaggle replay kaydeder.",
    )
    parser.add_argument(
        "--save-trace-every",
        type=int,
        default=1,
        help="0 kapatır; N her N episode'da tam state/action/token trace kaydeder.",
    )
    parser.add_argument(
        "--log-step-every",
        type=int,
        default=100,
        help="0 kapatır; uzun oyunlarda her N adımda ilerleme logu yazar.",
    )


def _add_lifecycle_credit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--lifecycle-credit-mode",
        choices=tuple(mode.value for mode in LifecycleCreditMode),
        default=LifecycleCreditMode.ECONOMIC_SPAN.value,
        help=(
            "Lifecycle reward: economic_span crop/trade/HIRE ekonomisini "
            "korunumlu semantik action span'larına ayırır; causal_chain eski "
            "token-level split-credit v2 yoludur; off krediyi kapatır; "
            "harvest_value hasat değerini global reward'a ekler; "
            "terminal modunda economic_span için shaping_beta 0 olmalıdır."
        ),
    )
    parser.add_argument(
        "--lifecycle-credit-coef",
        type=float,
        default=1.0,
        help="Global veya token-level lifecycle credit çarpanı; off iken etkisizdir.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Kaggriculture masked autoregressive PPO",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser(
        "train",
        help="Scripted veya gecikmeli self-play rakibine karşı PPO eğit.",
    )
    _add_environment_arguments(train, allow_self_play=True)
    _add_artifact_arguments(train)
    _add_lifecycle_credit_arguments(train)
    train.add_argument("--episodes", type=int, default=100)
    train.add_argument(
        "--episodes-per-update",
        type=int,
        default=16,
        help=(
            "Update başına episode. Terminal ağırlıklı hedefte bir episode "
            "yaklaşık tek bir bağımsız örnek sayılır; düşük değerler gradyan "
            "varyansını çok yükseltir."
        ),
    )
    train.add_argument("--ppo-epochs", type=int, default=3)
    train.add_argument("--minibatch-size", type=int, default=256)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--gamma", type=float, default=0.999)
    train.add_argument("--gae-lambda", type=float, default=0.95)
    train.add_argument("--clip-ratio", type=float, default=0.20)
    train.add_argument("--value-clip", type=float, default=0.20)
    train.add_argument("--value-coef", type=float, default=0.50)
    train.add_argument("--entropy-coef", type=float, default=0.01)
    train.add_argument(
        "--entropy-schedule",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Müfredat aşamasına göre entropy katsayısı uygular. "
            "--no-entropy-schedule sabit --entropy-coef kullanır."
        ),
    )
    train.add_argument(
        "--entropy-head-weights",
        type=float,
        nargs=4,
        metavar=("UNIT", "MARKET", "ITEM", "QUANTITY"),
        default=[1.0, 1.0, 1.0, 1.0],
        help=(
            "Entropy bonusunun kafa başına çarpanı. Hareket kafası doğal "
            "olarak yüksek entropili kalırken market kafası çökebilir; tek bir "
            "global katsayı bu ikisini ayıramaz."
        ),
    )
    train.add_argument("--entropy-wheat-loop", type=float, default=0.030)
    train.add_argument("--entropy-wheat-hire", type=float, default=0.025)
    train.add_argument("--entropy-wheat-land", type=float, default=0.020)
    train.add_argument("--entropy-wheat-carrot", type=float, default=0.015)
    train.add_argument("--entropy-full-start", type=float, default=0.012)
    train.add_argument("--entropy-full-end", type=float, default=0.005)
    train.add_argument("--entropy-full-decay-episodes", type=int, default=3000)
    train.add_argument("--target-kl", type=float, default=0.020)
    train.add_argument(
        "--kl-skip-multiplier",
        type=float,
        default=2.0,
        help=(
            "KL'i target_kl'in bu katından fazla olan minibatch atlanır. "
            "Update yalnızca epoch ORTALAMASI target_kl'i aşarsa durur."
        ),
    )
    train.add_argument("--max-grad-norm", type=float, default=1.00)
    train.add_argument(
        "--lifecycle-policy-coef",
        type=float,
        default=1.0,
        help="Token-level lifecycle PPO loss katsayısı.",
    )
    train.add_argument(
        "--span-policy-coef",
        type=float,
        default=0.0,
        help="İsteğe bağlı lifecycle yardımcı loss. 0 iken kredi yalnız loglanır.",
    )
    train.add_argument(
        "--global-actor-coef",
        type=float,
        default=1.0,
        help=(
            "Economic span modunda terminal global span PPO katsayısı. "
            "Ekonomik modda pozitif olmalı; bütün gerçek kararları eğitir."
        ),
    )
    train.add_argument(
        "--span-credit-role-weights",
        type=float,
        nargs=4,
        metavar=("PURCHASE", "FARM", "TRADE", "HIRE"),
        default=[1.0, 1.0, 1.0, 1.0],
        help="PURCHASE/FARM/TRADE/HIRE local span loss ağırlıkları.",
    )
    train.add_argument(
        "--normalize-token-advantages",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Lifecycle token kredilerini batch içinde merkezleyip ana "
            "avantajlarla aynı ölçeğe taşır."
        ),
    )
    train.add_argument(
        "--shaping-beta",
        type=float,
        default=0.0,
        help=(
            "Ekonomik potential shaping katsayısı; cash_progress modunda "
            "nakit hariç varlıkların ilerleme kredisi."
        ),
    )
    train.add_argument("--reward-mode", choices=("terminal", "cash_progress"), default="terminal")
    train.add_argument("--realized-value-coef", type=float, default=0.0,
                       help="Critic için gerçek base getirisinin Monte Carlo yardımcı loss katsayısı.")
    train.add_argument("--checkpoint-every", type=int, default=10)
    train.add_argument("--run-dir", type=Path)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--override-kl-on-resume", action="store_true",
                       help="Resume sırasında yalnız target-kl değişimine açık izin verir; değişiklik event'e yazılır.")
    train.add_argument(
        "--initialize-from-checkpoint",
        type=Path,
        help=(
            "Yeni bir run'ın learner ağırlıklarını bu checkpoint'ten "
            "başlatır; optimizer/update/RNG taşınmaz. --resume ile birlikte "
            "kullanılamaz."
        ),
    )
    train.add_argument("--device", default="auto")
    train.add_argument("--torch-threads", type=int, default=1)
    train.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO")
    train.add_argument(
        "--self-play-lag-updates",
        type=int,
        default=100,
        help=(
            "Self-play rakibinin learner'dan en az kaç PPO update geride "
            "olacağı. 16 episode/update ile 100 update yaklaşık 1600 "
            "episode'dur."
        ),
    )
    train.add_argument(
        "--self-play-snapshot-interval",
        type=int,
        default=10,
        help="Learner modelini kaç update'te bir rakip havuzuna kaydeder.",
    )
    train.add_argument(
        "--self-play-opponent-deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Donmuş rakibin argmax oynaması; varsayılan stochastic sampling.",
    )
    train.add_argument(
        "--deterministic-actions",
        action="store_true",
        help="Yalnızca smoke/debug için; normal PPO eğitiminde kullanmayın.",
    )
    train.add_argument(
        "--curriculum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Buğday döngüsü müfredatı ve aşamalı aksiyon maskesi. "
            "Economic v1 smoke'da varsayılan açıktır; --no-curriculum "
            "yalnız ayrı full-action deneyidir."
        ),
    )
    train.add_argument(
        "--curriculum-probe-probability",
        type=float,
        default=0.0,
        help="Müfredat açıkken tam yasal maskeli probe episode oranı.",
    )
    train.add_argument(
        "--curriculum-episodes-per-stage",
        type=int,
        default=100,
    )
    train.add_argument(
        "--curriculum-starting-wheat",
        type=int,
        default=8,
        help="Müfredat aşamalarında learner deposuna eklenen buğday tohumu. 0 kapatır.",
    )

    evaluate = subparsers.add_parser("evaluate", help="Checkpoint'i scripted rakibe karşı oyna.")
    _add_environment_arguments(evaluate)
    _add_artifact_arguments(evaluate)
    _add_lifecycle_credit_arguments(evaluate)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--episodes", type=int, default=10)
    evaluate.add_argument("--output-dir", type=Path)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--torch-threads", type=int, default=1)
    evaluate.add_argument("--stochastic", action="store_true")
    evaluate.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO")

    render = subparsers.add_parser("render", help="Kayıtlı replay'i izlenebilir HTML'e çevir.")
    render.add_argument("replay", type=Path)
    render.add_argument("--output", type=Path)
    render.add_argument("--width", type=int, default=1200)
    render.add_argument("--height", type=int, default=800)
    render.add_argument("--open", action="store_true", dest="open_browser")
    return parser


def _validate_positive(name: str, value: int | float, *, allow_zero: bool = False) -> None:
    valid = value >= 0 if allow_zero else value > 0
    if not valid:
        relation = "negatif olamaz" if allow_zero else "pozitif olmalı"
        raise ValueError(f"{name} {relation}.")


def _environment_configuration(args: argparse.Namespace) -> dict:
    return {
        "episodeSteps": args.episode_steps,
        "turnsPerDay": args.turns_per_day,
        "startingMoney": int(args.starting_money),
        "actTimeout": 10,
        "runTimeout": max(1200, args.episode_steps * 10),
    }


def _ppo_config(args: argparse.Namespace) -> PPOConfig:
    economic_span = (
        args.lifecycle_credit_mode == LifecycleCreditMode.ECONOMIC_SPAN.value
    )
    return PPOConfig(
        realized_value_coef=args.realized_value_coef,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        value_clip=args.value_clip,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        # STRUCTURAL tokenlar asla stochastic karar değildir; başa 1.0 eklenir.
        entropy_head_weights=(1.0, *args.entropy_head_weights),
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        kl_skip_multiplier=args.kl_skip_multiplier,
        normalize_token_advantages=args.normalize_token_advantages,
        lifecycle_policy_coef=args.lifecycle_policy_coef,
        use_action_span_policy=economic_span,
        global_actor_coef=args.global_actor_coef,
        span_policy_coef=args.span_policy_coef,
        head_balanced_entropy=economic_span,
        span_credit_role_weights=tuple(args.span_credit_role_weights),
    )


def _run_directory(args: argparse.Namespace) -> Path:
    if args.run_dir is not None:
        return args.run_dir.resolve()
    return (ROOT / "runs" / default_run_name(args.opponent)).resolve()


def _validate_resume_configuration(
    args: argparse.Namespace,
    checkpoint: dict,
) -> None:
    saved = checkpoint.get("run_configuration", {})
    for field, default in (("reward_mode", "terminal"), ("realized_value_coef", 0.0)):
        if saved.get(field, default) != getattr(args, field):
            raise ValueError(f"Resume {field} değişimini kabul etmez; yeni run başlatın.")
    fixed_fields = (
        "opponent",
        "episode_steps",
        "turns_per_day",
        "starting_money",
        "learner_side",
        "seed",
        "episodes_per_update",
        "ppo_epochs",
        "minibatch_size",
        "learning_rate",
        "gamma",
        "gae_lambda",
        "clip_ratio",
        "value_clip",
        "value_coef",
        "entropy_coef",
        "entropy_head_weights",
        "entropy_schedule",
        "entropy_wheat_loop",
        "entropy_wheat_hire",
        "entropy_wheat_land",
        "entropy_wheat_carrot",
        "entropy_full_start",
        "entropy_full_end",
        "entropy_full_decay_episodes",
        "target_kl",
        "kl_skip_multiplier",
        "max_grad_norm",
        "lifecycle_policy_coef",
        "span_policy_coef",
        "global_actor_coef",
        "normalize_token_advantages",
        "span_credit_role_weights",
        "shaping_beta",
        "lifecycle_credit_mode",
        "lifecycle_credit_coef",
        "deterministic_actions",
        "curriculum",
        "curriculum_probe_probability",
        "curriculum_episodes_per_stage",
        "curriculum_starting_wheat",
        "self_play_lag_updates",
        "self_play_snapshot_interval",
        "self_play_opponent_deterministic",
    )
    mismatches = []
    for field in fixed_fields:
        if field == "target_kl" and getattr(args, "override_kl_on_resume", False):
            continue
        if field in saved and saved[field] != getattr(args, field):
            mismatches.append(
                f"{field}: checkpoint={saved[field]!r}, CLI={getattr(args, field)!r}"
            )
    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(
            "Resume ayarları ilk koşuyla uyuşmuyor. Yalnızca toplam "
            f"--episodes ve artifact/log aralıklarını değiştirin. {details}"
        )
    saved_lifecycle = saved.get("lifecycle_credit", {})
    if args.lifecycle_credit_mode == LifecycleCreditMode.ECONOMIC_SPAN.value:
        if (
            saved.get("training_semantics_version")
            != TRAINING_SEMANTICS_VERSION
            or saved.get("economic_span_scheme_version")
            != ECONOMIC_SPAN_SCHEME_VERSION
            or saved_lifecycle.get("scheme_version")
            != ECONOMIC_SPAN_SCHEME_VERSION
        ):
            raise ValueError(
                "Checkpoint economic-span v5 (terminal actor ve STOP) "
                "semantiğini taşımıyor. Yeni --run-dir kullanın; mimari "
                "uyumluysa --initialize-from-checkpoint ile yalnız ağırlıkları alın."
            )
    if (
        saved.get("lifecycle_credit_mode") == LifecycleCreditMode.CAUSAL_CHAIN.value
        and saved_lifecycle.get("scheme_version")
        != CAUSAL_CHAIN_SCHEME_VERSION
    ):
        raise ValueError(
            "Checkpoint eski causal-chain reward semantiğini kullanıyor; "
            "purchase/plant split-credit v2 ile resume edilemez. Yeni bir "
            "--run-dir ile sıfırdan training başlatın."
        )
    saved_entropy_schedule = saved.get("entropy_schedule_config", {})
    if (
        args.entropy_schedule
        and saved_entropy_schedule.get("version") != ENTROPY_SCHEDULE_VERSION
    ):
        raise ValueError(
            "Checkpoint eski veya statik entropy semantiğini kullanıyor; "
            "curriculum entropy schedule v1 ile resume edilemez. Yeni bir "
            "--run-dir ile sıfırdan training başlatın."
        )


def _head_entropy_rollup(
    summaries: list[dict],
    value_key: str,
) -> tuple[dict[str, float], dict[str, int]]:
    """Episode head ortalamalarını karar sayısıyla ağırlıklı birleştirir."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for summary in summaries:
        values = summary.get(value_key, {})
        head_counts = summary.get("stochastic_decision_count_by_head", {})
        for head_name, count_value in head_counts.items():
            count = int(count_value)
            counts[head_name] = counts.get(head_name, 0) + count
            sums[head_name] = sums.get(head_name, 0.0) + (
                float(values.get(head_name, 0.0)) * count
            )
    return (
        {
            head_name: (sums.get(head_name, 0.0) / count if count else 0.0)
            for head_name, count in sorted(counts.items())
        },
        dict(sorted(counts.items())),
    )


def train(args: argparse.Namespace) -> int:
    if args.resume and args.initialize_from_checkpoint is not None:
        raise ValueError(
            "--resume ile --initialize-from-checkpoint birlikte kullanılamaz."
        )
    for name in (
        "episodes",
        "episodes_per_update",
        "ppo_epochs",
        "minibatch_size",
        "episode_steps",
        "turns_per_day",
        "checkpoint_every",
        "torch_threads",
        "entropy_full_decay_episodes",
    ):
        _validate_positive(name, getattr(args, name))
    _validate_positive("save_replay_every", args.save_replay_every, allow_zero=True)
    _validate_positive("save_trace_every", args.save_trace_every, allow_zero=True)
    _validate_positive("log_step_every", args.log_step_every, allow_zero=True)
    _validate_positive(
        "lifecycle_policy_coef",
        args.lifecycle_policy_coef,
        allow_zero=True,
    )
    _validate_positive(
        "lifecycle_credit_coef",
        args.lifecycle_credit_coef,
        allow_zero=True,
    )
    _validate_positive("span_policy_coef", args.span_policy_coef, allow_zero=True)
    _validate_positive("global_actor_coef", args.global_actor_coef, allow_zero=True)
    if any(weight < 0.0 for weight in args.span_credit_role_weights):
        raise ValueError("span_credit_role_weights negatif olamaz.")
    economic_span_enabled = (
        args.lifecycle_credit_mode == LifecycleCreditMode.ECONOMIC_SPAN.value
    )
    validate_terminal_actor_safety(_ppo_config(args))
    if args.reward_mode == "cash_progress":
        if args.gamma != 1.0:
            raise ValueError("cash_progress için --gamma 1 gerekli.")
        if args.span_policy_coef != 0 or args.lifecycle_credit_mode not in ("economic_span", "off"):
            raise ValueError("cash_progress yalnız audit/off lifecycle ve --span-policy-coef 0 kabul eder.")
    if economic_span_enabled and args.reward_mode == "terminal" and not np.isclose(args.shaping_beta, 0.0):
        raise ValueError(
            "economic_span v1 realized krediyle potential değerini iki kez "
            "saymamak için --shaping-beta 0 gerektirir."
        )
    if not 0.0 <= args.curriculum_probe_probability <= 1.0:
        raise ValueError("curriculum_probe_probability [0, 1] aralığında olmalı.")
    _validate_positive(
        "curriculum_episodes_per_stage",
        args.curriculum_episodes_per_stage,
    )
    _validate_positive(
        "curriculum_starting_wheat",
        args.curriculum_starting_wheat,
        allow_zero=True,
    )
    _validate_positive("self_play_lag_updates", args.self_play_lag_updates)
    _validate_positive(
        "self_play_snapshot_interval",
        args.self_play_snapshot_interval,
    )
    curriculum_config = CurriculumConfig(
        enabled=bool(args.curriculum),
        probe_probability=args.curriculum_probe_probability,
        episodes_per_stage=args.curriculum_episodes_per_stage,
        starting_wheat_seeds=args.curriculum_starting_wheat,
    )
    entropy_schedule_config = EntropyScheduleConfig(
        enabled=bool(args.entropy_schedule),
        wheat_loop=args.entropy_wheat_loop,
        wheat_hire=args.entropy_wheat_hire,
        wheat_land=args.entropy_wheat_land,
        wheat_carrot=args.entropy_wheat_carrot,
        full_start=args.entropy_full_start,
        full_end=args.entropy_full_end,
        full_decay_episodes=args.entropy_full_decay_episodes,
    )
    lifecycle_credit_config = LifecycleCreditConfig(
        mode=LifecycleCreditMode(args.lifecycle_credit_mode),
        coefficient=args.lifecycle_credit_coef,
    )
    self_play_config = LaggedSelfPlayConfig(
        lag_updates=args.self_play_lag_updates,
        snapshot_interval=args.self_play_snapshot_interval,
        deterministic=args.self_play_opponent_deterministic,
    )

    run_directory = _run_directory(args)
    initialization_checkpoint_path = (
        args.initialize_from_checkpoint.resolve()
        if args.initialize_from_checkpoint is not None
        else None
    )
    if (
        initialization_checkpoint_path is not None
        and not initialization_checkpoint_path.exists()
    ):
        raise FileNotFoundError(
            "Başlangıç checkpoint'i bulunamadı: "
            f"{initialization_checkpoint_path}"
        )
    if args.resume:
        checkpoint_path = run_directory / "checkpoints" / "latest.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Resume checkpoint bulunamadı: {checkpoint_path}")
    elif run_directory.exists() and any(run_directory.iterdir()):
        raise FileExistsError(
            f"Run directory boş değil: {run_directory}. "
            "Devam etmek için --resume kullanın."
        )

    configure_logging(run_directory, args.log_level)
    device = resolve_device(args.device)
    torch.set_num_threads(args.torch_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ppo_config = _ppo_config(args)
    reward_config = environment_reward_config(
        episode_steps=args.episode_steps,
        turns_per_day=args.turns_per_day,
        starting_money=args.starting_money,
        gamma=args.gamma,
        shaping_beta=args.shaping_beta,
        mode=args.reward_mode,
    )
    environment_configuration = _environment_configuration(args)
    model = KaggricultureActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    initialization_checkpoint: dict | None = None
    if initialization_checkpoint_path is not None:
        initialization_checkpoint = load_checkpoint(
            initialization_checkpoint_path,
            model,
            optimizer=None,
            device=device,
            restore_rng=False,
        )
    span_credit_scaler = RunningRoleRMS()
    generator = torch.Generator(device=device.type)
    generator.manual_seed(args.seed + 9173)
    run_configuration = {
        **vars(args),
        "run_dir": str(run_directory),
        "device_resolved": str(device),
        "ppo": asdict(ppo_config),
        "reward": asdict(reward_config),
        "lifecycle_credit": asdict(lifecycle_credit_config),
        "environment": environment_configuration,
        "curriculum_config": {
            "enabled": curriculum_config.enabled,
            "probe_probability": curriculum_config.probe_probability,
            "episodes_per_stage": curriculum_config.episodes_per_stage,
            "starting_wheat_seeds": curriculum_config.starting_wheat_seeds,
        },
        "entropy_schedule_config": asdict(entropy_schedule_config),
        "training_semantics_version": TRAINING_SEMANTICS_VERSION,
        "economic_span_scheme_version": (
            ECONOMIC_SPAN_SCHEME_VERSION if economic_span_enabled else None
        ),
        "initial_model": (
            {
                "checkpoint": str(initialization_checkpoint_path),
                "source_episode": int(initialization_checkpoint["episode"]),
                "source_update": int(initialization_checkpoint["update"]),
            }
            if initialization_checkpoint is not None
            else None
        ),
    }

    start_episode = 1
    update_number = 0
    total_environment_steps = 0
    checkpoint: dict | None = None
    if args.resume:
        checkpoint = load_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            device,
            restore_rng=True,
        )
        _validate_resume_configuration(args, checkpoint)
        saved_run_configuration = checkpoint.get("run_configuration", {})
        previous_kl = saved_run_configuration.get("target_kl")
        if previous_kl is not None and previous_kl != args.target_kl:
            append_jsonl(run_directory / "events.jsonl", {
                "event": "kl_threshold_override",
                "time": datetime.now().astimezone().isoformat(),
                "checkpoint": str(checkpoint_path),
                "checkpoint_episode": int(checkpoint["episode"]),
                "old_target_kl": previous_kl,
                "new_target_kl": args.target_kl,
                "new_skip_threshold": args.target_kl * args.kl_skip_multiplier,
            })
            LOGGER.info("Resume KL hedefi değişti: %.4f -> %.4f", previous_kl, args.target_kl)
        for metadata_field in ("initialize_from_checkpoint", "initial_model"):
            if metadata_field in saved_run_configuration:
                run_configuration[metadata_field] = saved_run_configuration[
                    metadata_field
                ]
        start_episode = int(checkpoint["episode"]) + 1
        update_number = int(checkpoint["update"])
        total_environment_steps = int(checkpoint["total_environment_steps"])
        if "sampling_generator_state" in checkpoint:
            generator.set_state(checkpoint["sampling_generator_state"].cpu())
        if economic_span_enabled:
            scaler_state = checkpoint.get("span_credit_scaler_state")
            if scaler_state is None:
                raise ValueError(
                    "Economic span checkpoint RunningRoleRMS state içermiyor; "
                    "eski checkpoint yeni optimizer semantiğiyle resume edilemez."
                )
            span_credit_scaler.load_state_dict(scaler_state)
        append_jsonl(
            run_directory / "events.jsonl",
            {
                "event": "resume",
                "time": datetime.now().astimezone().isoformat(),
                "checkpoint": str(checkpoint_path),
                "start_episode": start_episode,
            },
        )
        LOGGER.info("Checkpoint yüklendi: %s", checkpoint_path)
    else:
        write_json(run_directory / "config.json", run_configuration)
        if initialization_checkpoint is not None:
            append_jsonl(
                run_directory / "events.jsonl",
                {
                    "event": "model_initialized_from_checkpoint",
                    "time": datetime.now().astimezone().isoformat(),
                    **run_configuration["initial_model"],
                },
            )
            LOGGER.info(
                "Learner ağırlıkları checkpoint'ten başlatıldı | "
                "source_episode=%d source_update=%d checkpoint=%s",
                initialization_checkpoint["episode"],
                initialization_checkpoint["update"],
                initialization_checkpoint_path,
            )

    self_play_opponent: LaggedSelfPlayOpponent | None = None
    if args.opponent == SELF_PLAY_OPPONENT_NAME:
        resume_self_play_state = (
            checkpoint.get("self_play_state")
            if checkpoint is not None
            else None
        )
        if args.resume and resume_self_play_state is None:
            raise ValueError(
                "Self-play checkpoint self_play_state içermiyor; bu koşu "
                "güvenli ve aynı rakiple resume edilemez."
            )
        self_play_opponent = LaggedSelfPlayOpponent(
            model,
            run_directory=run_directory,
            device=device,
            config=self_play_config,
            seed=args.seed + 104_729,
            learner_update=update_number,
            resume_state=resume_self_play_state,
        )
        self_play_event = {
            "event": "self_play_resume" if args.resume else "self_play_initialized",
            "time": datetime.now().astimezone().isoformat(),
            **self_play_opponent.summary(),
            "config": asdict(self_play_config),
        }
        append_jsonl(run_directory / "events.jsonl", self_play_event)
        LOGGER.info(
            "Gecikmeli self-play hazır | learner_update=%d opponent_update=%d "
            "lag_target=%d snapshot_interval=%d deterministic=%s",
            update_number,
            self_play_opponent.opponent_update,
            self_play_config.lag_updates,
            self_play_config.snapshot_interval,
            self_play_config.deterministic,
        )

    if start_episode > args.episodes:
        LOGGER.info("Checkpoint episode=%d zaten hedef episodes=%d değerine ulaşmış.", start_episode - 1, args.episodes)
        return 0

    pending_records = []
    pending_entropy_coefficients: list[float] = []
    pending_episode_summaries: list[dict] = []
    last_episode = start_episode - 1
    metrics_path = run_directory / "metrics.jsonl"

    LOGGER.info(
        "Eğitim başladı | opponent=%s episodes=%d device=%s curriculum=%s "
        "entropy_schedule=%s run=%s",
        args.opponent,
        args.episodes,
        device,
        curriculum_config.enabled,
        entropy_schedule_config.enabled,
        run_directory,
    )
    for episode_number in range(start_episode, args.episodes + 1):
        episode_seed = args.seed + episode_number - 1
        learner_player = (
            (episode_number - 1) % 2
            if args.learner_side == "alternate"
            else int(args.learner_side)
        )
        save_replay = (
            args.save_replay_every > 0
            and episode_number % args.save_replay_every == 0
        )
        save_trace = (
            args.save_trace_every > 0
            and episode_number % args.save_trace_every == 0
        )
        episode_curriculum = resolve_episode_curriculum(
            curriculum_config,
            episode_number,
            episode_seed,
        )
        stage_episode_number = episode_in_stage(
            episode_number,
            episode_curriculum.stage_index,
            curriculum_config.episodes_per_stage,
        )
        episode_entropy_coefficient = entropy_schedule_config.coefficient(
            episode_curriculum.stage_name,
            stage_episode_number,
            fixed_coefficient=args.entropy_coef,
        )
        episode_ppo_config = replace(
            ppo_config,
            entropy_coef=episode_entropy_coefficient,
        )
        opponent_policy = (
            partial(
                self_play_opponent.act,
                grammar_config=episode_curriculum.grammar_config,
            )
            if self_play_opponent is not None
            else None
        )
        result = collect_scripted_episode(
            model,
            opponent_name=args.opponent,
            episode_number=episode_number,
            seed=episode_seed,
            learner_player=learner_player,
            environment_configuration=environment_configuration,
            reward_config=reward_config,
            ppo_config=episode_ppo_config,
            output_root=run_directory,
            save_replay=save_replay,
            save_trace=save_trace,
            deterministic_actions=args.deterministic_actions,
            torch_generator=generator,
            log_step_every=args.log_step_every,
            environment_debug=args.environment_debug,
            episode_curriculum=episode_curriculum,
            lifecycle_credit_config=lifecycle_credit_config,
            opponent_policy=opponent_policy,
        )
        records = result.rollout.records()
        pending_records.extend(records)
        pending_entropy_coefficients.append(episode_entropy_coefficient)
        result.summary["entropy_coefficient"] = episode_entropy_coefficient
        result.summary["entropy_stage_episode"] = stage_episode_number
        if self_play_opponent is not None:
            result.summary.update(self_play_opponent.summary())
        pending_episode_summaries.append(result.summary)
        total_environment_steps += len(records)
        last_episode = episode_number
        write_json(result.episode_directory / "summary.json", result.summary)
        append_jsonl(metrics_path, result.summary)
        LOGGER.info(
            "Episode %d | seed=%d side=%d stage=%s probe=%s money=%.0f "
            "opponent=%.0f profit=%.0f reward=%.4f entropy=%.4f coef=%.4f steps/s=%.1f",
            episode_number,
            episode_seed,
            learner_player,
            result.summary["curriculum_stage"],
            result.summary["curriculum_probe"],
            result.summary["final_money"],
            result.summary["opponent_final_money"],
            result.summary["profit"],
            result.summary["training_reward_sum"],
            result.summary["mean_entropy"],
            episode_entropy_coefficient,
            result.summary["steps_per_second"],
        )

        update_due = (
            len(pending_records) > 0
            and (
                episode_number % args.episodes_per_update == 0
                or episode_number == args.episodes
            )
        )
        if not update_due:
            continue

        model.train()
        update_started = time.perf_counter()
        update_entropy_coefficient = float(np.mean(pending_entropy_coefficients))
        update_ppo_config = replace(
            ppo_config,
            entropy_coef=update_entropy_coefficient,
        )
        head_entropy, head_counts = _head_entropy_rollup(
            pending_episode_summaries,
            "mean_entropy_by_head",
        )
        normalized_head_entropy, _ = _head_entropy_rollup(
            pending_episode_summaries,
            "mean_normalized_entropy_by_head",
        )
        update_metrics = ppo_update(
            model,
            optimizer,
            pending_records,
            config=update_ppo_config,
            epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            device=device,
            generator=generator,
            span_credit_scaler=(
                span_credit_scaler if economic_span_enabled else None
            ),
        )
        update_number += 1
        self_play_refresh = (
            self_play_opponent.after_learner_update(model, update_number)
            if self_play_opponent is not None
            else None
        )
        update_event = {
            "event": "update",
            "episode": episode_number,
            "update": update_number,
            "records": len(pending_records),
            "total_environment_steps": total_environment_steps,
            "duration_seconds": time.perf_counter() - update_started,
            "entropy_coefficient": update_entropy_coefficient,
            "rollout_mean_entropy_by_head": head_entropy,
            "rollout_mean_normalized_entropy_by_head": normalized_head_entropy,
            "rollout_stochastic_decision_count_by_head": head_counts,
            **asdict(update_metrics),
        }
        if self_play_refresh is not None:
            update_event.update(
                {
                    "self_play_opponent_update": self_play_refresh.opponent_update,
                    "self_play_actual_lag_updates": (
                        self_play_refresh.actual_lag_updates
                    ),
                    "self_play_snapshot_saved": self_play_refresh.snapshot_saved,
                    "self_play_opponent_changed": self_play_refresh.opponent_changed,
                }
            )
            if (
                self_play_refresh.snapshot_saved
                or self_play_refresh.opponent_changed
            ):
                append_jsonl(
                    run_directory / "events.jsonl",
                    {
                        "event": "self_play_refresh",
                        "time": datetime.now().astimezone().isoformat(),
                        **asdict(self_play_refresh),
                    },
                )
                LOGGER.info(
                    "Self-play refresh | learner_update=%d opponent_update=%d "
                    "actual_lag=%d snapshot_saved=%s opponent_changed=%s",
                    self_play_refresh.learner_update,
                    self_play_refresh.opponent_update,
                    self_play_refresh.actual_lag_updates,
                    self_play_refresh.snapshot_saved,
                    self_play_refresh.opponent_changed,
                )
        append_jsonl(metrics_path, update_event)
        LOGGER.info(
            "Update %d | records=%d mb=%d/%d ep=%d/%d loss=%.5f policy=%.5f "
            "local_span=%.5f value=%.5f realized_value=%.5f "
            "entropy=%.4f coef=%.4f Hn[unit=%.3f market=%.3f item=%.3f qty=%.3f] "
            "KL=%.6f KLspan[u=%.6f m=%.6f p=%.6f f=%.6f t=%.6f h=%.6f] "
            "KLhead[u=%.6f m=%.6f i=%.6f q=%.6f] grad=%.4f%s",
            update_number,
            len(pending_records),
            update_metrics.minibatches,
            update_metrics.minibatches + update_metrics.skipped_minibatches,
            update_metrics.epochs_completed,
            args.ppo_epochs,
            update_metrics.total_loss,
            update_metrics.policy_loss,
            update_metrics.lifecycle_policy_loss,
            update_metrics.value_loss,
            update_metrics.realized_value_loss,
            update_metrics.entropy,
            update_entropy_coefficient,
            normalized_head_entropy.get("unit_operation", 0.0),
            normalized_head_entropy.get("market_operation", 0.0),
            normalized_head_entropy.get("item", 0.0),
            normalized_head_entropy.get("quantity", 0.0),
            update_metrics.approximate_kl,
            update_metrics.unit_span_kl,
            update_metrics.market_span_kl,
            update_metrics.purchase_credit_kl,
            update_metrics.farm_credit_kl,
            update_metrics.trade_credit_kl,
            update_metrics.hire_credit_kl,
            update_metrics.unit_operation_kl,
            update_metrics.market_operation_kl,
            update_metrics.item_kl,
            update_metrics.quantity_kl,
            update_metrics.gradient_norm,
            " KL-STOP" if update_metrics.stopped_for_kl else "",
        )
        pending_records.clear()
        pending_entropy_coefficients.clear()
        pending_episode_summaries.clear()

        payload = checkpoint_payload(
            model,
            optimizer,
            episode=episode_number,
            update=update_number,
            total_environment_steps=total_environment_steps,
            run_configuration=run_configuration,
            sampling_generator=generator,
            span_credit_scaler_state=(
                span_credit_scaler.state_dict()
                if economic_span_enabled
                else None
            ),
            self_play_state=(
                self_play_opponent.state_dict()
                if self_play_opponent is not None
                else None
            ),
        )
        checkpoint_path = save_checkpoint(
            run_directory,
            payload,
            numbered=(episode_number % args.checkpoint_every == 0 or episode_number == args.episodes),
        )
        LOGGER.info("Checkpoint kaydedildi: %s", checkpoint_path)

    append_jsonl(
        run_directory / "events.jsonl",
        {
            "event": "training_complete",
            "time": datetime.now().astimezone().isoformat(),
            "episode": last_episode,
            "update": update_number,
            "total_environment_steps": total_environment_steps,
        },
    )
    LOGGER.info("Eğitim tamamlandı: %s", run_directory)
    return 0


def evaluate(args: argparse.Namespace) -> int:
    _validate_positive("episodes", args.episodes)
    _validate_positive("episode_steps", args.episode_steps)
    _validate_positive("turns_per_day", args.turns_per_day)
    _validate_positive("save_replay_every", args.save_replay_every, allow_zero=True)
    _validate_positive("save_trace_every", args.save_trace_every, allow_zero=True)
    _validate_positive("log_step_every", args.log_step_every, allow_zero=True)
    _validate_positive(
        "lifecycle_credit_coef",
        args.lifecycle_credit_coef,
        allow_zero=True,
    )
    device = resolve_device(args.device)
    torch.set_num_threads(args.torch_threads)
    output_directory = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else args.checkpoint.resolve().parent.parent
        / "evaluations"
        / datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    )
    configure_logging(output_directory, args.log_level)
    model = KaggricultureActorCritic().to(device)
    checkpoint = load_checkpoint(
        args.checkpoint.resolve(),
        model,
        optimizer=None,
        device=device,
        restore_rng=False,
    )
    model.eval()
    checkpoint_config = checkpoint.get("run_configuration", {})
    gamma = float(checkpoint_config.get("ppo", {}).get("gamma", 0.999))
    shaping_beta = float(
        checkpoint_config.get("reward", {}).get("shaping_beta", 0.0)
    )
    ppo_config = PPOConfig(gamma=gamma)
    reward_config = environment_reward_config(
        episode_steps=args.episode_steps,
        turns_per_day=args.turns_per_day,
        starting_money=args.starting_money,
        gamma=gamma,
        shaping_beta=shaping_beta,
        mode=checkpoint_config.get("reward", {}).get("mode", "terminal"),
    )
    lifecycle_credit_config = LifecycleCreditConfig(
        mode=LifecycleCreditMode(args.lifecycle_credit_mode),
        coefficient=args.lifecycle_credit_coef,
    )
    environment_configuration = _environment_configuration(args)
    generator = torch.Generator(device=device.type)
    generator.manual_seed(args.seed + 123_457)
    summaries = []
    for episode_number in range(1, args.episodes + 1):
        learner_player = (
            (episode_number - 1) % 2
            if args.learner_side == "alternate"
            else int(args.learner_side)
        )
        result = collect_scripted_episode(
            model,
            opponent_name=args.opponent,
            episode_number=episode_number,
            seed=args.seed + episode_number - 1,
            learner_player=learner_player,
            environment_configuration=environment_configuration,
            reward_config=reward_config,
            ppo_config=ppo_config,
            output_root=output_directory,
            save_replay=(args.save_replay_every > 0 and episode_number % args.save_replay_every == 0),
            save_trace=(args.save_trace_every > 0 and episode_number % args.save_trace_every == 0),
            deterministic_actions=not args.stochastic,
            torch_generator=generator,
            log_step_every=args.log_step_every,
            environment_debug=args.environment_debug,
            lifecycle_credit_config=lifecycle_credit_config,
        )
        summaries.append(result.summary)
        append_jsonl(output_directory / "metrics.jsonl", result.summary)
        LOGGER.info(
            "Eval %d | money=%.0f opponent=%.0f side=%d",
            episode_number,
            result.summary["final_money"],
            result.summary["opponent_final_money"],
            learner_player,
        )
    aggregate = {
        "episodes": len(summaries),
        "checkpoint": str(args.checkpoint.resolve()),
        "opponent": args.opponent,
        "mean_final_money": float(np.mean([row["final_money"] for row in summaries])),
        "mean_opponent_money": float(
            np.mean([row["opponent_final_money"] for row in summaries])
        ),
        "mean_profit": float(np.mean([row["profit"] for row in summaries])),
        "win_rate_by_money": float(np.mean([row["won_by_money"] for row in summaries])),
    }
    write_json(output_directory / "evaluation_summary.json", aggregate)
    LOGGER.info("Evaluation tamamlandı: %s", output_directory)
    return 0


def render(args: argparse.Namespace) -> int:
    replay_path = args.replay.resolve()
    if args.output is None:
        base = replay_path if replay_path.is_dir() else replay_path.parent
        output_path = base / "replay.html"
    else:
        output_path = args.output.resolve()
    rendered = render_replay_to_html(
        replay_path,
        output_path,
        width=args.width,
        height=args.height,
    )
    print(rendered)
    if args.open_browser:
        webbrowser.open(rendered.as_uri())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            return train(args)
        if args.command == "evaluate":
            return evaluate(args)
        if args.command == "render":
            return render(args)
        parser.error(f"Bilinmeyen command: {args.command}")
    except KeyboardInterrupt:
        logging.getLogger("kaggriculture.training").warning("Kullanıcı tarafından durduruldu.")
        return 130
    except Exception:
        logging.getLogger("kaggriculture.training").exception("Komut başarısız.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
