"""Scripted rakiplere karşı PPO episode toplama, loglama ve checkpoint yardımcıları."""

from __future__ import annotations

import gzip
import json
import logging
import math
import os
import random
import shutil
import time
from collections import Counter
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import torch
from kaggle_environments import make

from action_encoding import ActionHeadType, ID_TO_TOKEN
from cash_accounting import CashLedger
from curriculum import EpisodeCurriculum, UNRESTRICTED_CURRICULUM, inject_starting_seeds
from lifecycle_credit import (
    LifecycleCreditAssigner,
    LifecycleCreditConfig,
    LifecycleCreditMode,
    total_lifecycle_credit,
)
from lifecycle_tracking import FarmLifecycleTracker
from market_execution import trace_market_executions
from model import KaggricultureActorCritic
from policy import SampledAction, sample_action
from ppo import PPOConfig, validate_terminal_actor_safety
from reward import RewardConfig
from span_credit import (
    EconomicSpanCreditAssigner,
    aggregate_training_credits,
    expected_normalized_economic_net,
    summarize_span_credits,
)
from training import EpisodeRollout


LOGGER = logging.getLogger("kaggriculture.training")

# v2: SpatialEncoder tek katmanlı 10x10 conv yerine 3x3 residual yığın.
CHECKPOINT_FORMAT_VERSION = 2
SCRIPTED_AGENTS = ("starter", "random", "pass")
OpponentPolicy = Callable[[Mapping[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    rollout: EpisodeRollout
    summary: dict[str, Any]
    episode_directory: Path
    replay_path: Path | None
    trace_path: Path | None
    lifecycle_events_path: Path
    plant_lifecycles_path: Path
    lifecycle_token_credits_path: Path
    action_spans_path: Path
    span_credit_assignments_path: Path
    span_training_credits_path: Path


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA istendi ancak torch.cuda.is_available() False.")
    return device


def configure_logging(run_directory: Path, level: str = "INFO") -> None:
    run_directory.mkdir(parents=True, exist_ok=True)
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Bilinmeyen log seviyesi: {level}")

    LOGGER.setLevel(numeric_level)
    LOGGER.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(
        run_directory / "train.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(console)
    LOGGER.addHandler(file_handler)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "value"):
        return value.value
    raise TypeError(f"JSON'a çevrilemeyen tip: {type(value).__name__}")


def write_json(path: Path, value: Any, *, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(
            value,
            file,
            ensure_ascii=False,
            indent=indent,
            default=_json_default,
        )
        file.write("\n")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, default=_json_default)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())


def write_gzip_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as file:
        json.dump(value, file, ensure_ascii=False, default=_json_default)
    os.replace(temporary, path)


def read_replay(path: Path) -> dict[str, Any]:
    if path.is_dir():
        gzip_path = path / "replay.json.gz"
        plain_path = path / "replay.json"
        path = gzip_path if gzip_path.exists() else plain_path
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as file:
            return json.load(file)
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _open_trace(path: Path | None) -> TextIO | None:
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    return gzip.open(path, "wt", encoding="utf-8", compresslevel=6)


def _trace_token_decisions(sample: SampledAction) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for index in range(len(sample.target_token_ids)):
        valid_ids = np.flatnonzero(sample.valid_token_masks[index])
        decisions.append(
            {
                "index": index,
                "input_token": ID_TO_TOKEN[int(sample.input_token_ids[index])].value,
                "target_token": ID_TO_TOKEN[int(sample.target_token_ids[index])].value,
                "head": ActionHeadType(int(sample.head_type_ids[index])).name,
                "context_unit_index": int(sample.context_unit_indices[index]),
                "is_stochastic_decision": bool(sample.decision_mask[index]),
                "log_probability": float(sample.token_log_probs[index]),
                "entropy": float(sample.token_entropies[index]),
                "normalized_entropy": _normalized_token_entropy(sample, index),
                "valid_tokens": [ID_TO_TOKEN[int(token_id)].value for token_id in valid_ids],
            }
        )
    return decisions


_LEARNABLE_HEADS = tuple(
    head for head in ActionHeadType if head is not ActionHeadType.STRUCTURAL
)


def _normalized_token_entropy(sample: SampledAction, index: int) -> float:
    """Token entropy'sini o adımdaki ulaşılabilir maksimuma böler."""
    valid_count = int(sample.valid_token_masks[index].sum())
    if not bool(sample.decision_mask[index]) or valid_count <= 1:
        return 0.0
    maximum = math.log(valid_count)
    return float(np.clip(float(sample.token_entropies[index]) / maximum, 0.0, 1.0))


def action_entropy_statistics(sample: SampledAction) -> dict[str, dict[str, float | int]]:
    """Bir action paketindeki keşfi head bazında özetler."""
    entropy_sums: Counter[str] = Counter()
    normalized_sums: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    for index, is_decision in enumerate(sample.decision_mask):
        if not bool(is_decision):
            continue
        head = ActionHeadType(int(sample.head_type_ids[index]))
        if head is ActionHeadType.STRUCTURAL:
            continue
        name = head.name.lower()
        entropy_sums[name] += float(sample.token_entropies[index])
        normalized_sums[name] += _normalized_token_entropy(sample, index)
        counts[name] += 1

    names = (head.name.lower() for head in _LEARNABLE_HEADS)
    return {
        name: {
            "decision_count": int(counts[name]),
            "mean_entropy": (
                float(entropy_sums[name] / counts[name]) if counts[name] else 0.0
            ),
            "mean_normalized_entropy": (
                float(normalized_sums[name] / counts[name]) if counts[name] else 0.0
            ),
        }
        for name in names
    }


def _count_operations(action: Mapping[str, Any], counter: Counter[str]) -> None:
    farmer = action.get("farmer", [])
    if farmer:
        counter[f"unit:{farmer[0]}"] += 1
    for hand_action in action.get("hands", []):
        if hand_action:
            counter[f"unit:{hand_action[0]}"] += 1
    for order in action.get("market", []):
        if order:
            counter[f"market:{order[0]}"] += 1


def _learner_player(episode_number: int, side: str) -> int:
    if side == "alternate":
        return (episode_number - 1) % 2
    return int(side)


def _prepare_episode_directory(
    output_root: Path,
    episode_directory: Path,
) -> None:
    """Yarım kalmış episode'u korur; tamamlanmış kaydı ezmez."""
    if episode_directory.exists():
        completion_marker = episode_directory / "complete.json"
        if completion_marker.exists():
            raise FileExistsError(
                f"Tamamlanmış episode zaten mevcut: {episode_directory}"
            )
        failed_directory = output_root / "failed_episodes"
        failed_directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        archive_path = failed_directory / f"{episode_directory.name}_{timestamp}"
        os.replace(episode_directory, archive_path)
        LOGGER.warning("Yarım episode kaydı arşivlendi: %s", archive_path)
    episode_directory.mkdir(parents=True, exist_ok=False)


def collect_scripted_episode(
    model: KaggricultureActorCritic,
    *,
    opponent_name: str,
    episode_number: int,
    seed: int,
    learner_player: int,
    environment_configuration: Mapping[str, Any],
    reward_config: RewardConfig,
    ppo_config: PPOConfig,
    output_root: Path,
    save_replay: bool,
    save_trace: bool,
    deterministic_actions: bool,
    torch_generator: torch.Generator | None,
    log_step_every: int = 100,
    environment_debug: bool = False,
    episode_curriculum: EpisodeCurriculum | None = None,
    lifecycle_credit_config: LifecycleCreditConfig | None = None,
    opponent_policy: OpponentPolicy | None = None,
) -> EpisodeResult:
    validate_terminal_actor_safety(ppo_config)
    if opponent_policy is None and opponent_name not in SCRIPTED_AGENTS:
        raise ValueError(
            f"Rakip {SCRIPTED_AGENTS} içinden biri olmalı; gelen={opponent_name!r}"
        )
    if learner_player not in (0, 1):
        raise ValueError("learner_player 0 veya 1 olmalı.")

    episode_directory = output_root / "episodes" / (
        f"episode_{episode_number:06d}_seed_{seed}_p{learner_player}"
    )
    _prepare_episode_directory(output_root, episode_directory)
    trace_path = episode_directory / "trajectory.jsonl.gz" if save_trace else None
    replay_path = episode_directory / "replay.json.gz" if save_replay else None

    configuration = dict(environment_configuration)
    configuration["seed"] = int(seed)
    env = make(
        "kaggriculture",
        configuration=configuration,
        debug=environment_debug,
    )
    opponent = (
        opponent_policy
        if opponent_policy is not None
        else env.agents[opponent_name]
    )
    curriculum = episode_curriculum or UNRESTRICTED_CURRICULUM
    inject_starting_seeds(
        env.state[learner_player].observation,
        curriculum.starting_seeds,
    )
    # Self-play'de iki taraf aynı curriculum başlangıcını görsün. Scripted
    # rakiplerin mevcut davranışını değiştirmemek için onlara tohum eklenmez.
    if opponent_policy is not None:
        inject_starting_seeds(
            env.state[1 - learner_player].observation,
            curriculum.starting_seeds,
        )
    lifecycle_tracker = FarmLifecycleTracker(
        deepcopy(env.state[learner_player].observation),
        episode_number=episode_number,
        player_index=learner_player,
        turns_per_day=int(configuration.get("turnsPerDay", 24)),
        shed_capacity=int(configuration.get("shedCapacity", 100)),
        reward_scale=reward_config.reward_scale,
    )
    lifecycle_credit_assigner = LifecycleCreditAssigner(lifecycle_credit_config)
    economic_span_enabled = (
        lifecycle_credit_assigner.config.mode
        is LifecycleCreditMode.ECONOMIC_SPAN
    )
    economic_credit_assigner = EconomicSpanCreditAssigner(
        coefficient=lifecycle_credit_assigner.config.coefficient,
        reward_scale=reward_config.reward_scale,
    )
    rollout = EpisodeRollout(reward_config, ppo_config)
    cash_ledger = CashLedger(
        [env.state[0].observation.farms[p].money for p in range(2)]
    )
    operation_counts: Counter[str] = Counter()
    opponent_operation_counts: Counter[str] = Counter()
    entropy_values: list[float] = []
    head_entropy_sums: Counter[str] = Counter()
    head_normalized_entropy_sums: Counter[str] = Counter()
    head_decision_counts: Counter[str] = Counter()
    value_predictions: list[float] = []
    action_log_probs: list[float] = []
    stochastic_decisions = 0
    token_count = 0
    started = time.perf_counter()

    trace_file = _open_trace(trace_path)
    try:
        while not env.done:
            pre_observation = deepcopy(env.state[learner_player].observation)
            opponent_player = 1 - learner_player
            opponent_observation = deepcopy(env.state[opponent_player].observation)

            sample = sample_action(
                model,
                pre_observation,
                grammar_config=curriculum.grammar_config,
                deterministic=deterministic_actions,
                generator=torch_generator,
            )
            opponent_action = opponent(opponent_observation)
            actions = [None, None]
            actions[learner_player] = sample.action
            actions[opponent_player] = opponent_action
            ordered_observations = [None, None]
            ordered_observations[learner_player] = pre_observation
            ordered_observations[opponent_player] = opponent_observation
            all_executions = trace_market_executions(
                ordered_observations, actions, configuration,
            )
            market_executions = all_executions[learner_player]
            env.step(actions)

            next_observation = deepcopy(env.state[learner_player].observation)
            cash_ledger.observe(
                int(pre_observation.get("step", len(rollout.transitions))),
                all_executions,
                [farm["money"] for farm in pre_observation["farms"]],
                [farm["money"] for farm in next_observation["farms"]],
            )
            done = env.done or env.state[learner_player].status != "ACTIVE"
            lifecycle_events = lifecycle_tracker.observe_transition(
                pre_observation,
                sample.action,
                next_observation,
                token_ids=sample.token_ids,
                market_executions=market_executions,
            )
            lifecycle_credit_assignments = lifecycle_credit_assigner.assign(
                lifecycle_events
            )
            lifecycle_credit = total_lifecycle_credit(
                lifecycle_credit_assignments
            )
            shaped = rollout.add(
                sample,
                pre_observation,
                next_observation,
                done,
                lifecycle_credit=lifecycle_credit,
            )

            _count_operations(sample.action, operation_counts)
            _count_operations(opponent_action, opponent_operation_counts)
            entropy_values.append(sample.mean_entropy)
            for head_name, statistics in action_entropy_statistics(sample).items():
                count = int(statistics["decision_count"])
                head_decision_counts[head_name] += count
                head_entropy_sums[head_name] += float(statistics["mean_entropy"]) * count
                head_normalized_entropy_sums[head_name] += (
                    float(statistics["mean_normalized_entropy"]) * count
                )
            value_predictions.append(sample.value)
            action_log_probs.append(sample.action_log_prob)
            stochastic_decisions += int(sample.decision_mask.sum())
            token_count += len(sample.token_ids)

            if trace_file is not None:
                trace_row = {
                    "episode": episode_number,
                    "environment_seed": seed,
                    "learner_player": learner_player,
                    # Kaggle ikinci oyuncunun observation'inda ``step`` alanını
                    # bazen göndermiyor; day/hour ise iki tarafta da mevcut.
                    "step": int(
                        pre_observation.get(
                            "step",
                            int(pre_observation["day"])
                            * int(configuration.get("turnsPerDay", 24))
                            + int(pre_observation["hour"]),
                        )
                    ),
                    "day": int(pre_observation["day"]),
                    "hour": int(pre_observation["hour"]),
                    "observation": pre_observation,
                    "learner_action": sample.action,
                    "opponent_action": opponent_action,
                    "next_observation": next_observation,
                    "token_ids": sample.token_ids,
                    "tokens": [
                        ID_TO_TOKEN[int(token_id)].value
                        for token_id in sample.token_ids
                    ],
                    "token_decisions": _trace_token_decisions(sample),
                    "action_log_probability": sample.action_log_prob,
                    "mean_entropy": sample.mean_entropy,
                    "value_prediction": sample.value,
                    "reward": asdict(shaped),
                    "lifecycle_events": [
                        {
                            "event_id": event.event_id,
                            "event_type": event.event_type.value,
                            "plant_id": event.plant_id,
                            "normalized_credit": event.normalized_credit,
                        }
                        for event in lifecycle_events
                    ],
                    "lifecycle_credit_assignments": [
                        asdict(assignment)
                        for assignment in lifecycle_credit_assignments
                    ],
                    "done": done,
                    "learner_status": env.state[learner_player].status,
                    "opponent_status": env.state[opponent_player].status,
                }
                json.dump(
                    trace_row,
                    trace_file,
                    ensure_ascii=False,
                    default=_json_default,
                )
                trace_file.write("\n")
                trace_file.flush()

            completed_steps = len(rollout.transitions)
            if log_step_every > 0 and completed_steps % log_step_every == 0:
                current_money = float(
                    next_observation["farms"][learner_player]["money"]
                )
                LOGGER.info(
                    "Episode %d progress | step=%d day=%s hour=%s "
                    "money=%.0f value=%.4f entropy=%.4f action=%s",
                    episode_number,
                    completed_steps,
                    next_observation.get("day"),
                    next_observation.get("hour"),
                    current_money,
                    sample.value,
                    sample.mean_entropy,
                    sample.action,
                )
            if done:
                break
    finally:
        if trace_file is not None:
            trace_file.close()

    elapsed = time.perf_counter() - started
    final_learner_money = float(env.state[learner_player].observation.farms[learner_player].money)
    final_opponent_money = float(
        env.state[1 - learner_player].observation.farms[1 - learner_player].money
    )
    lifecycle_token_assignments = (
        lifecycle_credit_assigner.assign_episode_token_credits(
            lifecycle_tracker.events
        )
    )
    credited_token_count = 0
    for assignment in lifecycle_token_assignments:
        credited_token_count += rollout.apply_token_credit(
            environment_step=assignment.step,
            token_start=assignment.token_start,
            token_end=assignment.token_end,
            amount=assignment.amount,
            role=assignment.role,
        )
    lifecycle_token_credits_path = (
        episode_directory / "lifecycle_token_credits.json"
    )
    write_json(
        lifecycle_token_credits_path,
        [asdict(assignment) for assignment in lifecycle_token_assignments],
    )

    span_credit_assignments = (
        economic_credit_assigner.assign_episode(lifecycle_tracker.events)
        if economic_span_enabled
        else ()
    )
    span_training_credits = aggregate_training_credits(span_credit_assignments)
    credited_span_count = 0
    for credit in span_training_credits:
        credited_span_count += int(
            rollout.apply_span_credit(
                action_id=credit.action_id,
                amount=credit.amount,
                role=credit.credit_role,
            )
        )
    action_spans_path = episode_directory / "action_spans.json"
    write_json(
        action_spans_path,
        [
            span.to_dict()
            for transition in rollout.transitions
            for span in transition.action_spans
        ],
    )
    span_credit_assignments_path = (
        episode_directory / "span_credit_assignments.json"
    )
    write_json(
        span_credit_assignments_path,
        [assignment.to_dict() for assignment in span_credit_assignments],
    )
    span_training_credits_path = episode_directory / "span_training_credits.json"
    write_json(
        span_training_credits_path,
        [credit.to_dict() for credit in span_training_credits],
    )
    span_credit_summary = summarize_span_credits(span_credit_assignments)
    expected_economic_net = (
        lifecycle_credit_assigner.config.coefficient
        * expected_normalized_economic_net(
            lifecycle_tracker.events,
            reward_scale=reward_config.reward_scale,
        )
        if economic_span_enabled
        else 0.0
    )
    economic_invariant_error = (
        span_credit_summary.total - expected_economic_net
    )
    if economic_span_enabled and abs(economic_invariant_error) > 1e-7:
        raise RuntimeError(
            "Economic span credit invariant bozuldu: "
            f"assigned={span_credit_summary.total:.10f}, "
            f"expected={expected_economic_net:.10f}, "
            f"error={economic_invariant_error:.10f}"
        )

    lifecycle_artifacts = lifecycle_tracker.write_artifacts(episode_directory)
    lifecycle_summary = lifecycle_tracker.summary()
    reward_rows = [transition.reward for transition in rollout.transitions]
    write_json(episode_directory / "cash_ledger.json", cash_ledger.to_dict())
    summary = {
        "cash_accounting": cash_ledger.summary(learner_player),
        "opponent_cash_accounting": cash_ledger.summary(1 - learner_player),
        "cash_minus_lifecycle_credit": (
            cash_ledger.summary(learner_player)["net_cash"] / reward_config.reward_scale
            - span_credit_summary.total
        ),
        "global_actor_coef": ppo_config.global_actor_coef,
        "span_policy_coef": ppo_config.span_policy_coef,
        "event": "episode",
        "terminal_statuses": [str(state.status) for state in env.state],
        "episode": episode_number,
        "environment_seed": seed,
        "opponent": opponent_name,
        "learner_player": learner_player,
        "steps": len(rollout.transitions),
        "duration_seconds": elapsed,
        "steps_per_second": len(rollout.transitions) / max(elapsed, 1e-9),
        "final_money": final_learner_money,
        "profit": final_learner_money - reward_config.starting_money,
        "opponent_final_money": final_opponent_money,
        "money_advantage": final_learner_money - final_opponent_money,
        "won_by_money": final_learner_money > final_opponent_money,
        "environment_reward": float(env.state[learner_player].reward or 0.0),
        "training_reward_sum": float(sum(row.total for row in reward_rows)),
        "base_reward_sum": float(sum(row.base for row in reward_rows)),
        "shaping_reward_sum": float(sum(row.shaping for row in reward_rows)),
        "lifecycle_credit_sum": float(
            sum(row.lifecycle_credit for row in reward_rows)
        ),
        "lifecycle_credit_mode": lifecycle_credit_assigner.config.mode.value,
        "lifecycle_credit_coefficient": (
            lifecycle_credit_assigner.config.coefficient
        ),
        "lifecycle_credit_scheme_version": (
            lifecycle_credit_assigner.config.scheme_version
        ),
        "lifecycle_token_assignment_count": len(
            lifecycle_token_assignments
        ),
        "lifecycle_credited_token_count": credited_token_count,
        "lifecycle_token_credit_sum": float(
            sum(assignment.amount for assignment in lifecycle_token_assignments)
        ),
        "lifecycle_token_positive_credit_sum": float(
            sum(
                max(assignment.amount, 0.0)
                for assignment in lifecycle_token_assignments
            )
        ),
        "lifecycle_token_negative_credit_sum": float(
            sum(
                min(assignment.amount, 0.0)
                for assignment in lifecycle_token_assignments
            )
        ),
        "economic_span_enabled": economic_span_enabled,
        "economic_span_credit_assignment_count": (
            span_credit_summary.assignment_count
        ),
        "economic_credited_span_count": credited_span_count,
        "economic_training_span_count": len(span_training_credits),
        "economic_span_credit_sum": span_credit_summary.total,
        "economic_span_credit_positive_sum": span_credit_summary.positive,
        "economic_span_credit_negative_sum": span_credit_summary.negative,
        "economic_span_credit_by_role": span_credit_summary.by_role,
        "economic_expected_normalized_net": expected_economic_net,
        "economic_credit_invariant_error": economic_invariant_error,
        "mean_entropy": float(np.mean(entropy_values)) if entropy_values else 0.0,
        "entropy_coefficient": float(ppo_config.entropy_coef),
        "mean_entropy_by_head": {
            head.name.lower(): (
                float(head_entropy_sums[head.name.lower()] / head_decision_counts[head.name.lower()])
                if head_decision_counts[head.name.lower()]
                else 0.0
            )
            for head in _LEARNABLE_HEADS
        },
        "mean_normalized_entropy_by_head": {
            head.name.lower(): (
                float(
                    head_normalized_entropy_sums[head.name.lower()]
                    / head_decision_counts[head.name.lower()]
                )
                if head_decision_counts[head.name.lower()]
                else 0.0
            )
            for head in _LEARNABLE_HEADS
        },
        "stochastic_decision_count_by_head": {
            head.name.lower(): int(head_decision_counts[head.name.lower()])
            for head in _LEARNABLE_HEADS
        },
        "mean_value_prediction": (
            float(np.mean(value_predictions)) if value_predictions else 0.0
        ),
        "mean_action_log_probability": (
            float(np.mean(action_log_probs)) if action_log_probs else 0.0
        ),
        "token_count": token_count,
        "stochastic_decision_count": stochastic_decisions,
        "learner_operation_counts": dict(sorted(operation_counts.items())),
        "opponent_operation_counts": dict(sorted(opponent_operation_counts.items())),
        "replay_path": str(replay_path) if replay_path is not None else None,
        "trace_path": str(trace_path) if trace_path is not None else None,
        "lifecycle_events_path": str(lifecycle_artifacts["events"]),
        "plant_lifecycles_path": str(lifecycle_artifacts["plants"]),
        "lifecycle_token_credits_path": str(lifecycle_token_credits_path),
        "action_spans_path": str(action_spans_path),
        "span_credit_assignments_path": str(span_credit_assignments_path),
        "span_training_credits_path": str(span_training_credits_path),
        "product_lots_path": str(lifecycle_artifacts["products"]),
        "trade_lifecycles_path": str(lifecycle_artifacts["trades"]),
        "worker_lifecycles_path": str(lifecycle_artifacts["workers"]),
        "lifecycle_event_count": lifecycle_summary["event_count"],
        "tracked_plant_count": lifecycle_summary["plant_count"],
        "tracked_harvested_units": lifecycle_summary["total_harvested_units"],
        "tracked_sale_revenue": lifecycle_summary[
            "total_attributed_sale_revenue"
        ],
        "curriculum_stage": curriculum.stage_name,
        "curriculum_stage_index": curriculum.stage_index,
        "curriculum_probe": curriculum.is_probe,
        "curriculum_starting_seeds": dict(curriculum.starting_seeds),
        "final_statuses": [state.status for state in env.state],
    }
    write_json(episode_directory / "summary.json", summary)
    if replay_path is not None:
        write_gzip_json(replay_path, env.toJSON())
    write_json(
        episode_directory / "complete.json",
        {
            "episode": episode_number,
            "environment_seed": seed,
            "completed_at": datetime.now().astimezone().isoformat(),
        },
    )

    return EpisodeResult(
        rollout=rollout,
        summary=summary,
        episode_directory=episode_directory,
        replay_path=replay_path,
        trace_path=trace_path,
        lifecycle_events_path=lifecycle_artifacts["events"],
        plant_lifecycles_path=lifecycle_artifacts["plants"],
        lifecycle_token_credits_path=lifecycle_token_credits_path,
        action_spans_path=action_spans_path,
        span_credit_assignments_path=span_credit_assignments_path,
        span_training_credits_path=span_training_credits_path,
    )


def checkpoint_payload(
    model: KaggricultureActorCritic,
    optimizer: torch.optim.Optimizer,
    *,
    episode: int,
    update: int,
    total_environment_steps: int,
    run_configuration: Mapping[str, Any],
    sampling_generator: torch.Generator | None = None,
    span_credit_scaler_state: Mapping[str, Any] | None = None,
    self_play_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "created_at": datetime.now().astimezone().isoformat(),
        "episode": episode,
        "update": update,
        "total_environment_steps": total_environment_steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "run_configuration": dict(run_configuration),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }
    if sampling_generator is not None:
        payload["sampling_generator_state"] = sampling_generator.get_state().cpu()
        payload["sampling_generator_device"] = str(sampling_generator.device)
    if span_credit_scaler_state is not None:
        payload["span_credit_scaler_state"] = dict(span_credit_scaler_state)
    if self_play_state is not None:
        payload["self_play_state"] = dict(self_play_state)
    if torch.cuda.is_available():
        payload["torch_cuda_random_states"] = torch.cuda.get_rng_state_all()
    return payload


def save_checkpoint(
    run_directory: Path,
    payload: Mapping[str, Any],
    *,
    numbered: bool,
) -> Path:
    checkpoint_directory = run_directory / "checkpoints"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    episode = int(payload["episode"])
    update = int(payload["update"])
    target = checkpoint_directory / f"episode_{episode:06d}_update_{update:06d}.pt"
    temporary = checkpoint_directory / f".{target.name}.tmp"
    torch.save(dict(payload), temporary)
    if numbered:
        os.replace(temporary, target)
        source = target
    else:
        source = temporary
    latest = checkpoint_directory / "latest.pt"
    latest_temporary = checkpoint_directory / ".latest.pt.tmp"
    shutil.copy2(source, latest_temporary)
    os.replace(latest_temporary, latest)
    if not numbered:
        temporary.unlink(missing_ok=True)
        return latest
    return target


def load_checkpoint(
    checkpoint_path: Path,
    model: KaggricultureActorCritic,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    *,
    restore_rng: bool,
) -> dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    saved_version = int(checkpoint.get("format_version", -1))
    if saved_version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "Desteklenmeyen checkpoint formatı: "
            f"{saved_version}; beklenen={CHECKPOINT_FORMAT_VERSION}. "
            "v1 checkpoint'leri tek katmanlı 10x10 SpatialEncoder ile "
            "eğitilmiştir ve residual encoder'a yüklenemez; yeni bir "
            "--run-dir ile sıfırdan eğitim başlatın."
        )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
    if restore_rng:
        random.setstate(checkpoint["python_random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"].cpu())
        if torch.cuda.is_available() and "torch_cuda_random_states" in checkpoint:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["torch_cuda_random_states"]]
            )
    return checkpoint


def render_replay_to_html(
    replay_path: Path,
    output_path: Path,
    *,
    width: int = 1200,
    height: int = 800,
) -> Path:
    replay = read_replay(replay_path)
    env = make(
        replay.get("name", "kaggriculture"),
        configuration=dict(replay["configuration"]),
        steps=replay["steps"],
        info=replay.get("info", {}),
    )
    html = env.render(
        mode="html",
        playing=True,
        controls=True,
        width=width,
        height=height,
    )
    if not isinstance(html, str) or not html.strip():
        raise RuntimeError("Kaggle renderer HTML üretmedi.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(html, encoding="utf-8")
    os.replace(temporary, output_path)
    return output_path


def default_run_name(opponent: str) -> str:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_vs_{opponent}"


def environment_reward_config(
    *,
    episode_steps: int,
    turns_per_day: int,
    starting_money: float,
    gamma: float,
    shaping_beta: float,
) -> RewardConfig:
    return RewardConfig(
        gamma=gamma,
        shaping_beta=shaping_beta,
        reward_scale=starting_money,
        starting_money=starting_money,
        season_days=max(1, math.ceil(episode_steps / turns_per_day)),
        turns_per_day=turns_per_day,
    )
