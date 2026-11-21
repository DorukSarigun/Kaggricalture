"""Token eğitimine dokunmadan işlem-head PPO çalıştırma CLI'ı."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import gzip
import json
import logging
from pathlib import Path
import random
import time

import numpy as np
import torch
from kaggle_environments import make

from action_encoding import ActionGrammarConfig
from cash_accounting import CashLedger
from market_execution import trace_market_executions
from lifecycle_tracking import FarmLifecycleTracker
from reward import RewardConfig
from structured_actions import StructuredActorCritic
from structured_training import propose_turn, settle_turn, assign_advantages, update_policy

FORMAT = "structured-operation-v1"
LOGGER = logging.getLogger(__name__)


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, default=json_default, indent=2), encoding="utf-8")


def append_json(path, value):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, default=json_default) + "\n")


def collect_episode(model, args, episode, reward_config, directory):
    directory.mkdir(parents=True, exist_ok=False)
    configuration = {"episodeSteps": args.episode_steps, "turnsPerDay": 24,
                     "startingMoney": 3000, "seed": args.seed + episode - 1}
    env = make("kaggriculture", configuration=configuration)
    learner = (episode - 1) % 2
    opponent = env.agents[args.opponent]
    ledger = CashLedger([float(f.money) for f in env.state[learner].observation.farms])
    lifecycle = FarmLifecycleTracker(deepcopy(env.state[learner].observation),
                                     episode_number=episode, player_index=learner)
    grammar = ActionGrammarConfig(turns_per_day=24)
    rows, turn_audit = [], []
    save = args.save_every > 0 and episode % args.save_every == 0
    trace = gzip.open(directory / "trace.jsonl.gz", "wt", encoding="utf-8") if save else None
    started = time.perf_counter()
    model.eval()
    try:
        while not env.done:
            observations = [deepcopy(s.observation) for s in env.state]
            # Kaggle framework step alanını yalnız oyuncu 0'a koyabilir.
            turn_step = int(observations[0].get("step", len(turn_audit)))
            for obs in observations:
                obs["step"] = turn_step
            before = observations[learner]
            action, transitions = propose_turn(model, before, reward_config, grammar)
            actions = [None, None]
            actions[learner] = action
            # Rakip emri ancak kendi karar paketi tamamlandıktan sonra alınır.
            actions[1-learner] = opponent(observations[1-learner])
            executions = trace_market_executions(observations, actions, configuration)
            env.step(actions)
            after = deepcopy(env.state[learner].observation)
            after["step"] = turn_step + 1
            if any(str(s.status) not in ("ACTIVE", "DONE") for s in env.state):
                raise RuntimeError("Ortam hata/timeout ile sonlandı.")
            ledger.observe(int(before["step"]), executions,
                           [f["money"] for f in before["farms"]],
                           [f["money"] for f in after["farms"]])
            actual, error = settle_turn(transitions, before, after, env.done, reward_config)
            lifecycle.observe_transition(before, action, after,
                                         market_executions=executions[learner])
            turn_audit.append({"step": int(before["step"]), "base": actual.base,
                               "shaping": actual.shaping, "total": actual.total,
                               "provisional_sum": sum(t.reward for t in transitions[:-1]),
                               "settlement": transitions[-1].reward, "error": error})
            rows.extend(transitions)
            if trace:
                trace.write(json.dumps({"observation": before, "next_observation": after,
                                        "learner_action": action, "opponent_action": actions[1-learner],
                                        "operations": [{"action": t.sample.action if t.sample else None,
                                                        "section": t.inputs.section, "prefix": t.inputs.prefix,
                                                        "unit_index": t.inputs.unit_index,
                                                        "market_order_index": t.inputs.order_index,
                                                        "value": t.value, "reward": t.reward,
                                                        "base_reward": t.base_reward,
                                                        "is_settlement": t.boundary,
                                                        "current_encoded": asdict(t.inputs.current)}
                                                       for t in transitions]}, default=json_default) + "\n")
    finally:
        if trace:
            trace.close()
    assign_advantages(rows, args.gae_lambda)
    final = env.state[learner].observation
    money = float(final.farms[learner].money)
    rival = float(final.farms[1-learner].money)
    total = sum(r.reward for r in rows)
    base = sum(r.base_reward for r in rows)
    # Yeni oyun hediye varlık içermediğinden toplam doğrudan son farktır.
    if abs(total - (money-rival)/3000) > 1e-8 or abs(base - (money-rival)/3000) > 1e-8:
        raise RuntimeError("Episode reward ekonomik hedefle eşleşmedi.")
    summary = {"event": "episode", "episode": episode, "seed": configuration["seed"],
               "learner": learner, "turns": len(turn_audit), "operations": len(rows)-len(turn_audit),
               "money": money, "opponent_money": rival, "won": money > rival,
               "total_reward": total, "base_reward": base,
               "max_settlement_error": max(abs(t["error"]) for t in turn_audit),
               "cash_error": ledger.max_reconciliation_error,
               "duration_seconds": time.perf_counter()-started}
    summary["lifecycle"] = lifecycle.summary()
    lifecycle.write_artifacts(directory)
    write_json(directory / "summary.json", summary)
    write_json(directory / "turn_rewards.json", turn_audit)
    write_json(directory / "cash_ledger.json", ledger.to_dict())
    # Her ayrı kararın A'sı ve ayrı gerçekleşmiş getirisi görünür.
    write_json(directory / "advantages.json", [
        {"index": i, "step": r.environment_step, "action": r.sample.action if r.sample else None,
         "section": r.inputs.section, "unit_index": r.inputs.unit_index,
         "market_order_index": r.inputs.order_index,
         "boundary": r.boundary, "value": r.value, "reward": r.reward,
         "advantage": r.advantage, "return_target": r.return_target,
         "realized_return_target": r.realized_target}
        for i, r in enumerate(rows)])
    if save:
        with gzip.open(directory / "replay.json.gz", "wt", encoding="utf-8") as f:
            json.dump(env.toJSON(), f, default=json_default)
    return rows, summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--episode-steps", type=int, default=720)
    p.add_argument("--episodes-per-update", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--minibatch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--target-kl", type=float, default=0.02)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--shaping-beta", type=float, default=0.25)
    p.add_argument("--realized-value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.02)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--opponent", choices=("starter", "random", "pass"), default="starter")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--resume", action="store_true")
    return p


def main():
    args = parser().parse_args()
    for name in ("episodes", "episodes_per_update", "epochs", "minibatch_size", "episode_steps"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} pozitif olmalı.")
    if args.episode_steps < 2 or not 0 <= args.gae_lambda <= 1 or args.save_every < 0:
        raise ValueError("Geçersiz horizon/lambda/save ayarı.")
    for name in ("learning_rate", "target_kl", "shaping_beta", "realized_value_coef", "entropy_coef"):
        x = getattr(args, name)
        if not np.isfinite(x) or x < 0 or (name in ("learning_rate", "target_kl") and x == 0):
            raise ValueError(f"Geçersiz {name}.")
    root = args.run_dir.resolve()
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise FileExistsError("Run klasörü boş değil.")
    if args.resume and not (root / "latest.pt").exists():
        raise FileNotFoundError("Resume checkpoint bulunamadı.")
    root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        handlers=[logging.FileHandler(root / "train.log"), logging.StreamHandler()])
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    config = {**vars(args), "run_dir": str(root), "format": FORMAT}
    reward_config = RewardConfig(mode="cash_progress", gamma=1, shaping_beta=args.shaping_beta,
                                 season_days=max(1, int(np.ceil(args.episode_steps/24))))
    model = StructuredActorCritic().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    first, update = 1, 0
    if args.resume:
        saved = torch.load(root / "latest.pt", map_location=args.device, weights_only=False)
        if saved.get("format") != FORMAT:
            raise ValueError("Bu checkpoint structured head modeline ait değil.")
        for name, value in config.items():
            if name not in ("episodes", "save_every", "resume") and saved["config"].get(name) != value:
                raise ValueError(f"Resume config farkı: {name}")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if args.device == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng"]])
        first, update = saved["episode"]+1, saved["update"]
        # Kısmi rollout'u sessizce ezme veya farklı RNG ile eski loga ekleme.
        if any(int(p.name.split("_")[-1]) >= first for p in (root/"episodes").glob("episode_*")):
            raise RuntimeError("Checkpoint sonrası episode kayıtları var; önce bunları ayrı arşivleyin.")
    else:
        write_json(root / "config.json", {**config, "reward": asdict(reward_config)})
    pending = []
    for episode in range(first, args.episodes+1):
        rows, summary = collect_episode(model, args, episode, reward_config,
                                        root / "episodes" / f"episode_{episode:06d}")
        pending.extend(rows)
        append_json(root / "metrics.jsonl", summary)
        LOGGER.info("Episode %d | money=%.0f opponent=%.0f operations=%d seconds=%.2f",
                    episode, summary["money"], summary["opponent_money"], summary["operations"], summary["duration_seconds"])
        if (episode-first+1) % args.episodes_per_update and episode != args.episodes:
            continue
        metrics = update_policy(model, optimizer, pending, epochs=args.epochs,
                                minibatch_size=args.minibatch_size, target_kl=args.target_kl,
                                entropy_coef=args.entropy_coef, realized_coef=args.realized_value_coef)
        update += 1
        append_json(root / "metrics.jsonl", {"event": "update", "episode": episode, "update": update, **metrics})
        LOGGER.info("Update %d | %s", update, metrics)
        checkpoint = {"format": FORMAT, "config": config, "episode": episode, "update": update,
                      "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                      "torch_rng": torch.get_rng_state(),
                      "cuda_rng": torch.cuda.get_rng_state_all() if args.device == "cuda" else []}
        torch.save(checkpoint, root / "latest.pt.tmp")
        (root / "latest.pt.tmp").replace(root / "latest.pt")
        pending.clear()
    LOGGER.info("Eğitim tamamlandı.")


if __name__ == "__main__":
    main()
