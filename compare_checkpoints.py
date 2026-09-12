"""Paired full-game checkpoint evaluation, without curriculum or seed gifts.

Every checkpoint plays both sides for every seed, against the same opponent.
This runs no optimizer. Old compatible weights may be evaluated but never
silently resumed under new training semantics.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import statistics

import numpy as np
import torch
from kaggle_environments.envs.kaggriculture import kaggriculture as game

from curriculum import UNRESTRICTED_CURRICULUM
from lifecycle_credit import LifecycleCreditConfig, LifecycleCreditMode
from model import KaggricultureActorCritic
from ppo import PPOConfig
from scripted_training import (collect_scripted_episode, configure_logging,
                               environment_reward_config, load_checkpoint,
                               resolve_device, write_json, append_jsonl)


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def evaluation_cases(seeds: list[int]) -> list[tuple[int, int]]:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be nonempty and unique.")
    return [(seed, side) for seed in seeds for side in (0, 1)]


def aggregate(rows: list[dict]) -> dict:
    counts = Counter()
    for row in rows:
        counts.update(row["learner_operation_counts"])
    return {
        "games": len(rows),
        "mean_final_money": statistics.mean(r["final_money"] for r in rows),
        "mean_money_advantage": statistics.mean(r["money_advantage"] for r in rows),
        "win_rate": statistics.mean(r["money_advantage"] > 0 for r in rows),
        "draw_rate": statistics.mean(r["money_advantage"] == 0 for r in rows),
        "mean_operation_counts": {k: v / len(rows) for k, v in sorted(counts.items())},
        "max_cash_error": max(r["cash_accounting"]["max_reconciliation_error"] for r in rows),
    }


def validate_full_game_summary(summary: dict) -> None:
    if (summary["steps"] != 719 or summary["terminal_statuses"] != ["DONE", "DONE"]
            or summary["curriculum_stage"] != "off" or summary["curriculum_starting_seeds"]):
        raise RuntimeError("Evaluation requires a completed 720-step game, no curriculum or gifts.")


def paired_comparison(reference: list[dict], candidate: list[dict]) -> dict:
    def indexed(rows):
        result = {(r["environment_seed"], r["learner_player"]): r["money_advantage"] for r in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate evaluation cases.")
        return result
    left, right = indexed(reference), indexed(candidate)
    if not left or left.keys() != right.keys():
        raise ValueError("Paired comparison requires identical seed/side cases.")
    seeds = sorted({seed for seed, _ in left})
    if any((seed, side) not in left for seed in seeds for side in (0, 1)):
        raise ValueError("Both sides required for every seed.")
    # Treat the two sides of one seed as a block, not independent samples.
    differences = [statistics.mean(right[seed, side] - left[seed, side] for side in (0, 1))
                   for seed in seeds]
    return {"seed_blocks": len(seeds), "mean_advantage_delta": statistics.mean(differences),
            "standard_error_over_seed_blocks": (
                statistics.stdev(differences) / len(seeds)**0.5 if len(seeds) > 1 else None),
            "per_seed_advantage_delta": dict(zip(map(str, seeds), differences))}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[31001, 31002, 31003, 31004, 31005])
    parser.add_argument("--opponent", choices=["starter", "random", "pass"], default="starter")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args(argv)
    cases = evaluation_cases(args.seeds)
    paths = [p.resolve(strict=True) for p in args.checkpoints]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate checkpoints.")
    device = resolve_device(args.device)
    torch.set_num_threads(1)
    # Preflight every checkpoint before starting a potentially long comparison.
    model = KaggricultureActorCritic().to(device)
    for path in paths:
        load_checkpoint(path, model, None, device, restore_rng=False)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    configure_logging(args.output_dir)
    configuration = {"episodeSteps": 720, "turnsPerDay": 24, "startingMoney": 3000,
                     "actTimeout": 10, "runTimeout": 7200}
    manifest = {"protocol": "paired-full-game-v1", "opponent": args.opponent,
                "cases": cases, "environment": configuration, "curriculum": False,
                "starting_seed_gifts": {}, "deterministic": not args.stochastic,
                "training": False, "device": str(device),
                "engine_sha256": sha256(Path(game.__file__)),
                "source_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
                "checkpoints": [{"path": str(p), "sha256": sha256(p)} for p in paths],
                "caution": "Small seed sets are diagnostics, not leaderboard evidence."}
    write_json(args.output_dir / "manifest.json", manifest)
    all_rows = []
    results = []
    for index, path in enumerate(paths):
        checkpoint = load_checkpoint(path, model, None, device, restore_rng=False)
        model.eval()
        root = args.output_dir / f"checkpoint_{index:02d}"
        rows = []
        for number, (seed, side) in enumerate(cases, 1):
            # Reset all RNG streams for every case, independent of earlier game length.
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            generator = torch.Generator(device=device.type).manual_seed(seed + 123457 + side)
            result = collect_scripted_episode(
                model, opponent_name=args.opponent, episode_number=number, seed=seed,
                learner_player=side, environment_configuration=configuration,
                reward_config=environment_reward_config(episode_steps=720, turns_per_day=24,
                    starting_money=3000, gamma=0.999, shaping_beta=0),
                ppo_config=PPOConfig(), output_root=root, save_replay=True,
                save_trace=False, deterministic_actions=not args.stochastic,
                torch_generator=generator, log_step_every=0,
                episode_curriculum=UNRESTRICTED_CURRICULUM,
                lifecycle_credit_config=LifecycleCreditConfig(mode=LifecycleCreditMode.OFF),
            )
            validate_full_game_summary(result.summary)
            rows.append(result.summary)
            append_jsonl(root / "metrics.jsonl", result.summary)
            print(f"{path.name}: seed={seed} side={side} gold={result.summary['final_money']:.0f} "
                  f"difference={result.summary['money_advantage']:.0f}", flush=True)
        all_rows.append(rows)
        summary = {"checkpoint": str(path), "training_episode": checkpoint["episode"],
                   **aggregate(rows)}
        write_json(root / "evaluation_summary.json", summary)
        results.append(summary)
        write_json(args.output_dir / "comparison.json", {
            "complete": index == len(paths)-1, "results": results,
            "paired_against_first": [paired_comparison(all_rows[0], r) for r in all_rows[1:]],
        })
    print(json.dumps(results, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
