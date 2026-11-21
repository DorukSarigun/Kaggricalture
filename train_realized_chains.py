"""Potential-free structured PPO with backwards, branch-split production-DAG credit."""
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
from lifecycle_tracking import FarmLifecycleTracker
from market_execution import trace_market_executions
from production_graph_credit import GraphCreditConfig, assign_graph_credit
from realized_training import (RealizedChainActorCritic, propose_chain_turn,
                               finish_chain_episode, update_chain_policy)
from train_structured import write_json, append_json, json_default

FORMAT = "structured-production-dag-v2"
LOGGER = logging.getLogger(__name__)


def collect_episode(model, args, episode, directory):
    directory.mkdir(parents=True, exist_ok=False)
    configuration = {"episodeSteps": args.episode_steps, "turnsPerDay": 24,
                     "startingMoney": 3000, "seed": args.seed+episode-1}
    env = make("kaggriculture", configuration=configuration)
    player = (episode-1) % 2
    opponent = env.agents[args.opponent]
    ledger = CashLedger([float(f.money) for f in env.state[player].observation.farms])
    lifecycle = FarmLifecycleTracker(deepcopy(env.state[player].observation),
                                     episode_number=episode, player_index=player, graph_tracking=True)
    grammar = ActionGrammarConfig(turns_per_day=24)
    credit_config = GraphCreditConfig(args.chain_decay, unsold_weighting=args.unsold_weighting)
    rows, refs, turns = [], {}, 0
    save = args.save_every > 0 and episode % args.save_every == 0
    trace = gzip.open(directory/"trace.jsonl.gz", "wt", encoding="utf-8") if save else None
    started = time.perf_counter()
    model.eval()
    try:
        while not env.done:
            observations = [deepcopy(s.observation) for s in env.state]
            step = int(observations[0].get("step", turns))
            for obs in observations:
                obs["step"] = step
            before = observations[player]
            action, decisions = propose_chain_turn(model, before, grammar)
            local_refs = {}
            for j, row in enumerate(decisions):
                if row.sample is None:
                    continue
                index = row.inputs.unit_index if row.inputs.section == "unit" else row.inputs.order_index
                key = (row.inputs.section, index)
                local_refs[key] = len(rows)+j
                refs[(step, *key)] = len(rows)+j
            actions = [None, None]
            actions[player] = action
            actions[1-player] = opponent(observations[1-player])
            executions = trace_market_executions(observations, actions, configuration)
            env.step(actions)
            after = deepcopy(env.state[player].observation)
            after["step"] = step+1
            if any(str(s.status) not in ("ACTIVE", "DONE") for s in env.state):
                raise RuntimeError("Environment error/timeout")
            ledger.observe(step, executions, [f["money"] for f in before["farms"]],
                           [f["money"] for f in after["farms"]])
            lifecycle.observe_transition(before, action, after, market_executions=executions[player],
                                         decision_refs=local_refs)
            if trace:
                trace.write(json.dumps({"observation": before, "next_observation": after,
                    "learner_action": action, "opponent_action": actions[1-player],
                    "decision_ids": [{"section": k[0], "index": k[1], "decision_id": v}
                                     for k, v in local_refs.items()]}, default=json_default)+"\n")
            rows.extend(decisions)
            turns += 1
    finally:
        if trace:
            trace.close()
    report = assign_graph_credit(lifecycle.events, ledger.entries, refs, player, credit_config)
    money, rival = ledger.final_money[player], ledger.final_money[1-player]
    finish_chain_episode(rows, money-rival, report, args.gae_lambda)
    if abs(sum(r.reward for r in rows) - (money-rival)/credit_config.reward_scale) > 1e-8:
        raise RuntimeError("Terminal reward mismatch")
    if abs(sum(r.chain_credit for r in rows) - report["net_cash"]/credit_config.reward_scale) > 1e-8:
        raise RuntimeError("Local credit mismatch")
    summary = {"event": "episode", "episode": episode, "seed": configuration["seed"],
        "learner": player, "turns": turns, "operations": sum(r.sample is not None for r in rows),
        "money": money, "opponent_money": rival, "won": money > rival,
        "terminal_reward": sum(r.reward for r in rows), "chain_credit_sum": sum(r.chain_credit for r in rows),
        "chain_count": sum(a["kind"] == "sale" for a in report["allocations"]),
        "sale_order_count": len({a["event_id"] for a in report["allocations"] if a["kind"] == "sale"}),
        "animal_sale_count": len({a["event_id"] for a in report["allocations"] if a["kind"] == "sale"
            and any(o.startswith("animal:") for o in a.get("origins", ()))}),
        "action_graph_nodes": len(report["graph"]["action_nodes"]),
        "material_graph_nodes": len(report["graph"]["material_nodes"]),
        "unknown_source_sale_revenue": report["unknown_source_sale_revenue"],
        "chain_reconciliation_error": report["reconciliation_error"],
        "cash_error": ledger.max_reconciliation_error,
        "duration_seconds": time.perf_counter()-started, "lifecycle": lifecycle.summary()}
    write_json(directory/"production_graph.json", report["graph"])
    write_json(directory/"chain_credit.json", {k: v for k, v in report.items() if k != "graph"})
    write_json(directory/"cash_ledger.json", ledger.to_dict())
    write_json(directory/"summary.json", summary)
    lifecycle.write_artifacts(directory)
    write_json(directory/"advantages.json", [{"decision_id": i, "step": r.environment_step,
        "action": r.sample.action if r.sample else None, "section": r.inputs.section,
        "unit_index": r.inputs.unit_index, "market_order_index": r.inputs.order_index,
        "boundary": r.boundary, "terminal_reward": r.reward, "terminal_value": r.value,
        "terminal_advantage": r.advantage, "terminal_return_target": r.return_target,
        "chain_credit": r.chain_credit, "chain_value": r.chain_value,
        "chain_advantage": r.chain_advantage} for i, r in enumerate(rows)])
    if save:
        with gzip.open(directory/"replay.json.gz", "wt", encoding="utf-8") as f:
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
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--target-kl", type=float, default=0.02)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--gae-lambda", type=float, default=1.0, help="Terminal objective only; not chain decay")
    p.add_argument("--chain-decay", type=float, default=0.8, help="Per backward action edge; split at branches")
    p.add_argument("--unsold-weighting", choices=("tree", "oldest", "purchase_only"), default="tree",
                   help="tree: unused stock on BUY, planted/placed losses on the care path; "
                        "purchase_only: every leftover on BUY; oldest: all leftovers backwards")
    p.add_argument("--terminal-policy-coef", type=float, default=1.0)
    p.add_argument("--chain-policy-coef", type=float, default=1.0)
    p.add_argument("--chain-value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.02)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--opponent", choices=("starter", "random", "pass"), default="starter")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--resume", action="store_true")
    return p


def main():
    args = parser().parse_args()
    GraphCreditConfig(args.chain_decay, unsold_weighting=args.unsold_weighting)
    for name in ("episodes", "episodes_per_update", "epochs", "minibatch_size"):
        if getattr(args, name) < 1:
            raise ValueError(f"Invalid {name}")
    if args.episode_steps < 2 or not 0 <= args.gae_lambda <= 1 or args.save_every < 0:
        raise ValueError("Invalid horizon/lambda/save")
    for name in ("learning_rate", "target_kl", "max_grad_norm", "terminal_policy_coef", "chain_policy_coef", "chain_value_coef"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(args.entropy_coef) or args.entropy_coef < 0:
        raise ValueError("Invalid entropy coefficient")
    root = args.run_dir.resolve()
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise FileExistsError("Run directory is not empty")
    if args.resume and not (root/"latest.pt").exists():
        raise FileNotFoundError("No resume checkpoint")
    root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        handlers=[logging.FileHandler(root/"train.log"), logging.StreamHandler()])
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    config = {**vars(args), "run_dir": str(root), "format": FORMAT,
              "reward": {"terminal": "final_gold_difference", "potential": False,
                         "chain": asdict(GraphCreditConfig(args.chain_decay, unsold_weighting=args.unsold_weighting)),
                         "gamma": 1.0}}
    model = RealizedChainActorCritic().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    first, update = 1, 0
    if args.resume:
        saved = torch.load(root/"latest.pt", map_location=args.device, weights_only=False)
        if saved.get("format") != FORMAT:
            raise ValueError("Reward/checkpoint semantics incompatible; use a NEW run")
        for name, value in config.items():
            if name not in ("episodes", "save_every", "resume") and saved["config"].get(name) != value:
                raise ValueError(f"Resume config mismatch: {name}")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if args.device == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng"]])
        first, update = saved["episode"]+1, saved["update"]
        if any(int(p.name.split("_")[-1]) >= first for p in (root/"episodes").glob("episode_*")):
            raise RuntimeError("Uncheckpointed episode artifacts exist; archive before resume")
    else:
        write_json(root/"config.json", config)
    pending = []
    for episode in range(first, args.episodes+1):
        rows, summary = collect_episode(model, args, episode, root/"episodes"/f"episode_{episode:06d}")
        pending.extend(rows)
        append_json(root/"metrics.jsonl", summary)
        LOGGER.info("Episode %d | money=%.0f opponent=%.0f terminal=%.4f chain=%.4f chains=%d unknown_sales=%.0f",
                    episode, summary["money"], summary["opponent_money"], summary["terminal_reward"],
                    summary["chain_credit_sum"], summary["chain_count"], summary["unknown_source_sale_revenue"])
        if (episode-first+1) % args.episodes_per_update and episode != args.episodes:
            continue
        metrics = update_chain_policy(model, optimizer, pending, epochs=args.epochs,
            minibatch_size=args.minibatch_size, target_kl=args.target_kl, max_grad_norm=args.max_grad_norm,
            entropy_coef=args.entropy_coef, chain_value_coef=args.chain_value_coef,
            terminal_policy_coef=args.terminal_policy_coef, chain_policy_coef=args.chain_policy_coef)
        update += 1
        append_json(root/"metrics.jsonl", {"event": "update", "episode": episode, "update": update, **metrics})
        LOGGER.info("Update %d | %s", update, metrics)
        checkpoint = {"format": FORMAT, "config": config, "episode": episode, "update": update,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if args.device == "cuda" else []}
        torch.save(checkpoint, root/"latest.pt.tmp")
        (root/"latest.pt.tmp").replace(root/"latest.pt")
        pending.clear()
    LOGGER.info("Training completed")


if __name__ == "__main__":
    main()
