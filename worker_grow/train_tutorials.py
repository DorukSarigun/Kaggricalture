"""Separate crew tutorial pretrain. Does not change train_crew / best_agent."""
import argparse
from copy import deepcopy
import gzip
import json
import logging
from pathlib import Path
import random
import numpy as np
import torch
from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as game

from market_execution import trace_market_executions
from worker_grow.crew import FeudalCrew, VERSION, propose_crew_turn
from worker_grow.delivery import CrewDeliveryTracker
from worker_grow.train_crew import (
    PASS, initialize_weights, prepare_run_dir, write_json,
)
from worker_grow.training import finish_episode, update_crew
from worker_grow.tutorials import (
    SLICES, attach_gifted_inventory, horizon, parse_slices, setup, slice_for_episode,
)


def collect(model, args, episode, directory):
    directory.mkdir(parents=True, exist_ok=False)
    slices = parse_slices(args.tutorial_slices)
    slice_name = slice_for_episode(episode, slices, args.slice_episodes)
    steps = horizon(slice_name, args.episode_steps)
    configuration = {
        "episodeSteps": steps, "turnsPerDay": 24,
        "seed": args.seed + episode, "weedSpawnChance": 0.,
    }
    env = make("kaggriculture", configuration=configuration)
    book = setup(env, slice_name, item=args.item, quantity=args.quantity)
    tasks = book.task_book
    tracker = CrewDeliveryTracker(
        deepcopy(env.state[0].observation), tasks, episode=episode,
        decay=args.credit_decay, grow_reward=args.grow_event_reward,
        delivery_bonus=args.delivery_bonus,
    )
    if slice_name == "drop":
        attach_gifted_inventory(tracker, book, args.item)
    rows = []
    state = None
    turn = 0
    save = args.save_every > 0 and episode % args.save_every == 0
    trace = gzip.open(directory / "trace.jsonl.gz", "wt", encoding="utf8") if save else None
    model.eval()
    try:
        while not env.done:
            observations = [deepcopy(s.observation) for s in env.state]
            for obs in observations:
                obs["step"] = turn
            before = observations[0]
            action, new_rows, state = propose_crew_turn(model, before, tasks, state)
            action = dict(action)
            action["market"] = []
            actions = [action, deepcopy(PASS)]
            executions = trace_market_executions(observations, actions, configuration)
            env.step(actions)
            if any(str(s.status) not in ("ACTIVE", "DONE") for s in env.state):
                raise RuntimeError("Environment failure")
            after = deepcopy(env.state[0].observation)
            after["step"] = turn + 1
            deliveries = tracker.observe_transition(
                before, action, after, market_executions=executions[0], decisions=new_rows,
            )
            book.sync_from_tasks(book.task_book, step=turn, prices=before["market"]["prices"])
            if trace:
                trace.write(json.dumps({
                    "observation": before, "action": action, "next_observation": after,
                    "decision_ids": [r.decision_id for r in new_rows],
                    "deliveries": deliveries, "tutorial_slice": slice_name,
                }) + "\n")
            rows.extend(new_rows)
            turn += 1
    finally:
        if trace:
            trace.close()
    report = tracker.ledger.finish()
    finish_episode(rows, report)
    book.sync_from_tasks(book.task_book, step=turn, prices=env.state[0].observation["market"]["prices"])
    book.expire_open(step=turn, prices=env.state[0].observation["market"]["prices"])
    summary = {
        "event": "episode", "episode": episode, "turns": turn, "decisions": len(rows),
        "crew_reward": report["reward"],
        "delivery_reward": report.get("delivery_reward", report["reward"]),
        "grow_reward": report.get("grow_reward", 0.),
        "grow_events": len(report.get("progress") or ()),
        "deliveries": len(report["deliveries"]),
        "delivered_units": sum(d["quantity"] for d in report["deliveries"]),
        "stage": "tutorials", "tutorial_slice": slice_name, "gifted_curriculum": True,
        "tasks": {k: v["status"] for k, v in report["tasks"].items()},
    }
    write_json(directory / "delivery_credit.json", report)
    write_json(directory / "summary.json", summary)
    write_json(directory / "decisions.json", [{
        "decision_id": r.decision_id, "step": r.step, "unit_index": r.unit_index,
        "action": r.action, "log_prob": r.log_prob, "value": r.value, "credit": r.credit,
        "advantage": r.credit - r.value, "allowed_ops": r.allowed_ops,
    } for r in rows])
    tracker.lifecycle.write_artifacts(directory)
    if save:
        with gzip.open(directory / "replay.json.gz", "wt", encoding="utf8") as f:
            json.dump(env.toJSON(), f)
    return rows, summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--episode-steps", type=int, default=None, help="Override per-slice horizon")
    p.add_argument("--episodes-per-update", type=int, default=4)
    p.add_argument("--tutorial-slices", default=",".join(SLICES))
    p.add_argument("--slice-episodes", type=int, default=20)
    p.add_argument("--item", choices=tuple(game.CROPS), default="WHEAT")
    p.add_argument("--quantity", type=int, default=1)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--sequence-length", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--credit-decay", type=float, default=.8)
    p.add_argument("--grow-event-reward", type=float, default=1.)
    p.add_argument("--delivery-bonus", type=float, default=2.,
                   help="Multiplier on first-quota shed delivery; grow events stay 1x")
    p.add_argument("--entropy-coef", type=float, default=.01)
    p.add_argument("--max-grad-norm", type=float, default=1.)
    p.add_argument("--target-kl", type=float, default=.02)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--init-crew", type=Path, help="Optional crew checkpoint; weights only")
    return p


def main():
    args = parser().parse_args()
    slices = parse_slices(args.tutorial_slices)
    if args.slice_episodes < 1:
        raise ValueError("Invalid slice-episodes")
    for name in ("episodes", "episodes_per_update", "quantity", "epochs", "sequence_length"):
        if getattr(args, name) < 1:
            raise ValueError(f"Invalid {name}")
    if args.episode_steps is not None and args.episode_steps < 2:
        raise ValueError("Invalid episode-steps")
    if args.save_every < 0 or not 0 < args.credit_decay <= 1:
        raise ValueError("Invalid tutorial run configuration")
    if not np.isfinite(args.grow_event_reward) or args.grow_event_reward < 0:
        raise ValueError("Invalid grow event reward")
    if not np.isfinite(args.delivery_bonus) or args.delivery_bonus <= 0:
        raise ValueError("Invalid delivery bonus")
    for name in ("learning_rate", "max_grad_norm", "target_kl", "entropy_coef"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"Invalid {name}")
    if min(args.learning_rate, args.max_grad_norm, args.target_kl) <= 0:
        raise ValueError("Zero optimizer parameter")
    if args.init_crew is not None and not args.init_crew.is_file():
        raise FileNotFoundError(args.init_crew)
    if args.resume and args.init_crew is not None:
        raise ValueError("Cannot init-crew and resume")
    root = prepare_run_dir(args.run_dir, resume=args.resume, overwrite=args.overwrite)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        handlers=[logging.FileHandler(root / "train.log"), logging.StreamHandler()])
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    model = FeudalCrew().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    config = {
        **vars(args), "run_dir": str(root), "version": VERSION,
        "mode": "tutorials", "reward": "grow-progress+first-delivery-v1",
        "site_selection": "crew", "slices": list(slices),
    }
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    first, update = 1, 0
    if args.resume:
        saved = torch.load(root / "latest.pt", map_location=args.device, weights_only=False)
        if saved.get("version") != VERSION:
            raise ValueError("Incompatible crew checkpoint")
        legacy = {"delivery_bonus": 1.}
        for k, v in config.items():
            if k in ("episodes", "save_every", "resume", "overwrite", "init_crew", "delivery_bonus"):
                continue
            if saved["config"].get(k) != v:
                raise ValueError(f"Resume mismatch: {k}")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
        if args.device == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng"]])
        first = saved["episode"] + 1
        update = saved["update"]
        if any(int(p.name.split("_")[-1]) >= first for p in (root / "episodes").glob("episode_*")):
            raise RuntimeError("Uncheckpointed episodes exist; archive them before resume")
        write_json(root / "config.json", config)
    else:
        initialize_weights(model, args.init_crew, VERSION, args.device)
        write_json(root / "config.json", config)
    pending = []
    for episode in range(first, args.episodes + 1):
        rows, summary = collect(model, args, episode, root / "episodes" / f"episode_{episode:06d}")
        pending.append(rows)
        with (root / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(summary) + "\n")
        logging.info(
            "Tutorial episode %d | slice=%s delivery_reward=%.3f grow_reward=%.3f delivered_units=%d grow_events=%d tasks=%s",
            episode, summary["tutorial_slice"], summary["delivery_reward"], summary["grow_reward"],
            summary["delivered_units"], summary["grow_events"], summary["tasks"],
        )
        if len(pending) < args.episodes_per_update and episode != args.episodes:
            continue
        metrics = update_crew(
            model, optimizer, pending, epochs=args.epochs, sequence_length=args.sequence_length,
            entropy_coef=args.entropy_coef, max_grad_norm=args.max_grad_norm, target_kl=args.target_kl,
        )
        update += 1
        pending.clear()
        with (root / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"event": "update", "update": update, **metrics}) + "\n")
        logging.info("Tutorial update %d | %s", update, metrics)
        torch.save({
            "version": VERSION, "config": config, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "episode": episode, "update": update,
            "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all() if args.device == "cuda" else [],
        }, root / "latest.pt.tmp")
        (root / "latest.pt.tmp").replace(root / "latest.pt")
    logging.info("Tutorial pretrain completed")


if __name__ == "__main__":
    main()
