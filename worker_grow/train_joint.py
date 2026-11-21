"""Joint manager + crew PPO: shared games, separate rewards and optimizers.

Collect with BOTH policies fixed, then update each from its own decisions.
No gifted assets, frozen crew, or cross-agent gradient propagation.
"""
import argparse
import logging
from pathlib import Path
import random

import numpy as np
import torch

from worker_grow.crew import FeudalCrew, VERSION as CREW_VERSION
from worker_grow.manager import ManagerActor, MANAGER_VERSION
from worker_grow.manager_workload import ADMISSION_VERSION, WORKLOAD_NAMES
from worker_grow.production_paths import PATH_ITEMS, config_issue_items, parse_issue_items
from worker_grow.issue_outcomes import (
    IssueOutcomeLedger, VERSION as TASK_COST_VERSION, DEFAULT_WAIT_COST, DEFAULT_MISSED_CARE_COST,
)
from worker_grow.issue_completion import (
    SETTLEMENT_RULES, DEFAULT_FULFILL_REWARD, DEFAULT_UNMET_REWARD, assign_issue_throughput,
)
from worker_grow.contracts import TASK_LIFECYCLE_VERSION
from worker_grow.manager_training import (
    OPTIMIZER_VERSION, append_jsonl, update_manager, write_json,
)
from worker_grow.train_manager import collect
from worker_grow.training import update_crew

VERSION = "feudal-joint-issue-gold-dag-v9"


def parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=10000)
    p.add_argument("--episode-steps", type=int, default=720)
    p.add_argument("--episodes-per-update", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--minibatch-size", type=int, default=64, help="Manager minibatch size")
    p.add_argument("--sequence-length", type=int, default=16, help="Crew recurrent chunk length")
    p.add_argument("--manager-learning-rate", type=float, default=3e-5)
    p.add_argument("--crew-learning-rate", type=float, default=3e-5)
    p.add_argument("--entropy-coef", type=float, default=.02, help="Both policies")
    p.add_argument("--target-kl", type=float, default=.02)
    p.add_argument("--max-grad-norm", type=float, default=1.)
    p.add_argument("--gae-lambda", type=float, default=1., help="Manager terminal GAE only")
    p.add_argument("--terminal-policy-coef", type=float, default=1.)
    p.add_argument("--chain-policy-coef", type=float, default=1.)
    p.add_argument("--chain-value-coef", type=float, default=.5)
    p.add_argument("--issue-wait-cost", type=float, default=DEFAULT_WAIT_COST)
    p.add_argument("--missed-care-cost", type=float, default=DEFAULT_MISSED_CARE_COST)
    p.add_argument("--task-cost-settlement", choices=SETTLEMENT_RULES, default='unfulfilled_fraction')
    p.add_argument("--fulfill-reward", type=float, default=DEFAULT_FULFILL_REWARD)
    p.add_argument("--unmet-reward", type=float, default=DEFAULT_UNMET_REWARD)
    p.add_argument("--throughput-policy-coef", type=float, default=1.)
    p.add_argument("--throughput-value-coef", type=float, default=.5)
    p.add_argument("--credit-decay", type=float, default=.8, help="Crew delivery graph decay")
    p.add_argument("--save-every", type=int, default=10, help="Replay interval; traces always saved")
    p.add_argument("--opponent", choices=("starter", "random", "pass", "best_agent"), default="starter")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--init-manager", type=Path, help="Optional standalone manager checkpoint; weights only")
    p.add_argument("--init-crew", type=Path, help="Optional standalone crew checkpoint; weights only")
    p.add_argument("--issue-items", default=None,
                   help="Comma-separated TASK_ITEMS to grow on this path "
                        f"(one of {', '.join(PATH_ITEMS)}). Omit for all eight products")
    return p


def validate(args):
    for key in ("episodes", "episodes_per_update", "epochs", "minibatch_size", "sequence_length"):
        if getattr(args, key) < 1:
            raise ValueError(f"Invalid {key}")
    IssueOutcomeLedger(wait_cost=args.issue_wait_cost, missed_care_cost=args.missed_care_cost)
    assign_issue_throughput({}, {}, fulfill_reward=args.fulfill_reward, unmet_reward=args.unmet_reward)
    for key in ("manager_learning_rate", "crew_learning_rate", "target_kl", "max_grad_norm",
                "terminal_policy_coef", "chain_policy_coef", "chain_value_coef",
                "throughput_value_coef"):
        if not np.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f"Invalid {key}")
    if not np.isfinite(args.throughput_policy_coef) or args.throughput_policy_coef < 0:
        raise ValueError("Invalid throughput_policy_coef")
    if (args.episode_steps < 2 or args.save_every < 0 or not 0 <= args.gae_lambda <= 1
            or not 0 < args.credit_decay <= 1 or not np.isfinite(args.entropy_coef)
            or args.entropy_coef < 0):
        raise ValueError("Invalid horizon/reward/entropy configuration")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    # Resume uses the two models in the joint checkpoint, not initialization files.
    if not args.resume:
        for path in (args.init_manager, args.init_crew):
            if path is not None and not path.is_file():
                raise FileNotFoundError(path)
    args.issue_items = parse_issue_items(args.issue_items)


def update_both(manager, crew, manager_optimizer, crew_optimizer, episodes, args, directory, update):
    """Each episode is (manager rows, crew rows, summary). No replay across updates."""
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "episodes.json", [ep[2]["episode"] for ep in episodes])
    manager_metrics = update_manager(
        manager, manager_optimizer, [r for ep in episodes for r in ep[0]],
        epochs=args.epochs, minibatch_size=args.minibatch_size,
        entropy_coef=args.entropy_coef, target_kl=args.target_kl,
        max_grad_norm=args.max_grad_norm, terminal_policy_coef=args.terminal_policy_coef,
        chain_policy_coef=args.chain_policy_coef, chain_value_coef=args.chain_value_coef,
        throughput_policy_coef=args.throughput_policy_coef,
        throughput_value_coef=args.throughput_value_coef,
        batch_log=directory / "manager_batches.jsonl", update_id=update,
    )
    crew_metrics = update_crew(
        crew, crew_optimizer, [ep[1] for ep in episodes], epochs=args.epochs,
        sequence_length=args.sequence_length, entropy_coef=args.entropy_coef,
        target_kl=args.target_kl, max_grad_norm=args.max_grad_norm,
        batch_log=directory / "crew_batches.jsonl", update_id=update,
    )
    return {"event": "update", "update": update, "episode": episodes[-1][2]["episode"],
            "manager": manager_metrics, "crew": crew_metrics}


def _initialize(model, path, version, device):
    if path is None:
        return
    saved = torch.load(path, map_location=device, weights_only=False)
    if saved.get("version") != version:
        raise ValueError(f"Incompatible initialization checkpoint: {path}")
    model.load_state_dict(saved["model"], strict=True)


def run(args):
    validate(args)
    root = args.run_dir.resolve()
    args.run_dir = root
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise FileExistsError("Use a NEW joint run directory")
    if args.resume and not (root / "latest.pt").is_file():
        raise FileNotFoundError("Missing joint checkpoint")
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    manager, crew = ManagerActor().to(args.device), FeudalCrew().to(args.device)
    manager_optimizer = torch.optim.Adam(manager.parameters(), lr=args.manager_learning_rate)
    crew_optimizer = torch.optim.Adam(crew.parameters(), lr=args.crew_learning_rate)
    config = {key: str(value.resolve()) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(version=VERSION, manager_version=MANAGER_VERSION, crew_version=CREW_VERSION,
                  site_selection="crew", issue_matching="oldest_compatible",
                  gifted_curriculum=False,
                  reward="manager-issue-sale-dag-v2+completion-dag-cost-v2+issue-throughput-v1+crew-first-delivery-v1",
                  issue_split="equal_children_own_issue_plus_consumed_batches",
                  production_cost_basis="subsequent_harvested_units", unsold_weighting="tree")
    config.update(workload_features=list(WORKLOAD_NAMES), log_schema="issue-deltas-v2",
                  task_lifecycle=TASK_LIFECYCLE_VERSION, admission=ADMISSION_VERSION,
                  task_cost=TASK_COST_VERSION, optimizer=OPTIMIZER_VERSION,
                  issue_items=config_issue_items(args.issue_items))
    first, update = 1, 0
    if args.resume:
        saved = torch.load(root / "latest.pt", map_location=args.device, weights_only=False)
        if saved.get("version") != VERSION:
            raise ValueError("Not a compatible joint checkpoint")
        for key, value in config.items():
            if key not in ("episodes", "save_every", "resume", "init_manager", "init_crew"):
                if saved["config"].get(key) != value:
                    raise ValueError(f"Resume configuration mismatch: {key}")
        for key in ("init_manager", "init_crew"):
            config[key] = saved["config"].get(key)
        first, update = saved["episode"] + 1, saved["update"]
        if any(int(p.name.split("_")[-1]) >= first for p in (root / "episodes").glob("episode_*")):
            raise RuntimeError("Uncheckpointed episodes exist; archive them before resume")
        if any(int(p.name.split("_")[-1]) > update for p in (root / "updates").glob("update_*")):
            raise RuntimeError("Uncheckpointed updates exist; archive them before resume")
        manager.load_state_dict(saved["manager"], strict=True)
        crew.load_state_dict(saved["crew"], strict=True)
        manager_optimizer.load_state_dict(saved["manager_optimizer"])
        crew_optimizer.load_state_dict(saved["crew_optimizer"])
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if args.device == "cuda":
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
    else:
        _initialize(manager, args.init_manager, MANAGER_VERSION, args.device)
        _initialize(crew, args.init_crew, CREW_VERSION, args.device)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "config.json", config)
    # Private logger prevents leaking file handlers between repeated test invocations.
    logger = logging.getLogger(f"joint:{root}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handlers = [logging.FileHandler(root / "train.log"), logging.StreamHandler()]
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        logger.addHandler(handler)
    try:
        pending = []
        logger.info("Joint training | BOTH agents trainable | device=%s | opponent=%s | issue_items=%s",
                    args.device, args.opponent, args.issue_items or "ALL")
        for episode in range(first, args.episodes + 1):
            result = collect(manager, crew, args, episode,
                             root / "episodes" / f"episode_{episode:06d}", joint=True)
            pending.append(result)
            summary = result[2]
            append_jsonl(root / "metrics.jsonl", {k: v for k, v in summary.items() if k != "issues"})
            logger.info("Episode %d | money=%.1f rival=%.1f delivery=%.2f units=%d issues=%s task_credit=%.5f throughput=%.3f task_costs=%s fulfilled=%d unmet_units=%d",
                        episode, summary["money"], summary["opponent_money"], summary["crew_reward"],
                        summary["delivered_units"], summary["issue_statuses"],
                        summary['task_credit_sum'], summary['throughput_credit_sum'], summary['task_cost_counts'],
                        summary['fulfilled_issue_count'], summary['unmet_issue_units'])
            if len(pending) < args.episodes_per_update and episode != args.episodes:
                continue
            update += 1
            metrics = update_both(manager, crew, manager_optimizer, crew_optimizer, pending, args,
                                  root / "updates" / f"update_{update:06d}", update)
            append_jsonl(root / "metrics.jsonl", metrics)
            logger.info("Update %d | manager=%s | crew=%s", update, metrics["manager"], metrics["crew"])
            payload = {"version": VERSION, "config": config, "episode": episode, "update": update,
                       "manager": manager.state_dict(), "crew": crew.state_dict(),
                       "manager_optimizer": manager_optimizer.state_dict(),
                       "crew_optimizer": crew_optimizer.state_dict(),
                       "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                       "torch_rng": torch.get_rng_state(),
                       "cuda_rng": torch.cuda.get_rng_state_all() if args.device == "cuda" else []}
            torch.save(payload, root / "latest.pt.tmp")
            (root / "latest.pt.tmp").replace(root / "latest.pt")
            pending.clear()
        logger.info("Joint training completed")
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()


if __name__ == "__main__":
    run(parser().parse_args())
