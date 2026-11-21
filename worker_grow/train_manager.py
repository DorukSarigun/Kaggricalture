"""Train the economic manager. Crew weights are not updated here.

Every environment move and every PPO minibatch is appended to disk.
Does not write into production-DAG or crew run directories.
"""
import argparse
from copy import copy, deepcopy
import gzip
import json
import logging
from pathlib import Path
import random

import numpy as np
import torch
from kaggle_environments import make

from cash_accounting import CashLedger
from market_execution import trace_market_executions
from worker_grow.crew import CrewCursor, FeudalCrew, propose_crew_turn
from worker_grow.delivery import CrewDeliveryTracker
from worker_grow.issues import IssueBook
from worker_grow.manager import MANAGER_VERSION, ManagerActor, propose_manager_contracts, propose_manager_market
from worker_grow.manager_workload import ADMISSION_VERSION, WORKLOAD_NAMES
from worker_grow.production_paths import (
    PATH_ITEMS, config_issue_items, grammar_for_issue_items, parse_issue_items,
)
from worker_grow.issue_outcomes import (
    IssueOutcomeLedger, VERSION as TASK_COST_VERSION, DEFAULT_WAIT_COST, DEFAULT_MISSED_CARE_COST,
)
from worker_grow.issue_completion import (
    SETTLEMENT_RULES, assign_issue_throughput, DEFAULT_FULFILL_REWARD, DEFAULT_UNMET_REWARD,
)
from worker_grow.contracts import TASK_LIFECYCLE_VERSION
from worker_grow.assignments import decision_metadata
from worker_grow.schedule import STRATEGIES, WALK_POLICIES, resolve_walk_policy
from worker_grow.manager_training import (
    OPTIMIZER_VERSION, append_jsonl, decision_record, finish_manager_episode,
    update_manager, write_json,
)
from worker_grow.scripted_manager import best_agent_opponent

PASS = {"farmer": ["PASS"], "hands": [], "market": []}
OPPONENTS = ("starter", "random", "pass", "best_agent")


def _make_opponent(name, env, configuration):
    if name == "best_agent":
        return best_agent_opponent(configuration)
    return env.agents[name]


def collect(manager, crew, args, episode, directory, *, joint=False):
    if joint and (crew is None or not all(p.requires_grad for p in crew.parameters())):
        raise ValueError("Joint rollout requires a trainable crew")
    directory.mkdir(parents=True, exist_ok=False)
    configuration = {"episodeSteps": args.episode_steps, "turnsPerDay": 24,
                     "startingMoney": 3000, "seed": args.seed + episode - 1}
    env = make("kaggriculture", configuration=configuration)
    player = (episode - 1) % 2
    opponent = _make_opponent(args.opponent, env, configuration)
    # Training the crew and realizing manager revenue are independent choices.
    book = IssueBook(realized_sales=True)
    tracker = CrewDeliveryTracker(deepcopy(env.state[player].observation), book.task_book,
                                  episode=episode, decay=getattr(args, "credit_decay", .8))
    ledger = CashLedger([float(farm.money) for farm in env.state[player].observation.farms])
    task_costs = IssueOutcomeLedger(wait_cost=getattr(args, 'issue_wait_cost', DEFAULT_WAIT_COST),
                                   missed_care_cost=getattr(args, 'missed_care_cost', DEFAULT_MISSED_CARE_COST))
    rows, crew_state, next_id, turn = [], None, 0, 0
    crew_rows = []
    economic_refs = {}
    previous_issues = {}
    moves_path = directory / "moves.jsonl"
    all_moves = args.run_dir / "all_moves.jsonl"
    manager.eval()
    if crew is not None:
        crew.eval()
    trace = gzip.open(directory / "trace.jsonl.gz", "wt", encoding="utf-8")
    try:
        while not env.done:
            observations = [deepcopy(state.observation) for state in env.state]
            for obs in observations:
                obs["step"] = turn
            before = observations[player]
            issue_items = parse_issue_items(getattr(args, "issue_items", None))
            contracts, next_id = propose_manager_contracts(
                manager, before, book, step=turn, day=int(before["day"]), next_id=next_id,
                issue_items=issue_items,
            )
            new_crew_rows = []
            if crew is None:
                crew_action = {
                    "farmer": ["PASS"],
                    "hands": [["PASS"] for _ in before["farms"][before["player"]]["hands"]],
                    "market": [],
                }
            else:
                crew_action, new_crew_rows, crew_state = propose_crew_turn(
                    crew, before, book.task_book, crew_state,
                    assignment_strategy=getattr(args, "assignment_strategy", None),
                    walk_policy=getattr(args, "walk_policy", None),
                )
            shadow = CrewCursor(before, book.task_book)
            for unit in (crew_action["farmer"], *crew_action["hands"]):
                shadow.apply(unit)
            market_book = copy(book)
            market_book.task_book = shadow.tasks
            market, market_rows, next_id = propose_manager_market(
                manager, shadow.observation(), step=turn, next_id=next_id,
                book=market_book, grammar_config=grammar_for_issue_items(issue_items),
            )
            for row in contracts + market_rows:
                row.episode = episode
            action = {"farmer": crew_action["farmer"], "hands": crew_action["hands"], "market": market}
            actions = [None, None]
            actions[player] = action
            actions[1 - player] = opponent(observations[1 - player])
            executions = trace_market_executions(observations, actions, configuration)
            env.step(actions)
            if any(str(state.status) not in ("ACTIVE", "DONE") for state in env.state):
                raise RuntimeError("Environment failure")
            after = deepcopy(env.state[player].observation)
            after["step"] = turn + 1
            lifecycle_start = len(tracker.lifecycle.events)
            deliveries = tracker.observe_transition(
                before, action, after, market_executions=executions[player],
                decisions=new_crew_rows,
                market_decisions=market_rows,
            )
            ledger.observe(turn, executions,
                           [farm["money"] for farm in before["farms"]],
                           [farm["money"] for farm in after["farms"]])
            book.sync_from_tasks(book.task_book, step=turn, prices=before["market"]["prices"])
            task_events = task_costs.observe(before, after, book,
                                            tracker.lifecycle.events[lifecycle_start:])
            orders=[r for r in market_rows if not r.boundary and r.action and r.action[0]!='STOP']
            economic_refs.update({(turn,'market',i):2*r.decision_id for i,r in enumerate(orders)})
            current_issues = {key: issue.as_dict() for key, issue in book.issues.items()}
            changed_issues = {key: value for key, value in current_issues.items()
                              if previous_issues.get(key) != value}
            previous_issues = current_issues
            move = {
                "episode": episode, "step": turn, "player": player,
                "contracts": [decision_record(row) for row in contracts],
                "market": [decision_record(row) for row in market_rows],
                "crew_action": crew_action, "action": action,
                "open_issue_count": len(book.open_issues()),
                "issue_updates": changed_issues,
                "task_cost_events": task_events,
                "crew_decisions": [
                    {"decision_id": r.decision_id, "unit_index": r.unit_index,
                     "action": r.action, **decision_metadata(r)} for r in new_crew_rows
                ],
                "crew_size": 1 + len(before["farms"][player]["hands"]),
                "money": [farm["money"] for farm in after["farms"]],
            }
            append_jsonl(moves_path, move)
            # Root index, not a second copy of every large observation/decision.
            append_jsonl(all_moves, {"episode": episode, "step": turn,
                                    "path": str(moves_path.relative_to(args.run_dir)),
                                    "money": move["money"]})
            if trace:
                trace.write(json.dumps({"step": turn, "observation": before,
                    "next_observation": after, "action": action,
                    "manager_decision_ids": [r.decision_id for r in contracts + market_rows],
                    "crew_decision_ids": [r.decision_id for r in new_crew_rows],
                    "deliveries": deliveries}) + "\n")
                crew_rows.extend(new_crew_rows)
            rows.extend(contracts)
            rows.extend(market_rows)
            turn += 1
    finally:
        if trace:
            trace.close()
    delivery_report = tracker.ledger.finish()
    book.sync_from_tasks(book.task_book, step=turn, prices=env.state[player].observation["market"]["prices"])
    book.expire_open(step=turn, prices=env.state[player].observation["market"]["prices"])
    closed = [issue for issue in book.issues.values() if not issue.open]
    money, rival = ledger.final_money[player], ledger.final_money[1 - player]
    from worker_grow.issue_credit import assign_issue_credit
    credit_report=assign_issue_credit(tracker.lifecycle.events,ledger.entries,
                                      economic_refs,player,book.issues)
    for issue_id,gold in credit_report['issue_gold'].items():
        book.issues[issue_id].realized_gold=gold
    task_report = task_costs.finish(book, after, tracker.lifecycle.events,
                                   rule=getattr(args, 'task_cost_settlement', 'unfulfilled_fraction'))
    throughput_report = assign_issue_throughput(
        task_report['completion'], book.issues,
        fulfill_reward=getattr(args, 'fulfill_reward', DEFAULT_FULFILL_REWARD),
        unmet_reward=getattr(args, 'unmet_reward', DEFAULT_UNMET_REWARD),
    )
    # Linked typed graph: no task cost is inserted into material cash edges.
    credit_report['graph']['task_outcome_graph'] = task_report['graph']
    credit_report['graph']['issue_throughput'] = {
        k: v for k, v in throughput_report.items() if k != 'sinks'
    }
    write_json(directory/'issue_production_graph.json',credit_report['graph'])
    write_json(directory/'issue_sale_credit.json',{k:v for k,v in credit_report.items() if k!='graph'})
    write_json(directory / 'issue_task_cost.json', task_report)
    write_json(directory / 'issue_completion.json', task_report['completion'])
    write_json(directory / 'issue_throughput.json', throughput_report)
    finish_manager_episode(rows, money - rival, closed, gae_lambda=args.gae_lambda,
                           credit_report=credit_report, task_report=task_report,
                           throughput_report=throughput_report)
    summary = {
        "event": "episode", "episode": episode, "turns": turn, "decisions": len(rows),
        "money": money, "opponent_money": rival, "terminal_reward": money - rival,
        "issue_gold": sum(credit_report['issue_gold'].values()),
        "issues": {issue.task_id: issue.as_dict() for issue in book.issues.values()},
        "open_at_end": 0,
        "issue_count": len(book.issues),
        "reward_contract": "manager-issue-sale-dag-v2+completion-dag-cost-v2+issue-throughput-v1",
        "task_credit_sum": task_report['task_credit'],
        "raw_task_credit_sum": task_report['raw_task_credit'],
        "throughput_credit_sum": throughput_report['throughput_credit'],
        "fulfilled_issue_count": sum(c['demand_fulfilled'] for c in task_report['completion'].values()),
        "fulfilled_units": throughput_report['fulfilled_units'],
        "unmet_issue_units": sum(c['unmet'] for c in task_report['completion'].values()),
        "task_cost_counts": task_report['counts'],
    }
    from worker_grow.training import finish_episode
    finish_episode(crew_rows, delivery_report)
    summary.update({
            "crew_decisions": len(crew_rows),
            "crew_stochastic_decisions": sum(r.stochastic for r in crew_rows),
            "crew_reward": delivery_report["reward"],
            "delivered_units": sum(d["quantity"] for d in delivery_report["deliveries"]),
            "crew_operations": {op: sum(r.action[0] == op for r in crew_rows)
                                for op in sorted({r.action[0] for r in crew_rows})},
            "issue_statuses": {status: sum(i.status.value == status for i in closed)
                               for status in sorted({i.status.value for i in closed})},
            "economic_credit_sum": sum(credit_report['decision_credit'].values()),
            "economic_net_cash": credit_report['net_cash'],
            "economic_reconciliation_error": credit_report['reconciliation_error'],
            "sale_count": sum(a['kind']=='sale' for a in credit_report['allocations']),
            "gifted_curriculum": False,
    })
    write_json(directory / "delivery_credit.json", delivery_report)
    write_json(directory / "crew_decisions.json", [
            {"decision_id": r.decision_id, "step": r.step, "unit_index": r.unit_index,
             "action": r.action, "log_prob": r.log_prob, "value": r.value,
             "credit": r.credit, "advantage": r.credit - r.value,
             "stochastic": r.stochastic, "allowed_ops": r.allowed_ops,
             "allowed_items": r.allowed_items, "quantity_limit": r.quantity_limit,
             **decision_metadata(r)}
            for r in crew_rows
    ])
    write_json(directory / "summary.json", summary)
    write_json(directory / "manager_decisions.json", [decision_record(row) for row in rows])
    write_json(directory / "issues.json", {key: issue.as_dict() for key, issue in book.issues.items()})
    write_json(directory / "cash_ledger.json", ledger.to_dict())
    tracker.lifecycle.write_artifacts(directory)
    if args.save_every > 0 and episode % args.save_every == 0:
        with gzip.open(directory / "replay.json.gz", "wt", encoding="utf-8") as file:
            json.dump(env.toJSON(), file)
    return (rows, crew_rows, summary) if joint else (rows, summary)


class _Help(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass


def parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=_Help,
        epilog=(
            "Example:\n"
            "  .venv/bin/python -m worker_grow.train_manager \\\n"
            "    --run-dir runs/feudal_manager_cuda_01 \\\n"
            "    --episodes 10000 --episode-steps 720 --episodes-per-update 4 \\\n"
            "    --epochs 3 --minibatch-size 64 --learning-rate 3e-5 \\\n"
            "    --gae-lambda 1.0 --terminal-policy-coef 1.0 --chain-policy-coef 1.0 \\\n"
            "    --chain-value-coef 0.5 --entropy-coef 0.02 --target-kl 0.02 \\\n"
            "    --max-grad-norm 1.0 --save-every 10 --crew pass --opponent starter \\\n"
            "    --device cuda --seed 2026\n"
        ),
    )
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Empty directory for this run; --resume to continue an existing one")
    p.add_argument("--episodes", type=int, default=10000,
                   help="Last episode index to train through (inclusive)")
    p.add_argument("--episode-steps", type=int, default=720,
                   help="Full Kaggriculture match length (30 days × 24 turns)")
    p.add_argument("--episodes-per-update", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--minibatch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--gae-lambda", type=float, default=1.0,
                   help="Terminal GAE λ; 1.0 is Monte Carlo on (own − rival) gold")
    p.add_argument("--terminal-policy-coef", type=float, default=1.0)
    p.add_argument("--chain-policy-coef", type=float, default=1.0,
                   help="Policy weight on realized DAG credit plus task costs (separate terminal target)")
    p.add_argument("--issue-wait-cost", type=float, default=DEFAULT_WAIT_COST,
                   help="Manager reward cost per unstarted issue per turn; 0 disables")
    p.add_argument("--missed-care-cost", type=float, default=DEFAULT_MISSED_CARE_COST,
                   help="Manager reward cost per missed daily WATER/FEED; 0 disables")
    p.add_argument("--task-cost-settlement", choices=SETTLEMENT_RULES, default='unfulfilled_fraction',
                   help="Scale task costs by unmet quota fraction, or preserve their full budget")
    p.add_argument("--fulfill-reward", type=float, default=DEFAULT_FULFILL_REWARD,
                   help="ISSUE throughput credit per fulfilled quota unit; 0 disables")
    p.add_argument("--unmet-reward", type=float, default=DEFAULT_UNMET_REWARD,
                   help="ISSUE throughput penalty per unmet quota unit; 0 disables")
    p.add_argument("--throughput-policy-coef", type=float, default=1.0,
                   help="Policy weight on fulfill/unmet ISSUE credit (separate from cash and task cost)")
    p.add_argument("--throughput-value-coef", type=float, default=0.5)
    p.add_argument("--chain-value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.02)
    p.add_argument("--target-kl", type=float, default=0.02)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--save-every", type=int, default=10,
                   help="Write replay.json.gz every N episodes; 0 disables replay. "
                        "moves.jsonl, decisions, issues, and every PPO batch are always written")
    p.add_argument("--crew", choices=("pass", "feudal", "scripted"), default="pass",
                   help="pass: no unit work. feudal: frozen learned crew. "
                        "scripted: frozen FeudalCrew body with a daily-budget planner")
    p.add_argument("--crew-checkpoint", type=Path, default=None,
                   help="Optional latest.pt for --crew feudal/scripted")
    p.add_argument("--assignment-strategy", choices=STRATEGIES, default=None,
                   help="Crew subtask chooser. Default: learned with feudal, tour with scripted")
    p.add_argument("--walk-policy", choices=WALK_POLICIES, default=None,
                   help="How the crew executes a chosen subtask. Default follows assignment")
    p.add_argument("--opponent", choices=OPPONENTS, default="starter",
                   help="Rival policy. best_agent is the third_party chassis, not a learned manager")
    p.add_argument("--init-manager", type=Path, default=None,
                   help="Load manager weights into a NEW run directory; not used with --resume")
    p.add_argument("--issue-items", default=None,
                   help="Comma-separated TASK_ITEMS to grow on this path "
                        f"(one of {', '.join(PATH_ITEMS)}). Omit for all eight products")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--resume", action="store_true")
    return p


def _validate(args):
    for name in ("episodes", "episodes_per_update", "epochs", "minibatch_size"):
        if getattr(args, name) < 1:
            raise ValueError(f"Invalid {name}")
    IssueOutcomeLedger(wait_cost=args.issue_wait_cost, missed_care_cost=args.missed_care_cost)
    assign_issue_throughput({}, {}, fulfill_reward=args.fulfill_reward, unmet_reward=args.unmet_reward)
    if args.episode_steps < 2 or not 0 <= args.gae_lambda <= 1 or args.save_every < 0:
        raise ValueError("Invalid horizon/lambda/save")
    for name in ("learning_rate", "target_kl", "max_grad_norm",
                 "terminal_policy_coef", "chain_policy_coef", "chain_value_coef",
                 "throughput_policy_coef", "throughput_value_coef"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
        if name not in ("throughput_policy_coef",) and getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(args.entropy_coef) or args.entropy_coef < 0:
        raise ValueError("Invalid entropy coefficient")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.crew == "pass":
        if args.assignment_strategy is not None or args.walk_policy is not None:
            raise ValueError("assignment-strategy/walk-policy require a crew")
    elif args.crew == "scripted":
        args.assignment_strategy = args.assignment_strategy or "tour"
        if args.assignment_strategy == "learned":
            raise ValueError("scripted crew cannot use the learned assignment head")
    else:
        args.assignment_strategy = args.assignment_strategy or "learned"
    if args.crew != "pass":
        args.walk_policy = resolve_walk_policy(args.assignment_strategy, args.walk_policy)
    if args.crew != "pass" and args.crew_checkpoint is not None and not args.crew_checkpoint.exists():
        raise FileNotFoundError(args.crew_checkpoint)
    if args.resume and args.init_manager is not None:
        raise ValueError("Use --resume on the same run, or --init-manager into a NEW directory")
    if args.init_manager is not None and not args.init_manager.is_file():
        raise FileNotFoundError(args.init_manager)
    args.issue_items = parse_issue_items(args.issue_items)


def main():
    args = parser().parse_args()
    _validate(args)
    root = args.run_dir.resolve()
    args.run_dir = root
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise FileExistsError("Use a NEW manager run directory")
    if args.resume and not (root / "latest.pt").exists():
        raise FileNotFoundError("No resume checkpoint")
    root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        handlers=[logging.FileHandler(root / "train.log"), logging.StreamHandler()])
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    manager = ManagerActor().to(args.device)
    optimizer = torch.optim.Adam(manager.parameters(), lr=args.learning_rate)
    crew = None
    if args.crew in ("feudal", "scripted"):
        crew = FeudalCrew().to(args.device)
        if args.crew_checkpoint is not None:
            saved_crew = torch.load(args.crew_checkpoint, map_location=args.device, weights_only=False)
            crew.load_state_dict(saved_crew["model"], strict=True)
        crew.eval()
        for parameter in crew.parameters():
            parameter.requires_grad_(False)
    config = {**vars(args), "run_dir": str(root),
              "crew_checkpoint": str(args.crew_checkpoint or ""),
              "init_manager": str(args.init_manager or ""),
              "version": MANAGER_VERSION, "crew_updated": False,
              "assignment_strategy": args.assignment_strategy, "walk_policy": args.walk_policy,
              "reward": "manager-issue-sale-dag-v2+completion-dag-cost-v2+issue-throughput-v1", "unsold_weighting": "tree",
              "producer_lifetime": True, "workload_features": list(WORKLOAD_NAMES),
              "task_lifecycle": TASK_LIFECYCLE_VERSION,
              "admission": ADMISSION_VERSION, "task_cost": TASK_COST_VERSION,
              "log_schema": "issue-deltas-v2", "optimizer": OPTIMIZER_VERSION,
              "vacuous_issue_list_price": False, "unfilled_market_list_price": False,
              "issue_items": config_issue_items(args.issue_items)}
    logging.info("Manager training | crew=%s assignment=%s walk=%s opponent=%s device=%s issue_items=%s",
                 args.crew, args.assignment_strategy, args.walk_policy, args.opponent, args.device,
                 args.issue_items or "ALL")
    first, update = 1, 0
    if args.resume:
        saved = torch.load(root / "latest.pt", map_location=args.device, weights_only=False)
        if saved.get("version") != MANAGER_VERSION:
            raise ValueError("Incompatible manager checkpoint")
        for name, value in config.items():
            if name not in ("episodes", "save_every", "resume", "init_manager") and saved["config"].get(name) != value:
                raise ValueError(f"Resume config mismatch: {name}")
        manager.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if args.device == "cuda" and saved.get("cuda_rng"):
            torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng"]])
        first, update = saved["episode"] + 1, saved["update"]
        if any(int(p.name.split("_")[-1]) >= first for p in (root / "episodes").glob("episode_*")):
            raise RuntimeError("Uncheckpointed episode artifacts exist; archive before resume")
    else:
        if args.init_manager is not None:
            started = torch.load(args.init_manager, map_location=args.device, weights_only=False)
            if started.get("version") != MANAGER_VERSION:
                raise ValueError("Incompatible manager checkpoint")
            manager.load_state_dict(started["model"], strict=True)
        write_json(root / "config.json", config)
    pending = []
    for episode in range(first, args.episodes + 1):
        rows, summary = collect(manager, crew, args, episode, root / "episodes" / f"episode_{episode:06d}")
        pending.extend(rows)
        append_jsonl(root / "metrics.jsonl", {k: v for k, v in summary.items() if k != "issues"})
        logging.info("Manager episode %d | money=%.1f rival=%.1f issues=%d sales=%d work=%s cash_error=%.3g task_credit=%.5f throughput=%.3f task_costs=%s fulfilled=%d unmet_units=%d",
                     episode, summary["money"], summary["opponent_money"], len(summary["issues"]),
                     summary["sale_count"], summary["crew_operations"], summary["economic_reconciliation_error"],
                     summary['task_credit_sum'], summary['throughput_credit_sum'], summary['task_cost_counts'],
                     summary['fulfilled_issue_count'], summary['unmet_issue_units'])
        if episode % args.episodes_per_update and episode != args.episodes:
            continue
        update += 1
        batch_log = root / "updates" / f"update_{update:06d}" / "batches.jsonl"
        metrics = update_manager(
            manager, optimizer, pending, epochs=args.epochs, minibatch_size=args.minibatch_size,
            entropy_coef=args.entropy_coef, target_kl=args.target_kl,
            max_grad_norm=args.max_grad_norm, batch_log=batch_log, update_id=update,
            terminal_policy_coef=args.terminal_policy_coef,
            chain_policy_coef=args.chain_policy_coef,
            chain_value_coef=args.chain_value_coef,
            throughput_policy_coef=args.throughput_policy_coef,
            throughput_value_coef=args.throughput_value_coef,
        )
        pending.clear()
        append_jsonl(root / "metrics.jsonl", {"event": "update", "update": update, **metrics})
        logging.info("Manager update %d | %s", update, metrics)
        payload = {
            "version": MANAGER_VERSION, "config": config, "model": manager.state_dict(),
            "optimizer": optimizer.state_dict(), "episode": episode, "update": update,
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
        }
        if args.device == "cuda":
            payload["cuda_rng"] = torch.cuda.get_rng_state_all()
        torch.save(payload, root / "latest.pt.tmp")
        (root / "latest.pt.tmp").replace(root / "latest.pt")


if __name__ == "__main__":
    main()
