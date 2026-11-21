import json
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as game

from cash_accounting import CashEntry
from observation_encoding import encode_observation
from worker_grow.contracts import TaskStatus
from worker_grow.issues import Issue, IssueBook, crew_size, issue_close_gold
from worker_grow.manager import (
    ManagerActor, ManagerDecision, propose_manager_contracts, propose_manager_market,
)
from worker_grow.manager_training import (
    apply_issue_rewards, apply_market_cash, apply_unfilled_market, finish_manager_episode,
    scaled_mse, stream_unit, update_manager,
)
from worker_grow.train_manager import collect, parser


def observation(hands=0):
    obs = {
        "player": 0,
        "step": 0,
        "day": 0,
        "hour": 0,
        "farms": [game._new_farm(10, 3000), game._new_farm(10, 3000)],
        "private": game._new_private(),
        "market": game._new_market(),
        "town": game._new_town(),
    }
    obs["farms"][0]["hands"] = [[0, 0] for _ in range(hands)]
    obs["private"]["inventories"] = [{} for _ in range(hands + 1)]
    return obs


class IssueBookTests(unittest.TestCase):
    def test_open_issues_are_not_capped_by_crew_size(self):
        obs = observation(hands=0)
        self.assertEqual(crew_size(obs), 1)
        book = IssueBook()
        for index, item in enumerate(("WHEAT", "CARROT", "MELON")):
            book.issue(obs, item=item, tile_xy=(index, 0), decision_id=index, step=0, day=0)
        self.assertEqual(len(book.open_issues()), 3)
        first = next(iter(book.open_issues()))
        book.close(first.task_id, TaskStatus.COMPLETE, step=3, prices={"WHEAT": 25},
                   delivered={"WHEAT": 1})
        self.assertEqual(len(book.open_issues()), 2)
        expired = book.expire_open(step=8, prices={"CARROT": 35, "MELON": 80})
        self.assertEqual(len(expired), 2)
        self.assertTrue(all(issue.status is TaskStatus.EXPIRED for issue in expired))

    def test_status_rewards_use_seed_cost_and_current_price(self):
        self.assertEqual(issue_close_gold(TaskStatus.FAILED, 10, {}, {"WHEAT": 25}), -10)
        self.assertEqual(issue_close_gold(TaskStatus.COMPLETE, 10, {"WHEAT": 1}, {"WHEAT": 25}), 15)
        self.assertEqual(issue_close_gold(TaskStatus.EXPIRED, 10, {"WHEAT": 0}, {"WHEAT": 25}), -10)
        self.assertEqual(issue_close_gold(TaskStatus.EXPIRED, 10, {"WHEAT": 1}, {"WHEAT": 25}), 15)

    def test_unprovisioned_failure_uses_catalog_input_price(self):
        self.assertEqual(
            issue_close_gold(TaskStatus.EXPIRED, 0, {}, {"WHEAT": 25}, item="WHEAT"),
            -game.CROPS["WHEAT"]["seed"],
        )
        self.assertEqual(
            issue_close_gold(TaskStatus.FAILED, 0, {}, {"MILK": 40}, item="COW"),
            -game.ANIMALS["COW"]["cost"],
        )
        book = IssueBook()
        issue = book.issue(observation(), item="WHEAT", decision_id=0, step=0, day=0)
        book.close(issue.task_id, TaskStatus.EXPIRED, step=8, prices={"WHEAT": 25})
        self.assertAlmostEqual(issue.close_gold, -game.CROPS["WHEAT"]["seed"])

    def test_provision_fifo_attaches_actual_buy_gold(self):
        book = IssueBook()
        obs = observation()
        book.observe_cash([
            CashEntry(0, 0, 0, "BUY_SEED", "WHEAT", 2, -24.0),
        ], player=0)
        issue = book.issue(obs, item="WHEAT", tile_xy=(0, 0), decision_id=0, step=1, day=0)
        self.assertAlmostEqual(issue.provision_cost, 12.0)
        leftover = book.pool.take("WHEAT", 1)
        self.assertAlmostEqual(leftover, 12.0)

    def test_issue_fields_are_complete(self):
        issue = Issue("issue:1", (3, 2), "WHEAT", (("WHEAT", 1),), 7, 4, 0)
        data = issue.as_dict()
        for key in ("task_id", "tile_xy", "item", "quotas", "decision_id", "opened_step",
                    "opened_day", "status", "producer_id", "delivered", "provision_cost",
                    "provision_lots", "closed_step", "close_gold"):
            self.assertIn(key, data)


class ManagerRewardTests(unittest.TestCase):
    def test_issue_and_market_credits_go_to_the_issuing_decision(self):
        rows = [
            ManagerDecision(None, None, 0.0, kind="contract", action=["ISSUE", "WHEAT", 0, 0],
                            decision_id=0, step=0),
            ManagerDecision(None, None, 0.0, kind="market", action=["BUY_SEED", "WHEAT", 1],
                            decision_id=1, step=0),
            ManagerDecision(None, None, 0.1, kind="boundary", boundary=True, decision_id=2, step=1),
        ]
        book = IssueBook()
        issue = book.add(Issue("issue:1", (0, 0), "WHEAT", (("WHEAT", 1),), 0, 0, 0,
                               provision_cost=10), observation())
        book.close("issue:1", TaskStatus.FAILED, step=2, prices={"WHEAT": 25})
        apply_issue_rewards(rows, [issue])
        apply_market_cash(rows, [CashEntry(0, 0, 0, "BUY_SEED", "WHEAT", 1, -10.0)], 0, 0)
        finish_manager_episode(rows, 100, [])
        self.assertAlmostEqual(rows[0].local_credit, -10)
        self.assertAlmostEqual(rows[1].local_credit, -10)
        self.assertAlmostEqual(rows[-1].reward, 100)
        self.assertTrue(rows[-1].done)

    def test_unfilled_buy_is_charged_catalog_price(self):
        from market_execution import MarketOrderExecution
        rows = [
            ManagerDecision(None, None, 0.0, kind="market", action=["BUY_SEED", "WHEAT", 3],
                            decision_id=1, step=0),
        ]
        executions = [MarketOrderExecution(
            player=0, order_index=0, operation="BUY_SEED", item="WHEAT",
            requested_quantity=3, executed_quantity=1, unit_prices=(10.0,),
        )]
        apply_unfilled_market(rows, executions, 0, 0, {"WHEAT": 25})
        self.assertAlmostEqual(rows[0].local_credit, -20)
        missed = [
            ManagerDecision(None, None, 0.0, kind="market", action=["BUY_ANIMAL", "COW", 1],
                            decision_id=2, step=1),
        ]
        apply_unfilled_market(missed, (), 0, 1, {})
        self.assertAlmostEqual(missed[0].local_credit, -game.ANIMALS["COW"]["cost"])


class ManagerPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(3)

    def test_contract_and_market_use_opponent_aware_encoder(self):
        model = ManagerActor()
        obs = observation()
        book = IssueBook()
        rows, next_id = propose_manager_contracts(model, obs, book, step=0, day=0)
        self.assertGreaterEqual(len(rows), 1)
        self.assertIn(rows[0].action[0], ("ISSUE", "STOP"))
        if rows[0].action[0] == "ISSUE":
            self.assertGreaterEqual(len(book.open_issues()), 1)
        market, market_rows, _ = propose_manager_market(model, obs, step=0, next_id=next_id)
        self.assertTrue(market_rows[-1].boundary)
        self.assertTrue(all(
            row.boundary or row.action[0] in (
                "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL", "HIRE", "BUY_LAND", "STOP",
            )
            for row in market_rows
        ))
        changed = deepcopy(obs)
        changed["farms"][1]["money"] = 1
        changed["farms"][1]["farmer"] = [9, 9]
        ctx_a, _ = model.contract_context([encode_observation(obs)])
        ctx_b, _ = model.contract_context([encode_observation(changed)])
        self.assertFalse(torch.allclose(ctx_a, ctx_b))

    def test_update_writes_every_batch(self):
        model = ManagerActor()
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
        obs = observation()
        rows, next_id = propose_manager_contracts(model, obs, IssueBook(), step=0, day=0)
        _, market_rows, _ = propose_manager_market(model, obs, step=0, next_id=next_id)
        rows.extend(market_rows)
        finish_manager_episode(rows, -30, [])
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "batches.jsonl"
            metrics = update_manager(model, optimizer, rows, epochs=1, minibatch_size=8,
                                     batch_log=log, update_id=1)
            lines = log.read_text(encoding="utf-8").strip().splitlines()
            self.assertGreaterEqual(len(lines), 1)
            payload = json.loads(lines[0])
            self.assertEqual(payload["event"], "batch")
            self.assertIn("decisions", payload)
            self.assertGreaterEqual(metrics["minibatches"], 1)

    def test_gold_stream_units_keep_logged_advantages_and_bound_value_loss(self):
        self.assertAlmostEqual(stream_unit([-160000.0] * 8), 160000.0)
        self.assertEqual(stream_unit([]), 1.0)
        pred = torch.zeros(4)
        target = torch.full((4,), -160000.0)
        self.assertAlmostEqual(float(scaled_mse(pred, target)), 0.5)
        model = ManagerActor()
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
        obs = observation()
        rows, next_id = propose_manager_contracts(model, obs, IssueBook(), step=0, day=0)
        _, market_rows, _ = propose_manager_market(model, obs, step=0, next_id=next_id)
        rows.extend(market_rows)
        for row in rows:
            if row.kind == "market" and row.action and row.action[0].startswith("BUY"):
                row.local_credit = -1200.0
                break
        finish_manager_episode(rows, -160000.0, [])
        self.assertEqual(rows[-1].reward, -160000.0)
        self.assertLess(rows[-1].advantage, -1000.0)
        self.assertTrue(any(row.chain_advantage < -100 for row in rows))
        metrics = update_manager(model, optimizer, rows, epochs=1, minibatch_size=8)
        self.assertLess(metrics["terminal_value_loss"], 10)
        self.assertLess(metrics["loss"], 100)
        self.assertGreater(metrics["terminal_adv_unit"], 1)


class ManagerCollectTests(unittest.TestCase):
    def test_parser_defaults_are_full_match_cuda(self):
        args = parser().parse_args(["--run-dir", "unused"])
        self.assertEqual(args.episodes, 10000)
        self.assertEqual(args.episode_steps, 720)
        self.assertEqual(args.episodes_per_update, 4)
        self.assertEqual(args.epochs, 3)
        self.assertEqual(args.minibatch_size, 64)
        self.assertEqual(args.device, "cuda")
        self.assertEqual(args.save_every, 10)
        self.assertEqual(args.opponent, "starter")
        self.assertEqual(args.crew, "pass")

    def test_scripted_crew_defaults_to_tour_greedy(self):
        from worker_grow.train_manager import _validate
        args = parser().parse_args([
            "--run-dir", "unused", "--crew", "scripted", "--device", "cpu",
        ])
        _validate(args)
        self.assertEqual(args.assignment_strategy, "tour")
        self.assertEqual(args.walk_policy, "greedy")
        with self.assertRaises(ValueError):
            bad = parser().parse_args([
                "--run-dir", "unused", "--crew", "scripted",
                "--assignment-strategy", "learned", "--device", "cpu",
            ])
            _validate(bad)

    def test_collect_scripted_crew_manager_does_not_command_units(self):
        from worker_grow.crew import FeudalCrew
        from worker_grow.train_manager import _validate
        torch.set_num_threads(1)
        torch.manual_seed(1)
        manager = ManagerActor()
        crew = FeudalCrew()
        crew.eval()
        for parameter in crew.parameters():
            parameter.requires_grad_(False)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = parser().parse_args([
                "--run-dir", str(root), "--episodes", "1", "--episode-steps", "4",
                "--save-every", "0", "--crew", "scripted", "--opponent", "pass",
                "--device", "cpu",
            ])
            _validate(args)
            args.run_dir = root
            rows, summary = collect(manager, crew, args, 1, root / "episodes" / "episode_000001")
            self.assertGreaterEqual(summary["turns"], 2)
            self.assertTrue(rows[-1].done)
            self.assertTrue(all(row.kind in ("contract", "market", "boundary") for row in rows))
            self.assertTrue(all(
                row.kind != "contract" or row.action[0] in ("ISSUE", "STOP") for row in rows
            ))
            self.assertTrue(all(
                row.kind != "market" or not row.action or row.action[0] in (
                    "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL", "HIRE", "BUY_LAND", "STOP",
                )
                for row in rows
            ))
            for row in rows:
                if row.kind == "contract" and row.action and row.action[0] == "ISSUE":
                    self.assertEqual(row.action, [row.action[0], row.action[1]])
            for issue in summary["issues"].values():
                self.assertIn(issue["status"], ("COMPLETE", "FAILED", "EXPIRED"))
                self.assertTrue(issue['realized_sales'])
                self.assertEqual(issue['close_gold'],0.)
            self.assertAlmostEqual(sum(r.local_credit for r in rows),summary['economic_net_cash'])
            self.assertEqual(summary['economic_reconciliation_error'],0.)
            folder=root/'episodes'/'episode_000001'
            self.assertTrue((folder/'trace.jsonl.gz').is_file())
            self.assertTrue((folder/'issue_sale_credit.json').is_file())
            moves=[json.loads(line) for line in (folder/'moves.jsonl').read_text().splitlines()]
            self.assertEqual(sum(len(m['crew_decisions']) for m in moves),summary['crew_decisions'])
            self.assertTrue(all('open_issues' not in m and 'issue_updates' in m for m in moves))
            self.assertTrue(all('subtask' in d and 'assignment_change' in d
                                for m in moves for d in m['crew_decisions']))

    def test_collect_records_every_move_and_closes_leftover_issues(self):
        torch.set_num_threads(1)
        torch.manual_seed(1)
        manager = ManagerActor()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = parser().parse_args([
                "--run-dir", str(root), "--episodes", "1", "--episode-steps", "4",
                "--save-every", "1", "--crew", "pass", "--opponent", "pass",
                "--device", "cpu",
            ])
            args.run_dir = root
            rows, summary = collect(manager, None, args, 1, root / "episodes" / "episode_000001")
            moves = (root / "episodes" / "episode_000001" / "moves.jsonl").read_text().strip().splitlines()
            self.assertEqual(len(moves), summary["turns"])
            self.assertGreaterEqual(summary["turns"], 2)
            self.assertTrue((root / "all_moves.jsonl").exists())
            self.assertTrue((root / "episodes" / "episode_000001" / "manager_decisions.json").exists())
            self.assertGreaterEqual(len(rows), summary["turns"])
            self.assertTrue(rows[-1].done)
            for issue in summary["issues"].values():
                self.assertIn(issue["status"], ("COMPLETE", "FAILED", "EXPIRED"))

    def test_parser_accepts_best_agent_opponent(self):
        args = parser().parse_args(["--run-dir", "unused", "--opponent", "best_agent"])
        self.assertEqual(args.opponent, "best_agent")

    def test_resume_and_init_manager_are_exclusive(self):
        from worker_grow.train_manager import _validate
        args = parser().parse_args([
            "--run-dir", "unused", "--resume", "--init-manager", "missing.pt", "--device", "cpu",
        ])
        with self.assertRaises(ValueError):
            _validate(args)

    def test_collect_against_best_agent_opponent(self):
        torch.set_num_threads(1)
        torch.manual_seed(1)
        manager = ManagerActor()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = parser().parse_args([
                "--run-dir", str(root), "--episodes", "1", "--episode-steps", "6",
                "--save-every", "0", "--crew", "pass", "--opponent", "best_agent",
                "--device", "cpu",
            ])
            args.run_dir = root
            rows, summary = collect(manager, None, args, 1, root / "episodes" / "episode_000001")
            self.assertGreaterEqual(summary["turns"], 2)
            self.assertTrue(rows[-1].done)
            self.assertNotEqual(summary["opponent_money"], 3000.0)


if __name__ == "__main__":
    unittest.main()
