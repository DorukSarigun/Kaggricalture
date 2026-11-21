from pathlib import Path
import tempfile
import unittest

import torch
from kaggle_environments.envs.kaggriculture import kaggriculture as game

from worker_grow.contracts import TASK_ITEMS
from worker_grow.issues import IssueBook
from worker_grow.manager import ManagerActor, market_cursor, propose_manager_contracts
from worker_grow.manager_workload import admission_items
from worker_grow.production_paths import (
    PATH_ITEMS, grammar_for_issue_items, market_item_allowlist, parse_issue_items,
    path_slug, sell_products,
)
from worker_grow.train_joint import parser as joint_parser, validate as joint_validate
from worker_grow.train_manager import _validate, collect, parser
from tests.test_manager import observation


class ParseIssueItemsTests(unittest.TestCase):
    def test_blank_is_unrestricted(self):
        self.assertIsNone(parse_issue_items(None))
        self.assertIsNone(parse_issue_items(""))
        self.assertIsNone(parse_issue_items("  "))

    def test_case_and_duplicates(self):
        self.assertEqual(parse_issue_items("carrot"), ("CARROT",))
        self.assertEqual(parse_issue_items("COW, cow, WHEAT"), ("COW", "WHEAT"))

    def test_tuple_is_idempotent(self):
        self.assertEqual(parse_issue_items(("CARROT",)), ("CARROT",))
        self.assertEqual(parse_issue_items(["cow", "COW"]), ("COW",))
        with self.assertRaises(ValueError):
            parse_issue_items("STONE")

    def test_every_task_item_is_a_path(self):
        self.assertEqual(PATH_ITEMS, TASK_ITEMS)
        for item in TASK_ITEMS:
            self.assertEqual(parse_issue_items(item), (item,))
            self.assertEqual(path_slug(item), item.lower())


class AdmissionAndMarketMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_admission_intersects_physical_sites_with_path(self):
        obs, book = observation(), IssueBook()
        self.assertIn("WHEAT", admission_items(obs, book))
        self.assertEqual(admission_items(obs, book, issue_items=("CARROT",)), ("CARROT",))
        self.assertEqual(admission_items(obs, book, issue_items=("COW",)), ("COW",))

    def test_carrot_path_issues_only_carrot(self):
        torch.manual_seed(3)
        manager = ManagerActor()
        obs = observation()
        book = IssueBook()
        with torch.no_grad():
            manager.contract_op.weight.zero_()
            manager.contract_op.bias.copy_(torch.tensor([20., -20.]))
        rows, _ = propose_manager_contracts(
            manager, obs, book, step=0, day=0, deterministic=True, issue_items=("CARROT",),
        )
        issued = [row.action[1] for row in rows if row.action and row.action[0] == "ISSUE"]
        self.assertTrue(issued)
        self.assertTrue(all(item == "CARROT" for item in issued))
        self.assertEqual(rows[0].allowed_items, ("CARROT",))
        self.assertEqual(rows[-1].action, ["STOP"])

    def test_carrot_market_cannot_buy_wheat_or_animals(self):
        cursor = market_cursor(observation(), config=grammar_for_issue_items(("CARROT",)))
        self.assertIn("BUY_SEED", cursor.operations())
        self.assertEqual(cursor.items("BUY_SEED"), ("CARROT",))
        self.assertNotIn("BUY_ANIMAL", cursor.operations())
        self.assertNotIn("BUY_PRODUCT", cursor.operations())
        self.assertEqual(cursor.items("BUY_ANIMAL"), ())
        self.assertEqual(cursor.items("BUY_PRODUCT"), ())

    def test_cow_market_buys_cow_and_feed_wheat_not_seeds(self):
        obs = observation()
        obs["private"]["shed"]["MILK"] = 1
        obs["private"]["shed"]["WHEAT"] = 4
        cursor = market_cursor(obs, config=grammar_for_issue_items(("COW",)))
        self.assertNotIn("BUY_SEED", cursor.operations())
        self.assertIn("BUY_ANIMAL", cursor.operations())
        self.assertEqual(cursor.items("BUY_ANIMAL"), ("COW",))
        self.assertIn("BUY_PRODUCT", cursor.operations())
        self.assertEqual(cursor.items("BUY_PRODUCT"), ("WHEAT",))
        self.assertIn("SELL", cursor.operations())
        self.assertEqual(cursor.items("SELL"), ("MILK",))
        self.assertNotIn("WHEAT", cursor.items("SELL"))
        obs = observation()
        obs["private"]["shed"]["MILK"] = 1
        cursor = market_cursor(obs, config=grammar_for_issue_items(("COW",)))
        self.assertNotIn("BUY_SEED", cursor.operations())
        self.assertIn("BUY_ANIMAL", cursor.operations())
        self.assertEqual(cursor.items("BUY_ANIMAL"), ("COW",))
        self.assertIn("BUY_PRODUCT", cursor.operations())
        self.assertEqual(cursor.items("BUY_PRODUCT"), ("WHEAT",))
        self.assertIn("SELL", cursor.operations())
        self.assertEqual(cursor.items("SELL"), ("MILK",))
        self.assertNotIn("WHEAT", cursor.items("SELL"))

    def test_allowlist_covers_each_product_output(self):
        for item in TASK_ITEMS:
            allow = market_item_allowlist((item,))
            if item in game.CROPS:
                self.assertEqual(allow["BUY_SEED"], frozenset({item}))
                self.assertFalse(allow["BUY_ANIMAL"])
                self.assertFalse(allow["BUY_PRODUCT"])
                self.assertEqual(allow["SELL"], frozenset({item}))
            else:
                self.assertFalse(allow["BUY_SEED"])
                self.assertEqual(allow["BUY_ANIMAL"], frozenset({item}))
                self.assertEqual(allow["BUY_PRODUCT"], frozenset({"WHEAT"}))
                self.assertEqual(allow["SELL"], frozenset(sell_products(item)))


class ProductPathCliTests(unittest.TestCase):
    def test_manager_and_joint_parse_issue_items(self):
        for p, validate in ((parser(), _validate), (joint_parser(), joint_validate)):
            args = p.parse_args(["--run-dir", "unused", "--device", "cpu", "--issue-items", "melon"])
            validate(args)
            self.assertEqual(args.issue_items, ("MELON",))
            unrestricted = p.parse_args(["--run-dir", "unused", "--device", "cpu"])
            validate(unrestricted)
            self.assertIsNone(unrestricted.issue_items)

    def test_unknown_issue_item_fails_validation(self):
        args = parser().parse_args(["--run-dir", "unused", "--device", "cpu", "--issue-items", "STONE"])
        with self.assertRaises(ValueError):
            _validate(args)

    def test_collect_carrot_path_never_issues_other_crops(self):
        torch.set_num_threads(1)
        torch.manual_seed(1)
        manager = ManagerActor()
        with torch.no_grad():
            manager.contract_op.weight.zero_()
            manager.contract_op.bias.copy_(torch.tensor([20., -20.]))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = parser().parse_args([
                "--run-dir", str(root), "--episodes", "1", "--episode-steps", "6",
                "--save-every", "0", "--crew", "pass", "--opponent", "pass",
                "--device", "cpu", "--issue-items", "CARROT",
            ])
            _validate(args)
            args.run_dir = root
            rows, summary = collect(manager, None, args, 1, root / "episodes" / "episode_000001")
            issued = [row.action[1] for row in rows
                      if row.kind == "contract" and row.action and row.action[0] == "ISSUE"]
            self.assertTrue(issued)
            self.assertTrue(all(item == "CARROT" for item in issued))
            self.assertTrue(all(issue["item"] == "CARROT" for issue in summary["issues"].values()))
            bought_seeds = [row.action[1] for row in rows
                            if row.kind == "market" and row.action[:1] == ["BUY_SEED"]]
            self.assertTrue(all(item == "CARROT" for item in bought_seeds))
            self.assertFalse(any(row.kind == "market" and row.action[:1] == ["BUY_ANIMAL"]
                                 for row in rows))


if __name__ == "__main__":
    unittest.main()
