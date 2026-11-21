import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import torch
from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as game

from worker_grow.contracts import TaskStatus
from worker_grow.issues import IssueBook, issue_unlock_limit
from worker_grow.scripted_manager import (
    BestAgentTeacher, apply_production_intents, best_agent_opponent, production_intents,
)
from worker_grow.train_crew import collect, parser, prepare_run_dir


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
    obs["farms"][0]["farmer"] = [0, 0]
    obs["farms"][0]["hands"] = [[1, 0] for _ in range(hands)]
    obs["private"]["inventories"] = [{} for _ in range(hands + 1)]
    return obs


class BestAgentOpponentTests(unittest.TestCase):
    def test_fresh_chassis_returns_a_legal_action(self):
        env = make("kaggriculture", configuration={
            "episodeSteps": 8, "turnsPerDay": 24, "startingMoney": 3000, "seed": 1,
        })
        opponent = best_agent_opponent(env.configuration)
        obs = deepcopy(env.state[1].observation)
        obs["step"] = 0
        action = opponent(obs)
        self.assertIn("farmer", action)
        self.assertIn("hands", action)
        self.assertIn("market", action)
        env.step([{"farmer": ["PASS"], "hands": [], "market": []}, action])
        self.assertTrue(all(str(state.status) in ("ACTIVE", "DONE") for state in env.state))


class FakeChassis:
    def __init__(self, action):
        self.action = action

    def __call__(self, observation, configuration=None):
        return deepcopy(self.action)


class ScriptedManagerTests(unittest.TestCase):
    def test_plant_intent_pins_planning_actor_tile(self):
        obs = observation()
        action = {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": [["BUY_SEED", "WHEAT", 1]]}
        intents = production_intents(obs, action)
        self.assertEqual(intents, (("WHEAT", (0, 0)),))
        book = IssueBook()
        opened = apply_production_intents(obs, book, intents, step=3, day=0)
        self.assertEqual(len(opened), 1)
        self.assertEqual(opened[0].item, "WHEAT")
        self.assertEqual(opened[0].tile_xy, (0, 0))
        self.assertEqual(opened[0].as_dict()["tile_xy"], [0, 0])

    def test_illegal_or_occupied_pin_skips_issue(self):
        obs = observation()
        book = IssueBook()
        first = apply_production_intents(obs, book, (("WHEAT", (0, 0)),), step=0, day=0)
        self.assertEqual(len(first), 1)
        again = apply_production_intents(obs, book, (("CARROT", (0, 0)),), step=1, day=0)
        self.assertEqual(again, [])
        obs["farms"][0]["tiles"][1][2] = "LOCKED"
        skipped = apply_production_intents(obs, book, (("MELON", (2, 1)),), step=2, day=0)
        self.assertEqual(skipped, [])

    def test_water_and_walk_do_not_open_issues(self):
        obs = observation()
        action = {"farmer": ["WATER"], "hands": [], "market": []}
        self.assertEqual(production_intents(obs, action), ())

    def test_issue_unlock_follows_episode_windows(self):
        self.assertIsNone(issue_unlock_limit(1, interval=0, initial=1))
        self.assertEqual(issue_unlock_limit(1, interval=10, initial=1), 1)
        self.assertEqual(issue_unlock_limit(10, interval=10, initial=1), 1)
        self.assertEqual(issue_unlock_limit(11, interval=10, initial=1), 2)
        self.assertEqual(issue_unlock_limit(20, interval=10, initial=1), 2)
        self.assertEqual(issue_unlock_limit(21, interval=10, initial=1), 3)

    def test_open_issues_respect_episode_unlock_limit(self):
        obs = observation()
        book = IssueBook()
        opened = apply_production_intents(
            obs, book, ("MELON", "WHEAT", "CARROT"), step=0, day=0, limit=1,
        )
        self.assertEqual([issue.item for issue in opened], ["MELON"])
        apply_production_intents(obs, book, ("WHEAT",), step=1, day=0, limit=1)
        self.assertEqual(len(book.open_issues()), 1)
        book.close(opened[0].task_id, TaskStatus.COMPLETE, step=2, prices={"MELON": 80},
                   delivered={"MELON": 1})
        more = apply_production_intents(obs, book, ("WHEAT",), step=3, day=0, limit=1)
        self.assertEqual(len(more), 1)
        self.assertEqual(more[0].item, "WHEAT")

    def test_all_plant_intents_open_even_beyond_crew_size(self):
        obs = observation(hands=0)
        book = IssueBook()
        opened = apply_production_intents(
            obs, book, ("MELON", "WHEAT", "CARROT"), step=0, day=0,
        )
        self.assertEqual([issue.item for issue in opened], ["MELON", "WHEAT", "CARROT"])
        self.assertEqual(len(book.open_issues()), 3)

    def test_pasture_uses_next_step_place_by_the_same_actor(self):
        obs = observation()
        action = {"farmer": ["BUILD_PASTURE"], "hands": [], "market": []}
        intents = production_intents(obs, action, lookahead=[["PLACE", "COW"]])
        self.assertEqual(intents, (("COW", (0, 0)),))

    def test_teacher_keeps_market_and_does_not_return_unit_work(self):
        obs = observation()
        market = [["HIRE"], ["BUY_SEED", "MELON", 1]]
        teacher = BestAgentTeacher(FakeChassis({
            "farmer": ["PLANT", "MELON"], "hands": [], "market": market,
        }))
        book = IssueBook()
        teacher.propose_contracts(obs, book, step=0, day=0)
        self.assertEqual(teacher.propose_market(obs), market)
        self.assertEqual(len(book.open_issues()), 1)
        self.assertEqual(book.open_issues()[0].tile_xy, (0, 0))

    def test_teacher_market_reserves_wheat_for_live_animals(self):
        from worker_grow.scripted_manager import protect_crew_market
        obs = observation()
        obs["farms"][0]["tiles"][3][4] = game._new_animal("COW", 0)
        obs["farms"][0]["tiles"][3][4]["fed_today"] = False
        obs["private"]["shed"]["WHEAT"] = 1
        kept = protect_crew_market(obs, [["SELL", "WHEAT", 1], ["HIRE"]])
        self.assertEqual(kept, [["HIRE"]])
        obs["private"]["shed"]["WHEAT"] = 0
        bought = protect_crew_market(obs, [["SELL", "WHEAT", 4]])
        self.assertEqual(bought[0], ["BUY_PRODUCT", "WHEAT", 1])
        self.assertNotIn(["SELL", "WHEAT", 4], bought)

    def test_ghost_pin_ignores_live_farmer(self):
        configuration = dict(episodeSteps=8, turnsPerDay=24, seed=77, weedSpawnChance=0.)
        env = make("kaggriculture", configuration=configuration)
        start = tuple(env.state[0].observation["farms"][0]["farmer"])
        self.assertNotEqual(start, (9, 9))
        teacher = BestAgentTeacher(FakeChassis({
            "farmer": ["PLANT", "WHEAT"], "hands": [], "market": [],
        }))
        teacher.attach_world(configuration)
        live = deepcopy(env.state[0].observation)
        live["step"] = 0
        live["farms"][0]["farmer"] = [9, 9]
        book = IssueBook()
        opened = teacher.propose_contracts(live, book, step=0, day=0)
        self.assertEqual(len(opened), 1)
        self.assertNotEqual(opened[0].tile_xy, (9, 9))
        self.assertNotIn(opened[0].tile_xy, {(4, 4), (5, 4), (4, 5), (5, 5)})

    def test_collect_best_agent_uses_teacher_market_not_gifted_plants(self):
        torch.set_num_threads(1)
        torch.manual_seed(1)
        from worker_grow.crew import FeudalCrew
        teacher = BestAgentTeacher(FakeChassis({
            "farmer": ["PLANT", "WHEAT"],
            "hands": [],
            "market": [["BUY_SEED", "WHEAT", 1]],
        }))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = parser().parse_args([
                "--run-dir", str(root), "--stage", "best_agent",
                "--episodes", "1", "--episode-steps", "4", "--save-every", "0",
            ])
            rows, summary = collect(FeudalCrew(), args, 1, root / "episode", teacher=teacher)
            self.assertFalse(summary["gifted_curriculum"])
            self.assertEqual(summary["issue_limit"], 1)
            self.assertEqual(list(summary["tasks"]), ["issue:1"])
            self.assertGreaterEqual(len(rows), 1)
            self.assertGreaterEqual(summary["turns"], 2)

    def test_overwrite_deletes_existing_run_and_cannot_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            root.mkdir()
            (root / "stale.txt").write_text("old", encoding="utf8")
            cleared = prepare_run_dir(root, overwrite=True)
            self.assertFalse((cleared / "stale.txt").exists())
            self.assertTrue(cleared.is_dir())
            with self.assertRaises(ValueError):
                prepare_run_dir(root, resume=True, overwrite=True)
            occupied = Path(tmp) / "occupied"
            occupied.mkdir()
            (occupied / "stale.txt").write_text("old", encoding="utf8")
            with self.assertRaises(FileExistsError):
                prepare_run_dir(occupied)


if __name__ == "__main__":
    unittest.main()
