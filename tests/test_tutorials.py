"""Tutorial scenes are a separate mode; they must not alter train_crew stages."""
import tempfile
from pathlib import Path
import unittest

import torch
from kaggle_environments import make

from worker_grow.crew import CrewCursor, FeudalCrew, VERSION
from worker_grow.delivery import CrewDeliveryTracker
from worker_grow.train_crew import initialize_weights, parser as crew_parser
from worker_grow.train_tutorials import collect, parser as tutorial_parser
from worker_grow.tutorials import (
    SHED_XY, SLICES, WALK_PLANT_XY, attach_gifted_inventory, demand_is_bound,
    demand_is_unbound, parse_slices, setup, slice_for_episode,
)


def env(steps=16):
    return make("kaggriculture", configuration={
        "episodeSteps": steps, "turnsPerDay": 24, "weedSpawnChance": 0., "seed": 77,
    })


class TutorialSceneTests(unittest.TestCase):
    def test_slice_schedule_and_parser_are_not_on_train_crew(self):
        self.assertEqual(slice_for_episode(1, SLICES, 20), "drop")
        self.assertEqual(slice_for_episode(20, SLICES, 20), "drop")
        self.assertEqual(slice_for_episode(21, SLICES, 20), "harvest")
        self.assertEqual(slice_for_episode(100, SLICES, 20), "plant")
        crew_args = crew_parser().parse_args(["--run-dir", "unused"])
        self.assertFalse(hasattr(crew_args, "tutorial_slices"))
        self.assertEqual(crew_args.stage, "mature")

    def test_drop_allows_shed_drop_on_gifted_wheat(self):
        e = env()
        book = setup(e, "drop")
        obs = e.state[0].observation
        obs["step"] = 0
        tracker = CrewDeliveryTracker(obs, book.task_book, grow_reward=1.)
        attach_gifted_inventory(tracker, book, "WHEAT")
        cursor = CrewCursor(obs, book.task_book)
        self.assertIn("DROP", cursor.operations())
        self.assertEqual(tuple(obs["farms"][0]["farmer"]), SHED_XY)
        self.assertEqual(obs["private"]["inventories"][0]["WHEAT"], 1)

    def test_harvest_and_water_bind_existing_plant(self):
        e = env()
        book = setup(e, "harvest")
        obs = e.state[0].observation
        obs["step"] = 0
        CrewDeliveryTracker(obs, book.task_book)
        self.assertTrue(demand_is_bound(book, "WHEAT", SHED_XY))
        cursor = CrewCursor(obs, book.task_book)
        self.assertIn("HARVEST", cursor.operations())

        e = env()
        book = setup(e, "water")
        obs = e.state[0].observation
        obs["step"] = 0
        CrewDeliveryTracker(obs, book.task_book)
        self.assertTrue(demand_is_bound(book, "WHEAT", SHED_XY))
        cursor = CrewCursor(obs, book.task_book)
        self.assertIn("WATER", cursor.operations())
        self.assertFalse(obs["farms"][0]["tiles"][SHED_XY[1]][SHED_XY[0]]["watered_today"])

    def test_walk_harvest_requires_travel_and_plant_is_unbound_issue(self):
        e = env()
        book = setup(e, "walk_harvest")
        obs = e.state[0].observation
        obs["step"] = 0
        CrewDeliveryTracker(obs, book.task_book)
        self.assertTrue(demand_is_bound(book, "WHEAT", WALK_PLANT_XY))
        self.assertEqual(tuple(obs["farms"][0]["farmer"]), SHED_XY)
        cursor = CrewCursor(obs, book.task_book)
        self.assertNotIn("HARVEST", cursor.operations())
        self.assertTrue(any(op in cursor.operations() for op in ("NORTH", "WEST", "SOUTH", "EAST")))

        e = env()
        book = setup(e, "plant")
        obs = e.state[0].observation
        obs["step"] = 0
        CrewDeliveryTracker(obs, book.task_book)
        self.assertTrue(demand_is_unbound(book, "WHEAT"))
        cursor = CrewCursor(obs, book.task_book)
        self.assertIn("PLANT", cursor.operations())
        self.assertEqual(cursor.items("PLANT"), ("WHEAT",))

    def test_collect_tutorial_episode_does_not_open_teacher_issues(self):
        torch.set_num_threads(1)
        torch.manual_seed(1)
        with tempfile.TemporaryDirectory() as tmp:
            args = tutorial_parser().parse_args([
                "--run-dir", tmp, "--episodes", "1", "--save-every", "0",
                "--tutorial-slices", "drop", "--slice-episodes", "1",
            ])
            rows, summary = collect(FeudalCrew(), args, 1, Path(tmp) / "episode")
            self.assertEqual(summary["stage"], "tutorials")
            self.assertEqual(summary["tutorial_slice"], "drop")
            self.assertEqual(len(summary["tasks"]), 1)
            self.assertGreaterEqual(len(rows), 1)
            self.assertTrue(all(r.action[0] != "HIRE" for r in rows))

    def test_init_crew_rejects_incompatible_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.pt"
            torch.save({"version": "feudal-contract-crew-v1", "model": {}}, path)
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                initialize_weights(FeudalCrew(), path, VERSION, "cpu")

    def test_init_crew_accepts_site_v2_weights(self):
        src = FeudalCrew()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v2.pt"
            legacy={k:v for k,v in src.state_dict().items()
                    if not k.startswith(('assignment_score.','assignment_condition.'))}
            torch.save({"version": "feudal-contract-crew-site-v2", "model": legacy}, path)
            dest = FeudalCrew()
            initialize_weights(dest, path, VERSION, "cpu")
            self.assertTrue(torch.equal(src.phi.weight, dest.phi.weight))


if __name__ == "__main__":
    unittest.main()
