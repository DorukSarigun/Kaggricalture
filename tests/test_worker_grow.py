import unittest

import numpy as np
import torch
from kaggle_environments.envs.kaggriculture import kaggriculture as game

from action_encoding import ActionGrammar
from worker_grow.encoding import encode_own_farm
from worker_grow.policy import WorkerGrowActor, sample_worker
from worker_grow.recipes import ANIMAL_UNIT_ACTIONS, CROP_UNIT_ACTIONS, required_unit_actions
from worker_grow.reward import REWARD_SCALE, grow_products, grow_step_reward


def observation():
    return {
        "player": 0,
        "step": 0,
        "day": 0,
        "hour": 0,
        "farms": [game._new_farm(10, 3000), game._new_farm(10, 3000)],
        "private": game._new_private(),
        "market": game._new_market(),
        "town": game._new_town(),
    }


class WorkerGrowTests(unittest.TestCase):
    def test_first_cycle_unit_counts_exclude_market(self):
        self.assertEqual(CROP_UNIT_ACTIONS["WHEAT"], 6)
        self.assertEqual(CROP_UNIT_ACTIONS["TOMATO"], 12)
        self.assertEqual(ANIMAL_UNIT_ACTIONS["GOOSE"], 9)
        self.assertEqual(required_unit_actions("EGG"), 9)
        self.assertEqual(required_unit_actions("WHEAT"), 6)

    def test_water_splits_current_wheat_price(self):
        obs = observation()
        obs["farms"][0]["farmer"] = [0, 0]
        obs["farms"][0]["hands"] = [[1, 0]]
        obs["private"]["inventories"].append({})
        obs["farms"][0]["tiles"][0][1] = game._new_plant("WHEAT", 0, 24)
        price = obs["market"]["prices"]["WHEAT"]
        result = grow_step_reward(obs, 1, ["WATER"])
        self.assertEqual(result["products"], ("WHEAT",))
        self.assertAlmostEqual(result["gold"], price / 6)
        self.assertAlmostEqual(result["reward"], (price / 6) / REWARD_SCALE)
        self.assertEqual(grow_products(obs, 1, ["PASS"]), ())
        self.assertEqual(grow_step_reward(obs, 1, ["SOUTH"])["reward"], 0.0)

    def test_price_change_moves_the_step_reward(self):
        obs = observation()
        obs["farms"][0]["hands"] = [[0, 0]]
        obs["private"]["inventories"].append({})
        obs["farms"][0]["tiles"][0][0] = game._new_plant("WHEAT", 0, 24)
        cheap = grow_step_reward(obs, 1, ["WATER"])["gold"]
        obs["market"]["prices"]["WHEAT"] = obs["market"]["prices"]["WHEAT"] + 12
        dear = grow_step_reward(obs, 1, ["WATER"])["gold"]
        self.assertAlmostEqual(dear - cheap, 12 / 6)

    def test_plant_and_feed_use_named_or_tile_product(self):
        obs = observation()
        obs["farms"][0]["hands"] = [[0, 0]]
        obs["private"]["inventories"].append({"WHEAT": 1})
        obs["private"]["seeds"]["WHEAT"] = 1
        self.assertEqual(grow_products(obs, 1, ["PLANT", "WHEAT"]), ("WHEAT",))
        obs["farms"][0]["tiles"][0][0] = {
            "kind": "COOP", "animal": "GOOSE", "yield_units": 0,
            "fed_today": False, "cared_today": False, "fertilizer_available": False,
            "placed_day": 0,
        }
        feed = grow_step_reward(obs, 1, ["FEED"])
        self.assertEqual(feed["products"], ("EGG",))
        self.assertAlmostEqual(feed["gold"], obs["market"]["prices"]["EGG"] / 9)

    def test_pasture_build_splits_both_animal_products(self):
        obs = observation()
        result = grow_step_reward(obs, 0, ["BUILD_PASTURE"])
        self.assertEqual(result["products"], ("MILK", "WOOL"))
        expected = obs["market"]["prices"]["MILK"] / 13 + obs["market"]["prices"]["WOOL"] / 11
        self.assertAlmostEqual(result["gold"], expected)

    def test_own_farm_encoding_has_no_opponent_maps(self):
        obs = observation()
        obs["farms"][0]["hands"] = [[2, 1]]
        obs["private"]["inventories"].append({})
        encoded = encode_own_farm(obs, 1)
        self.assertEqual(encoded.categorical_ids.shape[1:], (10, 10))
        self.assertEqual(tuple(encoded.unit_position_yx), (1, 2))
        self.assertFalse(hasattr(encoded, "opponent_categorical_ids"))

    def test_actor_reads_own_farm_and_samples_a_legal_unit_op(self):
        torch.manual_seed(0)
        obs = observation()
        obs["farms"][0]["hands"] = [[0, 0]]
        obs["private"]["inventories"].append({})
        grammar = ActionGrammar(obs)
        grammar.unit_index = 1
        model = WorkerGrowActor()
        sample = sample_worker(model, obs, 1, grammar)
        self.assertIn(sample.action[0], grammar._valid_unit_operations())
        context, values = model([sample.observation])
        self.assertEqual(context.shape[-1], 256)
        self.assertEqual(values.shape, (1,))


if __name__ == "__main__":
    unittest.main()
