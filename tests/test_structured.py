from copy import deepcopy
from unittest.mock import patch
import unittest

import numpy as np
import torch
from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as game

from action_encoding import ActionGrammar
from observation_encoding import encode_observation
from reward import RewardConfig, transition_reward
from structured_actions import (OperationCursor, StructuredActorCritic, head_input,
                                 sample_operation, action_statistics)
from structured_training import (DecisionTransition, propose_turn, settle_turn,
                                  assign_advantages, update_policy)


def observation():
    return {"player": 0, "step": 0, "day": 0, "hour": 0,
            "farms": [game._new_farm(10, 3000), game._new_farm(10, 3000)],
            "private": game._new_private(), "market": game._new_market(), "town": game._new_town()}


class StructuredTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        torch.set_num_threads(1)
        self.config = RewardConfig(mode="cash_progress", gamma=1, shaping_beta=0.25)

    def test_next_hand_sees_planted_tile_and_consumed_seed(self):
        obs = observation()
        obs["farms"][0]["farmer"] = [0, 0]
        obs["farms"][0]["hands"] = [[0, 0]]
        obs["private"]["inventories"].append({})
        obs["private"]["seeds"]["WHEAT"] = 1
        original = deepcopy(obs)
        cursor = OperationCursor(obs)
        before = encode_observation(obs)
        cursor.apply(["PLANT", "WHEAT"])
        after = head_input(cursor, before)
        self.assertEqual(after.unit_index, 1)
        self.assertEqual(cursor.observation()["private"]["seeds"]["WHEAT"], 0)
        self.assertIn("WATER", cursor.operations())
        self.assertNotIn("PLANT", cursor.operations())
        self.assertFalse(np.array_equal(before.own_categorical_ids, after.current.own_categorical_ids))
        cursor.apply(["WATER"])
        self.assertTrue(cursor.observation()["farms"][0]["tiles"][0][0]["watered_today"])
        self.assertEqual(obs, original)

    def test_market_updates_budget_and_hires_are_not_new_unit_turns(self):
        obs = observation()
        obs["farms"][0]["money"] = 100
        c = OperationCursor(obs)
        c.apply(["PASS"])
        c.apply(["BUY_SEED", "WHEAT", 3])
        self.assertEqual(c.quantity_limit("BUY_SEED", "WHEAT"), 7)
        with self.assertRaises(ValueError):
            c.apply(["BUY_SEED", "WHEAT", 8])
        c.apply(["HIRE"])
        self.assertEqual(c.section, "market")
        self.assertEqual(c.actions["hands"], [])
        self.assertEqual(c.rules.farm["hires_today"], 1)
        c.apply(["STOP"])
        self.assertTrue(c.finished)

    def test_no_token_push_and_teacher_forcing_matches(self):
        model = StructuredActorCritic()
        obs = observation()
        cursor = OperationCursor(obs)
        with patch.object(ActionGrammar, "push", side_effect=AssertionError("No tokens allowed")):
            sample = sample_operation(model, cursor, encode_observation(obs))
            cursor.apply(sample.action)
        contexts, _, _ = model([sample.inputs])
        logp, entropy = action_statistics(model, contexts[0], sample)
        self.assertAlmostEqual(float(logp.detach()), sample.log_prob, places=5)
        self.assertTrue(torch.isfinite(entropy))
        (-logp).backward()
        self.assertIsNotNone(model.unit_head.weight.grad)
        self.assertIsNone(model.market_head.weight.grad)
        for limit in (1, 3, 10, 137, 1000):
            d = model.quantity_distribution(contexts[0], "BUY_SEED", "WHEAT", limit)
            self.assertEqual(len(d.probs), limit)
            self.assertTrue(torch.all(d.probs > 0))
            self.assertAlmostEqual(float(d.probs.sum().detach()), 1, places=5)

    def test_opponent_market_correction_is_boundary_not_fake_execution(self):
        env = make("kaggriculture", configuration={"episodeSteps": 3})
        for state in env.state:
            state.observation.private.shed["WHEAT"] = 40
        before = deepcopy(env.state[0].observation)
        cursor = OperationCursor(before)
        transitions = []
        for action in (["PASS"], ["SELL", "WHEAT", 20], ["STOP"]):
            previous = cursor.observation()
            cursor.apply(action)
            reward = transition_reward(previous, cursor.observation(), False, self.config)
            transitions.append(DecisionTransition(None, None, 0, reward.total, 0, False))
        transitions.append(DecisionTransition(None, None, 0, 0, 0, True))
        predicted_money = cursor.rules.farm["money"]
        env.step([cursor.actions, {"farmer": ["PASS"], "hands": [], "market": [["SELL", "WHEAT", 20]]}])
        after = deepcopy(env.state[0].observation)
        self.assertNotEqual(predicted_money, after.farms[0].money)
        actual, error = settle_turn(transitions, before, after, env.done, self.config)
        self.assertAlmostEqual(error, 0)
        self.assertAlmostEqual(sum(t.reward for t in transitions), actual.total)
        self.assertTrue(all(t.base_reward == 0 for t in transitions[:-1]))
        self.assertAlmostEqual(transitions[-1].base_reward, actual.base)

    def test_advantages_are_per_operation_and_internal_lambda_is_one(self):
        rows = [DecisionTransition(None, None, 0.1, 0.2, 0, False),
                DecisionTransition(None, None, 0.3, -0.1, 0, False),
                DecisionTransition(None, None, 0.0, 0.9, 1.0, True, True)]
        assign_advantages(rows, gae_lambda=0.1)
        self.assertAlmostEqual(rows[0].advantage, 0.9)
        self.assertAlmostEqual(rows[1].advantage, 0.5)
        self.assertAlmostEqual(rows[0].realized_target, 1)
        self.assertAlmostEqual(rows[1].realized_target, 1)
        # Zero-effect internal operation insertion must not discount the past.
        inserted = [deepcopy(rows[0]), DecisionTransition(None, None, 0.7, 0, 0, False),
                    deepcopy(rows[1]), deepcopy(rows[2])]
        assign_advantages(inserted, gae_lambda=0.1)
        self.assertAlmostEqual(inserted[0].advantage, rows[0].advantage)

    def test_rollout_ppo_and_separate_realized_head(self):
        model = StructuredActorCritic()
        obs = observation()
        action, rows = propose_turn(model, obs, self.config)
        after = deepcopy(obs)
        after["farms"][0]["money"] = 3100
        settle_turn(rows, obs, after, True, self.config)
        assign_advantages(rows)
        before = model.realized_value_head.weight.detach().clone()
        result = update_policy(model, torch.optim.Adam(model.parameters(), lr=1e-4), rows,
                               epochs=1, minibatch_size=64)
        self.assertTrue(np.isfinite(result["loss"]))
        self.assertFalse(torch.equal(before, model.realized_value_head.weight))
        self.assertIsNone(rows[-1].sample)


if __name__ == "__main__":
    unittest.main()
