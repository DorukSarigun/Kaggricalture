from copy import deepcopy
import unittest

import torch
from kaggle_environments.envs.kaggriculture import kaggriculture as game

from model import KaggricultureActorCritic
from policy import sample_action
from ppo import PPOConfig
from reward import RewardConfig, transition_reward
from training import EpisodeRollout, ppo_update
from train_scripted import build_parser, _validate_resume_configuration


def observation():
    return {"player": 0, "step": 0, "day": 0, "hour": 0,
            "farms": [game._new_farm(10, 3000), game._new_farm(10, 3000)],
            "private": game._new_private(), "market": game._new_market(),
            "town": game._new_town()}


class CashProgressTests(unittest.TestCase):
    def setUp(self):
        self.config = RewardConfig(mode="cash_progress", gamma=1, shaping_beta=0.25)

    def test_unused_seed_cost_and_terminal_correction(self):
        start = observation()
        bought = deepcopy(start)
        bought["farms"][0]["money"] -= 100
        bought["private"]["seeds"]["WHEAT"] = 10
        first = transition_reward(start, bought, False, self.config)
        last = transition_reward(bought, bought, True, self.config)
        self.assertAlmostEqual(first.base, -100 / 3000)
        self.assertLess(first.total, 0)
        self.assertLess(last.shaping, 0)
        self.assertAlmostEqual(first.total + last.total, -100 / 3000)

    def test_care_noop_loss_and_sale_conserve_money(self):
        states = [observation()]
        bought = deepcopy(states[-1])
        bought["farms"][0]["money"] -= 10
        bought["private"]["seeds"]["WHEAT"] = 1
        states.append(bought)
        planted = deepcopy(bought)
        planted["private"]["seeds"]["WHEAT"] = 0
        planted["farms"][0]["tiles"][0][0] = game._new_plant("WHEAT", 0, 24)
        states.append(planted)
        watered = deepcopy(planted)
        watered["farms"][0]["tiles"][0][0]["watered_today"] = True
        states.append(watered)
        self.assertGreater(transition_reward(planted, watered, False, self.config).shaping, 0)
        self.assertEqual(transition_reward(watered, watered, False, self.config).total, 0)
        lost = deepcopy(watered)
        lost["farms"][0]["tiles"][0][0] = {"kind": "WEED"}
        self.assertLess(transition_reward(watered, lost, False, self.config).shaping, 0)
        harvested = deepcopy(watered)
        harvested["farms"][0]["tiles"][0][0] = None
        harvested["private"]["shed"]["WHEAT"] = 4
        states.append(harvested)
        sold = deepcopy(harvested)
        price = game.market_price("WHEAT", sold["market"]["inventory"]["WHEAT"])
        game._commit_unit("SELL", "WHEAT", price, sold["farms"][0], sold["private"], sold["market"])
        states.append(sold)
        sale = transition_reward(harvested, sold, False, self.config)
        self.assertAlmostEqual(sale.base, price / 3000)
        self.assertLess(sale.shaping, 0)
        rewards = [transition_reward(a, b, False, self.config) for a, b in zip(states, states[1:])]
        rewards.append(transition_reward(sold, sold, True, self.config))
        self.assertAlmostEqual(sum(r.shaping for r in rewards), 0)
        self.assertAlmostEqual(sum(r.total for r in rewards), (price - 10) / 3000)

    def test_opponent_cash_and_initial_assets(self):
        start = observation()
        start["private"]["seeds"]["WHEAT"] = 2
        end = deepcopy(start)
        end["farms"][1]["money"] += 100
        result = transition_reward(start, end, True, self.config)
        self.assertAlmostEqual(result.base, -100 / 3000)
        self.assertAlmostEqual(result.shaping, -self.config.shaping_beta * result.current_potential)

    def test_realized_targets_and_optimizer(self):
        torch.manual_seed(9)
        model = KaggricultureActorCritic()
        # Value head'e tek gradient kaynağı yardımcı gerçekleşmiş getiri loss'u.
        ppo = PPOConfig(gamma=1, realized_value_coef=0.5, value_coef=0)
        rollout = EpisodeRollout(self.config, ppo)
        start = observation()
        middle = deepcopy(start)
        middle["step"] = 1
        middle["farms"][0]["money"] -= 10
        middle["private"]["seeds"]["WHEAT"] = 1
        end = deepcopy(middle)
        end["step"] = 2
        end["farms"][0]["money"] += 100
        sample = sample_action(model, start, deterministic=True)
        rollout.add(sample, start, middle, False)
        with self.assertRaisesRegex(ValueError, "tam episode"):
            rollout.records()
        rollout.add(sample_action(model, middle, deterministic=True), middle, end, True)
        records = rollout.records()
        self.assertAlmostEqual(records[0].realized_return_target, 90 / 3000)
        self.assertAlmostEqual(records[1].realized_return_target, 100 / 3000)
        self.assertGreater(records[1].potential_offset, 0)
        mc_rollout = EpisodeRollout(self.config, PPOConfig(gamma=1, gae_lambda=1))
        mc_rollout.transitions = rollout.transitions.copy()
        for record in mc_rollout.records():
            self.assertAlmostEqual(record.return_target + record.potential_offset,
                                   record.realized_return_target, places=6)
        before = model.value_head.network[-1].weight.detach().clone()
        metrics = ppo_update(model, torch.optim.Adam(model.parameters(), lr=1e-4), records,
                             config=ppo, epochs=1, minibatch_size=2)
        self.assertGreater(metrics.realized_value_loss, 0)
        self.assertFalse(torch.equal(before, model.value_head.network[-1].weight))

    def test_invalid_config(self):
        with self.assertRaises(ValueError):
            RewardConfig(mode="cash_progress", gamma=0.999)
        with self.assertRaises(ValueError):
            EpisodeRollout(self.config, PPOConfig(gamma=1, span_policy_coef=1))

    def test_new_reward_cannot_silently_resume_old_run(self):
        args = build_parser().parse_args(["train", "--reward-mode", "cash_progress"])
        with self.assertRaisesRegex(ValueError, "reward_mode"):
            _validate_resume_configuration(args, {"run_configuration": {}})
