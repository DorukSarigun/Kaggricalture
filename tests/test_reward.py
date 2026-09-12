from copy import deepcopy
import unittest

from kaggle_environments.envs.kaggriculture import kaggriculture as game

from reward import (
    RewardConfig,
    _one_time_growth_targets,
    economic_potential,
    economic_potential_breakdown,
    terminal_gold_difference_reward,
    terminal_profit_reward,
    transition_reward,
)


def make_observation(player: int = 0) -> dict:
    return {
        "player": player,
        "step": 0,
        "day": 0,
        "hour": 0,
        "farms": [game._new_farm(10, 3000), game._new_farm(10, 3000)],
        "private": game._new_private(),
        "market": game._new_market(),
        "town": game._new_town(),
    }


class RewardTests(unittest.TestCase):
    def test_initial_potential_is_zero(self) -> None:
        observation = make_observation()

        self.assertAlmostEqual(economic_potential(observation), 0.0)
        self.assertEqual(
            economic_potential_breakdown(observation).total,
            3000.0,
        )

    def test_potential_does_not_mutate_observation(self) -> None:
        observation = make_observation()
        observation["private"]["shed"]["WHEAT"] = 7
        original = deepcopy(observation)

        economic_potential_breakdown(observation)

        self.assertEqual(observation, original)

    def test_buy_then_immediate_liquidation_has_no_fake_gain(self) -> None:
        before = make_observation()
        after = deepcopy(before)
        item = "WHEAT"
        buy_price = game.market_price(
            item,
            after["market"]["inventory"][item] - 1,
        )
        committed = game._commit_unit(
            "BUY_PRODUCT",
            item,
            buy_price,
            after["farms"][0],
            after["private"],
            after["market"],
        )
        game._refresh_prices(after["market"])

        self.assertTrue(committed)
        self.assertAlmostEqual(
            economic_potential_breakdown(before).total,
            economic_potential_breakdown(after).total,
        )

    def test_selling_one_unit_preserves_cash_plus_remaining_liquidation(self) -> None:
        before = make_observation()
        before["private"]["shed"]["WHEAT"] = 7
        after = deepcopy(before)
        sell_price = game.market_price(
            "WHEAT",
            after["market"]["inventory"]["WHEAT"],
        )
        committed = game._commit_unit(
            "SELL",
            "WHEAT",
            sell_price,
            after["farms"][0],
            after["private"],
            after["market"],
        )
        game._refresh_prices(after["market"])

        self.assertTrue(committed)
        self.assertAlmostEqual(
            economic_potential_breakdown(before).total,
            economic_potential_breakdown(after).total,
        )

    def test_seed_with_no_time_to_yield_is_worthless(self) -> None:
        observation = make_observation()
        observation["day"] = 29
        observation["step"] = 29 * 24
        observation["private"]["seeds"]["MELON"] = 10

        breakdown = economic_potential_breakdown(observation)

        self.assertEqual(breakdown.seeds, 0.0)

    def test_terminal_reward_uses_opponent_gold_difference(self) -> None:
        observation = make_observation(player=0)
        observation["farms"][0]["money"] = 4500
        observation["farms"][1]["money"] = 3900

        self.assertEqual(terminal_gold_difference_reward(observation, False), 0.0)
        self.assertAlmostEqual(
            terminal_gold_difference_reward(observation, True),
            600 / 3000,
        )
        # Eski fonksiyon adı geriye uyumluluk için aynı hedefe yönlenir.
        self.assertAlmostEqual(terminal_profit_reward(observation, True), 600 / 3000)

    def test_terminal_reward_respects_player_perspective(self) -> None:
        observation = make_observation(player=1)
        observation["farms"][0]["money"] = 5000
        observation["farms"][1]["money"] = 3500

        self.assertAlmostEqual(
            terminal_gold_difference_reward(observation, True),
            -1500 / 3000,
        )

    def test_buying_seeds_is_value_neutral_and_earns_no_shaping(self) -> None:
        # 50 tohum tam olarak harcanan 500 parayı yerine koyar; Phi değişmez.
        before = make_observation()
        after = deepcopy(before)
        after["farms"][0]["money"] = 2500
        after["private"]["seeds"]["WHEAT"] = 50

        reward = transition_reward(before, after, done=False)

        self.assertEqual(reward.base, 0.0)
        self.assertAlmostEqual(reward.shaping, 0.0)
        self.assertAlmostEqual(reward.total, 0.0)

    def test_default_config_rewards_planting_within_the_step(self) -> None:
        """Varsayılan shaping ekimi gerçekleştiği adımda ödüllendirmeli.

        Terminal gold farkı tek başına 719 adımda tek skalerdir; bu test
        adım-başı kredinin gerçekten aktığını sabitler.
        """
        before = make_observation()
        before["private"]["seeds"]["WHEAT"] = 1
        after = deepcopy(before)
        after["private"]["seeds"]["WHEAT"] = 0
        after["farms"][0]["tiles"][0][0] = {
            "kind": "PLANT",
            "crop": "WHEAT",
            "planted_day": 0,
            "yield_units": 0,
            "fertilized_until_day": -1,
        }

        reward = transition_reward(before, after, done=False)

        self.assertGreater(RewardConfig().shaping_beta, 0.0)
        self.assertGreater(reward.next_potential, reward.current_potential)
        self.assertGreater(reward.shaping, 0.0)
        self.assertAlmostEqual(reward.total, reward.shaping)

    def test_default_config_penalises_losing_a_crop(self) -> None:
        before = make_observation()
        before["farms"][0]["tiles"][0][0] = {
            "kind": "PLANT",
            "crop": "WHEAT",
            "planted_day": 0,
            "yield_units": 0,
            "fertilized_until_day": -1,
        }
        after = deepcopy(before)
        after["farms"][0]["tiles"][0][0] = {"kind": "WEED"}

        reward = transition_reward(before, after, done=False)

        self.assertLess(reward.shaping, 0.0)

    def test_terminal_assets_are_not_counted_as_score(self) -> None:
        current = make_observation()
        terminal_a = deepcopy(current)
        terminal_b = deepcopy(current)
        terminal_b["private"]["shed"]["MELON"] = 50

        reward_a = transition_reward(current, terminal_a, done=True)
        reward_b = transition_reward(current, terminal_b, done=True)

        self.assertEqual(reward_a.base, reward_b.base)
        self.assertEqual(reward_a.next_potential, 0.0)
        self.assertEqual(reward_b.next_potential, 0.0)

    def test_shaping_telescopes_over_complete_episode(self) -> None:
        config = RewardConfig(gamma=0.999, shaping_beta=1.0)
        states = [make_observation() for _ in range(4)]
        states[0]["farms"][0]["money"] = 3100
        states[1]["farms"][0]["money"] = 2800
        states[1]["private"]["seeds"]["WHEAT"] = 20
        states[2]["farms"][0]["money"] = 4200
        states[2]["private"]["shed"]["CARROT"] = 3
        states[3]["farms"][0]["money"] = 5000

        discounted_shaping = 0.0
        for index in range(len(states) - 1):
            result = transition_reward(
                states[index],
                states[index + 1],
                done=index == len(states) - 2,
                config=config,
            )
            discounted_shaping += config.gamma**index * result.shaping

        expected = -economic_potential(states[0], config)
        self.assertAlmostEqual(discounted_shaping, expected, places=10)

    def test_growth_targets_come_from_object_type_data(self) -> None:
        for crop, expected in (
            ("WHEAT", (4, 4)),
            ("CARROT", (3, 3)),
            ("MELON", (6, 10)),
        ):
            with self.subTest(crop=crop):
                self.assertEqual(_one_time_growth_targets(crop), expected)

    def test_growth_targets_cover_every_one_time_crop(self) -> None:
        """Motor bir ürünü ``ongoing=False`` yaparsa sessiz KeyError olmamalı."""
        for crop, data in game.CROPS.items():
            if data["ongoing"]:
                continue
            with self.subTest(crop=crop):
                cap, target_age = _one_time_growth_targets(crop)
                self.assertGreater(cap, 0)
                self.assertGreater(target_age, 0)

    def test_growth_targets_reject_unknown_crop(self) -> None:
        with self.assertRaises(KeyError):
            _one_time_growth_targets("PUMPKIN")


if __name__ == "__main__":
    unittest.main()
