from copy import deepcopy
import unittest

import torch
from kaggle_environments import make

from action_encoding import (
    ActionGrammar,
    ActionGrammarConfig,
    ActionToken,
    decode_action_tokens,
    encode_action_tokens,
)
from curriculum import (
    CurriculumConfig,
    WHEAT_CURRICULUM_STAGES,
    grammar_config_for_stage,
    inject_starting_seeds,
    resolve_episode_curriculum,
    select_stage,
)
from model import KaggricultureActorCritic
from policy import sample_action
from tests.test_action_encoding import make_observation


def wheat_loop_config() -> ActionGrammarConfig:
    return grammar_config_for_stage(WHEAT_CURRICULUM_STAGES[0])


class CurriculumGrammarTests(unittest.TestCase):
    def test_default_config_still_allows_full_legal_set(self) -> None:
        observation = make_observation()
        observation["private"]["seeds"]["WHEAT"] = 1
        grammar = ActionGrammar(observation)
        grammar.push(ActionToken.BOS)
        grammar.push(ActionToken.FARMER)

        self.assertIn(ActionToken.UNIT_BUILD_COOP, grammar.valid_tokens())
        self.assertIn(ActionToken.UNIT_PLANT, grammar.valid_tokens())
        self.assertIn(ActionToken.UNIT_PASS, grammar.valid_tokens())

    def test_wheat_loop_hides_distractors_and_keeps_escape_tokens(self) -> None:
        observation = make_observation()
        observation["private"]["seeds"]["WHEAT"] = 2
        observation["private"]["seeds"]["MELON"] = 2
        grammar = ActionGrammar(observation, wheat_loop_config())
        grammar.push(ActionToken.BOS)
        grammar.push(ActionToken.FARMER)
        unit_ops = set(grammar.valid_tokens())

        self.assertIn(ActionToken.UNIT_PASS, unit_ops)
        self.assertIn(ActionToken.UNIT_PLANT, unit_ops)
        self.assertIn(ActionToken.UNIT_NORTH, unit_ops)
        self.assertNotIn(ActionToken.UNIT_BUILD_COOP, unit_ops)
        self.assertNotIn(ActionToken.UNIT_DIG, unit_ops)
        self.assertNotIn(ActionToken.UNIT_BUILD_PASTURE, unit_ops)

        grammar.push(ActionToken.UNIT_PLANT)
        self.assertEqual(grammar.valid_tokens(), (ActionToken.ITEM_WHEAT,))

    def test_pass_cannot_be_removed_from_unit_mask(self) -> None:
        config = ActionGrammarConfig(allowed_unit_ops=frozenset({"WATER"}))
        grammar = ActionGrammar(make_observation(), config)
        grammar.push(ActionToken.BOS)
        grammar.push(ActionToken.FARMER)
        tokens = set(grammar.valid_tokens())

        self.assertIn(ActionToken.UNIT_PASS, tokens)
        self.assertNotIn(ActionToken.UNIT_WATER, tokens)

    def test_market_keeps_stop_and_hides_hire_in_wheat_loop(self) -> None:
        observation = make_observation()
        grammar = ActionGrammar(observation, wheat_loop_config())
        for token in (
            ActionToken.BOS,
            ActionToken.FARMER,
            ActionToken.UNIT_PASS,
            ActionToken.END_ACTION,
            ActionToken.MARKET,
        ):
            grammar.push(token)

        market_tokens = set(grammar.valid_tokens())
        self.assertIn(ActionToken.STOP, market_tokens)
        self.assertIn(ActionToken.MARKET_BUY_SEED, market_tokens)
        self.assertNotIn(ActionToken.MARKET_HIRE, market_tokens)
        self.assertNotIn(ActionToken.MARKET_BUY_LAND, market_tokens)
        self.assertNotIn(ActionToken.MARKET_BUY_ANIMAL, market_tokens)

        grammar.push(ActionToken.MARKET_BUY_SEED)
        self.assertEqual(grammar.valid_tokens(), (ActionToken.ITEM_WHEAT,))

    def test_unknown_curriculum_names_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Bilinmeyen unit"):
            ActionGrammarConfig(allowed_unit_ops=frozenset({"TELEPORT"}))
        with self.assertRaisesRegex(ValueError, "Bilinmeyen market"):
            ActionGrammarConfig(allowed_market_ops=frozenset({"SHORT"}))
        with self.assertRaisesRegex(ValueError, "Bilinmeyen item"):
            ActionGrammarConfig(allowed_items=frozenset({"STONE"}))

    def test_restricted_grammar_cannot_encode_blocked_action(self) -> None:
        observation = make_observation()
        with self.assertRaisesRegex(ValueError, "BUILD_COOP"):
            encode_action_tokens(
                {"farmer": ["BUILD_COOP"], "hands": [], "market": []},
                observation,
                wheat_loop_config(),
            )

    def test_wheat_loop_sample_never_emits_blocked_ops(self) -> None:
        observation = make_observation()
        observation["private"]["seeds"]["WHEAT"] = 3
        generator = torch.Generator()
        generator.manual_seed(0)
        sample = sample_action(
            KaggricultureActorCritic(),
            observation,
            grammar_config=wheat_loop_config(),
            generator=generator,
        )
        farmer_op = sample.action["farmer"][0]
        self.assertIn(
            farmer_op,
            {"PASS", "NORTH", "SOUTH", "EAST", "WEST", "PLANT", "DROP"},
        )
        for order in sample.action["market"]:
            self.assertIn(order[0], {"BUY_SEED", "SELL"})
            if len(order) > 1:
                self.assertEqual(order[1], "WHEAT")

    def test_wheat_loop_round_trip_still_works(self) -> None:
        observation = make_observation()
        observation["private"]["seeds"]["WHEAT"] = 1
        action = {
            "farmer": ["PLANT", "WHEAT"],
            "hands": [],
            "market": [["BUY_SEED", "WHEAT", 1]],
        }
        token_ids = encode_action_tokens(action, observation, wheat_loop_config())
        self.assertEqual(
            decode_action_tokens(token_ids, observation, wheat_loop_config()),
            action,
        )


class CurriculumScheduleTests(unittest.TestCase):
    def test_disabled_curriculum_is_unrestricted(self) -> None:
        resolved = resolve_episode_curriculum(
            CurriculumConfig(enabled=False),
            episode_number=1,
            episode_seed=0,
        )
        self.assertEqual(resolved.stage_name, "off")
        self.assertFalse(resolved.is_probe)
        self.assertEqual(resolved.starting_seeds, {})
        self.assertIsNone(resolved.grammar_config.allowed_unit_ops)

    def test_stage_advances_by_episode_window(self) -> None:
        config = CurriculumConfig(enabled=True, episodes_per_stage=10, probe_probability=0.0)
        self.assertEqual(select_stage(config, 1).name, "wheat_loop")
        self.assertEqual(select_stage(config, 10).name, "wheat_loop")
        self.assertEqual(select_stage(config, 11).name, "wheat_hire")
        self.assertEqual(select_stage(config, 21).name, "wheat_land")
        self.assertEqual(select_stage(config, 31).name, "wheat_carrot")
        self.assertEqual(select_stage(config, 41).name, "full")
        self.assertEqual(select_stage(config, 10_000).name, "full")

        full = resolve_episode_curriculum(config, 41, episode_seed=1)
        self.assertEqual(full.starting_seeds, {})
        self.assertIsNone(full.grammar_config.allowed_unit_ops)

    def test_probe_uses_full_mask_but_keeps_stage_seeds(self) -> None:
        config = CurriculumConfig(
            enabled=True,
            episodes_per_stage=100,
            probe_probability=1.0,
            starting_wheat_seeds=8,
        )
        resolved = resolve_episode_curriculum(config, 1, episode_seed=3)
        self.assertTrue(resolved.is_probe)
        self.assertEqual(resolved.stage_name, "wheat_loop")
        self.assertEqual(resolved.starting_seeds, {"WHEAT": 8})
        self.assertIsNone(resolved.grammar_config.allowed_unit_ops)
        self.assertIsNone(resolved.grammar_config.allowed_market_ops)

    def test_probe_is_deterministic_for_episode_seed(self) -> None:
        config = CurriculumConfig(enabled=True, probe_probability=0.4)
        first = resolve_episode_curriculum(config, 7, episode_seed=99)
        second = resolve_episode_curriculum(config, 7, episode_seed=99)
        self.assertEqual(first.is_probe, second.is_probe)
        self.assertEqual(first.stage_name, second.stage_name)

    def test_hire_stage_opens_hire_token(self) -> None:
        config = grammar_config_for_stage(WHEAT_CURRICULUM_STAGES[1])
        grammar = ActionGrammar(make_observation(), config)
        for token in (
            ActionToken.BOS,
            ActionToken.FARMER,
            ActionToken.UNIT_PASS,
            ActionToken.END_ACTION,
            ActionToken.MARKET,
        ):
            grammar.push(token)
        self.assertIn(ActionToken.MARKET_HIRE, grammar.valid_tokens())
        self.assertNotIn(ActionToken.MARKET_BUY_LAND, grammar.valid_tokens())


class CurriculumSeedInjectionTests(unittest.TestCase):
    def test_inject_starting_seeds_mutates_live_environment(self) -> None:
        env = make(
            "kaggriculture",
            configuration={"episodeSteps": 4, "seed": 1},
        )
        learner = 0
        before = int(env.state[learner].observation.private["seeds"]["WHEAT"])
        inject_starting_seeds(
            env.state[learner].observation,
            {"WHEAT": 8},
        )
        after = int(env.state[learner].observation.private["seeds"]["WHEAT"])
        opponent = int(env.state[1].observation.private["seeds"]["WHEAT"])

        self.assertEqual(before, 0)
        self.assertEqual(after, 8)
        self.assertEqual(opponent, 0)

        observation = deepcopy(env.state[learner].observation)
        grammar = ActionGrammar(observation, wheat_loop_config())
        grammar.push(ActionToken.BOS)
        grammar.push(ActionToken.FARMER)
        self.assertIn(ActionToken.UNIT_PLANT, grammar.valid_tokens())


if __name__ == "__main__":
    unittest.main()
