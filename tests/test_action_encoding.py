from copy import deepcopy
import unittest

import numpy as np
from kaggle_environments.envs.kaggriculture import kaggriculture as game

from action_encoding import (
    ACTION_VOCAB_SIZE,
    ID_TO_TOKEN,
    TOKEN_TO_ID,
    ActionGrammar,
    ActionHeadType,
    ActionToken,
    TOKENS_BY_HEAD,
    decode_action_tokens,
    encode_action_tokens,
    teacher_forcing_masks,
)


def make_observation() -> dict:
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


class ActionEncodingTests(unittest.TestCase):
    def test_initial_mask_and_vocab(self) -> None:
        grammar = ActionGrammar(make_observation())
        self.assertEqual(grammar.valid_tokens(), (ActionToken.BOS,))
        self.assertEqual(ACTION_VOCAB_SIZE, 49)
        self.assertEqual(grammar.valid_token_mask().shape, (49,))
        self.assertEqual(grammar.valid_token_mask().dtype, np.bool_)

    def test_output_head_categories_are_complete_and_disjoint(self) -> None:
        classified = [
            token
            for tokens in TOKENS_BY_HEAD.values()
            for token in tokens
        ]
        self.assertEqual(len(classified), ACTION_VOCAB_SIZE)
        self.assertEqual(len(set(classified)), ACTION_VOCAB_SIZE)
        self.assertIn(
            ActionToken.STOP,
            TOKENS_BY_HEAD[ActionHeadType.MARKET_OPERATION],
        )
        self.assertNotIn(
            ActionToken.STOP,
            TOKENS_BY_HEAD[ActionHeadType.STRUCTURAL],
        )

    def test_action_round_trip(self) -> None:
        observation = make_observation()
        action = {
            "farmer": ["BUILD_COOP"],
            "hands": [],
            "market": [
                ["BUY_SEED", "WHEAT", 5],
                ["BUY_ANIMAL", "GOOSE", 1],
                ["HIRE"],
                ["BUY_LAND"],
            ],
        }

        token_ids = encode_action_tokens(action, observation)

        self.assertEqual(decode_action_tokens(token_ids, observation), action)
        self.assertEqual(token_ids.dtype, np.int64)
        self.assertEqual(token_ids[0], TOKEN_TO_ID[ActionToken.BOS])
        self.assertEqual(token_ids[-1], TOKEN_TO_ID[ActionToken.EOS])

        masks = teacher_forcing_masks(token_ids, observation)
        self.assertEqual(masks.shape, (len(token_ids), ACTION_VOCAB_SIZE))
        self.assertTrue(
            all(mask[token_id] for mask, token_id in zip(masks, token_ids))
        )

    def test_missing_hand_actions_become_pass(self) -> None:
        observation = make_observation()
        observation["farms"][0]["hands"] = [[3, 4], [2, 4]]
        observation["private"]["inventories"].extend([{}, {}])

        token_ids = encode_action_tokens(
            {"farmer": ["PASS"], "hands": [], "market": []},
            observation,
        )
        decoded = decode_action_tokens(token_ids, observation)

        self.assertEqual(decoded["hands"], [["PASS"], ["PASS"]])

    def test_previous_unit_changes_next_unit_mask(self) -> None:
        observation = make_observation()
        observation["farms"][0]["hands"] = [[4, 4]]
        observation["private"]["inventories"].append({"GOOSE": 1})
        original = deepcopy(observation)
        action = {
            "farmer": ["BUILD_COOP"],
            "hands": [["PLACE", "GOOSE", 1]],
            "market": [],
        }

        token_ids = encode_action_tokens(action, observation)

        self.assertEqual(decode_action_tokens(token_ids, observation), action)
        # Gölge simülasyon kullanıcının observation'ını değiştirmemeli.
        self.assertEqual(observation, original)

    def test_seed_is_reserved_autoregressively(self) -> None:
        observation = make_observation()
        observation["farms"][0]["hands"] = [[3, 4]]
        observation["private"]["inventories"].append({})
        observation["private"]["seeds"]["WHEAT"] = 1

        grammar = ActionGrammar(observation)
        for token in (
            ActionToken.BOS,
            ActionToken.FARMER,
            ActionToken.UNIT_PLANT,
            ActionToken.ITEM_WHEAT,
            ActionToken.END_ACTION,
            ActionToken.HAND,
        ):
            grammar.push(token)

        self.assertNotIn(ActionToken.UNIT_PLANT, grammar.valid_tokens())
        with self.assertRaises(ValueError):
            grammar.push(ActionToken.UNIT_PLANT)

    def test_all_is_canonical_for_non_bucket_quantity(self) -> None:
        observation = make_observation()
        observation["private"]["shed"]["WHEAT"] = 7
        action = {
            "farmer": ["PASS"],
            "hands": [],
            "market": [["SELL", "WHEAT", 7]],
        }

        token_ids = encode_action_tokens(action, observation)
        tokens = [ID_TO_TOKEN[int(token_id)] for token_id in token_ids]

        self.assertIn(ActionToken.QTY_ALL, tokens)
        self.assertEqual(decode_action_tokens(token_ids, observation), action)

    def test_non_representable_quantity_is_rejected(self) -> None:
        observation = make_observation()
        observation["private"]["shed"]["WHEAT"] = 7
        action = {
            "farmer": ["PASS"],
            "hands": [],
            "market": [["SELL", "WHEAT", 3]],
        }

        with self.assertRaisesRegex(ValueError, "temsil edilemiyor"):
            encode_action_tokens(action, observation)


if __name__ == "__main__":
    unittest.main()
