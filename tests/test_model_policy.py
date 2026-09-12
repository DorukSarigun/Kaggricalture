import unittest

import numpy as np
import torch
from kaggle_environments.envs.kaggriculture import kaggriculture as game

from action_encoding import (
    ACTION_VOCAB_SIZE,
    ActionHeadType,
    decoder_head_types_for_tokens,
    decoder_inputs_for_tokens,
    encode_action_tokens,
)
from model import (
    CategorizedActionHeads,
    GlobalEncoder,
    KaggricultureActorCritic,
    SpatialEncoder,
)
from observation_encoding import UNIT_FEATURE_NAMES, encode_observation
from policy import sample_action
from scripted_training import action_entropy_statistics


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


def add_batch(array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(array, dtype=dtype).unsqueeze(0)


class PolicyModelTests(unittest.TestCase):
    def test_sample_exposes_head_entropy_and_normalized_entropy(self) -> None:
        model = KaggricultureActorCritic()
        sample = sample_action(model, make_observation(), deterministic=True)
        statistics = action_entropy_statistics(sample)

        self.assertEqual(sample.token_entropies.shape, sample.decision_mask.shape)
        self.assertEqual(
            set(statistics),
            {"unit_operation", "market_operation", "item", "quantity"},
        )
        self.assertTrue(
            all(
                0.0 <= row["mean_normalized_entropy"] <= 1.0
                for row in statistics.values()
            )
        )

    def test_only_selected_output_head_receives_nonzero_gradient(self) -> None:
        heads = CategorizedActionHeads(hidden_dim=8)
        hidden = torch.randn(1, 1, 8, requires_grad=True)
        logits = heads(
            hidden,
            torch.tensor([[int(ActionHeadType.UNIT_OPERATION)]]),
        )

        logits.sum().backward()

        for name, head in heads.heads.items():
            gradient_sum = (
                0.0
                if head.weight.grad is None
                else float(head.weight.grad.abs().sum())
            )
            if name == ActionHeadType.UNIT_OPERATION.name.lower():
                self.assertGreater(gradient_sum, 0.0)
            else:
                self.assertEqual(gradient_sum, 0.0)

    def test_actor_critic_forward_and_gradients(self) -> None:
        observation = make_observation()
        observation["private"]["seeds"]["WHEAT"] = 2
        encoded = encode_observation(observation)
        action = {
            "farmer": ["PLANT", "WHEAT"],
            "hands": [],
            "market": [],
        }
        token_ids = encode_action_tokens(action, observation)
        input_ids, _, _, context_indices = decoder_inputs_for_tokens(
            token_ids,
            observation,
        )
        head_type_ids = decoder_head_types_for_tokens(token_ids, observation)

        model = KaggricultureActorCritic()
        output = model(
            add_batch(encoded.own_categorical_ids, torch.long),
            add_batch(encoded.own_numeric_features, torch.float32),
            add_batch(encoded.own_worker_positions, torch.float32),
            add_batch(encoded.opponent_categorical_ids, torch.long),
            add_batch(encoded.opponent_numeric_features, torch.float32),
            add_batch(encoded.opponent_worker_positions, torch.float32),
            add_batch(encoded.global_features, torch.float32),
            add_batch(encoded.own_unit_positions_yx, torch.long),
            add_batch(encoded.own_unit_features, torch.float32),
            torch.ones((1, 1), dtype=torch.bool),
            add_batch(input_ids, torch.long),
            add_batch(context_indices, torch.long),
            add_batch(head_type_ids, torch.long),
        )

        self.assertEqual(
            output.token_logits.shape,
            (1, len(input_ids), ACTION_VOCAB_SIZE),
        )
        self.assertEqual(output.value.shape, (1,))
        self.assertEqual(output.unit_contexts.shape, (1, 1, 128))

        loss = output.token_logits.mean() + output.value.mean()
        loss.backward()
        self.assertIsNotNone(
            model.state_encoder.farm_encoder.tile_encoder
            .embedding.kind_embedding.weight.grad
        )
        unit_head = model.action_decoder.output_heads.heads[
            ActionHeadType.UNIT_OPERATION.name.lower()
        ]
        self.assertIsNotNone(unit_head.weight.grad)
        self.assertIsNotNone(model.value_head.network[0].weight.grad)

    def test_unit_specific_inventories_remain_separate(self) -> None:
        observation = make_observation()
        observation["farms"][0]["hands"] = [[4, 4]]
        observation["private"]["inventories"] = [
            {"WHEAT": 1},
            {"MELON": 3},
        ]

        encoded = encode_observation(observation)

        self.assertEqual(encoded.own_unit_positions_yx.shape, (2, 2))
        self.assertEqual(
            encoded.own_unit_features.shape,
            (2, len(UNIT_FEATURE_NAMES)),
        )
        self.assertFalse(
            np.array_equal(
                encoded.own_unit_features[0],
                encoded.own_unit_features[1],
            )
        )

    def test_spatial_encoder_keeps_farm_shape(self) -> None:
        encoder = SpatialEncoder()
        farm = torch.zeros(2, 32, 10, 10)
        farm[0, 0, 0, 0] = 1.0
        farm[1, 0, 9, 9] = 1.0
        output = encoder(farm)

        self.assertEqual(tuple(output.shape), (2, 64, 10, 10))
        self.assertFalse(torch.allclose(output[0], output[1]))

    def test_spatial_encoder_is_translation_equivariant(self) -> None:
        """Simetrik 3x3 padding, girdiyi kaydırmanın çıktıyı da kaydırmasını sağlar.

        Eski tek katmanlı 10x10 conv (4, 5, 4, 5) asimetrik padding kullandığı
        için bu özelliği taşımıyordu. Kenar etkisini dışlamak adına alıcı
        alandan geniş bir harita üzerinde iç bölge karşılaştırılır.
        """
        encoder = SpatialEncoder().eval()
        base = torch.zeros(1, 32, 24, 24)
        base[0, :, 8, 8] = 1.0
        shifted = torch.roll(base, shifts=(1, 1), dims=(2, 3))

        with torch.no_grad():
            base_output = encoder(base)
            shifted_output = encoder(shifted)

        interior = base_output[..., 6:18, 6:18]
        shifted_interior = shifted_output[..., 7:19, 7:19]

        # Testin sabit bir çıktı üzerinden boş yere geçmediğini garanti eder.
        self.assertGreater(float(interior.std()), 1e-3)
        # Tolerans float32 birikim gürültüsü içindir; sinyal mertebesi ~1.
        torch.testing.assert_close(
            interior,
            shifted_interior,
            atol=1e-4,
            rtol=1e-3,
        )

    def test_spatial_encoder_mixes_neighbouring_tiles(self) -> None:
        encoder = SpatialEncoder().eval()
        farm = torch.zeros(1, 32, 10, 10)
        farm[0, :, 5, 5] = 1.0

        with torch.no_grad():
            reference = encoder(torch.zeros(1, 32, 10, 10))
            output = encoder(farm)

        difference = (output - reference).abs().sum(dim=1)[0]
        self.assertGreater(float(difference[4, 5]), 0.0)
        self.assertGreater(float(difference[5, 4]), 0.0)

    def test_global_encoder_output_is_signed(self) -> None:
        encoder = GlobalEncoder()
        first_linear = encoder.network[0]
        last_linear = encoder.network[-2]
        with torch.no_grad():
            first_linear.weight.zero_()
            first_linear.bias.zero_()
            last_linear.weight.zero_()
            last_linear.bias.fill_(-1.0)
            output = encoder(torch.zeros(2, encoder.input_dim))

        self.assertTrue(torch.all(output < 0.0))
        self.assertTrue(torch.all(output > -1.0))
        self.assertTrue(torch.all(output < 1.0))


if __name__ == "__main__":
    unittest.main()
