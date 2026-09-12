import unittest

import numpy as np
import torch
from kaggle_environments.envs.kaggriculture import kaggriculture as game

from action_encoding import ActionHeadType
from lifecycle_credit import LifecycleTokenCreditRole
from model import KaggricultureActorCritic
from policy import sample_action
from ppo import (
    PPOConfig,
    _group_balanced_ppo_loss,
    _normalize_token_advantages,
    actor_critic_ppo_loss,
    collate_ppo_batch,
    compute_gae,
    masked_sequence_statistics,
    optimizer_step,
    span_log_probabilities,
)
from span_credit import SpanCreditRole


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


class PPOTests(unittest.TestCase):
    def test_span_credit_weight_is_independent_of_decision_token_count(self) -> None:
        token_log_probs = torch.tensor([[-0.2, -0.3, -0.4, -0.5]])
        decision_mask = torch.ones_like(token_log_probs, dtype=torch.bool)
        starts = torch.tensor([[0, 3]])
        ends = torch.tensor([[2, 3]])
        mask = torch.tensor([[True, True]])
        old_span_log_probs = span_log_probabilities(
            token_log_probs,
            decision_mask,
            starts,
            ends,
            mask,
        )

        loss, kl, _, _ = _group_balanced_ppo_loss(
            old_span_log_probs.clone(),
            old_span_log_probs,
            torch.tensor([[0.75, 0.75]]),
            torch.tensor(
                [[int(SpanCreditRole.PURCHASE), int(SpanCreditRole.FARM)]]
            ),
            mask,
            clip_ratio=0.2,
            group_weights=(1.0, 1.0, 1.0, 1.0),
        )
        # Biri üç, diğeri tek stochastic token içerse de ratio=1 iken her
        # semantik action tam bir 0.75 kredi terimidir.
        self.assertAlmostEqual(loss.item(), -0.75, places=6)
        self.assertAlmostEqual(kl.item(), 0.0, places=6)

        padded_log_probs = torch.cat(
            [token_log_probs, torch.tensor([[-99.0]])], dim=1
        )
        padded_decisions = torch.cat(
            [decision_mask, torch.tensor([[False]])], dim=1
        )
        padded_spans = span_log_probabilities(
            padded_log_probs,
            padded_decisions,
            starts,
            ends,
            mask,
        )
        torch.testing.assert_close(padded_spans, old_span_log_probs)

    def test_zero_decision_span_has_no_actor_term(self) -> None:
        zero, kl, clip, groups = _group_balanced_ppo_loss(
            torch.zeros((1, 1)),
            torch.zeros((1, 1)),
            torch.ones((1, 1)),
            torch.zeros((1, 1), dtype=torch.long),
            torch.zeros((1, 1), dtype=torch.bool),
            clip_ratio=0.2,
            group_weights=(1.0, 1.0, 1.0, 1.0),
        )
        self.assertEqual(zero.item(), 0.0)
        self.assertEqual(kl.item(), 0.0)
        self.assertEqual(clip.item(), 0.0)
        self.assertEqual(groups, {})

    def test_gae_respects_terminal_boundary(self) -> None:
        rewards = torch.tensor([[1.0], [1.0]])
        values = torch.zeros_like(rewards)
        dones = torch.tensor([[False], [True]])

        advantages, returns = compute_gae(
            rewards,
            values,
            dones,
            next_values=torch.tensor([123.0]),
            gamma=1.0,
            gae_lambda=1.0,
        )

        torch.testing.assert_close(advantages[:, 0], torch.tensor([2.0, 1.0]))
        torch.testing.assert_close(returns, advantages)

    def test_sampled_log_prob_matches_teacher_forcing(self) -> None:
        torch.manual_seed(7)
        model = KaggricultureActorCritic()
        observation = make_observation()
        sample = sample_action(model, observation, deterministic=True)
        batch = collate_ppo_batch([sample], [1.0], [sample.value])

        output = model(
            batch.own_categorical_ids,
            batch.own_numeric_features,
            batch.own_worker_positions,
            batch.opponent_categorical_ids,
            batch.opponent_numeric_features,
            batch.opponent_worker_positions,
            batch.global_features,
            batch.unit_positions_yx,
            batch.unit_features,
            batch.unit_mask,
            batch.input_token_ids,
            batch.context_unit_indices,
            batch.head_type_ids,
        )
        statistics = masked_sequence_statistics(
            output.token_logits,
            batch.target_token_ids,
            batch.valid_token_masks,
            batch.sequence_mask,
            batch.decision_mask,
        )

        self.assertAlmostEqual(
            statistics.action_log_probs.item(),
            sample.action_log_prob,
            places=5,
        )
        self.assertGreaterEqual(sample.action_log_prob, -1e6)
        self.assertTrue(np.isfinite(sample.action_log_prob))

    def test_variable_units_and_sequences_collate(self) -> None:
        torch.manual_seed(11)
        model = KaggricultureActorCritic()
        observation_a = make_observation()
        observation_b = make_observation()
        observation_b["farms"][0]["hands"] = [[4, 4], [3, 4]]
        observation_b["private"]["inventories"].extend([{}, {}])

        samples = [
            sample_action(model, observation_a, deterministic=True),
            sample_action(model, observation_b, deterministic=True),
        ]
        batch = collate_ppo_batch(
            samples,
            advantages=[1.0, -0.5],
            returns=[samples[0].value + 1.0, samples[1].value - 0.5],
        )

        self.assertEqual(batch.unit_mask.shape, (2, 3))
        self.assertEqual(batch.unit_mask.sum(dim=1).tolist(), [1, 3])
        self.assertEqual(batch.input_token_ids.shape[0], 2)
        self.assertEqual(batch.valid_token_masks.shape[-1], 49)

    def test_ppo_loss_is_finite_and_updates_all_branches(self) -> None:
        torch.manual_seed(19)
        model = KaggricultureActorCritic()
        observations = [make_observation(), make_observation()]
        observations[1]["private"]["seeds"]["CARROT"] = 3
        samples = [
            sample_action(model, observation, deterministic=False)
            for observation in observations
        ]
        batch = collate_ppo_batch(
            samples,
            advantages=[0.7, -0.2],
            returns=[samples[0].value + 0.7, samples[1].value - 0.2],
        )
        config = PPOConfig()
        loss = actor_critic_ppo_loss(model, batch, config)

        self.assertTrue(torch.isfinite(loss.total))
        self.assertLess(abs(loss.approximate_kl.item()), 1e-5)

        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        gradient_norm = optimizer_step(model, optimizer, loss, config)
        self.assertTrue(np.isfinite(gradient_norm))
        self.assertTrue(
            any(
                head.weight.grad is not None
                for head in model.action_decoder.output_heads.heads.values()
            )
        )
        self.assertIsNotNone(model.value_head.network[0].weight.grad)

    def test_token_advantage_creates_separate_lifecycle_policy_loss(self) -> None:
        torch.manual_seed(23)
        model = KaggricultureActorCritic()
        sample = sample_action(model, make_observation(), deterministic=True)
        token_advantages = np.zeros(
            len(sample.target_token_ids),
            dtype=np.float32,
        )
        selected = int(np.flatnonzero(sample.decision_mask)[0])
        token_advantages[selected] = 0.25
        batch = collate_ppo_batch(
            [sample],
            advantages=[0.0],
            returns=[sample.value],
            token_advantages=[token_advantages],
        )
        loss = actor_critic_ppo_loss(
            model,
            batch,
            PPOConfig(value_coef=0.0, entropy_coef=0.0),
        )

        # Rollout policy ile teacher-forcing policy aynı: token ratio=1.
        self.assertAlmostEqual(loss.policy.item(), 0.0, places=6)
        self.assertAlmostEqual(loss.lifecycle_policy.item(), -0.25, places=5)
        self.assertAlmostEqual(loss.total.item(), -0.25, places=5)
        self.assertEqual(int(batch.token_credit_mask.sum()), 1)

        loss.total.backward()
        selected_head = ActionHeadType(int(sample.head_type_ids[selected]))
        for head_type, head in (
            model.action_decoder.output_heads.heads.items()
        ):
            gradient = head.weight.grad
            gradient_norm = (
                float(gradient.abs().sum()) if gradient is not None else 0.0
            )
            if head_type == selected_head.name.lower():
                self.assertGreater(gradient_norm, 0.0)
            else:
                self.assertEqual(gradient_norm, 0.0)

    @staticmethod
    def _credited_batch(samples, credits):
        rows = []
        for sample, credit in zip(samples, credits):
            row = np.zeros(len(sample.target_token_ids), dtype=np.float32)
            row[int(np.flatnonzero(sample.decision_mask)[0])] = credit
            rows.append(row)
        return collate_ppo_batch(
            samples,
            advantages=[0.0] * len(samples),
            returns=[sample.value for sample in samples],
            token_advantages=rows,
        )

    def _lifecycle_gradient_norm(self, model, batch, *, normalize):
        loss = actor_critic_ppo_loss(
            model,
            batch,
            PPOConfig(
                value_coef=0.0,
                entropy_coef=0.0,
                normalize_token_advantages=normalize,
            ),
        )
        gradients = torch.autograd.grad(
            loss.lifecycle_policy,
            [
                head.weight
                for head in model.action_decoder.output_heads.heads.values()
            ],
            allow_unused=True,
        )
        return float(
            sum(
                gradient.abs().sum()
                for gradient in gradients
                if gradient is not None
            )
        )

    def test_role_normalization_keeps_production_signal_alive(self) -> None:
        """purchase kredisi üretim rollerinden ~36x geniş yayılıyor.

        Tek havuzda normalize edilince global std'yi purchase belirliyor ve
        üretim kararları arasındaki fark ölçülemez hale geliyordu.
        """
        torch.manual_seed(5)
        purchase = torch.randn(64) * 0.029
        care = 0.0032 + torch.randn(64) * 0.0008
        advantages = torch.cat([purchase, care])
        roles = torch.cat(
            [
                torch.full((64,), int(LifecycleTokenCreditRole.PURCHASE.role_id)),
                torch.full((64,), int(LifecycleTokenCreditRole.CARE.role_id)),
            ]
        )

        pooled = _normalize_token_advantages(
            advantages, roles=None, min_role_samples=8
        )
        per_role = _normalize_token_advantages(
            advantages, roles=roles, min_role_samples=8
        )

        # Havuzlanmış ölçekte care sinyali ezilir; rol bazında birim varyansa döner.
        self.assertLess(pooled[64:].std(unbiased=False).item(), 0.1)
        self.assertAlmostEqual(
            per_role[64:].std(unbiased=False).item(), 1.0, places=4
        )
        self.assertAlmostEqual(
            per_role[64:].mean().item(), 0.0, places=5
        )
        # purchase kendi içinde hâlâ normalize.
        self.assertAlmostEqual(
            per_role[:64].std(unbiased=False).item(), 1.0, places=4
        )
        # Rol içi sıralama korunmalı: normalizasyon yalnızca kaydırıp ölçekler.
        self.assertTrue(
            torch.equal(care.argsort(), per_role[64:].argsort())
        )

    def test_role_normalization_falls_back_when_samples_are_scarce(self) -> None:
        advantages = torch.tensor([1.0, 2.0, 3.0, 10.0])
        roles = torch.tensor([0, 0, 0, 1])
        # transport gibi seyrek roller kendi gürültülü std'sine düşmemeli.
        result = _normalize_token_advantages(
            advantages, roles=roles, min_role_samples=8
        )
        pooled = _normalize_token_advantages(
            advantages, roles=None, min_role_samples=8
        )
        self.assertTrue(torch.allclose(result, pooled))

    def test_entropy_head_weights_target_a_single_head(self) -> None:
        """Market kafası çökerken hareket kafası yüksek entropide kalıyordu.

        Global katsayı ikisini ayıramadığı için ağırlık kafa bazında uygulanır.
        """
        torch.manual_seed(23)
        model = KaggricultureActorCritic()
        sample = sample_action(model, make_observation(), deterministic=True)
        batch = collate_ppo_batch(
            [sample],
            advantages=[0.0],
            returns=[sample.value],
        )
        present = set(int(head) for head in sample.head_type_ids[sample.decision_mask])
        self.assertIn(int(ActionHeadType.MARKET_OPERATION), present)
        self.assertIn(int(ActionHeadType.UNIT_OPERATION), present)

        def entropy_for(weights):
            return actor_critic_ppo_loss(
                model,
                batch,
                PPOConfig(
                    value_coef=0.0,
                    entropy_coef=1.0,
                    entropy_head_weights=weights,
                ),
            ).entropy.item()

        uniform = entropy_for((1.0, 1.0, 1.0, 1.0, 1.0))
        per_head = [
            entropy_for(tuple(1.0 if index == head else 0.0 for index in range(5)))
            for head in range(1, 5)
        ]
        market_only = per_head[int(ActionHeadType.MARKET_OPERATION) - 1]
        unit_only = per_head[int(ActionHeadType.UNIT_OPERATION) - 1]
        market_boosted = entropy_for((1.0, 1.0, 4.0, 1.0, 1.0))

        self.assertGreater(market_only, 0.0)
        self.assertGreater(unit_only, 0.0)
        # Kafa katkıları toplamı bütünü vermeli: ağırlıklar örtüşmez.
        self.assertAlmostEqual(sum(per_head), uniform, places=5)
        # Ağırlık yalnızca kendi kafasını ölçeklemeli.
        self.assertAlmostEqual(market_boosted - uniform, 3.0 * market_only, places=5)

    def test_equally_profitable_chains_get_no_lifecycle_gradient(self) -> None:
        """Baseline olmadan her kârlı zincir körlemesine pekiştirilirdi."""
        torch.manual_seed(23)
        model = KaggricultureActorCritic()
        samples = [
            sample_action(model, make_observation(), deterministic=True)
            for _ in range(2)
        ]
        batch = self._credited_batch(samples, [0.25, 0.25])
        self.assertEqual(int(batch.token_credit_mask.sum()), 2)

        centred = actor_critic_ppo_loss(
            model,
            batch,
            PPOConfig(value_coef=0.0, entropy_coef=0.0),
        )
        raw = actor_critic_ppo_loss(
            model,
            batch,
            PPOConfig(
                value_coef=0.0,
                entropy_coef=0.0,
                normalize_token_advantages=False,
            ),
        )

        self.assertAlmostEqual(centred.lifecycle_policy.item(), 0.0, places=6)
        self.assertAlmostEqual(raw.lifecycle_policy.item(), -0.25, places=5)

    def test_token_credit_normalization_removes_scale_sensitivity(self) -> None:
        """Ham krediler ana avantajlarla kıyaslanamaz bir ölçekte gelir.

        Normalize edildiğinde ``lifecycle_policy_coef`` sabit bir ağırlık olur;
        edilmediğinde efektif ağırlık kredilerin büyüklüğüyle sürüklenir.
        """
        torch.manual_seed(23)
        model = KaggricultureActorCritic()
        samples = [
            sample_action(model, make_observation(), deterministic=True)
            for _ in range(2)
        ]
        small = self._credited_batch(samples, [0.01, 0.02])
        large = self._credited_batch(samples, [1.00, 2.00])

        normalized = [
            self._lifecycle_gradient_norm(model, batch, normalize=True)
            for batch in (small, large)
        ]
        unnormalized = [
            self._lifecycle_gradient_norm(model, batch, normalize=False)
            for batch in (small, large)
        ]

        self.assertGreater(normalized[0], 0.0)
        self.assertAlmostEqual(normalized[0], normalized[1], places=5)
        self.assertGreater(unnormalized[1], 50.0 * unnormalized[0])


if __name__ == "__main__":
    unittest.main()
