from copy import deepcopy
from dataclasses import replace
import unittest

import numpy as np
import torch
from kaggle_environments.envs.kaggriculture import kaggriculture as game

from model import KaggricultureActorCritic
from policy import sample_action
from ppo import PPOConfig
from reward import RewardConfig
from training import EpisodeRollout, ppo_update


def make_observation(step: int = 0) -> dict:
    return {
        "player": 0,
        "step": step,
        "day": step // 24,
        "hour": step % 24,
        "farms": [game._new_farm(10, 3000), game._new_farm(10, 3000)],
        "private": game._new_private(),
        "market": game._new_market(),
        "town": game._new_town(),
    }


class TrainingTests(unittest.TestCase):
    def test_token_credit_only_reaches_stochastic_tokens_in_span(self) -> None:
        model = KaggricultureActorCritic()
        state = make_observation()
        sample = sample_action(model, state, deterministic=True)
        rollout = EpisodeRollout()
        rollout.add(sample, state, state, done=False)

        credited = rollout.apply_token_credit(
            environment_step=0,
            token_start=0,
            token_end=len(sample.target_token_ids) - 1,
            amount=0.25,
        )
        token_advantages = rollout.transitions[0].token_advantages

        self.assertEqual(credited, int(sample.decision_mask.sum()))
        np.testing.assert_allclose(
            token_advantages[sample.decision_mask],
            0.25,
        )
        np.testing.assert_allclose(
            token_advantages[~sample.decision_mask],
            0.0,
        )

    def test_lifecycle_credit_is_kept_separate_and_added_to_total(self) -> None:
        model = KaggricultureActorCritic()
        state = make_observation()
        rollout = EpisodeRollout()
        reward = rollout.add(
            sample_action(model, state, deterministic=True),
            state,
            state,
            done=False,
            lifecycle_credit=0.25,
        )

        self.assertAlmostEqual(reward.lifecycle_credit, 0.25)
        self.assertAlmostEqual(
            reward.total,
            reward.base + reward.shaping + reward.lifecycle_credit,
        )

    def test_reward_and_ppo_gamma_must_match(self) -> None:
        with self.assertRaisesRegex(ValueError, "aynı gamma"):
            EpisodeRollout(
                reward_config=RewardConfig(gamma=0.99),
                ppo_config=PPOConfig(gamma=0.999),
            )

    def test_episode_records_and_ppo_update(self) -> None:
        torch.manual_seed(31)
        model = KaggricultureActorCritic()
        rollout = EpisodeRollout()

        state_0 = make_observation(0)
        state_1 = deepcopy(state_0)
        state_1["step"] = 1
        state_1["hour"] = 1
        state_1["farms"][0]["money"] = 3200
        state_2 = deepcopy(state_1)
        state_2["step"] = 2
        state_2["hour"] = 2
        state_2["farms"][0]["money"] = 3500

        rollout.add(
            sample_action(model, state_0, deterministic=True),
            state_0,
            state_1,
            done=False,
        )
        rollout.add(
            sample_action(model, state_1, deterministic=True),
            state_1,
            state_2,
            done=True,
        )
        records = rollout.records()

        self.assertEqual(len(records), 2)
        self.assertTrue(all(np.isfinite(record.advantage) for record in records))
        self.assertTrue(all(np.isfinite(record.return_target) for record in records))

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        metrics = ppo_update(
            model,
            optimizer,
            records,
            epochs=2,
            minibatch_size=2,
        )
        self.assertGreaterEqual(metrics.minibatches, 1)
        self.assertTrue(np.isfinite(metrics.total_loss))
        self.assertTrue(np.isfinite(metrics.gradient_norm))

    def test_outlier_minibatch_is_skipped_not_fatal_to_the_update(self) -> None:
        """Tek bir aykırı minibatch tüm update'i çöpe atmamalı.

        Ölçülen davranış buydu: update başına uygulanan minibatch sayısı
        100'den 4.7'ye düşüyor ve model neredeyse hiç gradyan almıyordu.
        """
        torch.manual_seed(31)
        model = KaggricultureActorCritic()
        rollout = EpisodeRollout()
        states = [make_observation(step) for step in range(8)]
        for index, state in enumerate(states[:-1]):
            rollout.add(
                sample_action(model, state, deterministic=True),
                state,
                states[index + 1],
                done=index == len(states) - 2,
            )
        records = rollout.records()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # target_kl=0 her minibatch'i eşiğin üstüne iter; kl_skip_multiplier
        # sonsuz olduğu için hiçbiri atlanmaz, yalnızca epoch sonunda durulur.
        metrics = ppo_update(
            model,
            optimizer,
            records,
            config=PPOConfig(target_kl=0.0, kl_skip_multiplier=float("inf")),
            epochs=3,
            minibatch_size=2,
        )

        self.assertTrue(metrics.stopped_for_kl)
        self.assertEqual(metrics.skipped_minibatches, 0)
        # Eski davranışta bu 0 olurdu: ilk minibatch'te break edilirdi.
        self.assertGreaterEqual(metrics.minibatches, len(records) // 2)
        self.assertEqual(metrics.epochs_completed, 1)

    def test_all_minibatches_above_skip_threshold_raises(self) -> None:
        torch.manual_seed(31)
        model = KaggricultureActorCritic()
        rollout = EpisodeRollout()
        states = [make_observation(step) for step in range(4)]
        for index, state in enumerate(states[:-1]):
            rollout.add(
                sample_action(model, state, deterministic=True),
                state,
                states[index + 1],
                done=index == len(states) - 2,
            )
        # Kaydedilmiş log-prob'ları kaydırmak her minibatch'i eşiğin üstüne
        # çıkarır; bu durumda sessizce ilerlemek yerine hata verilmelidir.
        records = [
            replace(
                record,
                sample=replace(
                    record.sample,
                    action_log_prob=record.sample.action_log_prob + 5.0,
                ),
            )
            for record in rollout.records()
        ]
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        with self.assertRaisesRegex(RuntimeError, "KL eşiğini aştı"):
            ppo_update(
                model,
                optimizer,
                records,
                epochs=1,
                minibatch_size=1,
            )

    def test_terminal_closes_rollout(self) -> None:
        model = KaggricultureActorCritic()
        state = make_observation()
        rollout = EpisodeRollout()
        sample = sample_action(model, state, deterministic=True)
        rollout.add(sample, state, state, done=True)

        with self.assertRaisesRegex(RuntimeError, "Terminal"):
            rollout.add(sample, state, state, done=False)


if __name__ == "__main__":
    unittest.main()
