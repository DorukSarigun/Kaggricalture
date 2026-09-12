from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from action_encoding import ActionGrammarConfig
from lagged_self_play import LaggedSelfPlayConfig, LaggedSelfPlayOpponent
from lifecycle_credit import LifecycleCreditConfig, LifecycleCreditMode
from model import KaggricultureActorCritic
from ppo import PPOConfig
from scripted_training import collect_scripted_episode, environment_reward_config
from train_scripted import build_parser


def _first_parameter(model: KaggricultureActorCritic) -> torch.Tensor:
    return next(model.parameters()).detach().cpu().clone()


class LaggedSelfPlayTests(unittest.TestCase):
    def test_cli_defaults_keep_opponent_one_hundred_updates_behind(self) -> None:
        args = build_parser().parse_args(["train", "--opponent", "selfplay"])
        self.assertEqual(args.self_play_lag_updates, 100)
        self.assertEqual(args.self_play_snapshot_interval, 10)
        self.assertFalse(args.self_play_opponent_deterministic)

    def test_opponent_uses_old_snapshot_and_is_never_trainable(self) -> None:
        torch.manual_seed(17)
        learner = KaggricultureActorCritic()
        initial = _first_parameter(learner)

        with TemporaryDirectory() as directory:
            manager = LaggedSelfPlayOpponent(
                learner,
                run_directory=Path(directory),
                device=torch.device("cpu"),
                config=LaggedSelfPlayConfig(
                    lag_updates=2,
                    snapshot_interval=1,
                    deterministic=True,
                ),
                seed=91,
            )
            self.assertTrue(
                all(not parameter.requires_grad for parameter in manager.model.parameters())
            )
            self.assertTrue(torch.equal(_first_parameter(manager.model), initial))

            with torch.no_grad():
                for parameter in learner.parameters():
                    parameter.add_(1.0)
            first = manager.after_learner_update(learner, 1)
            self.assertEqual(first.opponent_update, 0)

            with torch.no_grad():
                for parameter in learner.parameters():
                    parameter.add_(1.0)
            second = manager.after_learner_update(learner, 2)
            self.assertEqual(second.opponent_update, 0)

            with torch.no_grad():
                for parameter in learner.parameters():
                    parameter.add_(1.0)
            third = manager.after_learner_update(learner, 3)
            self.assertEqual(third.opponent_update, 1)
            self.assertEqual(third.actual_lag_updates, 2)
            self.assertTrue(
                torch.equal(_first_parameter(manager.model), initial + 1.0)
            )
            self.assertFalse(
                torch.equal(_first_parameter(manager.model), _first_parameter(learner))
            )

            state = manager.state_dict()
            restored = LaggedSelfPlayOpponent(
                learner,
                run_directory=Path(directory),
                device=torch.device("cpu"),
                config=manager.config,
                seed=999,
                learner_update=3,
                resume_state=state,
            )
            self.assertEqual(restored.opponent_update, 1)
            self.assertTrue(
                torch.equal(_first_parameter(restored.model), initial + 1.0)
            )
            self.assertTrue(
                torch.equal(
                    restored.generator.get_state(),
                    manager.generator.get_state(),
                )
            )

    def test_episode_can_use_frozen_model_policy_as_opponent(self) -> None:
        learner = KaggricultureActorCritic()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manager = LaggedSelfPlayOpponent(
                learner,
                run_directory=root,
                device=torch.device("cpu"),
                config=LaggedSelfPlayConfig(
                    lag_updates=2,
                    snapshot_interval=1,
                    deterministic=True,
                ),
                seed=19,
            )
            result = collect_scripted_episode(
                learner,
                opponent_name="selfplay",
                opponent_policy=lambda observation: manager.act(
                    observation,
                    grammar_config=ActionGrammarConfig(),
                ),
                episode_number=1,
                seed=123,
                learner_player=0,
                environment_configuration={
                    "episodeSteps": 4,
                    "turnsPerDay": 2,
                    "startingMoney": 3000,
                },
                reward_config=environment_reward_config(
                    episode_steps=4,
                    turns_per_day=2,
                    starting_money=3000,
                    gamma=0.999,
                    shaping_beta=0.0,
                ),
                ppo_config=PPOConfig(use_action_span_policy=True),
                output_root=root,
                save_replay=False,
                save_trace=False,
                deterministic_actions=True,
                torch_generator=torch.Generator().manual_seed(23),
                log_step_every=0,
                lifecycle_credit_config=LifecycleCreditConfig(
                    mode=LifecycleCreditMode.ECONOMIC_SPAN
                ),
            )

        self.assertEqual(result.summary["opponent"], "selfplay")
        self.assertGreater(len(result.rollout.transitions), 0)


if __name__ == "__main__":
    unittest.main()
