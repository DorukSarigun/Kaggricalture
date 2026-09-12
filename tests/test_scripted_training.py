import random
from tempfile import TemporaryDirectory
from pathlib import Path
import unittest

import numpy as np
import torch

from lifecycle_credit import LifecycleCreditConfig, LifecycleCreditMode
from model import KaggricultureActorCritic
from ppo import PPOConfig
from scripted_training import (
    CHECKPOINT_FORMAT_VERSION,
    _prepare_episode_directory,
    checkpoint_payload,
    collect_scripted_episode,
    environment_reward_config,
    load_checkpoint,
    save_checkpoint,
)
from train_scripted import build_parser


class ScriptedTrainingTests(unittest.TestCase):
    def test_economic_cli_defaults_use_terminal_actor_and_audit_only_local(self) -> None:
        args = build_parser().parse_args(["train"])
        self.assertEqual(args.lifecycle_credit_mode, "economic_span")
        self.assertEqual(args.shaping_beta, 0.0)
        self.assertEqual(args.global_actor_coef, 1.0)
        self.assertEqual(args.span_policy_coef, 0.0)
        self.assertTrue(args.curriculum)
        self.assertEqual(args.curriculum_probe_probability, 0.0)
        self.assertEqual(len(args.span_credit_role_weights), 4)

    def test_economic_span_collection_writes_separate_audit_artifacts(self) -> None:
        with TemporaryDirectory() as directory:
            result = collect_scripted_episode(
                KaggricultureActorCritic(),
                opponent_name="pass",
                episode_number=1,
                seed=123,
                learner_player=0,
                environment_configuration={
                    "episodeSteps": 6,
                    "turnsPerDay": 3,
                    "startingMoney": 3000,
                },
                reward_config=environment_reward_config(
                    episode_steps=6,
                    turns_per_day=3,
                    starting_money=3000,
                    gamma=0.999,
                    shaping_beta=0.0,
                ),
                ppo_config=PPOConfig(use_action_span_policy=True),
                output_root=Path(directory),
                save_replay=False,
                save_trace=False,
                deterministic_actions=True,
                torch_generator=torch.Generator().manual_seed(99),
                log_step_every=0,
                lifecycle_credit_config=LifecycleCreditConfig(
                    mode=LifecycleCreditMode.ECONOMIC_SPAN
                ),
            )

            self.assertAlmostEqual(
                result.summary["economic_credit_invariant_error"], 0.0
            )
            self.assertTrue(result.action_spans_path.exists())
            self.assertTrue(result.span_credit_assignments_path.exists())
            self.assertTrue(result.span_training_credits_path.exists())
            self.assertTrue((result.episode_directory / "cash_ledger.json").exists())
            self.assertEqual(result.summary["cash_accounting"]["max_reconciliation_error"], 0)
        for record in result.rollout.records():
            action_ids = [credit.action_id for credit in record.span_credits]
            self.assertEqual(len(action_ids), len(set(action_ids)))

    def test_incomplete_episode_is_archived_but_complete_episode_is_protected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / "episodes" / "episode_000003_seed_9_p0"
            episode.mkdir(parents=True)
            (episode / "partial.txt").write_text("partial", encoding="utf-8")

            _prepare_episode_directory(root, episode)

            archived = list((root / "failed_episodes").iterdir())
            self.assertEqual(len(archived), 1)
            self.assertTrue((archived[0] / "partial.txt").exists())
            self.assertTrue(episode.exists())
            (episode / "complete.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "Tamamlanmış episode"):
                _prepare_episode_directory(root, episode)

    def test_checkpoint_restores_model_optimizer_and_sampling_generator(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        random.seed(13)
        np.random.seed(13)
        torch.manual_seed(13)

        model = KaggricultureActorCritic().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        generator = torch.Generator(device=device.type)
        generator.manual_seed(991)
        _ = torch.rand(5, generator=generator, device=device)
        expected_generator_state = generator.get_state().cpu().clone()

        payload = checkpoint_payload(
            model,
            optimizer,
            episode=7,
            update=3,
            total_environment_steps=1440,
            run_configuration={"opponent": "starter"},
            sampling_generator=generator,
        )

        with TemporaryDirectory() as directory:
            run_directory = Path(directory)
            checkpoint_path = save_checkpoint(
                run_directory,
                payload,
                numbered=True,
            )
            restored_model = KaggricultureActorCritic().to(device)
            restored_optimizer = torch.optim.Adam(restored_model.parameters())
            checkpoint = load_checkpoint(
                checkpoint_path,
                restored_model,
                restored_optimizer,
                device,
                restore_rng=True,
            )

        self.assertEqual(checkpoint["episode"], 7)
        self.assertEqual(checkpoint["update"], 3)
        self.assertTrue(
            torch.equal(
                checkpoint["sampling_generator_state"].cpu(),
                expected_generator_state,
            )
        )
        for expected, actual in zip(model.parameters(), restored_model.parameters()):
            self.assertTrue(torch.equal(expected, actual))

    def test_superseded_checkpoint_format_is_rejected_with_guidance(self) -> None:
        """v1 checkpoint'leri eski tek katmanlı SpatialEncoder ağırlıkları taşır."""
        device = torch.device("cpu")
        model = KaggricultureActorCritic()
        optimizer = torch.optim.Adam(model.parameters())
        payload = dict(
            checkpoint_payload(
                model,
                optimizer,
                episode=1,
                update=1,
                total_environment_steps=24,
                run_configuration={"opponent": "starter"},
            )
        )
        self.assertEqual(payload["format_version"], CHECKPOINT_FORMAT_VERSION)
        payload["format_version"] = 1

        with TemporaryDirectory() as directory:
            checkpoint_path = save_checkpoint(
                Path(directory),
                payload,
                numbered=False,
            )
            with self.assertRaises(ValueError) as caught:
                load_checkpoint(
                    checkpoint_path,
                    KaggricultureActorCritic(),
                    None,
                    device,
                    restore_rng=False,
                )

        self.assertIn("SpatialEncoder", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
