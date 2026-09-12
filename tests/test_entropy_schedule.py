import unittest

from entropy_schedule import EntropyScheduleConfig, episode_in_stage


class EntropyScheduleTests(unittest.TestCase):
    def test_curriculum_stage_coefficients_are_exact(self) -> None:
        schedule = EntropyScheduleConfig()

        self.assertEqual(
            schedule.coefficient("wheat_loop", 500, fixed_coefficient=0.01),
            0.030,
        )
        self.assertEqual(
            schedule.coefficient("wheat_hire", 1, fixed_coefficient=0.01),
            0.025,
        )
        self.assertEqual(
            schedule.coefficient("wheat_land", 1, fixed_coefficient=0.01),
            0.020,
        )
        self.assertEqual(
            schedule.coefficient("wheat_carrot", 1, fixed_coefficient=0.01),
            0.015,
        )

    def test_full_stage_decays_to_nonzero_floor(self) -> None:
        schedule = EntropyScheduleConfig(
            full_start=0.012,
            full_end=0.005,
            full_decay_episodes=3000,
        )

        self.assertEqual(
            schedule.coefficient("full", 1, fixed_coefficient=0.01),
            0.012,
        )
        self.assertEqual(
            schedule.coefficient("full", 3000, fixed_coefficient=0.01),
            0.005,
        )
        self.assertEqual(
            schedule.coefficient("full", 9999, fixed_coefficient=0.01),
            0.005,
        )

    def test_disabled_schedule_uses_fixed_coefficient(self) -> None:
        schedule = EntropyScheduleConfig(enabled=False)
        self.assertEqual(
            schedule.coefficient("off", 100, fixed_coefficient=0.017),
            0.017,
        )

    def test_episode_in_stage_resets_at_stage_boundary(self) -> None:
        self.assertEqual(episode_in_stage(1, 0, 750), 1)
        self.assertEqual(episode_in_stage(750, 0, 750), 750)
        self.assertEqual(episode_in_stage(751, 1, 750), 1)
        self.assertEqual(episode_in_stage(3001, 4, 750), 1)
        self.assertEqual(episode_in_stage(42, -1, 750), 42)


if __name__ == "__main__":
    unittest.main()
