from copy import deepcopy
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from kaggle_environments import make

from action_encoding import encode_action_tokens
from kinds import PlantTile, parse_tile
from lifecycle_tracking import FarmLifecycleTracker, LifecycleEventType
from lifecycle_credit import (
    LifecycleCreditAssigner,
    LifecycleCreditConfig,
    LifecycleCreditMode,
    LifecycleTokenCreditRole,
)
from tile_encoding import NUMERIC_FEATURE_NAMES, tile_to_numeric_features


PASS = {"farmer": ["PASS"], "hands": [], "market": []}


class LifecycleTrackingTests(unittest.TestCase):
    def _play(self, env, tracker, action) -> None:
        observation = deepcopy(env.state[0].observation)
        token_ids = encode_action_tokens(action, observation)
        env.step([action, PASS])
        next_observation = deepcopy(env.state[0].observation)
        tracker.observe_transition(
            observation,
            action,
            next_observation,
            token_ids=token_ids,
            transition_reward=0.0,
        )

    def test_seed_to_plant_to_harvest_to_sale_lineage(self) -> None:
        env = make(
            "kaggriculture",
            configuration={
                "episodeSteps": 30,
                "turnsPerDay": 4,
                "weedSpawnChance": 0.0,
                "seed": 77,
            },
        )
        tracker = FarmLifecycleTracker(
            env.state[0].observation,
            episode_number=12,
            player_index=0,
            turns_per_day=4,
        )
        actions = [
            {
                "farmer": ["PASS"],
                "hands": [],
                "market": [["BUY_SEED", "WHEAT", 1]],
            },
            {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": []},
            {"farmer": ["WATER"], "hands": [], "market": []},
            PASS,
            {"farmer": ["WATER"], "hands": [], "market": []},
            PASS,
            PASS,
            PASS,
            {"farmer": ["WATER"], "hands": [], "market": []},
            {"farmer": ["HARVEST"], "hands": [], "market": []},
            {"farmer": ["DROP"], "hands": [], "market": []},
            {
                "farmer": ["PASS"],
                "hands": [],
                "market": [["SELL", "WHEAT", 2]],
            },
        ]
        for action in actions:
            self._play(env, tracker, action)

        event_types = [event.event_type for event in tracker.events]
        self.assertEqual(event_types.count(LifecycleEventType.SEED_PURCHASED), 1)
        self.assertEqual(event_types.count(LifecycleEventType.PLANTED), 1)
        self.assertEqual(event_types.count(LifecycleEventType.WATERED), 3)
        self.assertEqual(event_types.count(LifecycleEventType.HARVESTED), 1)
        self.assertEqual(event_types.count(LifecycleEventType.PRODUCT_SOLD), 1)

        purchase = next(
            event
            for event in tracker.events
            if event.event_type is LifecycleEventType.SEED_PURCHASED
        )
        planted = next(
            event
            for event in tracker.events
            if event.event_type is LifecycleEventType.PLANTED
        )
        harvested = next(
            event
            for event in tracker.events
            if event.event_type is LifecycleEventType.HARVESTED
        )
        self.assertEqual(planted.seed_lot_id, purchase.seed_lot_id)
        self.assertEqual(planted.tile_xy, (4, 4))
        self.assertIn("UNIT:PLANT", planted.token_span.tokens)
        self.assertIn("MARKET:BUY_SEED", purchase.token_span.tokens)
        self.assertEqual(harvested.quantity, 2)
        self.assertEqual(harvested.normalized_credit, 52 / 3000)

        plants = tracker.plant_lifecycles()
        self.assertEqual(len(plants), 1)
        plant = plants[0]
        self.assertEqual(plant["status"], "HARVESTED")
        self.assertEqual(plant["water_steps"], [2, 4, 8])
        self.assertEqual(plant["harvested_units"], 2)
        self.assertEqual(plant["sold_units"], 2)
        self.assertEqual(plant["estimated_sale_revenue"], 51.0)
        self.assertEqual(plant["estimated_realized_profit"], 41.0)

        token_assignments = LifecycleCreditAssigner(
            LifecycleCreditConfig(
                mode=LifecycleCreditMode.CAUSAL_CHAIN,
                coefficient=1.0,
            )
        ).assign_episode_token_credits(tracker.events)
        by_role = {}
        for assignment in token_assignments:
            by_role.setdefault(assignment.role, []).append(assignment)

        # Purchase span tam order netini (+41/3000) alır. Bağımsız üretim
        # kolunda PLANT + 3 WATER + HARVEST + DROP + SELL = 7 aksiyon aynı
        # plant netini kendi aralarında eşit paylaşır.
        production_credit = (41 / 3000) / 7
        self.assertEqual(len(token_assignments), 8)
        self.assertAlmostEqual(
            by_role[LifecycleTokenCreditRole.PURCHASE][0].amount,
            41 / 3000,
        )
        production_assignments = [
            assignment
            for assignment in token_assignments
            if assignment.role is not LifecycleTokenCreditRole.PURCHASE
        ]
        self.assertTrue(
            all(
                abs(assignment.amount - production_credit) < 1e-9
                for assignment in production_assignments
            )
        )
        self.assertAlmostEqual(
            by_role[LifecycleTokenCreditRole.PLANT][0].amount,
            production_credit,
        )
        self.assertEqual(len(by_role[LifecycleTokenCreditRole.CARE]), 3)
        self.assertTrue(
            all(
                abs(assignment.amount - production_credit) < 1e-9
                for assignment in by_role[LifecycleTokenCreditRole.CARE]
            )
        )
        self.assertAlmostEqual(
            by_role[LifecycleTokenCreditRole.HARVEST][0].amount,
            production_credit,
        )
        self.assertAlmostEqual(
            by_role[LifecycleTokenCreditRole.TRANSPORT][0].amount,
            production_credit,
        )
        self.assertAlmostEqual(
            by_role[LifecycleTokenCreditRole.SELL][0].amount,
            production_credit,
        )
        # Tracker span'i BOS dahil token_ids'ye göre 5'te başlar; PPO
        # target_token_ids BOS'u içermediği için doğru indeks 4'tür.
        self.assertEqual(
            by_role[LifecycleTokenCreditRole.PURCHASE][0].token_start,
            purchase.token_span.start - 1,
        )

        with TemporaryDirectory() as directory:
            artifacts = tracker.write_artifacts(Path(directory))
            with gzip.open(artifacts["events"], "rt", encoding="utf-8") as file:
                event_rows = [json.loads(line) for line in file]
            plant_rows = json.loads(artifacts["plants"].read_text())
        self.assertEqual(len(event_rows), len(tracker.events))
        self.assertEqual(plant_rows[0]["plant_id"], plant["plant_id"])

    def test_unsold_seed_purchase_gets_full_order_cost_as_negative_credit(self) -> None:
        env = make(
            "kaggriculture",
            configuration={
                "episodeSteps": 4,
                "turnsPerDay": 4,
                "weedSpawnChance": 0.0,
                "seed": 91,
            },
        )
        tracker = FarmLifecycleTracker(
            env.state[0].observation,
            episode_number=13,
            player_index=0,
            turns_per_day=4,
        )
        self._play(
            env,
            tracker,
            {
                "farmer": ["PASS"],
                "hands": [],
                "market": [["BUY_SEED", "WHEAT", 2]],
            },
        )
        assignments = LifecycleCreditAssigner(
            LifecycleCreditConfig(mode=LifecycleCreditMode.CAUSAL_CHAIN)
        ).assign_episode_token_credits(tracker.events)
        self.assertEqual(len(assignments), 1)
        self.assertIs(assignments[0].role, LifecycleTokenCreditRole.PURCHASE)
        self.assertAlmostEqual(assignments[0].amount, -20 / 3000)

    def test_bought_product_resale_gets_no_lifecycle_sell_credit(self) -> None:
        """BUY_PRODUCT -> SELL döngüsü plant zinciri gibi ödüllendirilmez."""
        env = make(
            "kaggriculture",
            configuration={
                "episodeSteps": 6,
                "turnsPerDay": 4,
                "weedSpawnChance": 0.0,
                "seed": 92,
            },
        )
        tracker = FarmLifecycleTracker(
            env.state[0].observation,
            episode_number=14,
            player_index=0,
            turns_per_day=4,
        )
        self._play(
            env,
            tracker,
            {
                "farmer": ["PASS"],
                "hands": [],
                "market": [["BUY_PRODUCT", "WHEAT", 1]],
            },
        )
        self._play(
            env,
            tracker,
            {
                "farmer": ["PASS"],
                "hands": [],
                "market": [["SELL", "WHEAT", 1]],
            },
        )

        sale = next(
            event
            for event in tracker.events
            if event.event_type is LifecycleEventType.PRODUCT_SOLD
        )
        self.assertTrue(
            all(
                allocation.get("source_plant_id") is None
                for allocation in sale.data["allocations"]
            )
        )
        assignments = LifecycleCreditAssigner(
            LifecycleCreditConfig(mode=LifecycleCreditMode.CAUSAL_CHAIN)
        ).assign_episode_token_credits(tracker.events)
        self.assertFalse(
            any(
                assignment.role is LifecycleTokenCreditRole.SELL
                for assignment in assignments
            )
        )

    def test_planted_day_is_encoded_as_tile_age(self) -> None:
        raw_tile = {
            "kind": "PLANT",
            "crop": "WHEAT",
            "planted_day": 4,
            "watered_today": False,
            "consecutive_unwatered": 1,
            "yield_units": 1,
            "max_lifespan_step": 216,
            "fertilized_until_day": -1,
        }
        tile = parse_tile(raw_tile)
        self.assertIsInstance(tile, PlantTile)
        self.assertEqual(tile.planted_day, 4)

        features = tile_to_numeric_features(
            tile,
            current_day=7,
            current_step=168,
            market_prices={"WHEAT": 25},
        )
        age_index = NUMERIC_FEATURE_NAMES.index("age")
        self.assertAlmostEqual(float(features[age_index]), 3 / 30)


if __name__ == "__main__":
    unittest.main()
