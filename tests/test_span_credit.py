from copy import deepcopy
import unittest

from kaggle_environments import make

from action_encoding import encode_action_tokens
from lifecycle_tracking import FarmLifecycleTracker, LifecycleEventType
from market_execution import trace_market_executions
from span_credit import (
    EconomicSpanCreditAssigner,
    RunningRoleRMS,
    SpanCreditAssignment,
    SpanCreditRole,
    SpanTrainingCredit,
    aggregate_training_credits,
    expected_normalized_economic_net,
)


PASS = {"farmer": ["PASS"], "hands": [], "market": []}


class EconomicSpanCreditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.configuration = {
            "episodeSteps": 30,
            "turnsPerDay": 4,
            "weedSpawnChance": 0.0,
            "seed": 1701,
        }
        self.env = make("kaggriculture", configuration=self.configuration)
        self.tracker = FarmLifecycleTracker(
            self.env.state[0].observation,
            episode_number=21,
            player_index=0,
            turns_per_day=4,
        )

    def _play(self, action: dict) -> None:
        observations = [
            deepcopy(self.env.state[player].observation) for player in range(2)
        ]
        token_ids = encode_action_tokens(action, observations[0])
        executions = trace_market_executions(
            observations,
            [action, PASS],
            self.configuration,
        )[0]
        self.env.step([action, PASS])
        self.tracker.observe_transition(
            observations[0],
            action,
            deepcopy(self.env.state[0].observation),
            token_ids=token_ids,
            market_executions=executions,
        )

    def test_crop_and_trade_share_one_sell_span_without_double_counting(self) -> None:
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
            {
                "farmer": ["DROP"],
                "hands": [],
                "market": [["BUY_PRODUCT", "WHEAT", 1]],
            },
            {
                "farmer": ["PASS"],
                "hands": [],
                "market": [["SELL", "WHEAT", 3]],
            },
        ]
        for action in actions:
            self._play(action)

        events = self.tracker.events
        sale = next(
            event
            for event in events
            if event.event_type is LifecycleEventType.PRODUCT_SOLD
        )
        allocations = sale.data["allocations"]
        self.assertTrue(any(row.get("source_plant_id") for row in allocations))
        self.assertTrue(any(row.get("source_purchase_id") for row in allocations))

        assignments = EconomicSpanCreditAssigner().assign_episode(events)
        expected = expected_normalized_economic_net(events, reward_scale=3000.0)
        self.assertAlmostEqual(sum(row.amount for row in assignments), expected)

        sale_assignments = [
            row for row in assignments if row.target_event_id == sale.event_id
        ]
        self.assertEqual(
            {row.credit_role for row in sale_assignments},
            {SpanCreditRole.FARM, SpanCreditRole.TRADE},
        )
        training_credits = aggregate_training_credits(assignments)
        sale_credits = [
            row
            for row in training_credits
            if row.action_id == sale_assignments[0].action_id
        ]
        self.assertEqual(len(sale_credits), 1)
        self.assertAlmostEqual(
            sale_credits[0].amount,
            sum(row.amount for row in sale_assignments),
        )
        self.assertAlmostEqual(
            sum(row.amount for row in training_credits),
            sum(row.amount for row in assignments),
        )

        seed_purchase = next(
            event
            for event in events
            if event.event_type is LifecycleEventType.SEED_PURCHASED
        )
        seed_roles = {
            row.credit_role
            for row in assignments
            if row.target_event_id == seed_purchase.event_id
        }
        self.assertEqual(seed_roles, {SpanCreditRole.PURCHASE})

    def test_partial_trade_keeps_open_lot_cost_on_buy_span(self) -> None:
        self._play(
            {
                "farmer": ["PASS"],
                "hands": [],
                "market": [
                    ["BUY_PRODUCT", "WHEAT", 5],
                    ["SELL", "WHEAT", 1],
                ],
            }
        )
        events = self.tracker.events
        purchase = next(
            row
            for row in events
            if row.event_type is LifecycleEventType.PRODUCT_PURCHASED
        )
        sale = next(
            row for row in events if row.event_type is LifecycleEventType.PRODUCT_SOLD
        )
        assignments = EconomicSpanCreditAssigner().assign_episode(events)

        total_cost = float(purchase.data["total_cost"])
        revenue = float(sale.data["realized_revenue"])
        self.assertAlmostEqual(
            sum(row.amount for row in assignments),
            (revenue - total_cost) / 3000.0,
        )
        buy_rows = [
            row for row in assignments if row.target_event_id == purchase.event_id
        ]
        self.assertTrue(
            any(row.reason == "purchased_product_cost_without_realized_sale" for row in buy_rows)
        )
        trade = self.tracker.trade_lifecycles()[0]
        self.assertEqual(trade["sold_quantity"], 1)
        self.assertGreater(trade["open_cost"], 0.0)

    def test_hire_cost_and_next_turn_hand_identity_are_exact(self) -> None:
        before_money = float(self.env.state[0].observation["farms"][0]["money"])
        self._play(
            {
                "farmer": ["PASS"],
                "hands": [],
                "market": [["HIRE"], ["HIRE"]],
            }
        )
        after_farm = self.env.state[0].observation["farms"][0]
        hire_events = [
            row
            for row in self.tracker.events
            if row.event_type is LifecycleEventType.HIRED
        ]
        self.assertEqual(len(hire_events), 2)
        self.assertEqual([row.data["hand_index"] for row in hire_events], [0, 1])
        self.assertEqual([row.data["unit_index"] for row in hire_events], [1, 2])
        self.assertEqual(
            [row.data["hires_today_sequence"] for row in hire_events], [0, 1]
        )
        self.assertEqual(
            [row.data["initial_position"] for row in hire_events],
            list(after_farm["hands"]),
        )
        self.assertTrue(
            all(row.data["active_from_step"] == 1 for row in hire_events)
        )
        exact_cost = before_money - float(after_farm["money"])
        self.assertAlmostEqual(
            sum(float(row.data["cost"]) for row in hire_events), exact_cost
        )
        assignments = EconomicSpanCreditAssigner().assign_episode(hire_events)
        self.assertTrue(
            all(row.credit_role is SpanCreditRole.HIRE for row in assignments)
        )
        self.assertAlmostEqual(
            sum(row.amount for row in assignments), -exact_cost / 3000.0
        )

    def test_running_rms_never_changes_credit_sign(self) -> None:
        credits = (
            SpanTrainingCredit("purchase", SpanCreditRole.PURCHASE, -10.0),
            SpanTrainingCredit("farm", SpanCreditRole.FARM, 2.0),
            SpanTrainingCredit("trade", SpanCreditRole.TRADE, -3.0),
            SpanTrainingCredit("hire", SpanCreditRole.HIRE, -1.0),
        )
        scaler = RunningRoleRMS()
        scaler.update(credits)
        transformed = [scaler.transform(row) for row in credits]
        self.assertEqual([value > 0 for value in transformed], [False, True, False, False])
        self.assertEqual(scaler.state_dict()["version"], 2)

    def test_mixed_assignment_aggregation_uses_one_dominant_role(self) -> None:
        common = {
            "assignment_id": "a",
            "action_id": "sell",
            "source_event_id": "source",
            "target_event_id": "sale",
            "environment_step": 3,
            "token_start": 1,
            "token_end": 2,
            "source_id": "lot",
            "reason": "test",
        }
        assignments = (
            SpanCreditAssignment(
                **common,
                credit_role=SpanCreditRole.FARM,
                amount=0.2,
            ),
            SpanCreditAssignment(
                **{**common, "assignment_id": "b"},
                credit_role=SpanCreditRole.TRADE,
                amount=-0.5,
            ),
        )
        credits = aggregate_training_credits(assignments)
        self.assertEqual(len(credits), 1)
        self.assertAlmostEqual(credits[0].amount, -0.3)
        self.assertIs(credits[0].credit_role, SpanCreditRole.TRADE)


if __name__ == "__main__":
    unittest.main()
