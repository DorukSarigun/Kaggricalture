import unittest

from lifecycle_credit import (
    CAUSAL_CHAIN_SCHEME_VERSION,
    LifecycleCreditAssigner,
    LifecycleCreditConfig,
    LifecycleCreditMode,
    LifecycleTokenCreditRole,
    total_lifecycle_credit,
)
from lifecycle_tracking import (
    ActionTokenSpan,
    LifecycleEvent,
    LifecycleEventType,
)


def make_event(
    event_type: LifecycleEventType,
    *,
    normalized_credit: float = 0.0,
) -> LifecycleEvent:
    return LifecycleEvent(
        event_id=f"event-{event_type.value}",
        event_type=event_type,
        episode=1,
        step=42,
        day=1,
        hour=18,
        player=0,
        token_span=ActionTokenSpan(
            section="unit",
            index=0,
            start=1,
            end=4,
            token_ids=(1, 2, 3, 4),
            tokens=("FARMER", "HARVEST", "X_4", "END_ACTION"),
        ),
        plant_id="plant-1",
        crop="WHEAT",
        normalized_credit=normalized_credit,
    )


class LifecycleCreditTests(unittest.TestCase):
    def test_off_mode_never_assigns_credit(self) -> None:
        assigner = LifecycleCreditAssigner()
        self.assertEqual(
            assigner.assign(
                [make_event(LifecycleEventType.HARVESTED, normalized_credit=0.5)]
            ),
            (),
        )

    def test_harvest_mode_only_credits_harvest_events(self) -> None:
        assigner = LifecycleCreditAssigner(
            LifecycleCreditConfig(
                mode=LifecycleCreditMode.HARVEST_VALUE,
                coefficient=0.2,
            )
        )
        assignments = assigner.assign(
            [
                make_event(LifecycleEventType.WATERED),
                make_event(
                    LifecycleEventType.HARVESTED,
                    normalized_credit=0.5,
                ),
            ]
        )

        self.assertEqual(len(assignments), 1)
        self.assertAlmostEqual(assignments[0].amount, 0.1)
        self.assertEqual(assignments[0].token_start, 1)
        self.assertEqual(assignments[0].token_end, 4)
        self.assertAlmostEqual(total_lifecycle_credit(assignments), 0.1)

    def test_negative_coefficient_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "negatif"):
            LifecycleCreditConfig(coefficient=-0.1)

    def test_unknown_credit_scheme_version_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "scheme_version"):
            LifecycleCreditConfig(
                scheme_version=CAUSAL_CHAIN_SCHEME_VERSION - 1,
            )

    def test_purchase_and_each_plant_chain_have_independent_credit(self) -> None:
        span_index = 1

        def span(section: str) -> ActionTokenSpan:
            nonlocal span_index
            result = ActionTokenSpan(
                section=section,
                index=span_index,
                start=1,
                end=2,
                token_ids=(1, 2),
                tokens=("ACTION", "END"),
            )
            span_index += 1
            return result

        purchase = LifecycleEvent(
            event_id="purchase",
            event_type=LifecycleEventType.SEED_PURCHASED,
            episode=1,
            step=1,
            day=0,
            hour=1,
            player=0,
            token_span=span("market"),
            seed_lot_id="lot-1",
            crop="WHEAT",
            quantity=2,
            normalized_credit=-20 / 3000,
        )
        plant_1 = LifecycleEvent(
            event_id="plant-1-event",
            event_type=LifecycleEventType.PLANTED,
            episode=1,
            step=2,
            day=0,
            hour=2,
            player=0,
            token_span=span("unit"),
            plant_id="plant-1",
            seed_lot_id="lot-1",
            crop="WHEAT",
        )
        plant_2 = LifecycleEvent(
            event_id="plant-2-event",
            event_type=LifecycleEventType.PLANTED,
            episode=1,
            step=3,
            day=0,
            hour=3,
            player=0,
            token_span=span("unit"),
            plant_id="plant-2",
            seed_lot_id="lot-1",
            crop="WHEAT",
        )
        # Tek SELL emri iki plant'tan toplam 70$ gelir üretiyor. Her plant bu
        # ortak SELL span'ına kendi üretim sonucundan ayrı kredi yazmalı.
        sale = LifecycleEvent(
            event_id="sale",
            event_type=LifecycleEventType.PRODUCT_SOLD,
            episode=1,
            step=4,
            day=0,
            hour=4,
            player=0,
            token_span=span("market"),
            quantity=2,
            attributed_value=70,
            normalized_credit=70 / 3000,
            data={
                "allocations": [
                    {
                        "source_plant_id": "plant-1",
                        "quantity": 1,
                        "estimated_revenue": 30,
                    },
                    {
                        "source_plant_id": "plant-2",
                        "quantity": 1,
                        "estimated_revenue": 40,
                    },
                ]
            },
        )

        assignments = LifecycleCreditAssigner(
            LifecycleCreditConfig(mode=LifecycleCreditMode.CAUSAL_CHAIN)
        ).assign_episode_token_credits([purchase, plant_1, plant_2, sale])

        # Purchase: 70 - order'ın tam 20$ maliyeti = +50$.
        purchase_credit = next(
            row
            for row in assignments
            if row.role is LifecycleTokenCreditRole.PURCHASE
        )
        self.assertAlmostEqual(purchase_credit.amount, 50 / 3000)

        # Plant-1 neti 30-10=20 ve PLANT/SELL arasında ikiye bölünür.
        # Plant-2 neti 40-10=30 ve yine ikiye bölünür. Ortak SELL span'ı
        # iki plant'tan iki assignment alır; rollout bunları toplar.
        self.assertEqual(len(assignments), 5)
        self.assertEqual(
            sum(
                assignment.role is LifecycleTokenCreditRole.SELL
                for assignment in assignments
            ),
            2,
        )
        plant_1_credits = [
            row.amount for row in assignments if row.plant_id == "plant-1"
        ]
        plant_2_credits = [
            row.amount for row in assignments if row.plant_id == "plant-2"
        ]
        self.assertEqual(len(plant_1_credits), 2)
        self.assertEqual(len(plant_2_credits), 2)
        self.assertTrue(
            all(abs(amount - 10 / 3000) < 1e-9 for amount in plant_1_credits)
        )
        self.assertTrue(
            all(abs(amount - 15 / 3000) < 1e-9 for amount in plant_2_credits)
        )

    def test_bulk_purchase_quantity_is_penalized_by_unused_seed_cost(self) -> None:
        def span(section: str, index: int) -> ActionTokenSpan:
            return ActionTokenSpan(
                section=section,
                index=index,
                start=1,
                end=2,
                token_ids=(1, 2),
                tokens=("ACTION", "END"),
            )

        purchase = LifecycleEvent(
            event_id="purchase-50",
            event_type=LifecycleEventType.SEED_PURCHASED,
            episode=1,
            step=1,
            day=0,
            hour=1,
            player=0,
            token_span=span("market", 0),
            seed_lot_id="bulk-lot",
            crop="WHEAT",
            quantity=50,
            normalized_credit=-500 / 3000,
        )
        plant = LifecycleEvent(
            event_id="only-sold-plant",
            event_type=LifecycleEventType.PLANTED,
            episode=1,
            step=2,
            day=0,
            hour=2,
            player=0,
            token_span=span("unit", 0),
            plant_id="plant-1",
            seed_lot_id="bulk-lot",
            crop="WHEAT",
        )
        sale = LifecycleEvent(
            event_id="sale-51",
            event_type=LifecycleEventType.PRODUCT_SOLD,
            episode=1,
            step=3,
            day=0,
            hour=3,
            player=0,
            token_span=span("market", 1),
            quantity=1,
            attributed_value=51,
            normalized_credit=51 / 3000,
            data={
                "allocations": [
                    {
                        "source_plant_id": "plant-1",
                        "quantity": 1,
                        "estimated_revenue": 51,
                    }
                ]
            },
        )

        assignments = LifecycleCreditAssigner(
            LifecycleCreditConfig(mode=LifecycleCreditMode.CAUSAL_CHAIN)
        ).assign_episode_token_credits([purchase, plant, sale])

        purchase_credit = next(
            row
            for row in assignments
            if row.role is LifecycleTokenCreditRole.PURCHASE
        )
        self.assertAlmostEqual(purchase_credit.amount, -449 / 3000)
        production = [
            row
            for row in assignments
            if row.role is not LifecycleTokenCreditRole.PURCHASE
        ]
        self.assertEqual(len(production), 2)
        self.assertTrue(
            all(abs(row.amount - (41 / 3000) / 2) < 1e-9 for row in production)
        )


if __name__ == "__main__":
    unittest.main()
