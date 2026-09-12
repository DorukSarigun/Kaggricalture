"""Lifecycle event'lerini isteğe bağlı PPO reward kredisine dönüştürür.

Tracker yalnızca gerçekleşen olayları kaydeder. Bu modül ise bu olaylardan
hangilerinin eğitim reward'ina etki edeceğini belirleyen ayrı bir strategy'dir.
Varsayılan OFF modu mevcut terminal + potential-shaping hedefini değiştirmez.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable

from lifecycle_tracking import LifecycleEvent, LifecycleEventType
from span_credit import ECONOMIC_SPAN_SCHEME_VERSION


CAUSAL_CHAIN_SCHEME_VERSION = 2


class LifecycleCreditMode(StrEnum):
    OFF = "off"
    HARVEST_VALUE = "harvest_value"
    CAUSAL_CHAIN = "causal_chain"
    ECONOMIC_SPAN = "economic_span"


@dataclass(frozen=True, slots=True)
class LifecycleCreditConfig:
    mode: LifecycleCreditMode = LifecycleCreditMode.OFF
    coefficient: float = 1.0
    scheme_version: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", LifecycleCreditMode(self.mode))
        expected_scheme = (
            ECONOMIC_SPAN_SCHEME_VERSION
            if self.mode is LifecycleCreditMode.ECONOMIC_SPAN
            else CAUSAL_CHAIN_SCHEME_VERSION
        )
        if self.scheme_version is None:
            object.__setattr__(self, "scheme_version", expected_scheme)
        if not math.isfinite(self.coefficient) or self.coefficient < 0.0:
            raise ValueError("Lifecycle credit coefficient sonlu ve negatif olmayan bir sayı olmalı.")
        if self.scheme_version != expected_scheme:
            raise ValueError(
                "Desteklenmeyen lifecycle credit scheme_version: "
                f"{self.scheme_version}; beklenen={expected_scheme}"
            )


@dataclass(frozen=True, slots=True)
class LifecycleCreditAssignment:
    """Bir lifecycle event'inden mevcut environment adımına verilen kredi."""

    event_id: str
    event_type: LifecycleEventType
    step: int
    amount: float
    raw_normalized_credit: float
    plant_id: str | None
    token_start: int | None
    token_end: int | None
    reason: str


class LifecycleTokenCreditRole(StrEnum):
    PURCHASE = "purchase"
    PLANT = "plant"
    CARE = "care"
    HARVEST = "harvest"
    TRANSPORT = "transport"
    SELL = "sell"

    @property
    def role_id(self) -> int:
        """Tensor'a taşınabilir sabit tamsayı kimliği."""
        return TOKEN_CREDIT_ROLE_IDS[self]


TOKEN_CREDIT_ROLE_IDS: dict[LifecycleTokenCreditRole, int] = {
    role: index for index, role in enumerate(LifecycleTokenCreditRole)
}
# -1, kredi almamış tokenları işaretler.
NO_TOKEN_CREDIT_ROLE = -1


@dataclass(frozen=True, slots=True)
class LifecycleTokenCreditAssignment:
    """Tamamlanmış zincir sonucunu geçmişteki hedef token span'ına bağlar.

    ``token_start`` ve ``token_end`` PPO'nun ``target_token_ids`` dizisine
    göredir; tracker'ın BOS içeren tam token dizisine göre değildir.
    """

    source_event_id: str
    target_event_id: str
    target_event_type: LifecycleEventType
    role: LifecycleTokenCreditRole
    step: int
    amount: float
    plant_id: str | None
    seed_lot_id: str | None
    token_start: int
    token_end: int
    reason: str


class LifecycleCreditAssigner:
    """Strategy object: event listesi -> denetlenebilir reward atamaları."""

    def __init__(self, config: LifecycleCreditConfig | None = None) -> None:
        self.config = config or LifecycleCreditConfig()

    def assign(
        self,
        events: Iterable[LifecycleEvent],
    ) -> tuple[LifecycleCreditAssignment, ...]:
        if self.config.mode is LifecycleCreditMode.OFF:
            return ()

        assignments: list[LifecycleCreditAssignment] = []
        for event in events:
            if (
                self.config.mode is LifecycleCreditMode.HARVEST_VALUE
                and event.event_type is LifecycleEventType.HARVESTED
                and event.normalized_credit != 0.0
            ):
                span = event.token_span
                assignments.append(
                    LifecycleCreditAssignment(
                        event_id=event.event_id,
                        event_type=event.event_type,
                        step=event.step,
                        amount=self.config.coefficient * event.normalized_credit,
                        raw_normalized_credit=event.normalized_credit,
                        plant_id=event.plant_id,
                        token_start=span.start if span is not None else None,
                        token_end=span.end if span is not None else None,
                        reason="harvested_units_x_current_market_price",
                    )
                )
        return tuple(assignments)

    @staticmethod
    def _target_assignment(
        *,
        source_event: LifecycleEvent,
        target_event: LifecycleEvent,
        role: LifecycleTokenCreditRole,
        amount: float,
        reason: str,
        plant_id: str | None = None,
    ) -> LifecycleTokenCreditAssignment | None:
        span = target_event.token_span
        if span is None or amount == 0.0:
            return None
        # ActionTokenSpan, BOS dahil tam token dizisini indeksler. PPO ise
        # token_ids[1:] hedeflerini eğitir; bu nedenle tam olarak bir eksiltiriz.
        token_start = span.start - 1
        token_end = span.end - 1
        if token_start < 0 or token_end < token_start:
            raise ValueError(
                f"Geçersiz lifecycle token span: {span.start}..{span.end}"
            )
        return LifecycleTokenCreditAssignment(
            source_event_id=source_event.event_id,
            target_event_id=target_event.event_id,
            target_event_type=target_event.event_type,
            role=role,
            step=target_event.step,
            amount=float(amount),
            plant_id=plant_id or target_event.plant_id,
            seed_lot_id=target_event.seed_lot_id,
            token_start=token_start,
            token_end=token_end,
            reason=reason,
        )

    def assign_episode_token_credits(
        self,
        events: Iterable[LifecycleEvent],
    ) -> tuple[LifecycleTokenCreditAssignment, ...]:
        """Sipariş ekonomisi ile bitki üretim kredisini birbirinden ayırır.

        Satın alma span'ı miktar kararını tam order ekonomisiyle öğrenir::

            purchase_credit = descendant_sales - full_order_cost

        Dolayısıyla fazla veya hiç kullanılmayan tohumlar BUY_SEED/item/quantity
        tokenlarında negatif kalır. Her satılan plant'in üretim zinciri ise
        kardeş tohum israfından bağımsız hesaplanır::

            plant_net = plant_sales - one_seed_cost
            action_credit = plant_net / plant_action_count

        Bu ikinci kredi PLANT, WATER/FERTILIZE, HARVEST, shed'e DROP/PLACE ve
        SELL span'larına eşit dağıtılır. Satılmayan plant'in üretim
        aksiyonları kredi almaz; onun seed maliyeti purchase kredisinde kalır.

        Bu krediler global transition reward'ina eklenmez; ayrı token-level
        PPO loss'unda yalnızca ilgili stochastic tokenlara uygulanır.
        """
        if self.config.mode is not LifecycleCreditMode.CAUSAL_CHAIN:
            return ()

        event_rows = tuple(events)
        purchases_by_seed_lot = {
            event.seed_lot_id: event
            for event in event_rows
            if event.event_type is LifecycleEventType.SEED_PURCHASED
            and event.seed_lot_id is not None
        }
        plants = {
            event.plant_id: event
            for event in event_rows
            if event.event_type is LifecycleEventType.PLANTED
            and event.plant_id is not None
        }
        sale_credit_by_plant: dict[str, float] = {}
        sale_events_by_plant: dict[str, dict[str, LifecycleEvent]] = {}
        for sale in event_rows:
            if sale.event_type is not LifecycleEventType.PRODUCT_SOLD:
                continue
            for allocation in sale.data.get("allocations", ()):
                plant_id = allocation.get("source_plant_id")
                if plant_id not in plants:
                    continue
                revenue = float(allocation.get("estimated_revenue", 0.0))
                normalized_revenue = (
                    sale.normalized_credit * revenue / sale.attributed_value
                    if sale.attributed_value > 0.0
                    else 0.0
                )
                sale_credit_by_plant[plant_id] = (
                    sale_credit_by_plant.get(plant_id, 0.0)
                    + normalized_revenue
                )
                sale_events_by_plant.setdefault(plant_id, {})[sale.event_id] = sale

        assignments: list[LifecycleTokenCreditAssignment] = []
        # BUY_SEED + crop + quantity span'ı, order'ın tam maliyetini ve
        # yalnızca bu lot'tan gelen gerçek satışları görür. Satışı olmayan
        # order da bu nedenle tam maliyeti kadar negatif kredi alır.
        for seed_lot_id, purchase in purchases_by_seed_lot.items():
            descendant_revenue = sum(
                sale_credit_by_plant.get(plant_id, 0.0)
                for plant_id, plant in plants.items()
                if plant.seed_lot_id == seed_lot_id
            )
            purchase_net = float(purchase.normalized_credit) + descendant_revenue
            assignment = self._target_assignment(
                source_event=purchase,
                target_event=purchase,
                role=LifecycleTokenCreditRole.PURCHASE,
                amount=self.config.coefficient * purchase_net,
                reason="seed_lot_sales_minus_full_order_cost",
            )
            if assignment is not None:
                assignments.append(assignment)

        # Üretim kararları her satılan plant için ayrı hesaplanır. Böylece
        # aynı order'daki kullanılmayan kardeş seed'lerin maliyeti başarılı
        # PLANT/WATER/HARVEST/SELL kararlarına bulaşmaz.
        for plant_id, normalized_revenue in sale_credit_by_plant.items():
            if normalized_revenue <= 0.0:
                continue
            plant = plants[plant_id]
            purchase = purchases_by_seed_lot.get(plant.seed_lot_id)
            normalized_unit_cost = (
                -float(purchase.normalized_credit)
                / max(int(purchase.quantity or 0), 1)
                if purchase is not None
                else 0.0
            )
            plant_net = normalized_revenue - normalized_unit_cost

            # Aynı plant için bir action event'i allocation sayısından
            # bağımsız olarak bir kez sayılır.
            chain_actions: dict[
                str, tuple[LifecycleTokenCreditRole, LifecycleEvent]
            ] = {}
            if plant.token_span is not None:
                chain_actions[plant.event_id] = (
                    LifecycleTokenCreditRole.PLANT,
                    plant,
                )
            for event in event_rows:
                if event.token_span is None:
                    continue
                if event.plant_id == plant_id:
                    role = {
                        LifecycleEventType.WATERED: LifecycleTokenCreditRole.CARE,
                        LifecycleEventType.FERTILIZED: LifecycleTokenCreditRole.CARE,
                        LifecycleEventType.HARVESTED: LifecycleTokenCreditRole.HARVEST,
                    }.get(event.event_type)
                    if role is not None:
                        chain_actions[event.event_id] = (role, event)
                if (
                    event.event_type is LifecycleEventType.PRODUCT_MOVED
                    and event.data.get("destination") == "shed"
                    and any(
                        allocation.get("source_plant_id") == plant_id
                        for allocation in event.data.get("allocations", ())
                    )
                ):
                    chain_actions[event.event_id] = (
                        LifecycleTokenCreditRole.TRANSPORT,
                        event,
                    )
            for sale in sale_events_by_plant.get(plant_id, {}).values():
                if sale.token_span is not None:
                    chain_actions[sale.event_id] = (
                        LifecycleTokenCreditRole.SELL,
                        sale,
                    )

            if not chain_actions:
                continue
            equal_credit = (
                self.config.coefficient
                * plant_net
                / len(chain_actions)
            )
            source_sale = max(
                sale_events_by_plant[plant_id].values(),
                key=lambda event: (event.step, event.event_id),
            )
            for role, target_event in sorted(
                chain_actions.values(),
                key=lambda row: (row[1].step, row[1].event_id),
            ):
                assignment = self._target_assignment(
                    source_event=source_sale,
                    target_event=target_event,
                    role=role,
                    amount=equal_credit,
                    reason="plant_net_profit_equally_shared_across_production_actions",
                    plant_id=plant_id,
                )
                if assignment is not None:
                    assignments.append(assignment)

        return tuple(assignments)


def total_lifecycle_credit(
    assignments: Iterable[LifecycleCreditAssignment],
) -> float:
    return float(sum(assignment.amount for assignment in assignments))
