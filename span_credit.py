"""Korunumlu ekonomik action-span kredisi ve PPO ölçekleme bileşenleri.

Bu modül legacy ``causal_chain`` token kredisinden bağımsızdır. Crop, market
trading ve HIRE tutarlarını tek-sayım invariant'ıyla semantik action span'larına
bağlar; transition reward veya ayrı GAE üretmez.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import IntEnum, StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np

from lifecycle_tracking import (
    ActionTokenSpan,
    LifecycleEvent,
    LifecycleEventType,
    extract_action_token_spans,
)

if TYPE_CHECKING:
    from policy import SampledAction


ECONOMIC_SPAN_SCHEME_VERSION = 5


class ActionSection(IntEnum):
    UNIT = 0
    MARKET = 1


class SpanCreditRole(IntEnum):
    # BUY_SEED kendi kovasında tutulur. Böylece az sayıdaki pahalı/boşa
    # giden satın alma kararı binlerce WATER/FARM span'i tarafından ezilmez.
    PURCHASE = 0
    FARM = 1
    TRADE = 2
    HIRE = 3

    @property
    def label(self) -> str:
        return self.name.lower()


class AttributionMethod(StrEnum):
    FIFO = "fifo"


@dataclass(frozen=True, slots=True)
class ActionSpanRecord:
    action_id: str
    environment_step: int
    section: ActionSection
    operation: str
    actor_id: str | None
    token_start: int
    token_end: int
    old_log_probability: float
    entropy: float
    has_stochastic_decision: bool

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["section"] = self.section.name.lower()
        return value


@dataclass(frozen=True, slots=True)
class SpanCreditAssignment:
    assignment_id: str
    action_id: str
    source_event_id: str
    target_event_id: str
    environment_step: int
    token_start: int
    token_end: int
    credit_role: SpanCreditRole
    source_id: str
    amount: float
    reason: str
    attribution_method: AttributionMethod | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["credit_role"] = self.credit_role.label
        if self.attribution_method is not None:
            value["attribution_method"] = self.attribution_method.value
        return value


@dataclass(frozen=True, slots=True)
class SpanTrainingCredit:
    action_id: str
    credit_role: SpanCreditRole
    amount: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "credit_role": self.credit_role.label,
            "amount": float(self.amount),
        }


@dataclass(frozen=True, slots=True)
class EconomicCreditSummary:
    assignment_count: int
    total: float
    positive: float
    negative: float
    by_role: Mapping[str, Mapping[str, float | int]]


def action_id_for_span(step: int, span: ActionTokenSpan) -> str:
    return f"step-{int(step):06d}:{span.section}:{int(span.index)}"


def _target_indices(span: ActionTokenSpan) -> tuple[int, int]:
    # ActionTokenSpan BOS içeren token_ids dizisini indeksler. PPO hedef dizisi
    # token_ids[1:] olduğundan bir eksiltilir.
    start = int(span.start) - 1
    end = int(span.end) - 1
    if start < 0 or end < start:
        raise ValueError(f"Geçersiz action span: {span.start}..{span.end}")
    return start, end


def build_action_span_records(
    sample: SampledAction,
    environment_step: int,
) -> tuple[ActionSpanRecord, ...]:
    """Sample içindeki bütün unit ve market emirlerini semantik kayda çevirir."""

    spans = extract_action_token_spans(sample.token_ids)
    records: list[ActionSpanRecord] = []
    for (section_name, index), span in sorted(
        spans.items(),
        key=lambda row: (row[1].start, row[1].end),
    ):
        start, end = _target_indices(span)
        if end >= len(sample.target_token_ids):
            raise ValueError(
                f"Action span target dizisi dışında: {start}..{end}, "
                f"tokens={len(sample.target_token_ids)}"
            )
        selected = sample.decision_mask[start : end + 1]
        old_log_probability = float(
            sample.token_log_probs[start : end + 1][selected].sum()
        )
        entropies = sample.token_entropies[start : end + 1][selected]
        entropy = float(entropies.mean()) if len(entropies) else 0.0
        if section_name == "unit":
            actions = [sample.action.get("farmer", ["PASS"]), *sample.action.get("hands", [])]
            operation = (
                str(actions[index][0])
                if index < len(actions) and actions[index]
                else "PASS"
            )
            actor_id = "farmer" if index == 0 else f"hand:{index - 1}"
            section = ActionSection.UNIT
        elif section_name == "market":
            orders = sample.action.get("market", [])
            operation = (
                str(orders[index][0])
                if index < len(orders) and orders[index]
                else "STOP"
            )
            actor_id = None
            section = ActionSection.MARKET
        else:
            raise ValueError(f"Bilinmeyen action span section: {section_name!r}")
        records.append(
            ActionSpanRecord(
                action_id=action_id_for_span(environment_step, span),
                environment_step=int(environment_step),
                section=section,
                operation=operation,
                actor_id=actor_id,
                token_start=start,
                token_end=end,
                old_log_probability=old_log_probability,
                entropy=entropy,
                has_stochastic_decision=bool(selected.any()),
            )
        )
    return tuple(records)


class RunningRoleRMS:
    """Rol başına işaret koruyan, merkezlemeyen running RMS ölçeği."""

    def __init__(self, epsilon: float = 1e-8) -> None:
        if epsilon <= 0.0:
            raise ValueError("epsilon pozitif olmalı.")
        self.epsilon = float(epsilon)
        self._counts = {role: 0 for role in SpanCreditRole}
        self._sum_squares = {role: 0.0 for role in SpanCreditRole}

    def update(self, credits: Iterable[SpanTrainingCredit]) -> None:
        for credit in credits:
            amount = float(credit.amount)
            if not math.isfinite(amount):
                raise ValueError("Span credit sonlu olmalı.")
            role = SpanCreditRole(credit.credit_role)
            self._counts[role] += 1
            self._sum_squares[role] += amount * amount

    def scale(self, role: SpanCreditRole | int) -> float:
        role = SpanCreditRole(role)
        count = self._counts[role]
        if count == 0:
            return 1.0
        return max(math.sqrt(self._sum_squares[role] / count), self.epsilon)

    def transform(self, credit: SpanTrainingCredit) -> float:
        return float(credit.amount) / self.scale(credit.credit_role)

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 2,
            "epsilon": self.epsilon,
            "counts": {role.label: self._counts[role] for role in SpanCreditRole},
            "sum_squares": {
                role.label: self._sum_squares[role] for role in SpanCreditRole
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("version", -1)) != 2:
            raise ValueError("Desteklenmeyen RunningRoleRMS state sürümü.")
        self.epsilon = float(state.get("epsilon", self.epsilon))
        counts = state.get("counts", {})
        sum_squares = state.get("sum_squares", {})
        for role in SpanCreditRole:
            count = int(counts.get(role.label, 0))
            square_sum = float(sum_squares.get(role.label, 0.0))
            if count < 0 or square_sum < 0.0 or not math.isfinite(square_sum):
                raise ValueError("Geçersiz RunningRoleRMS state değeri.")
            self._counts[role] = count
            self._sum_squares[role] = square_sum


class EconomicSpanCreditAssigner:
    """Event journal'ını korunumlu FARM/TRADE/HIRE span kredisine dönüştürür."""

    def __init__(
        self,
        *,
        coefficient: float = 1.0,
        reward_scale: float = 3000.0,
    ) -> None:
        if not math.isfinite(coefficient) or coefficient < 0.0:
            raise ValueError("coefficient sonlu ve negatif olmayan olmalı.")
        if reward_scale <= 0.0:
            raise ValueError("reward_scale pozitif olmalı.")
        self.coefficient = float(coefficient)
        self.reward_scale = float(reward_scale)
        self._assignment_sequence = 0

    def _assignment(
        self,
        *,
        source_event: LifecycleEvent,
        target_event: LifecycleEvent,
        role: SpanCreditRole,
        source_id: str,
        amount: float,
        reason: str,
        attribution_method: AttributionMethod | None = None,
    ) -> SpanCreditAssignment | None:
        span = target_event.token_span
        if span is None or amount == 0.0:
            return None
        if not math.isfinite(amount):
            raise ValueError("Economic span credit sonlu olmalı.")
        start, end = _target_indices(span)
        self._assignment_sequence += 1
        return SpanCreditAssignment(
            assignment_id=f"span-credit-{self._assignment_sequence:08d}",
            action_id=action_id_for_span(target_event.step, span),
            source_event_id=source_event.event_id,
            target_event_id=target_event.event_id,
            environment_step=target_event.step,
            token_start=start,
            token_end=end,
            credit_role=role,
            source_id=source_id,
            amount=self.coefficient * float(amount),
            reason=reason,
            attribution_method=attribution_method,
        )

    @staticmethod
    def _sale_revenue_by_plant(
        events: Sequence[LifecycleEvent],
    ) -> tuple[dict[str, float], dict[str, dict[str, LifecycleEvent]]]:
        revenues: dict[str, float] = defaultdict(float)
        sale_events: dict[str, dict[str, LifecycleEvent]] = defaultdict(dict)
        for sale in events:
            if sale.event_type is not LifecycleEventType.PRODUCT_SOLD:
                continue
            for allocation in sale.data.get("allocations", ()):
                plant_id = allocation.get("source_plant_id")
                if not plant_id:
                    continue
                revenues[str(plant_id)] += float(
                    allocation.get("realized_revenue", allocation.get("estimated_revenue", 0.0))
                )
                sale_events[str(plant_id)][sale.event_id] = sale
        return dict(revenues), dict(sale_events)

    def _farm_assignments(
        self,
        events: Sequence[LifecycleEvent],
    ) -> list[SpanCreditAssignment]:
        purchases = {
            event.seed_lot_id: event
            for event in events
            if event.event_type is LifecycleEventType.SEED_PURCHASED
            and event.seed_lot_id is not None
        }
        plants = {
            event.plant_id: event
            for event in events
            if event.event_type is LifecycleEventType.PLANTED
            and event.plant_id is not None
        }
        revenue_by_plant, sale_events_by_plant = self._sale_revenue_by_plant(events)
        sold_plants_by_lot: dict[str, int] = defaultdict(int)
        assignments: list[SpanCreditAssignment] = []

        for plant_id, revenue in revenue_by_plant.items():
            if revenue <= 0.0 or plant_id not in plants:
                continue
            plant = plants[plant_id]
            purchase = purchases.get(plant.seed_lot_id)
            seed_unit_cost = float(plant.data.get("seed_unit_cost", 0.0))
            if purchase is not None:
                seed_unit_cost = float(
                    purchase.data.get("unit_cost", seed_unit_cost)
                )
                assert plant.seed_lot_id is not None
                sold_plants_by_lot[plant.seed_lot_id] += 1

            chain: dict[str, LifecycleEvent] = {}
            if purchase is not None and purchase.token_span is not None:
                chain[purchase.event_id] = purchase
            if plant.token_span is not None:
                chain[plant.event_id] = plant
            for event in events:
                if event.token_span is None:
                    continue
                if event.plant_id == plant_id and event.event_type in {
                    LifecycleEventType.WATERED,
                    LifecycleEventType.FERTILIZED,
                    LifecycleEventType.HARVESTED,
                }:
                    chain[event.event_id] = event
                elif (
                    event.event_type is LifecycleEventType.PRODUCT_MOVED
                    and event.data.get("destination") == "shed"
                    and any(
                        allocation.get("source_plant_id") == plant_id
                        for allocation in event.data.get("allocations", ())
                    )
                ):
                    chain[event.event_id] = event
            for sale in sale_events_by_plant.get(plant_id, {}).values():
                if sale.token_span is not None:
                    chain[sale.event_id] = sale

            if not chain:
                continue
            chain_net = (revenue - seed_unit_cost) / self.reward_scale
            equal_credit = chain_net / len(chain)
            source_sale = max(
                sale_events_by_plant[plant_id].values(),
                key=lambda event: (event.step, event.event_id),
            )
            for target in sorted(chain.values(), key=lambda event: (event.step, event.event_id)):
                assignment = self._assignment(
                    source_event=source_sale,
                    target_event=target,
                    role=(
                        SpanCreditRole.PURCHASE
                        if target.event_type is LifecycleEventType.SEED_PURCHASED
                        else SpanCreditRole.FARM
                    ),
                    source_id=plant_id,
                    amount=equal_credit,
                    reason=(
                        "seed_purchase_share_of_plant_realized_net"
                        if target.event_type is LifecycleEventType.SEED_PURCHASED
                        else "plant_realized_net_equally_shared_once"
                    ),
                    attribution_method=AttributionMethod.FIFO,
                )
                if assignment is not None:
                    assignments.append(assignment)

        # Satışa dönüşmeyen her ücretli seed maliyeti yalnız purchase span'ında
        # kalır. Satılmış plant'in seed maliyeti yukarıdaki chain_net içindedir.
        for seed_lot_id, purchase in purchases.items():
            quantity = max(int(purchase.quantity or 0), 0)
            sold_count = min(sold_plants_by_lot.get(seed_lot_id, 0), quantity)
            unsold_count = quantity - sold_count
            if unsold_count <= 0:
                continue
            unit_cost = float(purchase.data.get("unit_cost", 0.0))
            assignment = self._assignment(
                source_event=purchase,
                target_event=purchase,
                role=SpanCreditRole.PURCHASE,
                source_id=seed_lot_id,
                amount=-(unsold_count * unit_cost) / self.reward_scale,
                reason="unused_seed_credit",
                attribution_method=AttributionMethod.FIFO,
            )
            if assignment is not None:
                assignments.append(assignment)
        return assignments

    def _trade_assignments(
        self,
        events: Sequence[LifecycleEvent],
    ) -> list[SpanCreditAssignment]:
        purchases = {
            event.purchase_id: event
            for event in events
            if event.event_type is LifecycleEventType.PRODUCT_PURCHASED
            and event.purchase_id is not None
        }
        sold_cost_by_purchase: dict[str, float] = defaultdict(float)
        assignments: list[SpanCreditAssignment] = []

        for sale in events:
            if sale.event_type is not LifecycleEventType.PRODUCT_SOLD:
                continue
            for allocation in sale.data.get("allocations", ()):
                purchase_id = allocation.get("source_purchase_id")
                if purchase_id not in purchases:
                    continue
                purchase = purchases[str(purchase_id)]
                allocated_cost = float(allocation.get("allocated_cost", 0.0))
                realized_revenue = float(
                    allocation.get("realized_revenue", allocation.get("estimated_revenue", 0.0))
                )
                sold_cost_by_purchase[str(purchase_id)] += allocated_cost
                net = (realized_revenue - allocated_cost) / self.reward_scale
                targets = {
                    event.event_id: event
                    for event in (purchase, sale)
                    if event.token_span is not None
                }
                if not targets:
                    continue
                equal_credit = net / len(targets)
                for target in targets.values():
                    assignment = self._assignment(
                        source_event=sale,
                        target_event=target,
                        role=SpanCreditRole.TRADE,
                        source_id=str(purchase_id),
                        amount=equal_credit,
                        reason="realized_trade_net_shared_between_buy_and_sell",
                        attribution_method=AttributionMethod.FIFO,
                    )
                    if assignment is not None:
                        assignments.append(assignment)

        for purchase_id, purchase in purchases.items():
            total_cost = float(
                purchase.data.get("total_cost", -float(purchase.attributed_value))
            )
            unsold_cost = max(total_cost - sold_cost_by_purchase.get(str(purchase_id), 0.0), 0.0)
            if unsold_cost <= 1e-9:
                continue
            assignment = self._assignment(
                source_event=purchase,
                target_event=purchase,
                role=SpanCreditRole.TRADE,
                source_id=str(purchase_id),
                amount=-unsold_cost / self.reward_scale,
                reason="purchased_product_cost_without_realized_sale",
                attribution_method=AttributionMethod.FIFO,
            )
            if assignment is not None:
                assignments.append(assignment)
        return assignments

    def _hire_assignments(
        self,
        events: Sequence[LifecycleEvent],
    ) -> list[SpanCreditAssignment]:
        assignments: list[SpanCreditAssignment] = []
        for event in events:
            if event.event_type is not LifecycleEventType.HIRED:
                continue
            cost = float(event.data.get("cost", -float(event.attributed_value)))
            hire_id = event.hire_id or event.event_id
            assignment = self._assignment(
                source_event=event,
                target_event=event,
                role=SpanCreditRole.HIRE,
                source_id=hire_id,
                amount=-cost / self.reward_scale,
                reason="executed_hire_cost",
            )
            if assignment is not None:
                assignments.append(assignment)
        return assignments

    def assign_episode(
        self,
        events: Iterable[LifecycleEvent],
    ) -> tuple[SpanCreditAssignment, ...]:
        event_rows = tuple(events)
        self._assignment_sequence = 0
        return tuple(
            [
                *self._farm_assignments(event_rows),
                *self._trade_assignments(event_rows),
                *self._hire_assignments(event_rows),
            ]
        )


def aggregate_training_credits(
    assignments: Iterable[SpanCreditAssignment],
) -> tuple[SpanTrainingCredit, ...]:
    """Her action span için tam bir yerel PPO terimi üretir.

    Bir SELL aynı anda hem hasat hem market lotu satabilir. Attribution
    kayıtları ayrı kalır; fakat policy aynı span ratio'sunu iki kez yemez.
    Tutarlar action bazında toplanır. Role balancing etiketi, o action'daki
    mutlak ekonomik katkısı en büyük role verilir. Eşitlikte enum sırası
    deterministik tie-break'tir. BUY_SEED assignment'ları PURCHASE olarak
    üretildiğinden satın alma kovası FARM'dan ayrı kalır.
    """

    totals: dict[str, float] = defaultdict(float)
    role_totals: dict[str, dict[SpanCreditRole, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for assignment in assignments:
        amount = float(assignment.amount)
        totals[assignment.action_id] += amount
        role_totals[assignment.action_id][assignment.credit_role] += amount

    rows: list[SpanTrainingCredit] = []
    for action_id in sorted(totals):
        amount = totals[action_id]
        if math.isclose(amount, 0.0, abs_tol=1e-15):
            continue
        dominant_role = min(
            role_totals[action_id],
            key=lambda role: (-abs(role_totals[action_id][role]), int(role)),
        )
        rows.append(
            SpanTrainingCredit(
                action_id=action_id,
                credit_role=dominant_role,
                amount=amount,
            )
        )
    return tuple(rows)


def summarize_span_credits(
    assignments: Sequence[SpanCreditAssignment],
) -> EconomicCreditSummary:
    by_role: dict[str, dict[str, float | int]] = {}
    for role in SpanCreditRole:
        amounts = [
            assignment.amount
            for assignment in assignments
            if assignment.credit_role is role
        ]
        by_role[role.label] = {
            "count": len(amounts),
            "total": float(sum(amounts)),
            "positive": float(sum(max(amount, 0.0) for amount in amounts)),
            "negative": float(sum(min(amount, 0.0) for amount in amounts)),
        }
    amounts = [assignment.amount for assignment in assignments]
    return EconomicCreditSummary(
        assignment_count=len(assignments),
        total=float(sum(amounts)),
        positive=float(sum(max(amount, 0.0) for amount in amounts)),
        negative=float(sum(min(amount, 0.0) for amount in amounts)),
        by_role=by_role,
    )


def expected_normalized_economic_net(
    events: Iterable[LifecycleEvent],
    *,
    reward_scale: float,
) -> float:
    """V1 kapsamındaki gelir-maliyet defterinin bağımsız invariant değeri."""

    if reward_scale <= 0.0:
        raise ValueError("reward_scale pozitif olmalı.")
    revenue = 0.0
    cost = 0.0
    for event in events:
        if event.event_type in {
            LifecycleEventType.SEED_PURCHASED,
            LifecycleEventType.PRODUCT_PURCHASED,
            LifecycleEventType.HIRED,
        }:
            cost += max(-float(event.attributed_value), 0.0)
        elif event.event_type is LifecycleEventType.PRODUCT_SOLD:
            for allocation in event.data.get("allocations", ()):
                if (
                    allocation.get("source_plant_id") is not None
                    or allocation.get("source_purchase_id") is not None
                ):
                    revenue += float(
                        allocation.get(
                            "realized_revenue",
                            allocation.get("estimated_revenue", 0.0),
                        )
                    )
    return (revenue - cost) / reward_scale
