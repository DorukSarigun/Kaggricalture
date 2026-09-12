"""Seed -> plant -> harvest -> sale yaşam döngüsü için event-sourcing tracker.

Oyun motoru aynı tür item'ları sayaçlarda birleştirir; gerçek nesne kimliği
sağlamaz. Bu modül observation veya environment'i değiştirmeden deterministik
sentetik lot/plant kimlikleri üretir. Birleşen inventory'lerden satış kaynağı
FIFO muhasebe kuralıyla atanır.
"""

from __future__ import annotations

import gzip
import json
import os
from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from kaggle_environments.envs.kaggriculture import kaggriculture as game

from action_encoding import ActionToken, ID_TO_TOKEN
from market_execution import MarketOrderExecution


class LifecycleEventType(StrEnum):
    INITIAL_SEED_LOT = "INITIAL_SEED_LOT"
    SEED_PURCHASED = "SEED_PURCHASED"
    PLANTED = "PLANTED"
    WATERED = "WATERED"
    FERTILIZED = "FERTILIZED"
    HARVESTED = "HARVESTED"
    PLANT_LOST = "PLANT_LOST"
    PRODUCT_PURCHASED = "PRODUCT_PURCHASED"
    PRODUCT_MOVED = "PRODUCT_MOVED"
    PRODUCT_SOLD = "PRODUCT_SOLD"
    HIRED = "HIRED"
    INVENTORY_RECONCILED = "INVENTORY_RECONCILED"


@dataclass(frozen=True, slots=True)
class ActionTokenSpan:
    section: str
    index: int
    start: int
    end: int
    token_ids: tuple[int, ...]
    tokens: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LifecycleEvent:
    event_id: str
    event_type: LifecycleEventType
    episode: int
    step: int
    day: int
    hour: int
    player: int
    token_span: ActionTokenSpan | None = None
    plant_id: str | None = None
    seed_lot_id: str | None = None
    product_lot_id: str | None = None
    purchase_id: str | None = None
    hire_id: str | None = None
    crop: str | None = None
    tile_xy: tuple[int, int] | None = None
    quantity: int | None = None
    attributed_value: float = 0.0
    normalized_credit: float = 0.0
    transition_reward: float | None = None
    data: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["event_type"] = self.event_type.value
        return value


class LifecycleEventSink(Protocol):
    """Observer pattern: tracker event üretir, sink event'i saklar/yayınlar."""

    def emit(self, event: LifecycleEvent) -> None: ...


class InMemoryEventJournal:
    def __init__(self) -> None:
        self.events: list[LifecycleEvent] = []

    def emit(self, event: LifecycleEvent) -> None:
        self.events.append(event)


@dataclass(slots=True)
class _SeedLotState:
    lot_id: str
    crop: str
    remaining: int
    unit_cost: float


@dataclass(slots=True)
class _ProductLotState:
    lot_id: str
    item: str
    source_plant_id: str | None
    source_purchase_id: str | None
    unit_cost: float
    remaining: int
    location: str


def extract_action_token_spans(
    token_ids: Sequence[int],
) -> dict[tuple[str, int], ActionTokenSpan]:
    """Canonical action token dizisini unit ve market-order span'lerine ayırır."""
    ids = [int(token_id) for token_id in token_ids]
    tokens = [ID_TO_TOKEN[token_id] for token_id in ids]
    spans: dict[tuple[str, int], ActionTokenSpan] = {}
    unit_index = 0
    market_index = 0
    index = 0
    in_market = False

    while index < len(tokens):
        token = tokens[index]
        if token in {ActionToken.FARMER, ActionToken.HAND}:
            end = index
            while end < len(tokens) and tokens[end] is not ActionToken.END_ACTION:
                end += 1
            if end >= len(tokens):
                break
            span_ids = tuple(ids[index : end + 1])
            spans[("unit", unit_index)] = ActionTokenSpan(
                section="unit",
                index=unit_index,
                start=index,
                end=end,
                token_ids=span_ids,
                tokens=tuple(ID_TO_TOKEN[token_id].value for token_id in span_ids),
            )
            unit_index += 1
            index = end + 1
            continue
        if token is ActionToken.MARKET:
            in_market = True
            index += 1
            continue
        if in_market and token is ActionToken.STOP:
            spans[("market", market_index)] = ActionTokenSpan(
                section="market", index=market_index, start=index, end=index,
                token_ids=(ids[index],), tokens=(token.value,),
            )
            break
        if in_market and token in {
            ActionToken.MARKET_BUY_SEED,
            ActionToken.MARKET_BUY_PRODUCT,
            ActionToken.MARKET_BUY_ANIMAL,
            ActionToken.MARKET_SELL,
            ActionToken.MARKET_HIRE,
            ActionToken.MARKET_BUY_LAND,
        }:
            end = index
            while end < len(tokens) and tokens[end] is not ActionToken.END_ORDER:
                end += 1
            if end >= len(tokens):
                break
            span_ids = tuple(ids[index : end + 1])
            spans[("market", market_index)] = ActionTokenSpan(
                section="market",
                index=market_index,
                start=index,
                end=end,
                token_ids=span_ids,
                tokens=tuple(ID_TO_TOKEN[token_id].value for token_id in span_ids),
            )
            market_index += 1
            index = end + 1
            continue
        index += 1
    return spans


class FarmLifecycleTracker:
    """Tek oyuncunun ekonomik nesne yaşam döngüsünü transition bazında izler."""

    def __init__(
        self,
        initial_observation: Mapping[str, Any],
        *,
        episode_number: int,
        player_index: int | None = None,
        turns_per_day: int = 24,
        shed_capacity: int = 100,
        reward_scale: float = 3000.0,
        sink: LifecycleEventSink | None = None,
    ) -> None:
        self.episode_number = int(episode_number)
        self.player_index = (
            int(initial_observation["player"])
            if player_index is None
            else int(player_index)
        )
        self.turns_per_day = int(turns_per_day)
        self.shed_capacity = int(shed_capacity)
        self.reward_scale = float(reward_scale)
        self.journal = InMemoryEventJournal()
        self._external_sink = sink
        self._sequence = 0
        self._seed_lots: dict[str, _SeedLotState] = {}
        self._seed_queues: dict[str, deque[str]] = defaultdict(deque)
        self._product_lots: dict[str, _ProductLotState] = {}
        self._product_queues: dict[tuple[str, str], deque[str]] = defaultdict(deque)
        self._active_plants: dict[tuple[int, int], str] = {}
        self._plant_identity: dict[str, tuple[str, int]] = {}
        self._bootstrap(initial_observation)

    @property
    def events(self) -> tuple[LifecycleEvent, ...]:
        return tuple(self.journal.events)

    def _next_id(self, kind: str) -> str:
        self._sequence += 1
        return f"ep{self.episode_number:06d}-{kind}-{self._sequence:06d}"

    def _clock(self, observation: Mapping[str, Any]) -> tuple[int, int, int]:
        day = int(observation["day"])
        hour = int(observation["hour"])
        step = int(observation.get("step", day * self.turns_per_day + hour))
        return step, day, hour

    def _emit(
        self,
        event_type: LifecycleEventType,
        observation: Mapping[str, Any],
        **kwargs: Any,
    ) -> LifecycleEvent:
        step, day, hour = self._clock(observation)
        event = LifecycleEvent(
            event_id=self._next_id("event"),
            event_type=event_type,
            episode=self.episode_number,
            step=step,
            day=day,
            hour=hour,
            player=self.player_index,
            **kwargs,
        )
        self.journal.emit(event)
        if self._external_sink is not None:
            self._external_sink.emit(event)
        return event

    def _bootstrap(self, observation: Mapping[str, Any]) -> None:
        private = observation["private"]
        for crop in game.CROPS:
            quantity = int(private["seeds"].get(crop, 0))
            if quantity > 0:
                lot_id = self._add_seed_lot(crop, quantity, 0.0)
                self._emit(
                    LifecycleEventType.INITIAL_SEED_LOT,
                    observation,
                    seed_lot_id=lot_id,
                    crop=crop,
                    quantity=quantity,
                    data={"source": "initial_observation", "unit_cost": 0.0},
                )

        farm = observation["farms"][self.player_index]
        for y, row in enumerate(farm["tiles"]):
            for x, tile in enumerate(row):
                if isinstance(tile, Mapping) and tile.get("kind") == "PLANT":
                    self._observe_existing_plant(observation, x, y, tile)

        for item in game.PRODUCTS:
            shed_quantity = int(private["shed"].get(item, 0))
            if shed_quantity > 0:
                self._add_product_lot(
                    item,
                    shed_quantity,
                    location="shed",
                    source_plant_id=None,
                )
        for unit_index, inventory in enumerate(private["inventories"]):
            for item in game.PRODUCTS:
                quantity = int(inventory.get(item, 0))
                if quantity > 0:
                    self._add_product_lot(
                        item,
                        quantity,
                        location=f"unit:{unit_index}",
                        source_plant_id=None,
                    )

    def _add_seed_lot(self, crop: str, quantity: int, unit_cost: float) -> str:
        lot_id = self._next_id("seedlot")
        self._seed_lots[lot_id] = _SeedLotState(
            lot_id=lot_id,
            crop=crop,
            remaining=int(quantity),
            unit_cost=float(unit_cost),
        )
        self._seed_queues[crop].append(lot_id)
        return lot_id

    def _consume_seed(self, crop: str) -> tuple[str | None, float]:
        queue = self._seed_queues[crop]
        while queue:
            lot = self._seed_lots[queue[0]]
            if lot.remaining <= 0:
                queue.popleft()
                continue
            lot.remaining -= 1
            return lot.lot_id, lot.unit_cost
        return None, 0.0

    def _add_product_lot(
        self,
        item: str,
        quantity: int,
        *,
        location: str,
        source_plant_id: str | None,
        source_purchase_id: str | None = None,
        unit_cost: float = 0.0,
    ) -> str:
        lot_id = self._next_id("productlot")
        self._product_lots[lot_id] = _ProductLotState(
            lot_id=lot_id,
            item=item,
            source_plant_id=source_plant_id,
            source_purchase_id=source_purchase_id,
            unit_cost=float(unit_cost),
            remaining=int(quantity),
            location=location,
        )
        self._product_queues[(location, item)].append(lot_id)
        return lot_id

    def _take_product(
        self,
        location: str,
        item: str,
        quantity: int,
    ) -> list[dict[str, Any]]:
        queue = self._product_queues[(location, item)]
        remaining = int(quantity)
        allocations: list[dict[str, Any]] = []
        while remaining > 0 and queue:
            lot = self._product_lots[queue[0]]
            if lot.remaining <= 0 or lot.location != location:
                queue.popleft()
                continue
            take = min(remaining, lot.remaining)
            lot.remaining -= take
            remaining -= take
            allocations.append(
                {
                    "product_lot_id": lot.lot_id,
                    "source_plant_id": lot.source_plant_id,
                    "source_purchase_id": lot.source_purchase_id,
                    "unit_cost": lot.unit_cost,
                    "allocated_cost": take * lot.unit_cost,
                    "quantity": take,
                }
            )
            if lot.remaining <= 0:
                queue.popleft()
        if remaining > 0:
            lot_id = self._add_product_lot(
                item,
                remaining,
                location=location,
                source_plant_id=None,
            )
            lot = self._product_lots[lot_id]
            lot.remaining = 0
            self._product_queues[(location, item)].pop()
            allocations.append(
                {
                    "product_lot_id": lot_id,
                    "source_plant_id": None,
                    "source_purchase_id": None,
                    "unit_cost": 0.0,
                    "allocated_cost": 0.0,
                    "quantity": remaining,
                    "reconciled": True,
                }
            )
        return allocations

    def _move_product(
        self,
        source: str,
        destination: str,
        item: str,
        quantity: int,
    ) -> list[dict[str, Any]]:
        allocations = self._take_product(source, item, quantity)
        for allocation in allocations:
            source_lot = self._product_lots[allocation["product_lot_id"]]
            destination_lot_id = self._add_product_lot(
                item,
                int(allocation["quantity"]),
                location=destination,
                source_plant_id=source_lot.source_plant_id,
                source_purchase_id=source_lot.source_purchase_id,
                unit_cost=source_lot.unit_cost,
            )
            allocation["destination_product_lot_id"] = destination_lot_id
        return allocations

    @staticmethod
    def _same_plant(left: Any, right: Any) -> bool:
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and left.get("kind") == right.get("kind") == "PLANT"
            and left.get("crop") == right.get("crop")
            and int(left.get("planted_day", -1)) == int(right.get("planted_day", -2))
        )

    def _observe_existing_plant(
        self,
        observation: Mapping[str, Any],
        x: int,
        y: int,
        tile: Mapping[str, Any],
    ) -> str:
        coordinate = (int(x), int(y))
        existing = self._active_plants.get(coordinate)
        identity = (str(tile["crop"]), int(tile["planted_day"]))
        if existing is not None and self._plant_identity.get(existing) == identity:
            return existing
        plant_id = self._next_id("plant")
        self._active_plants[coordinate] = plant_id
        self._plant_identity[plant_id] = identity
        self._emit(
            LifecycleEventType.PLANTED,
            observation,
            plant_id=plant_id,
            crop=str(tile["crop"]),
            tile_xy=coordinate,
            data={
                "planted_day": int(tile["planted_day"]),
                "source": "observed_existing_tile",
                "seed_unit_cost": 0.0,
            },
        )
        return plant_id

    def _plant_for_tile(
        self,
        observation: Mapping[str, Any],
        x: int,
        y: int,
        tile: Mapping[str, Any],
    ) -> str:
        plant_id = self._active_plants.get((x, y))
        identity = (str(tile["crop"]), int(tile["planted_day"]))
        if plant_id is None or self._plant_identity.get(plant_id) != identity:
            return self._observe_existing_plant(observation, x, y, tile)
        return plant_id

    def _emit_product_move(
        self,
        observation: Mapping[str, Any],
        *,
        source: str,
        destination: str,
        item: str,
        quantity: int,
        token_span: ActionTokenSpan | None,
        transition_reward: float | None,
    ) -> None:
        if quantity <= 0:
            return
        allocations = self._move_product(source, destination, item, quantity)
        self._emit(
            LifecycleEventType.PRODUCT_MOVED,
            observation,
            token_span=token_span,
            crop=item if item in game.CROPS else None,
            quantity=quantity,
            transition_reward=transition_reward,
            data={
                "source": source,
                "destination": destination,
                "allocations": allocations,
                "attribution_method": "synthetic_fifo",
            },
        )

    def _tracked_product_quantity(self, location: str, item: str) -> int:
        return sum(
            lot.remaining
            for lot in self._product_lots.values()
            if lot.location == location and lot.item == item and lot.remaining > 0
        )

    def _reconcile_product_inventories(
        self,
        observation: Mapping[str, Any],
        next_observation: Mapping[str, Any],
        transition_reward: float | None,
    ) -> None:
        """Gece DROP ve dış kaynaklı lotları next observation ile uzlaştırır."""
        private = next_observation["private"]
        locations = [
            "shed",
            *(f"unit:{index}" for index in range(len(private["inventories"]))),
        ]
        for item in game.PRODUCTS:
            desired = {
                "shed": int(private["shed"].get(item, 0)),
                **{
                    f"unit:{index}": int(inventory.get(item, 0))
                    for index, inventory in enumerate(private["inventories"])
                },
            }
            current_locations = {
                lot.location
                for lot in self._product_lots.values()
                if lot.item == item and lot.remaining > 0
            }
            all_locations = list(dict.fromkeys([*locations, *sorted(current_locations)]))

            for destination in all_locations:
                deficit = desired.get(destination, 0) - self._tracked_product_quantity(
                    destination, item
                )
                if deficit <= 0:
                    continue
                for source in all_locations:
                    if source == destination or deficit <= 0:
                        continue
                    surplus = self._tracked_product_quantity(source, item) - desired.get(
                        source, 0
                    )
                    if surplus <= 0:
                        continue
                    moved = min(deficit, surplus)
                    allocations = self._move_product(
                        source,
                        destination,
                        item,
                        moved,
                    )
                    self._emit(
                        LifecycleEventType.INVENTORY_RECONCILED,
                        observation,
                        crop=item if item in game.CROPS else None,
                        quantity=moved,
                        transition_reward=transition_reward,
                        data={
                            "source": source,
                            "destination": destination,
                            "allocations": allocations,
                            "reason": "observation_location_reconciliation",
                        },
                    )
                    deficit -= moved
                if deficit > 0:
                    lot_id = self._add_product_lot(
                        item,
                        deficit,
                        location=destination,
                        source_plant_id=None,
                    )
                    self._emit(
                        LifecycleEventType.INVENTORY_RECONCILED,
                        observation,
                        product_lot_id=lot_id,
                        crop=item if item in game.CROPS else None,
                        quantity=deficit,
                        transition_reward=transition_reward,
                        data={
                            "source": "external_or_unobserved",
                            "destination": destination,
                            "reason": "observation_quantity_increase",
                        },
                    )

            for source in all_locations:
                surplus = self._tracked_product_quantity(source, item) - desired.get(
                    source, 0
                )
                if surplus <= 0:
                    continue
                allocations = self._take_product(source, item, surplus)
                self._emit(
                    LifecycleEventType.INVENTORY_RECONCILED,
                    observation,
                    crop=item if item in game.CROPS else None,
                    quantity=surplus,
                    transition_reward=transition_reward,
                    data={
                        "source": source,
                        "destination": "lost_or_consumed",
                        "allocations": allocations,
                        "reason": "observation_quantity_decrease",
                    },
                )

    def _observe_traced_market(
        self,
        observation: Mapping[str, Any],
        next_observation: Mapping[str, Any],
        executions: Sequence[MarketOrderExecution],
        token_spans: Mapping[tuple[str, int], ActionTokenSpan],
        transition_reward: float | None,
    ) -> None:
        """Gerçek lockstep replay çıktısını kaynak-korumalı lotlara uygular."""

        farm = observation["farms"][self.player_index]
        next_farm = next_observation["farms"][self.player_index]
        existing_hand_count = len(farm["hands"])
        hires_today_before = int(farm.get("hires_today", 0))
        hire_ordinal = 0

        for execution in sorted(executions, key=lambda row: row.order_index):
            token_span = token_spans.get(("market", execution.order_index))
            operation = execution.operation
            item = execution.item
            quantity = int(execution.executed_quantity)
            if quantity <= 0:
                continue

            if operation == "BUY_SEED" and item in game.CROPS:
                total_cost = execution.total_value
                unit_cost = total_cost / quantity
                lot_id = self._add_seed_lot(item, quantity, unit_cost)
                self._emit(
                    LifecycleEventType.SEED_PURCHASED,
                    observation,
                    token_span=token_span,
                    seed_lot_id=lot_id,
                    crop=item,
                    quantity=quantity,
                    attributed_value=-total_cost,
                    normalized_credit=-total_cost / self.reward_scale,
                    transition_reward=transition_reward,
                    data={
                        "item": item,
                        "requested_quantity": execution.requested_quantity,
                        "unit_cost": unit_cost,
                        "unit_prices": list(execution.unit_prices),
                        "total_cost": total_cost,
                        "execution_source": "lockstep_market_replay",
                    },
                )
                continue

            if operation == "BUY_PRODUCT" and item in game.PRODUCTS:
                purchase_id = self._next_id("purchase")
                lot_ids: list[str] = []
                for unit_cost in execution.unit_prices:
                    lot_ids.append(
                        self._add_product_lot(
                            item,
                            1,
                            location="shed",
                            source_plant_id=None,
                            source_purchase_id=purchase_id,
                            unit_cost=float(unit_cost),
                        )
                    )
                total_cost = execution.total_value
                self._emit(
                    LifecycleEventType.PRODUCT_PURCHASED,
                    observation,
                    token_span=token_span,
                    product_lot_id=lot_ids[0] if lot_ids else None,
                    purchase_id=purchase_id,
                    crop=item if item in game.CROPS else None,
                    quantity=quantity,
                    attributed_value=-total_cost,
                    normalized_credit=-total_cost / self.reward_scale,
                    transition_reward=transition_reward,
                    data={
                        "item": item,
                        "requested_quantity": execution.requested_quantity,
                        "product_lot_ids": lot_ids,
                        "unit_prices": list(execution.unit_prices),
                        "total_cost": total_cost,
                        "execution_source": "lockstep_market_replay",
                        "attribution_method": "fifo",
                    },
                )
                continue

            if operation == "SELL" and item in game.PRODUCTS:
                allocations: list[dict[str, Any]] = []
                realized_revenue = 0.0
                for price in execution.unit_prices:
                    unit_allocations = self._take_product("shed", item, 1)
                    realized_revenue += float(price)
                    for allocation in unit_allocations:
                        allocation["realized_revenue"] = float(price)
                        # Eski read-model ve artifact okuyucularıyla uyumluluk.
                        allocation["estimated_revenue"] = float(price)
                        allocations.append(allocation)
                self._emit(
                    LifecycleEventType.PRODUCT_SOLD,
                    observation,
                    token_span=token_span,
                    crop=item if item in game.CROPS else None,
                    quantity=quantity,
                    attributed_value=realized_revenue,
                    normalized_credit=realized_revenue / self.reward_scale,
                    transition_reward=transition_reward,
                    data={
                        "requested_quantity": execution.requested_quantity,
                        "realized_revenue": realized_revenue,
                        "estimated_revenue": realized_revenue,
                        "unit_prices": list(execution.unit_prices),
                        "allocations": allocations,
                        "attribution_method": "fifo",
                        "revenue_is_estimate": False,
                        "execution_source": "lockstep_market_replay",
                    },
                )
                continue

            if operation == "HIRE":
                cost = execution.total_value
                hire_id = self._next_id("hire")
                expected_hand_index = existing_hand_count + hire_ordinal
                matched_next_hand = expected_hand_index < len(next_farm["hands"])
                crossed_day_boundary = int(next_observation["day"]) != int(
                    observation["day"]
                )
                if not matched_next_hand and not crossed_day_boundary:
                    raise RuntimeError(
                        "Traced HIRE next observation'daki yeni hand ile "
                        f"eşleşmedi: hand_index={expected_hand_index}, "
                        f"hands={len(next_farm['hands'])}"
                    )
                hires_today_sequence = hires_today_before + hire_ordinal
                hire_ordinal += 1
                self._emit(
                    LifecycleEventType.HIRED,
                    observation,
                    token_span=token_span,
                    hire_id=hire_id,
                    quantity=1,
                    attributed_value=-cost,
                    normalized_credit=-cost / self.reward_scale,
                    transition_reward=transition_reward,
                    data={
                        "cost": cost,
                        "hire_order_index": execution.order_index,
                        "hires_today_sequence": hires_today_sequence,
                        "expected_hand_index": expected_hand_index,
                        "hand_index": (
                            expected_hand_index if matched_next_hand else None
                        ),
                        "unit_index": (
                            expected_hand_index + 1 if matched_next_hand else None
                        ),
                        "initial_position": (
                            list(next_farm["hands"][expected_hand_index])
                            if matched_next_hand
                            else None
                        ),
                        "verified_in_next_observation": matched_next_hand,
                        "expired_before_active": (
                            crossed_day_boundary and not matched_next_hand
                        ),
                        "active_from_step": (
                            self._clock(observation)[0] + 1
                            if matched_next_hand
                            else None
                        ),
                        "expires_after_day": int(observation["day"]),
                        "execution_source": "lockstep_market_replay",
                    },
                )

    def observe_transition(
        self,
        observation: Mapping[str, Any],
        action: Mapping[str, Any],
        next_observation: Mapping[str, Any],
        *,
        token_ids: Sequence[int] = (),
        transition_reward: float | None = None,
        market_executions: Sequence[MarketOrderExecution] | None = None,
    ) -> tuple[LifecycleEvent, ...]:
        """Bir environment geçişini izler ve o geçişteki event'leri döndürür."""
        event_start = len(self.journal.events)
        token_spans = (
            extract_action_token_spans(token_ids) if len(token_ids) > 0 else {}
        )
        farm = deepcopy(observation["farms"][self.player_index])
        private = deepcopy(observation["private"])
        next_farm = next_observation["farms"][self.player_index]
        day = int(observation["day"])
        board_size = len(farm["tiles"])

        unit_actions = [action.get("farmer", ["PASS"]), *action.get("hands", [])]
        num_units = 1 + len(farm["hands"])
        unit_actions.extend([["PASS"]] * max(0, num_units - len(unit_actions)))
        unit_actions = unit_actions[:num_units]

        plant_demand: Counter[str] = Counter()
        for unit_action in unit_actions:
            if unit_action and unit_action[0] == "PLANT" and len(unit_action) >= 2:
                plant_demand[str(unit_action[1])] += 1
        blocked_crops = {
            crop
            for crop, quantity in plant_demand.items()
            if quantity > int(private["seeds"].get(crop, 0))
        }
        successful_plants: Counter[str] = Counter()

        for unit_index, raw_action in enumerate(unit_actions):
            unit_action = list(raw_action) if raw_action else ["PASS"]
            if (
                unit_action[0] == "PLANT"
                and len(unit_action) >= 2
                and unit_action[1] in blocked_crops
            ):
                unit_action = ["PASS"]
            position = game._farmer_position(farm, unit_index)
            if position is None:
                continue
            x, y = map(int, position)
            before_tile = deepcopy(farm["tiles"][y][x])
            before_inventory = deepcopy(game._farmer_inventory(private, unit_index))
            before_shed = deepcopy(private["shed"])
            token_span = token_spans.get(("unit", unit_index))

            game._apply_unit_action(
                farm,
                private,
                unit_index,
                unit_action,
                board_size,
                day,
                self.turns_per_day,
                self.shed_capacity,
            )
            after_tile = deepcopy(farm["tiles"][y][x])
            after_inventory = deepcopy(game._farmer_inventory(private, unit_index))
            op = str(unit_action[0])

            if op == "PLANT" and not self._same_plant(before_tile, after_tile):
                if isinstance(after_tile, Mapping) and after_tile.get("kind") == "PLANT":
                    crop = str(after_tile["crop"])
                    seed_lot_id, seed_unit_cost = self._consume_seed(crop)
                    plant_id = self._next_id("plant")
                    self._active_plants[(x, y)] = plant_id
                    self._plant_identity[plant_id] = (
                        crop,
                        int(after_tile["planted_day"]),
                    )
                    successful_plants[crop] += 1
                    self._emit(
                        LifecycleEventType.PLANTED,
                        observation,
                        token_span=token_span,
                        plant_id=plant_id,
                        seed_lot_id=seed_lot_id,
                        crop=crop,
                        tile_xy=(x, y),
                        quantity=1,
                        transition_reward=transition_reward,
                        data={
                            "planted_day": int(after_tile["planted_day"]),
                            "unit_index": unit_index,
                            "seed_unit_cost": seed_unit_cost,
                            "source": "unit_action",
                        },
                    )

            elif op == "WATER" and self._same_plant(before_tile, after_tile):
                if not bool(before_tile.get("watered_today")) and bool(
                    after_tile.get("watered_today")
                ):
                    plant_id = self._plant_for_tile(observation, x, y, before_tile)
                    self._emit(
                        LifecycleEventType.WATERED,
                        observation,
                        token_span=token_span,
                        plant_id=plant_id,
                        crop=str(before_tile["crop"]),
                        tile_xy=(x, y),
                        transition_reward=transition_reward,
                        data={"unit_index": unit_index},
                    )

            elif op == "FERTILIZE" and self._same_plant(before_tile, after_tile):
                if int(after_tile.get("fertilized_until_day", -1)) > int(
                    before_tile.get("fertilized_until_day", -1)
                ):
                    plant_id = self._plant_for_tile(observation, x, y, before_tile)
                    self._emit(
                        LifecycleEventType.FERTILIZED,
                        observation,
                        token_span=token_span,
                        plant_id=plant_id,
                        crop=str(before_tile["crop"]),
                        tile_xy=(x, y),
                        transition_reward=transition_reward,
                        data={
                            "unit_index": unit_index,
                            "fertilized_until_day": int(
                                after_tile["fertilized_until_day"]
                            ),
                        },
                    )

            elif op == "HARVEST" and isinstance(before_tile, Mapping):
                crop = before_tile.get("crop")
                if crop in game.CROPS:
                    harvested = max(
                        int(after_inventory.get(crop, 0))
                        - int(before_inventory.get(crop, 0)),
                        0,
                    )
                    if harvested > 0:
                        plant_id = self._plant_for_tile(
                            observation, x, y, before_tile
                        )
                        product_lot_id = self._add_product_lot(
                            str(crop),
                            harvested,
                            location=f"unit:{unit_index}",
                            source_plant_id=plant_id,
                        )
                        unit_price = float(observation["market"]["prices"][crop])
                        harvest_value = harvested * unit_price
                        closed = not bool(game.CROPS[crop]["ongoing"])
                        self._emit(
                            LifecycleEventType.HARVESTED,
                            observation,
                            token_span=token_span,
                            plant_id=plant_id,
                            product_lot_id=product_lot_id,
                            crop=str(crop),
                            tile_xy=(x, y),
                            quantity=harvested,
                            attributed_value=harvest_value,
                            normalized_credit=harvest_value / self.reward_scale,
                            transition_reward=transition_reward,
                            data={
                                "unit_index": unit_index,
                                "market_unit_price_at_harvest": unit_price,
                                "plant_closed": closed,
                                "credit_type": "harvest_market_value",
                            },
                        )
                        if closed:
                            self._active_plants.pop((x, y), None)

            elif op == "DIG" and isinstance(before_tile, Mapping):
                if before_tile.get("kind") == "PLANT" and not self._same_plant(
                    before_tile, after_tile
                ):
                    plant_id = self._plant_for_tile(observation, x, y, before_tile)
                    self._active_plants.pop((x, y), None)
                    self._emit(
                        LifecycleEventType.PLANT_LOST,
                        observation,
                        token_span=token_span,
                        plant_id=plant_id,
                        crop=str(before_tile["crop"]),
                        tile_xy=(x, y),
                        transition_reward=transition_reward,
                        data={"reason": "DUG", "unit_index": unit_index},
                    )

            if op in {"DROP", "PLACE"}:
                for item in game.PRODUCTS:
                    moved = max(
                        int(private["shed"].get(item, 0))
                        - int(before_shed.get(item, 0)),
                        0,
                    )
                    self._emit_product_move(
                        observation,
                        source=f"unit:{unit_index}",
                        destination="shed",
                        item=item,
                        quantity=moved,
                        token_span=token_span,
                        transition_reward=transition_reward,
                    )
            elif op == "PICKUP":
                for item in game.PRODUCTS:
                    moved = max(
                        int(after_inventory.get(item, 0))
                        - int(before_inventory.get(item, 0)),
                        0,
                    )
                    self._emit_product_move(
                        observation,
                        source="shed",
                        destination=f"unit:{unit_index}",
                        item=item,
                        quantity=moved,
                        token_span=token_span,
                        transition_reward=transition_reward,
                    )

        # Unit fazı ekimden önce seed tüketir, market fazı sonra seed ekler.
        # Bu denklem aynı turdaki PLANT + BUY_SEED durumunda da gerçek miktardır.
        before_seeds = observation["private"]["seeds"]
        after_seeds = next_observation["private"]["seeds"]
        purchased_by_crop = {
            crop: max(
                int(after_seeds.get(crop, 0))
                - int(before_seeds.get(crop, 0))
                + successful_plants[crop],
                0,
            )
            for crop in game.CROPS
        }
        legacy_market_orders = (
            action.get("market", []) if market_executions is None else []
        )
        for market_index, order in enumerate(legacy_market_orders):
            if not order:
                continue
            op = str(order[0])
            token_span = token_spans.get(("market", market_index))
            if op == "BUY_SEED" and len(order) >= 3:
                crop = str(order[1])
                executed = min(int(order[2]), purchased_by_crop.get(crop, 0))
                if executed > 0:
                    purchased_by_crop[crop] -= executed
                    unit_cost = float(game.CROPS[crop]["seed"])
                    lot_id = self._add_seed_lot(crop, executed, unit_cost)
                    self._emit(
                        LifecycleEventType.SEED_PURCHASED,
                        observation,
                        token_span=token_span,
                        seed_lot_id=lot_id,
                        crop=crop,
                        quantity=executed,
                        attributed_value=-executed * unit_cost,
                        normalized_credit=-executed * unit_cost / self.reward_scale,
                        transition_reward=transition_reward,
                        data={
                            "requested_quantity": int(order[2]),
                            "unit_cost": unit_cost,
                            "total_cost": executed * unit_cost,
                        },
                    )

        # Satılan harvest lotları FIFO ile shed'den düşer. Rakip aynı market
        # sırasında fiyatı etkileyebildiği için gelir bir attribution tahminidir.
        market_inventory = {
            item: int(observation["market"]["inventory"][item])
            for item in game.PRODUCTS
        }
        available_shed = {
            item: int(private["shed"].get(item, 0)) for item in game.PRODUCTS
        }
        for market_index, order in enumerate(legacy_market_orders):
            if not order or order[0] != "SELL" or len(order) < 3:
                continue
            item = str(order[1])
            sold = min(max(int(order[2]), 0), available_shed.get(item, 0))
            if sold <= 0:
                continue
            available_shed[item] -= sold
            allocations: list[dict[str, Any]] = []
            estimated_revenue = 0.0
            for _ in range(sold):
                unit_allocations = self._take_product("shed", item, 1)
                price = float(
                    game.market_price(
                        item,
                        market_inventory[item],
                        observation["market"].get("params"),
                    )
                )
                estimated_revenue += price
                if price > 1:
                    market_inventory[item] += 1
                for allocation in unit_allocations:
                    allocation["estimated_revenue"] = price
                    allocations.append(allocation)
            self._emit(
                LifecycleEventType.PRODUCT_SOLD,
                observation,
                token_span=token_spans.get(("market", market_index)),
                crop=item if item in game.CROPS else None,
                quantity=sold,
                attributed_value=estimated_revenue,
                normalized_credit=estimated_revenue / self.reward_scale,
                transition_reward=transition_reward,
                data={
                    "requested_quantity": int(order[2]),
                    "estimated_revenue": estimated_revenue,
                    "allocations": allocations,
                    "attribution_method": "synthetic_fifo_current_market_curve",
                    "revenue_is_estimate": True,
                },
            )

        if market_executions is not None:
            self._observe_traced_market(
                observation,
                next_observation,
                market_executions,
                token_spans,
                transition_reward,
            )

        # Gece yenilemesi veya decay ile kaybolan bitkileri next observation'dan bul.
        for (x, y), plant_id in list(self._active_plants.items()):
            next_tile = next_farm["tiles"][y][x]
            crop, planted_day = self._plant_identity[plant_id]
            if (
                isinstance(next_tile, Mapping)
                and next_tile.get("kind") == "PLANT"
                and next_tile.get("crop") == crop
                and int(next_tile.get("planted_day", -1)) == planted_day
            ):
                continue
            self._active_plants.pop((x, y), None)
            self._emit(
                LifecycleEventType.PLANT_LOST,
                observation,
                plant_id=plant_id,
                crop=crop,
                tile_xy=(x, y),
                transition_reward=transition_reward,
                data={
                    "reason": (
                        "BECAME_WEED"
                        if isinstance(next_tile, Mapping)
                        and next_tile.get("kind") == "WEED"
                        else "REMOVED_OR_REPLACED"
                    )
                },
            )

        for y, row in enumerate(next_farm["tiles"]):
            for x, tile in enumerate(row):
                if isinstance(tile, Mapping) and tile.get("kind") == "PLANT":
                    self._plant_for_tile(next_observation, x, y, tile)

        self._reconcile_product_inventories(
            observation,
            next_observation,
            transition_reward,
        )

        return tuple(self.journal.events[event_start:])

    def plant_lifecycles(self) -> list[dict[str, Any]]:
        """Event journal'ından plant read-model/projection üretir."""
        plants: dict[str, dict[str, Any]] = {}
        for event in self.journal.events:
            if event.event_type is LifecycleEventType.PLANTED and event.plant_id:
                plants[event.plant_id] = {
                    "plant_id": event.plant_id,
                    "crop": event.crop,
                    "tile_xy": list(event.tile_xy) if event.tile_xy else None,
                    "planted_step": event.step,
                    "planted_day": event.data.get("planted_day", event.day),
                    "seed_lot_id": event.seed_lot_id,
                    "seed_unit_cost": float(event.data.get("seed_unit_cost", 0.0)),
                    "status": "ACTIVE",
                    "water_steps": [],
                    "fertilize_steps": [],
                    "harvest_events": [],
                    "harvested_units": 0,
                    "harvest_market_value": 0.0,
                    "sold_units": 0,
                    "estimated_sale_revenue": 0.0,
                    "event_ids": [event.event_id],
                }
                continue
            if event.plant_id and event.plant_id in plants:
                plant = plants[event.plant_id]
                plant["event_ids"].append(event.event_id)
                if event.event_type is LifecycleEventType.WATERED:
                    plant["water_steps"].append(event.step)
                elif event.event_type is LifecycleEventType.FERTILIZED:
                    plant["fertilize_steps"].append(event.step)
                elif event.event_type is LifecycleEventType.HARVESTED:
                    plant["harvest_events"].append(
                        {
                            "step": event.step,
                            "quantity": event.quantity,
                            "market_value": event.attributed_value,
                            "product_lot_id": event.product_lot_id,
                        }
                    )
                    plant["harvested_units"] += int(event.quantity or 0)
                    plant["harvest_market_value"] += event.attributed_value
                    if event.data.get("plant_closed"):
                        plant["status"] = "HARVESTED"
                elif event.event_type is LifecycleEventType.PLANT_LOST:
                    plant["status"] = str(event.data.get("reason", "LOST"))

            if event.event_type is LifecycleEventType.PRODUCT_SOLD:
                for allocation in event.data.get("allocations", []):
                    source_plant_id = allocation.get("source_plant_id")
                    if source_plant_id not in plants:
                        continue
                    plant = plants[source_plant_id]
                    plant["sold_units"] += int(allocation["quantity"])
                    plant["estimated_sale_revenue"] += float(
                        allocation.get("estimated_revenue", 0.0)
                    )
                    if event.event_id not in plant["event_ids"]:
                        plant["event_ids"].append(event.event_id)

        for plant in plants.values():
            plant["estimated_realized_profit"] = (
                plant["estimated_sale_revenue"] - plant["seed_unit_cost"]
            )
        return list(plants.values())

    def product_lots(self) -> list[dict[str, Any]]:
        """Sentetik product lotlarının episode sonu projection'ı."""

        return [
            {
                "product_lot_id": lot.lot_id,
                "item": lot.item,
                "source_plant_id": lot.source_plant_id,
                "source_purchase_id": lot.source_purchase_id,
                "unit_cost": lot.unit_cost,
                "remaining": lot.remaining,
                "location": lot.location,
                "attribution_method": "fifo",
            }
            for lot in sorted(self._product_lots.values(), key=lambda row: row.lot_id)
        ]

    def trade_lifecycles(self) -> list[dict[str, Any]]:
        """BUY_PRODUCT lotlarını FIFO satış allocation'larıyla birleştirir."""

        trades: dict[str, dict[str, Any]] = {}
        for event in self.journal.events:
            if (
                event.event_type is LifecycleEventType.PRODUCT_PURCHASED
                and event.purchase_id is not None
            ):
                trades[event.purchase_id] = {
                    "purchase_id": event.purchase_id,
                    "item": event.data.get("item", event.crop),
                    "purchase_step": event.step,
                    "quantity": int(event.quantity or 0),
                    "total_cost": float(event.data.get("total_cost", 0.0)),
                    "sold_quantity": 0,
                    "realized_revenue": 0.0,
                    "allocated_cost": 0.0,
                    "sale_event_ids": [],
                    "attribution_method": "fifo",
                }
        for event in self.journal.events:
            if event.event_type is not LifecycleEventType.PRODUCT_SOLD:
                continue
            touched: set[str] = set()
            for allocation in event.data.get("allocations", ()):
                purchase_id = allocation.get("source_purchase_id")
                if purchase_id not in trades:
                    continue
                trade = trades[str(purchase_id)]
                trade["sold_quantity"] += int(allocation.get("quantity", 0))
                trade["realized_revenue"] += float(
                    allocation.get(
                        "realized_revenue",
                        allocation.get("estimated_revenue", 0.0),
                    )
                )
                trade["allocated_cost"] += float(
                    allocation.get("allocated_cost", 0.0)
                )
                touched.add(str(purchase_id))
            for purchase_id in touched:
                trades[purchase_id]["sale_event_ids"].append(event.event_id)
        for trade in trades.values():
            trade["open_cost"] = max(
                trade["total_cost"] - trade["allocated_cost"],
                0.0,
            )
            trade["realized_net_after_open_cost"] = (
                trade["realized_revenue"] - trade["total_cost"]
            )
        return list(trades.values())

    def worker_lifecycles(self) -> list[dict[str, Any]]:
        """V1'de günlük HIRE maliyetlerinin denetlenebilir projection'ı."""

        return [
            {
                "hire_id": event.hire_id,
                "step": event.step,
                "day": event.day,
                "cost": float(event.data.get("cost", 0.0)),
                "active_from_step": event.data.get("active_from_step"),
                "expires_after_day": event.data.get("expires_after_day"),
                "hires_today_sequence": event.data.get("hires_today_sequence"),
                "expected_hand_index": event.data.get("expected_hand_index"),
                "hand_index": event.data.get("hand_index"),
                "unit_index": event.data.get("unit_index"),
                "initial_position": event.data.get("initial_position"),
                "verified_in_next_observation": event.data.get(
                    "verified_in_next_observation", False
                ),
                "expired_before_active": event.data.get(
                    "expired_before_active", False
                ),
                "event_id": event.event_id,
            }
            for event in self.journal.events
            if event.event_type is LifecycleEventType.HIRED
        ]

    def summary(self) -> dict[str, Any]:
        lifecycles = self.plant_lifecycles()
        counts = Counter(event.event_type.value for event in self.journal.events)
        return {
            "episode": self.episode_number,
            "player": self.player_index,
            "identity_model": "synthetic_lot_ids",
            "inventory_attribution": "FIFO",
            "event_count": len(self.journal.events),
            "event_counts": dict(sorted(counts.items())),
            "plant_count": len(lifecycles),
            "harvested_plant_count": sum(
                plant["harvested_units"] > 0 for plant in lifecycles
            ),
            "total_harvested_units": sum(
                plant["harvested_units"] for plant in lifecycles
            ),
            "total_harvest_market_value": sum(
                plant["harvest_market_value"] for plant in lifecycles
            ),
            "total_attributed_sale_revenue": sum(
                plant["estimated_sale_revenue"] for plant in lifecycles
            ),
        }

    def write_artifacts(self, episode_directory: Path) -> dict[str, Path]:
        episode_directory.mkdir(parents=True, exist_ok=True)
        events_path = episode_directory / "lifecycle_events.jsonl.gz"
        plants_path = episode_directory / "plant_lifecycles.json"
        products_path = episode_directory / "product_lots.json"
        trades_path = episode_directory / "trade_lifecycles.json"
        workers_path = episode_directory / "worker_lifecycles.json"
        summary_path = episode_directory / "lifecycle_summary.json"

        temporary_events = events_path.with_name(f".{events_path.name}.tmp")
        with gzip.open(
            temporary_events,
            "wt",
            encoding="utf-8",
            compresslevel=6,
        ) as file:
            for event in self.journal.events:
                json.dump(event.to_dict(), file, ensure_ascii=False)
                file.write("\n")
        os.replace(temporary_events, events_path)

        self._write_json(plants_path, self.plant_lifecycles())
        self._write_json(products_path, self.product_lots())
        self._write_json(trades_path, self.trade_lifecycles())
        self._write_json(workers_path, self.worker_lifecycles())
        self._write_json(summary_path, self.summary())
        return {
            "events": events_path,
            "plants": plants_path,
            "products": products_path,
            "trades": trades_path,
            "workers": workers_path,
            "summary": summary_path,
        }

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(value, file, ensure_ascii=False, indent=2)
            file.write("\n")
        os.replace(temporary, path)
