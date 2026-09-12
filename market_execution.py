"""Kaggriculture market fazını yan etkisiz biçimde yeniden oynatıp izler.

Environment unit aksiyonlarını marketten önce uygular ve iki oyuncunun market
emirlerini order-index bazında lockstep yürütür. Observation yalnızca turun
sonunu verdiği için BUY_PRODUCT/SELL/HIRE maliyetlerini doğru action span'ına
bağlamak amacıyla aynı deterministik faz burada kopyalar üzerinde replay edilir.

Bu modül environment state'ini değiştirmez. Sonuçlar gerçek next observation
ile entegrasyon testlerinde karşılaştırılır.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from kaggle_environments.envs.kaggriculture import kaggriculture as game


@dataclass(frozen=True, slots=True)
class MarketOrderExecution:
    """Tek oyuncunun tek market order'ında gerçekten execute edilen bölüm."""

    player: int
    order_index: int
    operation: str
    item: str | None
    requested_quantity: int
    executed_quantity: int
    unit_prices: tuple[float, ...]

    @property
    def total_value(self) -> float:
        return float(sum(self.unit_prices))


@dataclass(slots=True)
class _ExecutionBuilder:
    player: int
    order_index: int
    operation: str
    item: str | None
    requested_quantity: int
    unit_prices: list[float]

    def freeze(self) -> MarketOrderExecution:
        return MarketOrderExecution(
            player=self.player,
            order_index=self.order_index,
            operation=self.operation,
            item=self.item,
            requested_quantity=self.requested_quantity,
            executed_quantity=len(self.unit_prices),
            unit_prices=tuple(self.unit_prices),
        )


def _config_value(configuration: Mapping[str, Any], name: str, default: Any) -> Any:
    value = configuration.get(name, default)
    return default if value is None else value


def _simulate_units(
    farms: list[dict[str, Any]],
    privates: list[dict[str, Any]],
    actions: Sequence[Mapping[str, Any]],
    *,
    day: int,
    board_size: int,
    turns_per_day: int,
    shed_capacity: int,
) -> None:
    """Interpreter'ın market öncesi unit fazını kopyalar üzerinde uygular."""

    for player, raw_action in enumerate(actions):
        action = raw_action if isinstance(raw_action, Mapping) else {}
        farmer_action = action.get("farmer", ["PASS"])
        hands_actions = action.get("hands", [])
        if not isinstance(hands_actions, list):
            hands_actions = []

        unit_actions = [farmer_action, *hands_actions]
        plant_demand: dict[str, int] = {}
        for unit_action in unit_actions:
            if (
                isinstance(unit_action, list)
                and len(unit_action) >= 2
                and unit_action[0] == "PLANT"
            ):
                crop = str(unit_action[1])
                plant_demand[crop] = plant_demand.get(crop, 0) + 1
        seeds = privates[player].get("seeds", {})
        blocked = {
            crop
            for crop, quantity in plant_demand.items()
            if quantity > int(seeds.get(crop, 0))
        }

        def allowed(unit_action: Any) -> list[Any]:
            if (
                isinstance(unit_action, list)
                and len(unit_action) >= 2
                and unit_action[0] == "PLANT"
                and unit_action[1] in blocked
            ):
                return ["PASS"]
            return list(unit_action) if isinstance(unit_action, list) else ["PASS"]

        game._apply_unit_action(
            farms[player],
            privates[player],
            0,
            allowed(farmer_action),
            board_size,
            day,
            turns_per_day,
            shed_capacity,
        )
        for hand_index, hand_action in enumerate(hands_actions):
            game._apply_unit_action(
                farms[player],
                privates[player],
                hand_index + 1,
                allowed(hand_action),
                board_size,
                day,
                turns_per_day,
                shed_capacity,
            )


def trace_market_executions(
    observations: Sequence[Mapping[str, Any]],
    actions: Sequence[Mapping[str, Any]],
    configuration: Mapping[str, Any],
) -> tuple[tuple[MarketOrderExecution, ...], ...]:
    """İki oyuncunun market execution'larını environment ile aynı sırada üretir.

    ``observations`` player sırasındadır ve her satır kendi private alanını
    taşımalıdır. Dönen dış tuple da player sırasındadır. Başarısız veya sıfır
    execute edilen order'lar kredi kaydı üretmemek için sonuçta yer almaz.
    """

    if len(observations) != 2 or len(actions) != 2:
        raise ValueError("Market trace tam iki observation ve iki action gerektirir.")

    shared = observations[0]
    farms = deepcopy(list(shared["farms"]))
    privates = [deepcopy(dict(observation["private"])) for observation in observations]
    market = deepcopy(dict(shared["market"]))
    board_size = int(_config_value(configuration, "boardSize", len(farms[0]["tiles"])))
    turns_per_day = max(1, int(_config_value(configuration, "turnsPerDay", 24)))
    shed_capacity = int(_config_value(configuration, "shedCapacity", 100))
    max_orders = max(1, int(_config_value(configuration, "maxMarketOrdersPerTurn", 10)))
    hire_multiplier = int(
        _config_value(configuration, "farmHandCostMult", game.FARM_HAND_COST_MULT)
    )
    day = int(shared.get("day", 0))

    _simulate_units(
        farms,
        privates,
        actions,
        day=day,
        board_size=board_size,
        turns_per_day=turns_per_day,
        shed_capacity=shed_capacity,
    )

    queues: list[list[Any]] = []
    for action in actions:
        market_orders = action.get("market", []) if isinstance(action, Mapping) else []
        queue = list(market_orders) if isinstance(market_orders, list) else []
        queues.append(queue[:max_orders])

    results: list[list[MarketOrderExecution]] = [[], []]
    max_length = max((len(queue) for queue in queues), default=0)
    for order_index in range(max_length):
        order_states: list[dict[str, Any] | None] = []
        builders: list[_ExecutionBuilder | None] = []
        for player, queue in enumerate(queues):
            raw_order = queue[order_index] if order_index < len(queue) else None
            order_state = game._parse_order(raw_order)
            order_states.append(order_state)
            if order_state is None:
                builders.append(None)
                continue
            operation = str(order_state["type"])
            item = order_state.get("item")
            builders.append(
                _ExecutionBuilder(
                    player=player,
                    order_index=order_index,
                    operation=operation,
                    item=str(item) if item is not None else None,
                    requested_quantity=int(order_state.get("remaining", 1)),
                    unit_prices=[],
                )
            )

        # Atomic order'lar environment'taki gibi player sırasıyla uygulanır.
        for player, order_state in enumerate(order_states):
            if order_state is None:
                continue
            operation = str(order_state["type"])
            if operation == "HIRE":
                before_money = float(farms[player]["money"])
                before_hands = len(farms[player]["hands"])
                game._do_hire(
                    farms[player],
                    privates[player],
                    board_size,
                    hire_multiplier,
                )
                if len(farms[player]["hands"]) == before_hands + 1:
                    assert builders[player] is not None
                    builders[player].unit_prices.append(
                        before_money - float(farms[player]["money"])
                    )
                order_states[player] = None
            elif operation == "BUY_LAND":
                before_money = float(farms[player]["money"])
                before_land = len(farms[player]["unlocked_quadrants"])
                game._do_buy_land(farms[player], board_size)
                if len(farms[player]["unlocked_quadrants"]) == before_land + 1:
                    assert builders[player] is not None
                    builders[player].unit_prices.append(
                        before_money - float(farms[player]["money"])
                    )
                order_states[player] = None

        while True:
            quoted: list[tuple[str, str, float, dict[str, Any]] | None] = [None, None]
            for player, order_state in enumerate(order_states):
                if order_state is None or int(order_state["remaining"]) <= 0:
                    continue
                operation = str(order_state["type"])
                item = str(order_state["item"])
                if operation == "SELL" and item in game.PRODUCTS:
                    price = game.market_price(
                        item,
                        market["inventory"][item],
                        market.get("params"),
                    )
                elif operation == "BUY_PRODUCT" and item in ("WHEAT", "FERTILIZER"):
                    price = game.market_price(
                        item,
                        market["inventory"][item] - 1,
                        market.get("params"),
                    )
                elif operation == "BUY_SEED" and item in game.CROPS:
                    price = game.CROPS[item]["seed"]
                elif operation == "BUY_ANIMAL" and item in game.ANIMALS:
                    price = game.ANIMALS[item]["cost"]
                else:
                    order_states[player] = None
                    continue
                quoted[player] = (operation, item, float(price), order_state)

            if all(quote is None for quote in quoted):
                break

            committed_any = False
            for player, quote in enumerate(quoted):
                if quote is None:
                    continue
                operation, item, price, order_state = quote
                committed = game._commit_unit(
                    operation,
                    item,
                    price,
                    farms[player],
                    privates[player],
                    market,
                    shed_capacity,
                )
                if committed:
                    order_state["remaining"] -= 1
                    committed_any = True
                    assert builders[player] is not None
                    builders[player].unit_prices.append(price)
                else:
                    order_states[player] = None
            if not committed_any:
                break

        game._refresh_prices(market)
        for player, builder in enumerate(builders):
            if builder is not None and builder.unit_prices:
                results[player].append(builder.freeze())

    return tuple(tuple(player_results) for player_results in results)
