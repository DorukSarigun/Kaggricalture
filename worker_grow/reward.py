"""Her işçi adımında: yetiştirme hamlesinin ürünü × (güncel fiyat / gereken aksiyon)."""
from collections.abc import Mapping, Sequence
from typing import Any

from kaggle_environments.envs.kaggriculture import kaggriculture as game

from worker_grow.recipes import GROW_OPS, market_product, required_unit_actions

REWARD_SCALE = 3000.0


def current_price(observation: Mapping[str, Any], product: str) -> float:
    prices = observation["market"]["prices"]
    if product in prices:
        return float(prices[product])
    inventory = observation["market"]["inventory"][product]
    return float(game.market_price(product, inventory, observation["market"].get("params")))


def _tile_at(farm: Mapping[str, Any], position_xy: Sequence[int]) -> Any:
    x, y = int(position_xy[0]), int(position_xy[1])
    return farm["tiles"][y][x]


def _unit_position(farm: Mapping[str, Any], unit_index: int) -> list[int]:
    if unit_index == 0:
        return list(farm["farmer"])
    return list(farm["hands"][unit_index - 1])


def grow_products(observation: Mapping[str, Any], unit_index: int, action: Sequence[Any]) -> tuple[str, ...]:
    """Bu unit aksiyonunun büyüttüğü market ürünleri. Market emri yoksa boş."""
    if not action:
        return ()
    op = action[0]
    if op not in GROW_OPS:
        return ()
    player = int(observation["player"])
    farm = observation["farms"][player]
    tile = _tile_at(farm, _unit_position(farm, unit_index))
    named = market_product(str(action[1])) if len(action) > 1 else None

    if op == "COLLECT_FERTILIZER":
        return ("FERTILIZER",)
    if op == "BUILD_COOP":
        return ("EGG",)
    if op == "BUILD_PASTURE":
        return ("MILK", "WOOL")
    if named:
        return (named,)
    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
        product = market_product(str(tile["crop"]))
        return (product,) if product else ()
    if isinstance(tile, dict) and "animal" in tile:
        product = market_product(str(tile["animal"]))
        return (product,) if product else ()
    return ()


def grow_step_reward(
    observation: Mapping[str, Any],
    unit_index: int,
    action: Sequence[Any],
    *,
    scale: float = REWARD_SCALE,
) -> dict[str, Any]:
    """Yetiştirme hamlesi yoksa 0. Birden fazla üründe fiyatlar ayrı bölünüp toplanır."""
    products = grow_products(observation, unit_index, action)
    shares = []
    gold = 0.0
    for product in products:
        price = current_price(observation, product)
        actions_needed = required_unit_actions(product)
        share = price / actions_needed
        shares.append({"product": product, "price": price, "actions_needed": actions_needed, "gold": share})
        gold += share
    return {
        "products": products,
        "shares": shares,
        "gold": gold,
        "reward": gold / scale,
        "grow": bool(products),
    }
