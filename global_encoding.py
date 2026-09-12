import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


MARKET_PRODUCTS = (
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
)

SHED_ITEMS = MARKET_PRODUCTS + (
    "GOOSE",
    "COW",
    "SHEEP",
)

SEED_ITEMS = (
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
)

SHOP_TYPES = (
    "BAKERY",
    "PIZZA_SHOP",
    "BRUNCH_SPOT",
    "YARN_STORE",
    "ICE_CREAM_SHOP",
    "PET_CAFE",
    "SMOOTHIE_SHOP",
    "FARMERS_MARKET",
)

GLOBAL_FEATURE_NAMES = (
    "season_progress",
    "hour_sin",
    "hour_cos",
    *(f"market_price_{item}" for item in MARKET_PRODUCTS),
    *(f"market_inventory_{item}" for item in MARKET_PRODUCTS),
    "own_money",
    "opponent_money",
    *(f"shed_{item}" for item in SHED_ITEMS),
    *(f"seed_{item}" for item in SEED_ITEMS),
    *(f"carried_{item}" for item in SHED_ITEMS),
    "own_hand_count",
    "opponent_hand_count",
    *(f"shop_count_{shop}" for shop in SHOP_TYPES),
)

MARKET_PARAMS_PATH = Path(__file__).parent / "data" / "market_params.json"
with MARKET_PARAMS_PATH.open(encoding="utf-8") as file:
    MARKET_PARAMS = json.load(file)["curves"]


def _saturating_scale(value: int | float, scale: int | float) -> float:
    value = max(float(value), 0.0)
    return value / (value + scale)


def _price_signal(price: int | float, base_price: int | float) -> float:
    return _saturating_scale(price, base_price)


def _sum_inventories(
    inventories: list[Mapping[str, int]],
) -> dict[str, int]:
    return {
        item: sum(inventory.get(item, 0) for inventory in inventories)
        for item in SHED_ITEMS
    }


def encode_global_observation(observation: Mapping[str, Any]) -> np.ndarray:
    """Observation'ı sabit boyutlu, oyuncu perspektifli global vektöre çevirir."""
    player_index = int(observation["player"])
    farms = observation["farms"]
    own_farm = farms[player_index]
    opponent_farm = farms[1 - player_index]

    market = observation["market"]
    prices = market["prices"]
    market_inventory = market["inventory"]

    private = observation["private"]
    shed = private["shed"]
    seeds = private["seeds"]
    carried = _sum_inventories(private["inventories"])

    hour_angle = 2 * math.pi * observation["hour"] / 24
    shop_counts = {
        shop: observation["town"]["unlocked_shops"].count(shop)
        for shop in SHOP_TYPES
    }
    current_step = observation.get(
        "step",
        observation["day"] * 24 + observation["hour"],
    )

    features = [
        current_step / 720,
        math.sin(hour_angle),
        math.cos(hour_angle),
        *(
            _price_signal(prices[item], MARKET_PARAMS[item]["base"])
            for item in MARKET_PRODUCTS
        ),
        *(
            _saturating_scale(
                market_inventory[item],
                MARKET_PARAMS[item]["I0"],
            )
            for item in MARKET_PRODUCTS
        ),
        _saturating_scale(own_farm["money"], 3000),
        _saturating_scale(opponent_farm["money"], 3000),
        *(shed.get(item, 0) / 100 for item in SHED_ITEMS),
        *(_saturating_scale(seeds.get(item, 0), 10) for item in SEED_ITEMS),
        *(_saturating_scale(carried[item], 10) for item in SHED_ITEMS),
        _saturating_scale(len(own_farm["hands"]), 4),
        _saturating_scale(len(opponent_farm["hands"]), 4),
        *(shop_counts[shop] / 8 for shop in SHOP_TYPES),
    ]

    return np.asarray(features, dtype=np.float32)
