import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from kinds import AnimalType, CropType, Tile, TileKind


SEASON_DAYS = 30
GAME_STEPS = 720
MAX_NEGLECT_DAYS = 2
FERTILIZER_DURATION_DAYS = 3
MAX_PENDING_CARE_BONUS = 3

DYNAMIC_FEATURE_NAMES = (
    "age",
    "basic_need_done_today",
    "neglect_streak",
    "yield_fill",
    "market_price_vs_base",
    "lifespan_known",
    "lifespan_remaining",
    "plant_fertilizer_remaining",
    "animal_cared_today",
    "animal_fertilizer_available",
    "animal_pending_care_bonus",
    "animal_occupied",
)

STATIC_FEATURE_NAMES = (
    "type_is_ongoing",
    "type_is_indefinite",
    "type_purchase_cost",
    "type_base_market_price",
    "type_time_to_first_yield",
    "type_has_time_to_max_yield",
    "type_time_to_max_yield",
    "type_subsequent_interval",
    "type_subsequent_count",
    "type_max_units",
    "type_max_unfertilized_units",
    "type_action_cost",
    "type_build_action_cost",
    "type_yield_per_tile_per_day",
)

NUMERIC_FEATURE_NAMES = DYNAMIC_FEATURE_NAMES + STATIC_FEATURE_NAMES

OBJECT_TYPES_PATH = Path(__file__).parent / "data" / "object_types.json"
with OBJECT_TYPES_PATH.open(encoding="utf-8") as file:
    OBJECT_TYPE_ITEMS = json.load(file)["items"]


def _static_features(product_type: CropType | AnimalType) -> np.ndarray:
    item = OBJECT_TYPE_ITEMS[product_type.value]
    time_to_max = item["time_to_max_yield_days"]
    max_units = item["max_yield"] or item["max_held"]

    return np.array(
        [
            float(item["yield_type"] == "ongoing"),
            float(item["indefinite"]),
            item["seed_cost"] / 500,
            item["base_market_price"] / 250,
            item["time_to_first_yield_days"] / 10,
            float(time_to_max is not None),
            (time_to_max or 0) / 16,
            (item["subsequent_interval_days"] or 0) / 3,
            (item["subsequent_count"] or 0) / 4,
            max_units / 6,
            (item["max_yield_unfertilized"] or 0) / 6,
            item["action_cost"],
            item["build_action_cost"],
            item["yield_per_tile_per_day"],
        ],
        dtype=np.float32,
    )


STATIC_FEATURES_BY_TYPE = {
    product_type: _static_features(product_type)
    for product_type in (*CropType, *AnimalType)
}
EMPTY_STATIC_FEATURES = np.zeros(len(STATIC_FEATURE_NAMES), dtype=np.float32)

MARKET_PRODUCT_BY_TYPE = {
    **{crop: crop.value for crop in CropType},
    AnimalType.GOOSE: "EGG",
    AnimalType.COW: "MILK",
    AnimalType.SHEEP: "WOOL",
}

YIELD_CAPS = {
    CropType.WHEAT: 6,
    CropType.CARROT: 4,
    CropType.TOMATO: 4,
    CropType.STRAWBERRY: 4,
    CropType.MELON: 6,
    AnimalType.GOOSE: 4,
    AnimalType.COW: 6,
    AnimalType.SHEEP: 6,
}


TILE_CATEGORIES = {
    kind: index
    for index, kind in enumerate(TileKind)
}

CROP_CATEGORIES = {
    None: 0,
    **{
        crop: index
        for index, crop in enumerate(CropType, start=1)
    },
}

ANIMAL_CATEGORIES = {
    None: 0,
    **{
        animal: index
        for index, animal in enumerate(AnimalType, start=1)
    },
}


def tile_to_ids(tile: Tile) -> np.ndarray:
    """Bir Tile nesnesini [kind_id, crop_id, animal_id] dizisine çevirir."""
    return np.array(
        [
            TILE_CATEGORIES[tile.kind],
            CROP_CATEGORIES[getattr(tile, "crop", None)],
            ANIMAL_CATEGORIES[getattr(tile, "animal", None)],
        ],
        dtype=np.int64,
    )


def tile_grid_to_ids(tile_grid: list[list[Tile]]) -> np.ndarray:
    """Tile gridini embedding girdileri olan (3, height, width) ID'lerine çevirir."""
    encoded = np.array(
        [
            [tile_to_ids(tile) for tile in row]
            for row in tile_grid
        ],
        dtype=np.int64,
    )

    return encoded.transpose(2, 0, 1)


def tile_to_numeric_features(
    tile: Tile,
    current_day: int,
    current_step: int,
    market_prices: Mapping[str, int | float],
) -> np.ndarray:
    """Bir tile'in dinamik durumunu 0-1 aralığındaki sayısal özelliklere çevirir."""
    planted_day = getattr(tile, "planted_day", None)
    placed_day = getattr(tile, "placed_day", None)
    origin_day = planted_day if planted_day is not None else placed_day
    age = 0.0 if origin_day is None else (current_day - origin_day) / SEASON_DAYS

    basic_need_done = max(
        float(getattr(tile, "watered_today", False)),
        float(getattr(tile, "fed_today", False)),
    )
    neglect_streak = max(
        getattr(tile, "consecutive_unwatered", 0),
        getattr(tile, "consecutive_unfed", 0),
    ) / MAX_NEGLECT_DAYS

    crop = getattr(tile, "crop", None)
    animal = getattr(tile, "animal", None)
    product_type = crop if crop is not None else animal
    yield_cap = YIELD_CAPS.get(product_type, 1)
    yield_fill = getattr(tile, "yield_units", 0) / yield_cap

    market_product = MARKET_PRODUCT_BY_TYPE.get(product_type)
    if market_product is None:
        market_price_vs_base = 0.0
    else:
        market_price = market_prices[market_product]
        base_price = OBJECT_TYPE_ITEMS[product_type.value]["base_market_price"]
        market_price_vs_base = market_price / (market_price + base_price)

    max_lifespan_step = getattr(tile, "max_lifespan_step", -1)
    lifespan_known = float(max_lifespan_step >= 0)
    lifespan_remaining = (
        max(max_lifespan_step - current_step, 0) / GAME_STEPS
        if max_lifespan_step >= 0
        else 0.0
    )

    fertilized_until_day = getattr(tile, "fertilized_until_day", -1)
    plant_fertilizer_remaining = (
        max(fertilized_until_day - current_day + 1, 0)
        / FERTILIZER_DURATION_DAYS
    )

    dynamic_features = np.array(
        [
            age,
            basic_need_done,
            neglect_streak,
            yield_fill,
            market_price_vs_base,
            lifespan_known,
            lifespan_remaining,
            plant_fertilizer_remaining,
            float(getattr(tile, "cared_today", False)),
            float(getattr(tile, "fertilizer_available", False)),
            getattr(tile, "pending_care_bonus", 0) / MAX_PENDING_CARE_BONUS,
            float(animal is not None),
        ],
        dtype=np.float32,
    )
    static_features = STATIC_FEATURES_BY_TYPE.get(
        product_type,
        EMPTY_STATIC_FEATURES,
    )

    return np.clip(
        np.concatenate(
            [
                dynamic_features,
                static_features,
            ],
        ),
        0.0,
        1.0,
    )


def tile_grid_to_numeric_features(
    tile_grid: list[list[Tile]],
    current_day: int,
    current_step: int,
    market_prices: Mapping[str, int | float],
) -> np.ndarray:
    """Tile gridini (numeric_features, height, width) biçimine çevirir."""
    encoded = np.array(
        [
            [
                tile_to_numeric_features(
                    tile,
                    current_day,
                    current_step,
                    market_prices,
                )
                for tile in row
            ]
            for row in tile_grid
        ],
        dtype=np.float32,
    )

    return encoded.transpose(2, 0, 1)


def encode_tile_grid(
    tile_grid: list[list[Tile]],
    current_day: int,
    current_step: int,
    market_prices: Mapping[str, int | float],
) -> tuple[np.ndarray, np.ndarray]:
    """Embedding ID'lerini ve CNN'e eklenecek sayısal kanalları birlikte üretir."""
    categorical_ids = tile_grid_to_ids(tile_grid)
    numeric_features = tile_grid_to_numeric_features(
        tile_grid,
        current_day,
        current_step,
        market_prices,
    )
    return categorical_ids, numeric_features
