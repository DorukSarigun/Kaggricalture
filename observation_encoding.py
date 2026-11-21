from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from global_encoding import SHED_ITEMS, encode_global_observation
from kinds import parse_tile_grid
from tile_encoding import encode_tile_grid


@dataclass(frozen=True, slots=True)
class EncodedObservation:
    own_categorical_ids: np.ndarray
    own_numeric_features: np.ndarray
    own_worker_positions: np.ndarray
    opponent_categorical_ids: np.ndarray
    opponent_numeric_features: np.ndarray
    opponent_worker_positions: np.ndarray
    global_features: np.ndarray
    own_unit_positions_yx: np.ndarray
    own_unit_features: np.ndarray


UNIT_FEATURE_NAMES = (
    "is_main_farmer",
    *(f"carried_{item}" for item in SHED_ITEMS),
)


def encode_worker_positions(farm: Mapping[str, Any]) -> np.ndarray:
    """Kanallar: ana farmer için 0/1, her karedeki hand sayısı (ölçeksiz)."""
    tile_rows = farm["tiles"]
    height = len(tile_rows)
    width = len(tile_rows[0])
    positions = np.zeros((2, height, width), dtype=np.float32)

    farmer_x, farmer_y = farm["farmer"]
    positions[0, farmer_y, farmer_x] = 1.0

    for hand_x, hand_y in farm["hands"]:
        positions[1, hand_y, hand_x] += 1.0

    return positions


def encode_own_units(
    observation: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Farmer ve hand'leri sıralı, unit-specific policy girdilerine çevirir.

    Pozisyonlar CNN gather için ``(y, x)`` sırasındadır. Unit sırası oyun
    aksiyonuyla aynıdır: index 0 ana farmer, devamı ``hands`` listesidir.
    """
    player_index = int(observation["player"])
    farm = observation["farms"][player_index]
    private = observation["private"]

    positions_xy = [farm["farmer"], *farm["hands"]]
    inventories = private["inventories"]
    if len(inventories) < len(positions_xy):
        raise ValueError(
            f"{len(positions_xy)} unit var ama yalnızca {len(inventories)} "
            "inventory kaydı var."
        )

    positions_yx = np.asarray(
        [(int(y), int(x)) for x, y in positions_xy],
        dtype=np.int64,
    )
    features = np.asarray(
        [
            [
                float(unit_index == 0),
                *(
                    max(float(inventories[unit_index].get(item, 0)), 0.0)
                    / (
                        max(float(inventories[unit_index].get(item, 0)), 0.0)
                        + 10.0
                    )
                    for item in SHED_ITEMS
                ),
            ]
            for unit_index in range(len(positions_xy))
        ],
        dtype=np.float32,
    )
    return positions_yx, features


def _encode_farm(
    farm: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current_step = observation.get(
        "step",
        observation["day"] * 24 + observation["hour"],
    )
    tile_grid = parse_tile_grid(farm["tiles"])
    categorical_ids, numeric_features = encode_tile_grid(
        tile_grid=tile_grid,
        current_day=observation["day"],
        current_step=current_step,
        market_prices=observation["market"]["prices"],
    )
    worker_positions = encode_worker_positions(farm)
    return categorical_ids, numeric_features, worker_positions


def encode_observation(
    observation: Mapping[str, Any],
) -> EncodedObservation:
    """Observation'ı own/opponent farm ve global girdilere ayırır."""
    player_index = int(observation["player"])
    opponent_index = 1 - player_index

    own = _encode_farm(observation["farms"][player_index], observation)
    opponent = _encode_farm(observation["farms"][opponent_index], observation)
    own_unit_positions_yx, own_unit_features = encode_own_units(observation)

    return EncodedObservation(
        own_categorical_ids=own[0],
        own_numeric_features=own[1],
        own_worker_positions=own[2],
        opponent_categorical_ids=opponent[0],
        opponent_numeric_features=opponent[1],
        opponent_worker_positions=opponent[2],
        global_features=encode_global_observation(observation),
        own_unit_positions_yx=own_unit_positions_yx,
        own_unit_features=own_unit_features,
    )
