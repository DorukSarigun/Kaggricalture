"""İşçi CNN girdisi: yalnız kendi tarlası, rakip farm yok."""
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from kinds import parse_tile_grid
from observation_encoding import encode_own_units, encode_worker_positions
from tile_encoding import encode_tile_grid


@dataclass(frozen=True, slots=True)
class OwnFarmObservation:
    categorical_ids: np.ndarray
    numeric_features: np.ndarray
    worker_positions: np.ndarray
    unit_position_yx: np.ndarray
    unit_features: np.ndarray
    time_features: np.ndarray


def encode_own_farm(observation: Mapping[str, Any], unit_index: int) -> OwnFarmObservation:
    player = int(observation["player"])
    farm = observation["farms"][player]
    step = observation.get("step", observation["day"] * 24 + observation["hour"])
    tiles = parse_tile_grid(farm["tiles"])
    categorical_ids, numeric_features = encode_tile_grid(
        tile_grid=tiles,
        current_day=observation["day"],
        current_step=step,
        market_prices=observation["market"]["prices"],
    )
    positions_yx, unit_features = encode_own_units(observation)
    if not 0 <= unit_index < len(positions_yx):
        raise IndexError(f"unit_index={unit_index} bu tarlada yok")
    hour = int(observation.get("hour", step % 24))
    day = int(observation["day"])
    angle = 2 * np.pi * hour / 24
    time_features = np.array(
        [day / 30.0, np.sin(angle), np.cos(angle), unit_index / 10.0],
        dtype=np.float32,
    )
    return OwnFarmObservation(
        categorical_ids=categorical_ids,
        numeric_features=numeric_features,
        worker_positions=encode_worker_positions(farm),
        unit_position_yx=positions_yx[unit_index].astype(np.int64),
        unit_features=unit_features[unit_index],
        time_features=time_features,
    )
