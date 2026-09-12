from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum, auto
from functools import singledispatch
from typing import Any, ClassVar


class UpperStrEnum(StrEnum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name


class TileKind(UpperStrEnum):
    EMPTY = auto()
    LOCKED = auto()
    WEED = auto()
    PLANT = auto()
    COOP = auto()
    PASTURE = auto()


class CropType(UpperStrEnum):
    WHEAT = auto()
    CARROT = auto()
    TOMATO = auto()
    STRAWBERRY = auto()
    MELON = auto()


class AnimalType(UpperStrEnum):
    GOOSE = auto()
    COW = auto()
    SHEEP = auto()


ANIMAL_BY_VALUE: dict[str | None, AnimalType | None] = {
    None: None,
    **{animal.value: animal for animal in AnimalType},
}


@dataclass(frozen=True, slots=True)
class Tile:
    @classmethod
    def from_raw(cls, raw_tile: Any) -> "Tile":
        return cls()


tile_registry: dict[TileKind, type[Tile]] = {}


def register_tile_kind(cls: type[Tile]) -> type[Tile]:
    tile_registry[cls.kind] = cls
    return cls


@register_tile_kind
@dataclass(frozen=True, slots=True)
class EmptyTile(Tile):
    kind: ClassVar[TileKind] = TileKind.EMPTY


@register_tile_kind
@dataclass(frozen=True, slots=True)
class LockedTile(Tile):
    kind: ClassVar[TileKind] = TileKind.LOCKED


@register_tile_kind
@dataclass(frozen=True, slots=True)
class WeedTile(Tile):
    kind: ClassVar[TileKind] = TileKind.WEED


@register_tile_kind
@dataclass(frozen=True, slots=True)
class PlantTile(Tile):
    kind: ClassVar[TileKind] = TileKind.PLANT

    crop: CropType
    planted_day: int
    watered_today: bool
    consecutive_unwatered: int
    yield_units: int
    max_lifespan_step: int
    fertilized_until_day: int

    @classmethod
    def from_raw(cls, raw_tile: Mapping[str, Any]) -> "PlantTile":
        data = dict(raw_tile)
        data.pop("kind")
        data["crop"] = CropType(data["crop"])
        return cls(**data)


@dataclass(frozen=True, slots=True)
class AnimalStructureTile(Tile):
    animal: AnimalType | None = None
    placed_day: int | None = None
    yield_units: int = 0
    fed_today: bool = False
    consecutive_unfed: int = 0
    cared_today: bool = False
    fertilizer_available: bool = False
    pending_care_bonus: int = 0

    @classmethod
    def from_raw(cls, raw_tile: Mapping[str, Any]) -> "AnimalStructureTile":
        data = dict(raw_tile)
        data.pop("kind")
        data["animal"] = ANIMAL_BY_VALUE[data.get("animal")]
        return cls(**data)


@register_tile_kind
@dataclass(frozen=True, slots=True)
class CoopTile(AnimalStructureTile):
    kind: ClassVar[TileKind] = TileKind.COOP


@register_tile_kind
@dataclass(frozen=True, slots=True)
class PastureTile(AnimalStructureTile):
    kind: ClassVar[TileKind] = TileKind.PASTURE


@singledispatch
def parse_tile(raw_tile: Any) -> Tile:
    raise TypeError(f"Desteklenmeyen tile formatı: {raw_tile!r}")


@parse_tile.register(type(None))
def _parse_empty_tile(raw_tile: None) -> Tile:
    tile_class = tile_registry[TileKind.EMPTY]
    return tile_class.from_raw(raw_tile)


@parse_tile.register(str)
def _parse_string_tile(raw_tile: str) -> Tile:
    tile_class = tile_registry[TileKind(raw_tile)]
    return tile_class.from_raw(raw_tile)


@parse_tile.register(Mapping)
def _parse_mapping_tile(raw_tile: Mapping[str, Any]) -> Tile:
    kind = TileKind(raw_tile["kind"])
    tile_class = tile_registry[kind]
    return tile_class.from_raw(raw_tile)


def parse_tile_grid(tile_rows: list[list[Any]]) -> list[list[Tile]]:
    return [[parse_tile(raw_tile) for raw_tile in row] for row in tile_rows]
