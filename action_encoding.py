"""Kaggriculture bileşik aksiyonları için otoregresif token grameri.

Ortam tek bir ``Discrete(N)`` aksiyon beklemez. Bir farmer aksiyonu, değişken
sayıda hand aksiyonu ve en fazla 10 market emrinden oluşan yapısal bir sözlük
bekler. Bu modül o sözlüğü, dil modeli gibi sırayla üretilebilen sonlu bir
token dizisine dönüştürür.

Gramer yapısal tokenları zorunlu tutar ve observation'a göre geçersiz anlamsal
tokenları maskeler. Tamamlanan unit aksiyonları gerçek observation'a dokunmadan
bir gölge durum üzerinde uygulanır. Bu nedenle aynı turdaki sonraki hand,
önceki unit'in yaptığı değişikliği dikkate alır.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, IntEnum, StrEnum, auto
from typing import Any

import numpy as np
from kaggle_environments.envs.kaggriculture import kaggriculture as game


class ActionToken(StrEnum):
    # Yapısal tokenlar
    BOS = "<BOS>"
    FARMER = "<FARMER>"
    HAND = "<HAND>"
    MARKET = "<MARKET>"
    END_ACTION = "<END_ACTION>"
    END_ORDER = "<END_ORDER>"
    STOP = "<STOP>"
    EOS = "<EOS>"

    # Farmer / hand operasyonları
    UNIT_NORTH = "UNIT:NORTH"
    UNIT_SOUTH = "UNIT:SOUTH"
    UNIT_EAST = "UNIT:EAST"
    UNIT_WEST = "UNIT:WEST"
    UNIT_PASS = "UNIT:PASS"
    UNIT_DROP = "UNIT:DROP"
    UNIT_PICKUP = "UNIT:PICKUP"
    UNIT_PLANT = "UNIT:PLANT"
    UNIT_WATER = "UNIT:WATER"
    UNIT_HARVEST = "UNIT:HARVEST"
    UNIT_FERTILIZE = "UNIT:FERTILIZE"
    UNIT_BUILD_COOP = "UNIT:BUILD_COOP"
    UNIT_BUILD_PASTURE = "UNIT:BUILD_PASTURE"
    UNIT_DIG = "UNIT:DIG"
    UNIT_PLACE = "UNIT:PLACE"
    UNIT_FEED = "UNIT:FEED"
    UNIT_COLLECT_FERTILIZER = "UNIT:COLLECT_FERTILIZER"
    UNIT_CARE = "UNIT:CARE"

    # Market operasyonları
    MARKET_BUY_SEED = "MARKET:BUY_SEED"
    MARKET_BUY_PRODUCT = "MARKET:BUY_PRODUCT"
    MARKET_BUY_ANIMAL = "MARKET:BUY_ANIMAL"
    MARKET_SELL = "MARKET:SELL"
    MARKET_HIRE = "MARKET:HIRE"
    MARKET_BUY_LAND = "MARKET:BUY_LAND"

    # Operasyon parametreleri
    ITEM_WHEAT = "ITEM:WHEAT"
    ITEM_CARROT = "ITEM:CARROT"
    ITEM_TOMATO = "ITEM:TOMATO"
    ITEM_STRAWBERRY = "ITEM:STRAWBERRY"
    ITEM_MELON = "ITEM:MELON"
    ITEM_EGG = "ITEM:EGG"
    ITEM_MILK = "ITEM:MILK"
    ITEM_WOOL = "ITEM:WOOL"
    ITEM_FERTILIZER = "ITEM:FERTILIZER"
    ITEM_GOOSE = "ITEM:GOOSE"
    ITEM_COW = "ITEM:COW"
    ITEM_SHEEP = "ITEM:SHEEP"

    QTY_0 = "QTY:0"
    QTY_1 = "QTY:1"
    QTY_2 = "QTY:2"
    QTY_3 = "QTY:3"
    QTY_4 = "QTY:4"
    QTY_5 = "QTY:5"
    QTY_6 = "QTY:6"
    QTY_7 = "QTY:7"
    QTY_8 = "QTY:8"
    QTY_9 = "QTY:9"
    END_QUANTITY = "<END_QUANTITY>"


ACTION_TOKENS = tuple(ActionToken)
TOKEN_TO_ID = {token: index for index, token in enumerate(ACTION_TOKENS)}
ID_TO_TOKEN = {index: token for token, index in TOKEN_TO_ID.items()}
ACTION_VOCAB_SIZE = len(ACTION_TOKENS)


class ActionHeadType(IntEnum):
    """Bir sonraki stochastic tokenı hangi output head'in üreteceği."""

    STRUCTURAL = 0
    UNIT_OPERATION = 1
    MARKET_OPERATION = 2
    ITEM = 3
    QUANTITY = 4


UNIT_OP_TO_TOKEN = {
    "NORTH": ActionToken.UNIT_NORTH,
    "SOUTH": ActionToken.UNIT_SOUTH,
    "EAST": ActionToken.UNIT_EAST,
    "WEST": ActionToken.UNIT_WEST,
    "PASS": ActionToken.UNIT_PASS,
    # DROP motor tarafından destekleniyor; JSON açıklamasında atlanmış.
    "DROP": ActionToken.UNIT_DROP,
    "PICKUP": ActionToken.UNIT_PICKUP,
    "PLANT": ActionToken.UNIT_PLANT,
    "WATER": ActionToken.UNIT_WATER,
    "HARVEST": ActionToken.UNIT_HARVEST,
    "FERTILIZE": ActionToken.UNIT_FERTILIZE,
    "BUILD_COOP": ActionToken.UNIT_BUILD_COOP,
    "BUILD_PASTURE": ActionToken.UNIT_BUILD_PASTURE,
    "DIG": ActionToken.UNIT_DIG,
    "PLACE": ActionToken.UNIT_PLACE,
    "FEED": ActionToken.UNIT_FEED,
    "COLLECT_FERTILIZER": ActionToken.UNIT_COLLECT_FERTILIZER,
    "CARE": ActionToken.UNIT_CARE,
}
TOKEN_TO_UNIT_OP = {token: op for op, token in UNIT_OP_TO_TOKEN.items()}

MARKET_OP_TO_TOKEN = {
    "BUY_SEED": ActionToken.MARKET_BUY_SEED,
    "BUY_PRODUCT": ActionToken.MARKET_BUY_PRODUCT,
    "BUY_ANIMAL": ActionToken.MARKET_BUY_ANIMAL,
    "SELL": ActionToken.MARKET_SELL,
    "HIRE": ActionToken.MARKET_HIRE,
    "BUY_LAND": ActionToken.MARKET_BUY_LAND,
}
TOKEN_TO_MARKET_OP = {token: op for op, token in MARKET_OP_TO_TOKEN.items()}

ITEM_TO_TOKEN = {
    "WHEAT": ActionToken.ITEM_WHEAT,
    "CARROT": ActionToken.ITEM_CARROT,
    "TOMATO": ActionToken.ITEM_TOMATO,
    "STRAWBERRY": ActionToken.ITEM_STRAWBERRY,
    "MELON": ActionToken.ITEM_MELON,
    "EGG": ActionToken.ITEM_EGG,
    "MILK": ActionToken.ITEM_MILK,
    "WOOL": ActionToken.ITEM_WOOL,
    "FERTILIZER": ActionToken.ITEM_FERTILIZER,
    "GOOSE": ActionToken.ITEM_GOOSE,
    "COW": ActionToken.ITEM_COW,
    "SHEEP": ActionToken.ITEM_SHEEP,
}
TOKEN_TO_ITEM = {token: item for item, token in ITEM_TO_TOKEN.items()}

DIGIT_TO_TOKEN = {digit: ActionToken(f"QTY:{digit}") for digit in range(10)}
TOKEN_TO_DIGIT = {token: digit for digit, token in DIGIT_TO_TOKEN.items()}

UNIT_OPERATION_TOKENS = tuple(UNIT_OP_TO_TOKEN.values())
# STOP bir delimiter olsa da MARKET_OP_OR_STOP aşamasında gerçek bir policy
# kararıdır; bu nedenle market head'ine aittir.
MARKET_OPERATION_TOKENS = (
    *MARKET_OP_TO_TOKEN.values(),
    ActionToken.STOP,
)
ITEM_TOKENS = tuple(ITEM_TO_TOKEN.values())
QUANTITY_TOKENS = (
    *DIGIT_TO_TOKEN.values(),
    ActionToken.END_QUANTITY,
)
STRUCTURAL_TOKENS = (
    ActionToken.BOS,
    ActionToken.FARMER,
    ActionToken.HAND,
    ActionToken.MARKET,
    ActionToken.END_ACTION,
    ActionToken.END_ORDER,
    ActionToken.EOS,
)
TOKENS_BY_HEAD = {
    ActionHeadType.STRUCTURAL: STRUCTURAL_TOKENS,
    ActionHeadType.UNIT_OPERATION: UNIT_OPERATION_TOKENS,
    ActionHeadType.MARKET_OPERATION: MARKET_OPERATION_TOKENS,
    ActionHeadType.ITEM: ITEM_TOKENS,
    ActionHeadType.QUANTITY: QUANTITY_TOKENS,
}

_CLASSIFIED_ACTION_TOKENS = tuple(
    token
    for tokens in TOKENS_BY_HEAD.values()
    for token in tokens
)
if (
    set(_CLASSIFIED_ACTION_TOKENS) != set(ActionToken)
    or len(_CLASSIFIED_ACTION_TOKENS) != len(set(_CLASSIFIED_ACTION_TOKENS))
):
    raise RuntimeError("Action token head sınıflandırması eksik veya hatalı.")

HEAD_TYPE_BY_TOKEN = {
    token: head_type
    for head_type, tokens in TOKENS_BY_HEAD.items()
    for token in tokens
}
HEAD_TYPE_ID_BY_TOKEN_ID = np.asarray(
    [int(HEAD_TYPE_BY_TOKEN[token]) for token in ACTION_TOKENS],
    dtype=np.int64,
)

UNIT_NO_ARGUMENT_OPS = {
    "NORTH",
    "SOUTH",
    "EAST",
    "WEST",
    "PASS",
    "DROP",
    "WATER",
    "HARVEST",
    "FERTILIZE",
    "BUILD_COOP",
    "BUILD_PASTURE",
    "DIG",
    "FEED",
    "COLLECT_FERTILIZER",
    "CARE",
}
UNIT_ITEM_OPS = {"PLANT", "PICKUP", "PLACE"}
MARKET_NO_ARGUMENT_OPS = {"HIRE", "BUY_LAND"}
MARKET_ITEM_QUANTITY_OPS = {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"}


@dataclass(frozen=True, slots=True)
class ActionGrammarConfig:
    turns_per_day: int = 24
    shed_capacity: int = 100
    max_market_orders: int = 10
    farm_hand_cost_mult: int = 1
    # None = kısıt yok. Boş küme yalnızca kaçış tokenlarını (PASS / STOP) bırakır.
    # PASS ve STOP her zaman açık kalır; dizi tamamlanabilsin.
    allowed_unit_ops: frozenset[str] | None = None
    allowed_market_ops: frozenset[str] | None = None
    allowed_items: frozenset[str] | None = None
    # Per market operation. None = use allowed_items. Missing op → no items.
    allowed_market_items: Mapping[str, frozenset[str]] | None = None

    def __post_init__(self) -> None:
        if self.allowed_unit_ops is not None:
            unknown = self.allowed_unit_ops - set(UNIT_OP_TO_TOKEN)
            if unknown:
                names = ", ".join(sorted(unknown))
                raise ValueError(f"Bilinmeyen unit operasyonu: {names}")
        if self.allowed_market_ops is not None:
            unknown = self.allowed_market_ops - set(MARKET_OP_TO_TOKEN)
            if unknown:
                names = ", ".join(sorted(unknown))
                raise ValueError(f"Bilinmeyen market operasyonu: {names}")
        if self.allowed_items is not None:
            unknown = self.allowed_items - set(ITEM_TO_TOKEN)
            if unknown:
                names = ", ".join(sorted(unknown))
                raise ValueError(f"Bilinmeyen item: {names}")
        if self.allowed_market_items is not None:
            normalized: dict[str, frozenset[str]] = {}
            for op, items in self.allowed_market_items.items():
                if op not in MARKET_ITEM_QUANTITY_OPS:
                    raise ValueError(f"Bilinmeyen market operasyonu: {op}")
                frozen = frozenset(items)
                unknown = frozen - set(ITEM_TO_TOKEN)
                if unknown:
                    names = ", ".join(sorted(unknown))
                    raise ValueError(f"Bilinmeyen item: {names}")
                normalized[op] = frozen
            object.__setattr__(self, "allowed_market_items", normalized)


class _Phase(Enum):
    BOS = auto()
    FARMER_MARKER = auto()
    UNIT_MARKER = auto()
    UNIT_OP = auto()
    UNIT_ITEM = auto()
    UNIT_QUANTITY = auto()
    UNIT_END = auto()
    MARKET_OP_OR_STOP = auto()
    MARKET_ITEM = auto()
    MARKET_QUANTITY = auto()
    MARKET_END = auto()
    EOS = auto()
    DONE = auto()


def _as_token(token_or_id: ActionToken | int) -> ActionToken:
    if isinstance(token_or_id, ActionToken):
        return token_or_id
    if isinstance(token_or_id, (int, np.integer)):
        try:
            return ID_TO_TOKEN[int(token_or_id)]
        except KeyError:
            raise ValueError(f"Bilinmeyen action token id: {token_or_id}") from None
    raise TypeError(f"Token, ActionToken veya int olmalı; gelen: {token_or_id!r}")


class ActionGrammar:
    """Tek bir observation için durum tutan otoregresif aksiyon grameri."""

    def __init__(
        self,
        observation: Mapping[str, Any],
        config: ActionGrammarConfig | None = None,
    ) -> None:
        self.config = config or ActionGrammarConfig()
        self.player_index = int(observation["player"])
        self.farm = deepcopy(observation["farms"][self.player_index])
        self.private = deepcopy(observation["private"])
        self.market = deepcopy(observation["market"])
        self.day = int(observation["day"])
        self.board_size = len(self.farm["tiles"])

        self.num_units = 1 + len(self.farm["hands"])
        self.unit_index = 0
        self.market_order_count = 0

        self.phase = _Phase.BOS
        self.current_op: str | None = None
        self.current_item: str | None = None
        self.current_quantity: int | None = None

        self.unit_actions: list[list[Any]] = []
        self.market_actions: list[list[Any]] = []
        self.tokens: list[ActionToken] = []

    def _allows_unit_op(self, op: str) -> bool:
        allowed = self.config.allowed_unit_ops
        if allowed is None:
            return True
        return op == "PASS" or op in allowed

    def _allows_market_op(self, op: str) -> bool:
        allowed = self.config.allowed_market_ops
        if allowed is None:
            return True
        return op in allowed

    def _allows_item(self, item: str, op: str | None = None) -> bool:
        op = self.current_op if op is None else op
        per_op = self.config.allowed_market_items
        if per_op is not None and op in MARKET_ITEM_QUANTITY_OPS:
            allowed = per_op.get(op)
            return allowed is not None and item in allowed
        allowed = self.config.allowed_items
        if allowed is None:
            return True
        return item in allowed

    @property
    def is_finished(self) -> bool:
        return self.phase is _Phase.DONE

    def context_unit_index(self) -> int:
        """Bir sonraki tokenın unit context index'i; global/market için -1."""
        if self.phase is _Phase.FARMER_MARKER:
            return 0
        if self.phase is _Phase.UNIT_MARKER:
            return self.unit_index if self.unit_index < self.num_units else -1
        if self.phase in {
            _Phase.UNIT_OP,
            _Phase.UNIT_ITEM,
            _Phase.UNIT_QUANTITY,
            _Phase.UNIT_END,
        }:
            return self.unit_index
        return -1

    def next_head_type(self) -> ActionHeadType:
        """Bir sonraki token için kullanılacak output head kategorisi."""
        if self.phase is _Phase.UNIT_OP:
            return ActionHeadType.UNIT_OPERATION
        if self.phase is _Phase.MARKET_OP_OR_STOP:
            return ActionHeadType.MARKET_OPERATION
        if self.phase in {_Phase.UNIT_ITEM, _Phase.MARKET_ITEM}:
            return ActionHeadType.ITEM
        if self.phase in {_Phase.UNIT_QUANTITY, _Phase.MARKET_QUANTITY}:
            return ActionHeadType.QUANTITY
        return ActionHeadType.STRUCTURAL

    def valid_tokens(self) -> tuple[ActionToken, ...]:
        if self.phase is _Phase.BOS:
            return (ActionToken.BOS,)
        if self.phase is _Phase.FARMER_MARKER:
            return (ActionToken.FARMER,)
        if self.phase is _Phase.UNIT_MARKER:
            if self.unit_index < self.num_units:
                return (ActionToken.HAND,)
            return (ActionToken.MARKET,)
        if self.phase is _Phase.UNIT_OP:
            return tuple(
                UNIT_OP_TO_TOKEN[op] for op in self._valid_unit_operations()
            )
        if self.phase is _Phase.UNIT_ITEM:
            return tuple(ITEM_TO_TOKEN[item] for item in self._valid_unit_items())
        if self.phase is _Phase.UNIT_QUANTITY:
            return self._valid_quantity_tokens(self._current_quantity_limit())
        if self.phase is _Phase.UNIT_END:
            return (ActionToken.END_ACTION,)
        if self.phase is _Phase.MARKET_OP_OR_STOP:
            tokens: list[ActionToken] = [ActionToken.STOP]
            if self.market_order_count < self.config.max_market_orders:
                tokens.extend(
                    MARKET_OP_TO_TOKEN[op]
                    for op in self._valid_market_operations()
                )
            return tuple(tokens)
        if self.phase is _Phase.MARKET_ITEM:
            return tuple(ITEM_TO_TOKEN[item] for item in self._valid_market_items())
        if self.phase is _Phase.MARKET_QUANTITY:
            return self._valid_quantity_tokens(self._current_quantity_limit())
        if self.phase is _Phase.MARKET_END:
            return (ActionToken.END_ORDER,)
        if self.phase is _Phase.EOS:
            return (ActionToken.EOS,)
        return ()

    def valid_token_ids(self) -> tuple[int, ...]:
        return tuple(TOKEN_TO_ID[token] for token in self.valid_tokens())

    def valid_token_mask(self) -> np.ndarray:
        valid_tokens = self.valid_tokens()
        head_type = self.next_head_type()
        if any(
            HEAD_TYPE_BY_TOKEN[token] is not head_type
            for token in valid_tokens
        ):
            raise RuntimeError(
                f"{self.phase.name} gramer maskesi {head_type.name} dışında "
                "bir token açtı."
            )
        mask = np.zeros(ACTION_VOCAB_SIZE, dtype=np.bool_)
        mask[[TOKEN_TO_ID[token] for token in valid_tokens]] = True
        return mask

    def push(self, token_or_id: ActionToken | int) -> None:
        token = _as_token(token_or_id)
        valid = self.valid_tokens()
        if token not in valid:
            expected = ", ".join(item.value for item in valid)
            raise ValueError(
                f"{token.value} tokenı {self.phase.name} aşamasında geçersiz. "
                f"Beklenenlerden biri: {expected or '<hiçbiri>'}"
            )

        self.tokens.append(token)

        if self.phase is _Phase.BOS:
            self.phase = _Phase.FARMER_MARKER
            return
        if self.phase is _Phase.FARMER_MARKER:
            self.phase = _Phase.UNIT_OP
            return
        if self.phase is _Phase.UNIT_MARKER:
            self.phase = (
                _Phase.UNIT_OP if token is ActionToken.HAND
                else _Phase.MARKET_OP_OR_STOP
            )
            return
        if self.phase is _Phase.UNIT_OP:
            self.current_op = TOKEN_TO_UNIT_OP[token]
            if self.current_op in UNIT_NO_ARGUMENT_OPS:
                self.phase = _Phase.UNIT_END
            elif self.current_op in UNIT_ITEM_OPS:
                self.phase = _Phase.UNIT_ITEM
            else:  # pragma: no cover - sabit tabloların tutarlılık koruması
                raise RuntimeError(f"Eksik unit grameri: {self.current_op}")
            return
        if self.phase is _Phase.UNIT_ITEM:
            self.current_item = TOKEN_TO_ITEM[token]
            self.phase = (
                _Phase.UNIT_END
                if self.current_op == "PLANT"
                else _Phase.UNIT_QUANTITY
            )
            return
        if self.phase is _Phase.UNIT_QUANTITY:
            if self._consume_quantity_token(token):
                self.phase = _Phase.UNIT_END
            return
        if self.phase is _Phase.UNIT_END:
            action = self._finish_current_action()
            self.unit_actions.append(action)
            self._apply_shadow_unit_action(action)
            self.unit_index += 1
            self._clear_current_action()
            self.phase = _Phase.UNIT_MARKER
            return
        if self.phase is _Phase.MARKET_OP_OR_STOP:
            if token is ActionToken.STOP:
                self.phase = _Phase.EOS
                return
            self.current_op = TOKEN_TO_MARKET_OP[token]
            self.phase = (
                _Phase.MARKET_END
                if self.current_op in MARKET_NO_ARGUMENT_OPS
                else _Phase.MARKET_ITEM
            )
            return
        if self.phase is _Phase.MARKET_ITEM:
            self.current_item = TOKEN_TO_ITEM[token]
            self.phase = _Phase.MARKET_QUANTITY
            return
        if self.phase is _Phase.MARKET_QUANTITY:
            if self._consume_quantity_token(token):
                self.phase = _Phase.MARKET_END
            return
        if self.phase is _Phase.MARKET_END:
            order = self._finish_current_action()
            self.market_actions.append(order)
            self._apply_shadow_market_order(order)
            self.market_order_count += 1
            self._clear_current_action()
            self.phase = _Phase.MARKET_OP_OR_STOP
            return
        if self.phase is _Phase.EOS:
            self.phase = _Phase.DONE
            return

        raise RuntimeError(f"Beklenmeyen gramer aşaması: {self.phase}")

    def to_action(self) -> dict[str, list[Any]]:
        if not self.is_finished:
            raise ValueError("Token dizisi henüz EOS ile tamamlanmadı.")
        return {
            "farmer": list(self.unit_actions[0]),
            "hands": [list(action) for action in self.unit_actions[1:]],
            "market": [list(order) for order in self.market_actions],
        }

    def _position(self) -> tuple[int, int]:
        if self.unit_index == 0:
            position = self.farm["farmer"]
        else:
            position = self.farm["hands"][self.unit_index - 1]
        return int(position[0]), int(position[1])

    def _inventory(self) -> dict[str, int]:
        inventories = self.private["inventories"]
        while len(inventories) <= self.unit_index:
            inventories.append({})
        return inventories[self.unit_index]

    def _is_shed_access(self, x: int, y: int) -> bool:
        half = self.board_size // 2
        return (x, y) in {
            (half - 1, half - 1),
            (half, half - 1),
            (half - 1, half),
            (half, half),
        }

    def _shed_room(self) -> int:
        return max(
            0,
            self.config.shed_capacity - sum(self.private["shed"].values()),
        )

    def _valid_unit_operations(self) -> tuple[str, ...]:
        x, y = self._position()
        tile = self.farm["tiles"][y][x]
        inventory = self._inventory()
        operations: list[str] = ["PASS"]

        for op, (dx, dy) in game.FARMER_MOVES.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                operations.append(op)

        at_shed = self._is_shed_access(x, y)
        if at_shed and any(n > 0 for n in inventory.values()):
            operations.append("DROP")
        if at_shed and any(
            n > 0 and self._allows_item(item)
            for item, n in self.private["shed"].items()
        ):
            operations.append("PICKUP")
        if self._valid_place_items():
            operations.append("PLACE")

        # Aşağıdaki tile operasyonları LOCKED tile'da daima no-op'tur.
        if tile == "LOCKED":
            return self._filter_unit_operations(operations)

        if tile is None:
            if any(
                n > 0 and self._allows_item(crop)
                for crop, n in self.private["seeds"].items()
            ):
                operations.append("PLANT")
            operations.extend(("BUILD_COOP", "BUILD_PASTURE"))

        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
            if self._allows_item(str(tile["crop"])):
                if not tile.get("watered_today", False):
                    operations.append("WATER")
                crop_data = game.CROPS[tile["crop"]]
                mature = self.day - tile["planted_day"] >= crop_data["first_yield_day"]
                if tile.get("yield_units", 0) > 0 and mature:
                    operations.append("HARVEST")
                if inventory.get("FERTILIZER", 0) > 0:
                    operations.append("FERTILIZE")

        if isinstance(tile, dict) and "animal" in tile:
            if self._allows_item(str(tile["animal"])):
                if tile.get("yield_units", 0) > 0:
                    operations.append("HARVEST")
                if not tile.get("fed_today", False) and inventory.get("WHEAT", 0) > 0:
                    operations.append("FEED")
                if tile.get("fertilizer_available", False):
                    operations.append("COLLECT_FERTILIZER")
                if not tile.get("cared_today", False):
                    operations.append("CARE")

        if tile is not None and not (
            isinstance(tile, dict) and "animal" in tile
        ):
            operations.append("DIG")

        return self._filter_unit_operations(operations)

    def _filter_unit_operations(self, operations: Sequence[str]) -> tuple[str, ...]:
        legal = {op for op in operations if self._allows_unit_op(op)}
        legal.add("PASS")
        return tuple(op for op in UNIT_OP_TO_TOKEN if op in legal)

    def _valid_place_items(self) -> tuple[str, ...]:
        x, y = self._position()
        tile = self.farm["tiles"][y][x]
        inventory = self._inventory()
        items: list[str] = []

        # Uygun boş structure üzerinde hayvan yerleştirme shed alanı istemez.
        for animal, animal_data in game.ANIMALS.items():
            if (
                inventory.get(animal, 0) > 0
                and isinstance(tile, dict)
                and tile.get("kind") == animal_data["structure"]
                and "animal" not in tile
            ):
                items.append(animal)

        # PLACE aynı zamanda shed'e kontrollü miktarda eşya bırakabilir.
        if self._is_shed_access(x, y) and self._shed_room() > 0:
            items.extend(item for item, n in inventory.items() if n > 0)

        return tuple(
            item for item in ITEM_TO_TOKEN
            if item in set(items) and self._allows_item(item)
        )

    def _valid_unit_items(self) -> tuple[str, ...]:
        if self.current_op == "PLANT":
            return tuple(
                crop for crop in game.CROPS
                if self.private["seeds"].get(crop, 0) > 0
                and self._allows_item(crop)
            )
        if self.current_op == "PICKUP":
            return tuple(
                item for item in ITEM_TO_TOKEN
                if self.private["shed"].get(item, 0) > 0
                and self._allows_item(item)
            )
        if self.current_op == "PLACE":
            return self._valid_place_items()
        raise RuntimeError(f"{self.current_op} için item beklenmiyordu.")

    def _valid_market_operations(self) -> tuple[str, ...]:
        operations: list[str] = []
        if any(
            self._allows_item(item, "SELL")
            and self._market_quantity_limit("SELL", item) > 0
            for item in game.PRODUCTS
        ):
            operations.append("SELL")
        if any(
            self._allows_item(crop, "BUY_SEED")
            and self._market_quantity_limit("BUY_SEED", crop) > 0
            for crop in game.CROPS
        ):
            operations.append("BUY_SEED")
        if any(
            self._allows_item(item, "BUY_PRODUCT")
            and self._market_quantity_limit("BUY_PRODUCT", item) > 0
            for item in ("WHEAT", "FERTILIZER")
        ):
            operations.append("BUY_PRODUCT")
        if any(
            self._allows_item(animal, "BUY_ANIMAL")
            and self._market_quantity_limit("BUY_ANIMAL", animal) > 0
            for animal in game.ANIMALS
        ):
            operations.append("BUY_ANIMAL")

        hire_cost = self.config.farm_hand_cost_mult * game._fib(
            int(self.farm["hires_today"])
        )
        if self.farm["money"] >= hire_cost:
            operations.append("HIRE")

        extra_land = len(self.farm["unlocked_quadrants"]) - 1
        if (
            extra_land < len(game.LAND_PRICES)
            and self.farm["money"] >= game.LAND_PRICES[extra_land]
        ):
            operations.append("BUY_LAND")

        allowed = {
            op for op in operations
            if self._allows_market_op(op)
        }
        return tuple(op for op in MARKET_OP_TO_TOKEN if op in allowed)

    def _valid_market_items(self) -> tuple[str, ...]:
        if self.current_op == "BUY_SEED":
            candidates = game.CROPS
        elif self.current_op == "BUY_PRODUCT":
            candidates = ("WHEAT", "FERTILIZER")
        elif self.current_op == "BUY_ANIMAL":
            candidates = game.ANIMALS
        elif self.current_op == "SELL":
            candidates = game.PRODUCTS
        else:
            raise RuntimeError(f"{self.current_op} için market item beklenmiyordu.")
        return tuple(
            item for item in candidates
            if self._allows_item(item)
            and self._market_quantity_limit(self.current_op, item) > 0
        )

    def _current_quantity_limit(self) -> int:
        if self.current_op == "PICKUP":
            return int(self.private["shed"].get(self.current_item, 0))
        if self.current_op == "PLACE":
            return self._place_quantity_limit(self.current_item)
        if self.current_op in MARKET_ITEM_QUANTITY_OPS:
            return self._market_quantity_limit(self.current_op, self.current_item)
        raise RuntimeError(f"{self.current_op} için miktar beklenmiyordu.")

    def _place_quantity_limit(self, item: str | None) -> int:
        if item is None:
            return 0
        x, y = self._position()
        tile = self.farm["tiles"][y][x]
        inventory = self._inventory()

        if (
            item in game.ANIMALS
            and isinstance(tile, dict)
            and tile.get("kind") == game.ANIMALS[item]["structure"]
            and "animal" not in tile
        ):
            return 1
        if self._is_shed_access(x, y):
            return min(int(inventory.get(item, 0)), self._shed_room())
        return 0

    def _market_quantity_limit(self, op: str, item: str | None) -> int:
        if item is None:
            return 0
        money = float(self.farm["money"])
        if op == "SELL":
            return int(self.private["shed"].get(item, 0))
        if op == "BUY_SEED":
            return int(money // game.CROPS[item]["seed"])
        if op == "BUY_ANIMAL":
            return min(
                self._shed_room(),
                int(money // game.ANIMALS[item]["cost"]),
            )
        if op == "BUY_PRODUCT":
            # Fiyat her satın alınan birimden sonra değişebilir; oda en fazla
            # shed_capacity olduğu için kesin limiti küçük bir döngüyle buluyoruz.
            inventory = int(self.market["inventory"][item])
            count = 0
            for _ in range(self._shed_room()):
                price = game.market_price(
                    item,
                    inventory - 1,
                    self.market.get("params"),
                )
                if money < price:
                    break
                money -= price
                inventory -= 1
                count += 1
            return count
        return 0

    def _valid_quantity_tokens(self, limit: int) -> tuple[ActionToken, ...]:
        if limit <= 0:
            return ()
        prefix = self.current_quantity or 0
        tokens = [token for digit, token in DIGIT_TO_TOKEN.items()
                  if (prefix > 0 or digit > 0) and prefix * 10 + digit <= limit]
        if prefix > 0:
            tokens.append(ActionToken.END_QUANTITY)
        return tuple(tokens)

    def _consume_quantity_token(self, token: ActionToken) -> bool:
        if token is ActionToken.END_QUANTITY:
            return True
        self.current_quantity = (self.current_quantity or 0) * 10 + TOKEN_TO_DIGIT[token]
        return False

    def _finish_current_action(self) -> list[Any]:
        if self.current_op is None:
            raise RuntimeError("Operasyon seçilmeden aksiyon tamamlandı.")
        action: list[Any] = [self.current_op]
        if self.current_item is not None:
            action.append(self.current_item)
        if self.current_quantity is not None:
            action.append(self.current_quantity)
        return action

    def _clear_current_action(self) -> None:
        self.current_op = None
        self.current_item = None
        self.current_quantity = None

    def _apply_shadow_unit_action(self, action: list[Any]) -> None:
        # Oyun motorunun kendi fonksiyonunu kopya farm/private üzerinde çağırız.
        # Böylece maske kuralları motor semantiğinden ayrışmaz.
        game._apply_unit_action(
            self.farm,
            self.private,
            self.unit_index,
            action,
            self.board_size,
            self.day,
            self.config.turns_per_day,
            self.config.shed_capacity,
        )

    def _apply_shadow_market_order(self, order: list[Any]) -> None:
        # Rakibin aynı market sırasında vereceği emir henüz bilinemez. Bu gölge
        # hesap yalnızca kendi prefix'imizi izler; gerçek motor fiyat değişirse emri
        # karşılayabildiği birim kadar kısmi olarak uygular.
        op = order[0]
        if op == "HIRE":
            game._do_hire(
                self.farm,
                self.private,
                self.board_size,
                self.config.farm_hand_cost_mult,
            )
            return
        if op == "BUY_LAND":
            game._do_buy_land(self.farm, self.board_size)
            return

        item, quantity = order[1], int(order[2])
        for _ in range(quantity):
            if op == "SELL":
                price = game.market_price(
                    item,
                    self.market["inventory"][item],
                    self.market.get("params"),
                )
            elif op == "BUY_PRODUCT":
                price = game.market_price(
                    item,
                    self.market["inventory"][item] - 1,
                    self.market.get("params"),
                )
            elif op == "BUY_SEED":
                price = game.CROPS[item]["seed"]
            elif op == "BUY_ANIMAL":
                price = game.ANIMALS[item]["cost"]
            else:  # pragma: no cover - gramer bu duruma izin vermez
                raise RuntimeError(f"Bilinmeyen market operasyonu: {op}")

            committed = game._commit_unit(
                op,
                item,
                price,
                self.farm,
                self.private,
                self.market,
                self.config.shed_capacity,
            )
            if not committed:
                break
        game._refresh_prices(self.market)


def decode_action_tokens(
    token_ids: Sequence[int | ActionToken],
    observation: Mapping[str, Any],
    config: ActionGrammarConfig | None = None,
) -> dict[str, list[Any]]:
    """Tam bir token dizisini Kaggle environment aksiyonuna dönüştürür."""
    grammar = ActionGrammar(observation, config)
    for token in token_ids:
        grammar.push(token)
    return grammar.to_action()


def _push_quantity(
    grammar: ActionGrammar,
    quantity: int,
) -> None:
    if quantity <= 0:
        raise ValueError(f"Aksiyon miktarı pozitif olmalı; gelen: {quantity}")
    limit = grammar._current_quantity_limit()
    if quantity > limit:
        raise ValueError(f"Miktar {quantity}, kullanılabilir sınır {limit} değerini aşıyor.")
    for digit in str(quantity):
        grammar.push(DIGIT_TO_TOKEN[int(digit)])
    grammar.push(ActionToken.END_QUANTITY)


def _push_raw_unit_action(grammar: ActionGrammar, raw_action: Sequence[Any]) -> None:
    if not raw_action:
        raw_action = ["PASS"]
    op = str(raw_action[0])
    try:
        grammar.push(UNIT_OP_TO_TOKEN[op])
    except KeyError:
        raise ValueError(f"Bilinmeyen unit operasyonu: {op!r}") from None

    if op in UNIT_ITEM_OPS:
        if len(raw_action) < 2:
            raise ValueError(f"{op} bir item parametresi bekliyor.")
        item = str(raw_action[1])
        try:
            grammar.push(ITEM_TO_TOKEN[item])
        except KeyError:
            raise ValueError(f"Bilinmeyen item: {item!r}") from None

    if op in {"PICKUP", "PLACE"}:
        quantity = int(raw_action[2]) if len(raw_action) >= 3 else 1
        _push_quantity(grammar, quantity)

    grammar.push(ActionToken.END_ACTION)


def _push_raw_market_order(grammar: ActionGrammar, raw_order: Sequence[Any]) -> None:
    if not raw_order:
        raise ValueError("Boş market emri tokenlaştırılamaz.")
    op = str(raw_order[0])
    try:
        grammar.push(MARKET_OP_TO_TOKEN[op])
    except KeyError:
        raise ValueError(f"Bilinmeyen market operasyonu: {op!r}") from None

    if op in MARKET_ITEM_QUANTITY_OPS:
        if len(raw_order) < 3:
            raise ValueError(f"{op}, item ve miktar parametresi bekliyor.")
        item = str(raw_order[1])
        try:
            grammar.push(ITEM_TO_TOKEN[item])
        except KeyError:
            raise ValueError(f"Bilinmeyen item: {item!r}") from None
        _push_quantity(grammar, int(raw_order[2]))

    grammar.push(ActionToken.END_ORDER)


def encode_action_tokens(
    action: Mapping[str, Any],
    observation: Mapping[str, Any],
    config: ActionGrammarConfig | None = None,
) -> np.ndarray:
    """Geçerli bir Kaggle aksiyonunu canonical token id dizisine çevirir.

    Eksik hand aksiyonları ortam davranışıyla aynı olacak şekilde PASS olarak
    tamamlanır. Miktarlar başında sıfır olmayan ondalık rakamlar ve
    END_QUANTITY ile temsil edilir; kaynak sınırındaki her tam sayı geçerlidir.
    """
    grammar = ActionGrammar(observation, config)
    grammar.push(ActionToken.BOS)
    grammar.push(ActionToken.FARMER)
    _push_raw_unit_action(grammar, action.get("farmer", ["PASS"]))

    raw_hands = action.get("hands", [])
    if not isinstance(raw_hands, Sequence) or isinstance(raw_hands, (str, bytes)):
        raise TypeError("action['hands'] bir aksiyon listesi olmalı.")
    expected_hands = grammar.num_units - 1
    if len(raw_hands) > expected_hands:
        raise ValueError(
            f"Observation'da {expected_hands} hand var ama {len(raw_hands)} "
            "hand aksiyonu verildi."
        )
    for hand_index in range(expected_hands):
        grammar.push(ActionToken.HAND)
        hand_action = raw_hands[hand_index] if hand_index < len(raw_hands) else ["PASS"]
        _push_raw_unit_action(grammar, hand_action)

    grammar.push(ActionToken.MARKET)
    raw_market = action.get("market", [])
    if not isinstance(raw_market, Sequence) or isinstance(raw_market, (str, bytes)):
        raise TypeError("action['market'] bir emir listesi olmalı.")
    if len(raw_market) > grammar.config.max_market_orders:
        raise ValueError(
            f"En fazla {grammar.config.max_market_orders} market emri tokenlaştırılabilir."
        )
    for order in raw_market:
        _push_raw_market_order(grammar, order)
    grammar.push(ActionToken.STOP)
    grammar.push(ActionToken.EOS)
    return np.asarray([TOKEN_TO_ID[token] for token in grammar.tokens], dtype=np.int64)


def teacher_forcing_masks(
    token_ids: Sequence[int | ActionToken],
    observation: Mapping[str, Any],
    config: ActionGrammarConfig | None = None,
) -> np.ndarray:
    """Her hedef token için, token seçilmeden önceki geçerlilik maskesi.

    Sonuç ``(sequence_length, ACTION_VOCAB_SIZE)`` boyutundadır. PPO veya
    teacher forcing sırasında geçersiz logitlere ``-inf`` vermek için kullanılır.
    """
    grammar = ActionGrammar(observation, config)
    masks: list[np.ndarray] = []
    for token in token_ids:
        masks.append(grammar.valid_token_mask())
        grammar.push(token)
    if not grammar.is_finished:
        raise ValueError("Teacher-forcing token dizisi EOS ile tamamlanmalı.")
    return np.stack(masks)


def decoder_inputs_for_tokens(
    token_ids: Sequence[int | ActionToken],
    observation: Mapping[str, Any],
    config: ActionGrammarConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Teacher forcing için shifted decoder girdilerini hazırlar.

    Dönenler sırasıyla ``input_ids``, ``target_ids``, target geçerlilik
    maskeleri ve her target için aktif unit index'idir. Yapısal/global tokenlarda
    context index ``-1`` olur.
    """
    if len(token_ids) < 2:
        raise ValueError("Decoder dizisi en az BOS ve bir hedef token içermeli.")

    normalized = np.asarray(
        [TOKEN_TO_ID[_as_token(token)] for token in token_ids],
        dtype=np.int64,
    )
    grammar = ActionGrammar(observation, config)
    masks: list[np.ndarray] = []
    context_indices: list[int] = []

    # token[i] prefix'e eklendikten sonra decoder token[i+1]'i tahmin eder.
    for input_token, target_token in zip(normalized[:-1], normalized[1:]):
        grammar.push(int(input_token))
        masks.append(grammar.valid_token_mask())
        context_indices.append(grammar.context_unit_index())
        if not grammar.valid_token_mask()[int(target_token)]:
            raise ValueError(
                f"{ID_TO_TOKEN[int(target_token)].value} hedefi prefix için geçersiz."
            )

    # Son target da replay edilerek dizinin gerçekten tamamlandığı doğrulanır.
    grammar.push(int(normalized[-1]))
    if not grammar.is_finished:
        raise ValueError("Decoder token dizisi EOS ile tamamlanmalı.")

    return (
        normalized[:-1],
        normalized[1:],
        np.stack(masks),
        np.asarray(context_indices, dtype=np.int64),
    )


def decoder_head_types_for_tokens(
    token_ids: Sequence[int | ActionToken],
    observation: Mapping[str, Any],
    config: ActionGrammarConfig | None = None,
) -> np.ndarray:
    """Her shifted target tokenın output head türünü döndürür."""
    if len(token_ids) < 2:
        raise ValueError("Decoder dizisi en az BOS ve bir hedef token içermeli.")
    normalized = [TOKEN_TO_ID[_as_token(token)] for token in token_ids]
    grammar = ActionGrammar(observation, config)
    head_types: list[int] = []
    for input_token, target_token in zip(normalized[:-1], normalized[1:]):
        grammar.push(input_token)
        head_types.append(int(grammar.next_head_type()))
        if target_token not in grammar.valid_token_ids():
            raise ValueError(
                f"{ID_TO_TOKEN[target_token].value} hedefi prefix için geçersiz."
            )
    grammar.push(normalized[-1])
    if not grammar.is_finished:
        raise ValueError("Decoder token dizisi EOS ile tamamlanmalı.")
    return np.asarray(head_types, dtype=np.int64)
