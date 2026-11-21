"""Token üretmeyen, tam işlem seçen koşullu head politikası.

Eski ActionGrammar yalnız yasal işlem kuralları ve motorun shadow adapter'ı
olarak kullanılır; push/token/decoder çağrısı yoktur. Shadow tahmindir, gerçek
rakip emri veya private observation policy'ye verilmez.
"""
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from action_encoding import (
    ActionGrammar, ActionGrammarConfig, UNIT_OP_TO_TOKEN, MARKET_OP_TO_TOKEN,
    ITEM_TO_TOKEN, UNIT_ITEM_OPS, MARKET_ITEM_QUANTITY_OPS,
)
from model import StateEncoder
from observation_encoding import EncodedObservation, UNIT_FEATURE_NAMES, encode_observation

UNIT_OPS = tuple(UNIT_OP_TO_TOKEN)
MARKET_OPS = (*MARKET_OP_TO_TOKEN, "STOP")
ITEMS = tuple(ITEM_TO_TOKEN)
ALL_OPS = (*UNIT_OPS, *MARKET_OPS)


class OperationCursor:
    """Bir turda farmer -> mevcut hands -> market; yeniden encode edilebilir."""
    def __init__(self, observation: dict, config: ActionGrammarConfig | None = None):
        self.original = deepcopy(observation)
        self.rules = ActionGrammar(observation, config)
        self.section = "unit"
        self.finished = False
        self.actions = {"farmer": ["PASS"], "hands": [], "market": []}
        self.prefix: list[list[Any]] = []

    def observation(self) -> dict:
        obs = deepcopy(self.original)
        obs["farms"][obs["player"]] = deepcopy(self.rules.farm)
        obs["private"] = deepcopy(self.rules.private)
        obs["market"] = deepcopy(self.rules.market)
        return obs

    def operations(self) -> tuple[str, ...]:
        if self.finished:
            return ()
        if self.section == "unit":
            return self.rules._valid_unit_operations()
        return (*self.rules._valid_market_operations(), "STOP")

    def items(self, operation: str) -> tuple[str, ...]:
        self.rules.current_op = operation
        needs_item = operation in (UNIT_ITEM_OPS if self.section == "unit" else MARKET_ITEM_QUANTITY_OPS)
        if not needs_item:
            return ()
        return (self.rules._valid_unit_items() if self.section == "unit"
                else self.rules._valid_market_items())

    def quantity_limit(self, operation: str, item: str | None) -> int:
        self.rules.current_op = operation
        self.rules.current_item = item
        if operation in {"PICKUP", "PLACE"} or operation in MARKET_ITEM_QUANTITY_OPS:
            return self.rules._current_quantity_limit()
        return 0

    def apply(self, action: list[Any]) -> None:
        op = action[0]
        if op not in self.operations():
            raise ValueError(f"Geçersiz işlem: {action}")
        valid_items = self.items(op)
        item = action[1] if valid_items and len(action) > 1 else None
        if valid_items and item not in valid_items:
            raise ValueError(f"Geçersiz ürün: {action}")
        limit = self.quantity_limit(op, item)
        expected = 1 + bool(valid_items) + bool(limit)
        if len(action) != expected:
            raise ValueError(f"İşlem parametre sayısı yanlış: {action}")
        if limit and (not isinstance(action[2], int) or not 1 <= action[2] <= limit):
            raise ValueError(f"Miktar 1..{limit} aralığında olmalı: {action}")
        self.prefix.append(list(action))
        if self.section == "unit":
            index = self.rules.unit_index
            self.rules._apply_shadow_unit_action(action)
            if index == 0:
                self.actions["farmer"] = list(action)
            else:
                self.actions["hands"].append(list(action))
            self.rules.unit_index += 1
            if self.rules.unit_index == self.rules.num_units:
                self.section = "market"
        elif op == "STOP":
            self.finished = True
        else:
            self.rules._apply_shadow_market_order(action)
            self.actions["market"].append(list(action))
            self.rules.market_order_count += 1
            if self.rules.market_order_count >= self.rules.config.max_market_orders:
                self.finished = True
        self.rules._clear_current_action()


@dataclass
class HeadInput:
    original: EncodedObservation
    current: EncodedObservation
    section: str
    unit_index: int
    order_index: int
    hires_today: int
    # Tamamlanmış işlemler sıralı prefix olarak saklanır; rakip aksiyonu yoktur.
    prefix: tuple[tuple[Any, ...], ...]


def head_input(cursor: OperationCursor, original: EncodedObservation,
               boundary: bool = False) -> HeadInput:
    return HeadInput(original, encode_observation(cursor.observation()),
                     "boundary" if boundary else cursor.section,
                     cursor.rules.unit_index, cursor.rules.market_order_count,
                     int(cursor.rules.farm["hires_today"]),
                     tuple(tuple(a) for a in cursor.prefix))


class StructuredActorCritic(nn.Module):
    """Ayrı operation/item head'leri ve tüm 1..limit miktarlarına aday skorları.

GRU token üretmez: yalnız tamamlanmış işlemlerin sıralı geçmişini özetler.
Bu, aynı shadow sonucu veren farklı emir sıralarını critic'e ayırt ettirir.
"""
    def __init__(self):
        super().__init__()
        self.encoder = StateEncoder()
        self.prefix_op = nn.Embedding(len(ALL_OPS), 24)
        self.prefix_item = nn.Embedding(len(ITEMS) + 1, 8)
        self.prefix_encoder = nn.GRU(33, 32, batch_first=True)
        self.context = nn.Sequential(
            nn.Linear(256 * 2 + 192 + len(UNIT_FEATURE_NAMES) + 6 + 32, 256),
            nn.LayerNorm(256), nn.Tanh(),
        )
        self.unit_head = nn.Linear(256, len(UNIT_OPS))
        self.market_head = nn.Linear(256, len(MARKET_OPS))
        self.operation_embedding = nn.Embedding(len(ALL_OPS), 32)
        self.item_head = nn.Sequential(nn.Linear(288, 128), nn.Tanh(), nn.Linear(128, len(ITEMS)))
        self.item_embedding = nn.Embedding(len(ITEMS), 16)
        self.quantity_head = nn.Sequential(nn.Linear(304, 128), nn.Tanh(), nn.Linear(128, 4))
        self.value_head = nn.Linear(256, 1)
        self.realized_value_head = nn.Linear(256, 1)

    def _encode(self, rows: list[EncodedObservation]):
        device = next(self.parameters()).device
        names = ("own_categorical_ids", "own_numeric_features", "own_worker_positions",
                 "opponent_categorical_ids", "opponent_numeric_features",
                 "opponent_worker_positions", "global_features")
        tensors = [torch.as_tensor(np.stack([getattr(r, n) for r in rows]), device=device,
                                   dtype=torch.long if "categorical" in n else torch.float32)
                   for n in names]
        return self.encoder(*tensors)

    def forward(self, inputs: list[HeadInput]):
        current = self._encode([r.current for r in inputs])
        original = self._encode([r.original for r in inputs])
        device = current.state_vector.device
        unit_vectors, spatial_vectors, phases, histories = [], [], [], []
        for i, row in enumerate(inputs):
            if row.section == "unit":
                y, x = row.current.own_unit_positions_yx[row.unit_index]
                spatial_vectors.append(current.policy_features[i, :, int(y), int(x)])
                unit_vectors.append(row.current.own_unit_features[row.unit_index])
            else:
                spatial_vectors.append(torch.zeros(192, device=device))
                unit_vectors.append(np.zeros(len(UNIT_FEATURE_NAMES), dtype=np.float32))
            phases.append([float(row.section == s) for s in ("unit", "market", "boundary")]
                          + [row.unit_index / 10, row.order_index / 10, row.hires_today / 10])
            if row.prefix:
                op_ids = torch.tensor([ALL_OPS.index(a[0]) for a in row.prefix], device=device)
                item_ids = torch.tensor([ITEMS.index(a[1]) + 1 if len(a) > 1 else 0
                                         for a in row.prefix], device=device)
                quantities = torch.tensor([np.log1p(a[2]) if len(a) > 2 else 0
                                            for a in row.prefix], dtype=torch.float32, device=device)
                sequence = torch.cat([self.prefix_op(op_ids), self.prefix_item(item_ids),
                                      quantities[:, None]], dim=-1)[None]
                _, hidden = self.prefix_encoder(sequence)
                histories.append(hidden[0, 0])
            else:
                histories.append(torch.zeros(32, device=device))
        context = self.context(torch.cat([
            original.state_vector, current.state_vector, torch.stack(spatial_vectors),
            torch.as_tensor(np.stack(unit_vectors), device=device),
            torch.tensor(phases, dtype=torch.float32, device=device), torch.stack(histories)
        ], dim=-1))
        return context, self.value_head(context).squeeze(-1), self.realized_value_head(context).squeeze(-1)

    @staticmethod
    def distribution(logits: torch.Tensor, names: tuple[str, ...], allowed: tuple[str, ...]):
        mask = torch.tensor([n in allowed for n in names], device=logits.device)
        if not mask.any():
            raise ValueError("Head için geçerli seçenek yok.")
        return Categorical(logits=logits.masked_fill(~mask, -torch.inf))

    def operation_distribution(self, context, section, allowed):
        return self.distribution(self.unit_head(context) if section == "unit" else self.market_head(context),
                                 UNIT_OPS if section == "unit" else MARKET_OPS, allowed)

    def item_distribution(self, context, operation, allowed):
        op_id = torch.tensor(ALL_OPS.index(operation), device=context.device)
        conditioned = torch.cat([context, self.operation_embedding(op_id)])
        return self.distribution(self.item_head(conditioned), ITEMS, allowed)

    def quantity_distribution(self, context, operation, item, limit):
        if limit < 1:
            raise ValueError("Miktar sınırı pozitif olmalı.")
        op = self.operation_embedding(torch.tensor(ALL_OPS.index(operation), device=context.device))
        obj = self.item_embedding(torch.tensor(ITEMS.index(item), device=context.device))
        coefficients = self.quantity_head(torch.cat([context, op, obj]))
        # Her tam sayı ayrı adaydır; ağın parametre sayısı limit ile büyümez.
        q = torch.arange(1, limit + 1, device=context.device, dtype=context.dtype)
        fraction = q / limit
        features = torch.stack([fraction, fraction.square(), torch.sqrt(fraction),
                                torch.log1p(q) / np.log1p(limit)], dim=-1)
        return Categorical(logits=features @ coefficients)


@dataclass
class OperationSample:
    inputs: HeadInput
    action: list[Any]
    allowed_ops: tuple[str, ...]
    allowed_items: tuple[str, ...]
    quantity_limit: int
    log_prob: float
    value: float
    realized_value: float
    stochastic: bool
    throughput_value: float = 0.0


def action_statistics(model, context, sample):
    """Aynı tam işlem için koşullu log olasılıkların toplamını yeniden hesaplar."""
    names = UNIT_OPS if sample.inputs.section == "unit" else MARKET_OPS
    op = sample.action[0]
    d = model.operation_distribution(context, sample.inputs.section, sample.allowed_ops)
    logp = d.log_prob(torch.tensor(names.index(op), device=context.device))
    entropies = [d.entropy()] if len(sample.allowed_ops) > 1 else []
    if sample.allowed_items:
        d = model.item_distribution(context, op, sample.allowed_items)
        logp = logp + d.log_prob(torch.tensor(ITEMS.index(sample.action[1]), device=context.device))
        if len(sample.allowed_items) > 1:
            entropies.append(d.entropy())
    if sample.quantity_limit:
        d = model.quantity_distribution(context, op, sample.action[1], sample.quantity_limit)
        logp = logp + d.log_prob(torch.tensor(sample.action[2] - 1, device=context.device))
        if sample.quantity_limit > 1:
            entropies.append(d.entropy())
    return logp, torch.stack(entropies).mean() if entropies else context.sum() * 0


@torch.no_grad()
def sample_operation(model, cursor, original, deterministic=False, *, inputs=None):
    inputs = head_input(cursor, original) if inputs is None else inputs
    packed = model([inputs])
    contexts, values, realized = packed[0], packed[1], packed[2]
    context = contexts[0]
    def choose(distribution):
        return int(distribution.logits.argmax() if deterministic else distribution.sample())
    allowed = cursor.operations()
    names = UNIT_OPS if cursor.section == "unit" else MARKET_OPS
    op = names[choose(model.operation_distribution(context, cursor.section, allowed))]
    action = [op]
    items = cursor.items(op)
    if items:
        action.append(ITEMS[choose(model.item_distribution(context, op, items))])
    limit = cursor.quantity_limit(op, action[1] if items else None)
    if limit:
        action.append(choose(model.quantity_distribution(context, op, action[1], limit)) + 1)
    sample = OperationSample(inputs, action, allowed, items, limit, 0.0,
                             float(values[0]), float(realized[0]),
                             len(allowed) > 1 or len(items) > 1 or limit > 1,
                             float(packed[3][0]) if len(packed) > 3 else 0.0)
    sample.log_prob = float(action_statistics(model, context, sample)[0])
    return sample
