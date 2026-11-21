"""Kendi tarlasına CNN; market head yok."""
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from action_encoding import ITEM_TO_TOKEN, UNIT_ITEM_OPS, UNIT_OP_TO_TOKEN
from model import FarmEncoder, FarmPooling
from observation_encoding import UNIT_FEATURE_NAMES
from worker_grow.encoding import OwnFarmObservation, encode_own_farm

UNIT_OPS = tuple(UNIT_OP_TO_TOKEN)
ITEMS = tuple(ITEM_TO_TOKEN)


@dataclass
class WorkerSample:
    observation: OwnFarmObservation
    action: list[Any]
    allowed_ops: tuple[str, ...]
    allowed_items: tuple[str, ...]
    quantity_limit: int
    log_prob: float
    value: float
    stochastic: bool


class WorkerGrowActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.farm_encoder = FarmEncoder()
        self.farm_pooling = FarmPooling(
            input_channels=self.farm_encoder.output_channels,
            output_dim=64,
        )
        spatial = self.farm_encoder.output_channels
        self.context = nn.Sequential(
            nn.Linear(64 + spatial + len(UNIT_FEATURE_NAMES) + 4, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
        )
        self.unit_head = nn.Linear(256, len(UNIT_OPS))
        self.operation_embedding = nn.Embedding(len(UNIT_OPS), 32)
        self.item_head = nn.Sequential(nn.Linear(288, 128), nn.Tanh(), nn.Linear(128, len(ITEMS)))
        self.item_embedding = nn.Embedding(len(ITEMS), 16)
        self.quantity_head = nn.Sequential(nn.Linear(304, 128), nn.Tanh(), nn.Linear(128, 4))
        self.value_head = nn.Linear(256, 1)

    def encode(self, rows: list[OwnFarmObservation]):
        device = next(self.parameters()).device
        ids = torch.as_tensor(np.stack([r.categorical_ids for r in rows]), device=device, dtype=torch.long)
        numeric = torch.as_tensor(np.stack([r.numeric_features for r in rows]), device=device)
        workers = torch.as_tensor(np.stack([r.worker_positions for r in rows]), device=device)
        farm_map = self.farm_encoder(ids, numeric, workers)
        summary = self.farm_pooling(farm_map)
        spatial = []
        for i, row in enumerate(rows):
            y, x = int(row.unit_position_yx[0]), int(row.unit_position_yx[1])
            spatial.append(farm_map[i, :, y, x])
        extras = torch.as_tensor(
            np.stack([np.concatenate([r.unit_features, r.time_features]) for r in rows]),
            device=device,
        )
        return self.context(torch.cat([summary, torch.stack(spatial), extras], dim=-1))

    def forward(self, rows: list[OwnFarmObservation]):
        context = self.encode(rows)
        return context, self.value_head(context).squeeze(-1)

    @staticmethod
    def _masked(logits: torch.Tensor, names: tuple[str, ...], allowed: tuple[str, ...]):
        mask = torch.tensor([name in allowed for name in names], device=logits.device)
        if not mask.any():
            raise ValueError("İşçi için yasal seçenek yok")
        return Categorical(logits=logits.masked_fill(~mask, -torch.inf))

    def operation_distribution(self, context, allowed):
        return self._masked(self.unit_head(context), UNIT_OPS, allowed)

    def item_distribution(self, context, operation, allowed):
        op = self.operation_embedding(torch.tensor(UNIT_OPS.index(operation), device=context.device))
        return self._masked(self.item_head(torch.cat([context, op])), ITEMS, allowed)

    def quantity_distribution(self, context, operation, item, limit):
        if limit < 1:
            raise ValueError("Miktar sınırı pozitif olmalı")
        op = self.operation_embedding(torch.tensor(UNIT_OPS.index(operation), device=context.device))
        obj = self.item_embedding(torch.tensor(ITEMS.index(item), device=context.device))
        coefficients = self.quantity_head(torch.cat([context, op, obj]))
        q = torch.arange(1, limit + 1, device=context.device, dtype=context.dtype)
        fraction = q / limit
        features = torch.stack(
            [fraction, fraction.square(), torch.sqrt(fraction), torch.log1p(q) / np.log1p(limit)],
            dim=-1,
        )
        return Categorical(logits=features @ coefficients)


def worker_statistics(model: WorkerGrowActor, context: torch.Tensor, sample: WorkerSample):
    op = sample.action[0]
    dist = model.operation_distribution(context, sample.allowed_ops)
    logp = dist.log_prob(torch.tensor(UNIT_OPS.index(op), device=context.device))
    entropies = [dist.entropy()] if len(sample.allowed_ops) > 1 else []
    if sample.allowed_items:
        dist = model.item_distribution(context, op, sample.allowed_items)
        logp = logp + dist.log_prob(torch.tensor(ITEMS.index(sample.action[1]), device=context.device))
        if len(sample.allowed_items) > 1:
            entropies.append(dist.entropy())
    if sample.quantity_limit:
        dist = model.quantity_distribution(context, op, sample.action[1], sample.quantity_limit)
        logp = logp + dist.log_prob(torch.tensor(sample.action[2] - 1, device=context.device))
        if sample.quantity_limit > 1:
            entropies.append(dist.entropy())
    return logp, torch.stack(entropies).mean() if entropies else context.sum() * 0


@torch.no_grad()
def sample_worker(model: WorkerGrowActor, observation: dict, unit_index: int, grammar, deterministic=False):
    """grammar, o unit_index'e ayarlanmış ActionGrammar olmalı."""
    encoded = encode_own_farm(observation, unit_index)
    context, value = model([encoded])
    context = context[0]

    def choose(distribution):
        return int(distribution.logits.argmax() if deterministic else distribution.sample())

    allowed = grammar._valid_unit_operations()
    op = UNIT_OPS[choose(model.operation_distribution(context, allowed))]
    action: list[Any] = [op]
    items = ()
    if op in UNIT_ITEM_OPS:
        grammar.current_op = op
        items = grammar._valid_unit_items()
        action.append(ITEMS[choose(model.item_distribution(context, op, items))])
    limit = 0
    if op in {"PICKUP", "PLACE"}:
        grammar.current_op = op
        grammar.current_item = action[1] if len(action) > 1 else None
        limit = grammar._current_quantity_limit()
        if limit:
            action.append(choose(model.quantity_distribution(context, op, action[1], limit)) + 1)
    sample = WorkerSample(
        encoded, action, allowed, items, limit, 0.0, float(value[0]),
        len(allowed) > 1 or len(items) > 1 or limit > 1,
    )
    sample.log_prob = float(worker_statistics(model, context, sample)[0])
    return sample
