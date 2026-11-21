"""Economic manager on the original opponent-aware encoder.

Unit heads are never sampled. Contracts are ISSUE/STOP; market uses the
existing StructuredActorCritic heads after the crew has moved.
"""
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np
import torch
from torch import nn

from observation_encoding import encode_observation
from structured_actions import (
    HeadInput, OperationCursor, OperationSample, StructuredActorCritic,
    action_statistics, head_input, sample_operation,
)
from worker_grow.contracts import TASK_ITEMS
from worker_grow.issues import IssueBook, default_quota
from worker_grow.manager_workload import (
    WORKLOAD_DIM, admission_items, workload_features,
)

CONTRACT_OPS = ("ISSUE", "STOP")
MANAGER_VERSION = "feudal-manager-issue-gold-dag-v7"


def market_cursor(observation, prefix=(), config=None):
    cursor = OperationCursor(observation, config)
    cursor.section = "market"
    cursor.rules.unit_index = cursor.rules.num_units
    for order in prefix:
        cursor.apply(order)
    return cursor


@dataclass
class ManagerDecision:
    inputs: HeadInput
    sample: OperationSample | None
    value: float
    reward: float = 0.0
    base_reward: float = 0.0
    boundary: bool = False
    done: bool = False
    advantage: float = 0.0
    return_target: float = 0.0
    realized_target: float = 0.0
    local_credit: float = 0.0
    task_credit: float = 0.0
    throughput_credit: float = 0.0
    chain_value: float = 0.0
    chain_advantage: float = 0.0
    throughput_value: float = 0.0
    throughput_advantage: float = 0.0
    kind: str = "market"
    action: list[Any] = field(default_factory=list)
    log_prob: float = 0.0
    decision_id: int = 0
    step: int = 0
    stochastic: bool = True
    allowed_ops: tuple = ()
    allowed_items: tuple = ()
    encoded: Any = None
    episode: int = 0


@dataclass
class ManagerHeadInput(HeadInput):
    workload: tuple[float, ...] = (0.0,) * WORKLOAD_DIM


def with_workload(inputs, observation, book):
    values = {f.name: getattr(inputs, f.name) for f in fields(HeadInput)}
    return ManagerHeadInput(**values, workload=workload_features(
        observation, book))


class ManagerBody(StructuredActorCritic):
    """Demand conditions BOTH the policy and critics, with frozen PPO inputs."""

    def __init__(self):
        super().__init__()
        self.demand_context = nn.Sequential(
            nn.Linear(256 + WORKLOAD_DIM, 256), nn.LayerNorm(256), nn.Tanh(),
        )
        self.throughput_value_head = nn.Linear(256, 1)

    def forward(self, inputs):
        context, _, _ = super().forward(inputs)
        demand = torch.tensor([getattr(r, "workload", (0.,) * WORKLOAD_DIM) for r in inputs],
                              device=context.device, dtype=context.dtype)
        context = self.demand_context(torch.cat((context, demand), dim=-1))
        return (context, self.value_head(context).squeeze(-1),
                self.realized_value_head(context).squeeze(-1),
                self.throughput_value_head(context).squeeze(-1))


class ManagerActor(nn.Module):
    """Same StateEncoder + market body as the legacy structured agent."""

    def __init__(self):
        super().__init__()
        self.body = ManagerBody()
        self.contract_op = nn.Linear(256, len(CONTRACT_OPS))
        self.item_score = nn.Linear(256, len(TASK_ITEMS))

    @property
    def device(self):
        return next(self.parameters()).device

    def encode_state(self, encoded):
        return self.body.encoder(
            *tuple(
                torch.as_tensor(
                    np.stack([getattr(row, name) for row in encoded]),
                    device=self.device,
                    dtype=torch.long if "categorical" in name else torch.float32,
                )
                for name in (
                    "own_categorical_ids", "own_numeric_features", "own_worker_positions",
                    "opponent_categorical_ids", "opponent_numeric_features",
                    "opponent_worker_positions", "global_features",
                )
            )
        )

    def contract_context(self, encoded_rows):
        state = self.encode_state(encoded_rows)
        return state.state_vector, state.policy_features

    def contract_distributions(self, context, allowed_items, allowed_ops=CONTRACT_OPS):
        op = self.body.distribution(self.contract_op(context), CONTRACT_OPS, allowed_ops)
        item = self.body.distribution(self.item_score(context), TASK_ITEMS, allowed_items or TASK_ITEMS[:1])
        return op, item


def contract_logprob(model, context, policy_features, row):
    op, item = model.contract_distributions(context, row.allowed_items, row.allowed_ops)
    logp = op.log_prob(torch.tensor(CONTRACT_OPS.index(row.action[0]), device=context.device))
    entropy = [op.entropy()]
    if row.action[0] == "ISSUE":
        logp = logp + item.log_prob(torch.tensor(TASK_ITEMS.index(row.action[1]), device=context.device))
        entropy.append(item.entropy())
    return logp, torch.stack(entropy).mean()


@torch.no_grad()
def propose_manager_contracts(model, observation, book: IssueBook, *, step, day,
                              next_id=0, deterministic=False, issue_items=None):
    encoded = encode_observation(observation)
    rows = []
    while True:
        inputs = with_workload(_contract_input(observation, encoded), observation, book)
        contexts, value, local, throughput = model.body([inputs])
        context = contexts[0]
        allowed_items = admission_items(observation, book, issue_items=issue_items)
        allowed_ops = CONTRACT_OPS if allowed_items else ("STOP",)
        op_d, item_d = model.contract_distributions(context, allowed_items, allowed_ops)
        def choose(dist):
            return int(dist.logits.argmax() if deterministic else dist.sample())
        op = CONTRACT_OPS[choose(op_d)]
        action = [op]
        if op == "ISSUE":
            item = TASK_ITEMS[choose(item_d)]
            action.append(item)
        row = ManagerDecision(
            inputs, None, float(value[0]),
            kind="contract", action=action, decision_id=next_id, step=step,
            stochastic=len(allowed_ops) > 1, allowed_ops=allowed_ops, allowed_items=allowed_items,
            encoded=encoded, chain_value=float(local[0]), throughput_value=float(throughput[0]),
        )
        row.log_prob = float(contract_logprob(model, context, None, row)[0])
        next_id += 1
        rows.append(row)
        if op == "STOP":
            break
        book.issue(observation, item=action[1],
                   decision_id=row.decision_id, step=step, day=day, quotas=default_quota(action[1]))
        # ISSUE has no tile head yet; teacher-pinned sites are crew-only training.
    return rows, next_id


def _contract_input(observation, encoded=None):
    encoded = encoded or encode_observation(observation)
    cursor = market_cursor(observation)
    # Structured body's unit/market/boundary flags are all zero for contracts.
    # Never alias an ISSUE decision with the actual end-of-turn settlement.
    return HeadInput(encoded, encoded, "contract", cursor.rules.num_units, 0,
                     int(cursor.rules.farm["hires_today"]), ())


@torch.no_grad()
def propose_manager_market(model, observation, *, step, next_id=0, deterministic=False,
                           book=None, grammar_config=None):
    book = IssueBook() if book is None else book
    original = encode_observation(observation)
    cursor = market_cursor(observation, config=grammar_config)
    rows = []
    while not cursor.finished:
        inputs = with_workload(head_input(cursor, original), cursor.observation(), book)
        sample = sample_operation(model.body, cursor, original, deterministic, inputs=inputs)
        cursor.apply(sample.action)
        rows.append(ManagerDecision(
            sample.inputs, sample, sample.value, kind="market", action=list(sample.action),
            log_prob=sample.log_prob, decision_id=next_id, step=step,
            stochastic=sample.stochastic, allowed_ops=sample.allowed_ops,
            allowed_items=sample.allowed_items, chain_value=sample.realized_value,
            throughput_value=sample.throughput_value,
        ))
        next_id += 1
    boundary = with_workload(head_input(cursor, original, boundary=True), cursor.observation(), book)
    _, value, local, throughput = model.body([boundary])
    rows.append(ManagerDecision(
        boundary, None, float(value[0]), boundary=True, kind="boundary",
        action=[], decision_id=next_id, step=step, stochastic=False,
        chain_value=float(local[0]), throughput_value=float(throughput[0]),
    ))
    return cursor.actions["market"], rows, next_id + 1


def manager_action_statistics(model, row, context=None):
    if row.kind == "contract":
        if context is None:
            contexts, *_ = model.body([row.inputs])
            context = contexts[0]
        return contract_logprob(model, context, None, row)
    if row.sample is None:
        raise ValueError("Market statistics require a sampled order")
    if context is None:
        context, *_ = model.body([row.inputs])
        context = context[0]
    return action_statistics(model.body, context, row.sample)
