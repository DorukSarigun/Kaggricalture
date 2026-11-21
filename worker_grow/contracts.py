"""Persistent, manager-owned production orders; never issue movement commands."""
from dataclasses import dataclass, field
from collections.abc import Mapping
from enum import StrEnum
import numpy as np
from kaggle_environments.envs.kaggriculture import kaggriculture as game

TASK_ITEMS = (*game.CROPS, *game.ANIMALS)
SPATIAL_CHANNELS = len(TASK_ITEMS) + 2
TASK_CHANNELS = SPATIAL_CHANNELS + len(TASK_ITEMS)  # plus broadcast unassigned demand
BOARD_SIZE = 10
GOAL_DIM = TASK_CHANNELS * BOARD_SIZE * BOARD_SIZE + 1  # final coordinate = idle
TASK_LIFECYCLE_VERSION = 'production-occupancy-v2'


class TaskStatus(StrEnum):
    PLANNED = "PLANNED"
    ACTIVE = "ACTIVE"
    DELIVERING = "DELIVERING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


@dataclass(frozen=True)
class ProductionTask:
    task_id: str
    tile_xy: tuple[int, int] | None
    item: str
    # Explicit manager demand. Fertilizer is paid only when requested.
    quotas: tuple[tuple[str, int], ...]
    # Retain producer assignment after delivery quota; economic provenance is
    # independent of whether its outputs are delivered, consumed or sold.
    producer_lifetime: bool = False

    def __post_init__(self):
        if self.tile_xy is not None:
            object.__setattr__(self,'tile_xy',tuple(self.tile_xy))
        quotas=self.quotas.items() if isinstance(self.quotas,Mapping) else self.quotas
        object.__setattr__(self,'quotas',tuple(tuple(pair) for pair in quotas))
        if not self.task_id or self.item not in TASK_ITEMS:
            raise ValueError("Invalid task identity/item")
        if self.tile_xy is not None and (len(self.tile_xy) != 2 or any(type(v) is not int or not 0 <= v < BOARD_SIZE for v in self.tile_xy)):
            raise ValueError("Invalid task tile")
        products = ({self.item} if self.item in game.CROPS
                    else {game.ANIMALS[self.item]['product'], 'FERTILIZER'})
        if (not self.quotas or len(dict(self.quotas)) != len(self.quotas)
                or any(p not in products or type(q) is not int or q <= 0 for p, q in self.quotas)):
            raise ValueError("Invalid production quotas")


@dataclass
class TaskRecord:
    task: ProductionTask
    status: TaskStatus = TaskStatus.PLANNED
    producer_id: str | None = None
    delivered: dict[str, int] = field(default_factory=dict)
    production_closed: bool = False
    total_delivered: dict[str, int] = field(default_factory=dict)
    assigned_tile: tuple[int, int] | None = None
    bound_step: int | None = None
    bound_decision_id: int | None = None
    # Verified intermediate use, not a depot delivery or an extra reward.
    consumed: dict[str, int] = field(default_factory=dict)

    @property
    def tile_xy(self):
        return self.assigned_tile if self.assigned_tile is not None else self.task.tile_xy

    @property
    def open(self):
        return self.status in (TaskStatus.PLANNED, TaskStatus.ACTIVE, TaskStatus.DELIVERING)

    @property
    def needs_production(self):
        return self.open and not self.production_closed

    @property
    def reserves_tile(self):
        return self.needs_production and self.tile_xy is not None

    @property
    def fulfilled(self):
        return all(self.delivered.get(p, 0) + self.consumed.get(p, 0) >= q
                   for p, q in self.task.quotas)

    def complete_if_fulfilled(self):
        if self.open and self.fulfilled and (not self.task.producer_lifetime or self.production_closed):
            self.status = TaskStatus.COMPLETE

    def close_production(self):
        self.production_closed = True
        if self.open:
            self.status = TaskStatus.DELIVERING
            self.complete_if_fulfilled()


class TaskBook:
    def __init__(self, tasks=()):
        self.records = {}
        for task in tasks:
            self.add(task)

    def add(self, task):
        if task.task_id in self.records:
            raise ValueError("Task identity cannot be reused")
        if task.tile_xy is not None and any(r.reserves_tile and r.tile_xy == task.tile_xy for r in self.records.values()):
            raise ValueError("Tile already has an open task")
        self.records[task.task_id] = TaskRecord(task)

    def at(self, xy):
        # Historical tile coordinates remain on old jobs for lot/DAG provenance.
        return next((r for r in self.records.values() if r.reserves_tile and r.tile_xy == tuple(xy)), None)

    def candidate(self, xy, items):
        """Bound job first; otherwise oldest unassigned compatible issue (FIFO)."""
        r = self.at(xy)
        if r is not None:
            return r if r.producer_id is None and r.task.item in items else None
        return self.unassigned_candidate(items)

    def unassigned_candidate(self, items):
        """Oldest unbound demand, independent of the worker's current square."""
        return next((r for r in self.records.values() if r.needs_production and r.tile_xy is None
                     and r.producer_id is None and r.task.item in items), None)

    def bind(self, xy, items, *, step=None, decision_id=None):
        r = self.candidate(xy, items)
        if r is None:
            return None
        if r.tile_xy is None:
            tile = tuple(xy)
            if len(tile) != 2 or any(type(v) is not int or not 0 <= v < BOARD_SIZE for v in tile):
                raise ValueError('Invalid crew-selected tile')
            r.assigned_tile = tile
        if decision_id is not None and r.bound_decision_id is None:
            r.bound_step = step
            r.bound_decision_id = decision_id
        return r

    def cancel(self, task_id):
        self.records[task_id].status = TaskStatus.CANCELLED

    def planes(self):
        result = np.zeros((TASK_CHANNELS, BOARD_SIZE, BOARD_SIZE), np.float32)
        for r in self.records.values():
            if not r.needs_production:
                continue
            if r.tile_xy is None:
                result[SPATIAL_CHANNELS + TASK_ITEMS.index(r.task.item)] += 1
                continue
            x, y = r.tile_xy
            result[TASK_ITEMS.index(r.task.item), y, x] = 1
            total = sum(q for _, q in r.task.quotas)
            result[len(TASK_ITEMS), y, x] = sum(max(0, q-r.delivered.get(p, 0)) for p, q in r.task.quotas)/total
            result[len(TASK_ITEMS)+1, y, x] = float(r.producer_id is not None)
        return result

    def goal(self):
        planes = self.planes().reshape(-1)
        vector = np.append(planes, float(not np.any(planes))).astype(np.float32)
        return vector / np.linalg.norm(vector)

    def resource_state(self, record, obs):
        """Diagnostic only: unavailable resources are not a crew failure penalty."""
        if not record.open:
            return record.status.value
        if record.producer_id:
            return record.status.value
        xy = record.tile_xy
        if xy is not None and obs['farms'][obs['player']]['tiles'][xy[1]][xy[0]] == 'LOCKED':
            return 'WAITING_LAND'
        private, item = obs['private'], record.task.item
        available = (private['seeds'].get(item, 0) if item in game.CROPS else
                     private['shed'].get(item, 0)+sum(i.get(item, 0) for i in private['inventories']))
        return 'READY' if available else 'WAITING_RESOURCE'
