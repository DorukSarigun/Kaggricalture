"""Manager production issues. Open count is not capped by crew size."""
from dataclasses import dataclass, field
from kaggle_environments.envs.kaggriculture import kaggriculture as game

from worker_grow.contracts import ProductionTask, TaskBook, TaskStatus, TASK_ITEMS

OPEN = (TaskStatus.PLANNED, TaskStatus.ACTIVE, TaskStatus.DELIVERING)
CLOSED = (TaskStatus.COMPLETE, TaskStatus.FAILED, TaskStatus.EXPIRED)
REWARD_SCALE = 3000.0


def crew_size(observation) -> int:
    farm = observation["farms"][int(observation["player"])]
    return 1 + len(farm["hands"])


def issue_unlock_limit(episode, *, interval, initial=1):
    """Concurrent open-issue allowance for a crew episode. interval=0 disables."""
    if interval == 0:
        return None
    if episode < 1 or interval < 1 or initial < 1:
        raise ValueError("Invalid issue unlock schedule")
    return initial + (episode - 1) // interval


def legal_issue_tiles(observation, occupied) -> tuple[tuple[int, int], ...]:
    farm = observation["farms"][int(observation["player"])]
    tiles = farm["tiles"]
    height, width = len(tiles), len(tiles[0])
    free = []
    for y in range(height):
        for x in range(width):
            if (x, y) in occupied:
                continue
            if tiles[y][x] is None:
                free.append((x, y))
    return tuple(free)


SHED_DOORS = ((4, 4), (5, 4), (4, 5), (5, 5))


def production_pin(observation, occupied, tile_xy):
    """Ghost feet on a shed door still need a plantable neighbor."""
    if tile_xy is None:
        return None
    tile_xy = (int(tile_xy[0]), int(tile_xy[1]))
    if tile_xy in occupied:
        return None
    legal = legal_issue_tiles(observation, occupied)
    if tile_xy in legal and tile_xy not in SHED_DOORS:
        return tile_xy
    if tile_xy not in SHED_DOORS:
        return None
    plantable = tuple(xy for xy in legal if xy not in SHED_DOORS)
    if not plantable:
        return None
    return min(plantable, key=lambda xy: abs(xy[0] - tile_xy[0]) + abs(xy[1] - tile_xy[1]))


@dataclass
class Issue:
    task_id: str
    tile_xy: tuple[int, int] | None
    item: str
    quotas: tuple[tuple[str, int], ...]
    decision_id: int
    opened_step: int
    opened_day: int
    status: TaskStatus = TaskStatus.PLANNED
    producer_id: str | None = None
    delivered: dict = field(default_factory=dict)
    provision_cost: float = 0.0
    provision_lots: list = field(default_factory=list)
    closed_step: int | None = None
    close_gold: float | None = None
    realized_sales: bool = False
    realized_gold: float = 0.0
    bound_step: int | None = None
    bound_decision_id: int | None = None
    production_closed: bool = False
    consumed: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.tile_xy is not None:
            self.tile_xy = (int(self.tile_xy[0]), int(self.tile_xy[1]))
        if self.item not in TASK_ITEMS:
            raise ValueError(f"Invalid issue item: {self.item}")
        if self.decision_id < 0 or self.opened_step < 0:
            raise ValueError("Issue identity/step must be non-negative")

    @property
    def open(self) -> bool:
        return self.status in OPEN

    def production_task(self) -> ProductionTask:
        return ProductionTask(self.task_id, self.tile_xy, self.item, self.quotas,
                              producer_lifetime=self.realized_sales)

    def as_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "tile_xy": list(self.tile_xy) if self.tile_xy is not None else None,
            "bound_step": self.bound_step,
            "bound_decision_id": self.bound_decision_id,
            "item": self.item,
            "quotas": [list(pair) for pair in self.quotas],
            "decision_id": self.decision_id,
            "opened_step": self.opened_step,
            "opened_day": self.opened_day,
            "status": self.status.value,
            "producer_id": self.producer_id,
            "delivered": dict(self.delivered),
            "provision_cost": self.provision_cost,
            "provision_lots": list(self.provision_lots),
            "closed_step": self.closed_step,
            "close_gold": self.close_gold,
            "realized_sales": self.realized_sales,
            "realized_gold": self.realized_gold,
            "production_closed": self.production_closed,
            "consumed": dict(self.consumed),
        }


class ProvisionPool:
    """FIFO actual purchase gold, one unit at a time."""

    def __init__(self):
        self.queues: dict[str, list[list[float]]] = {}

    def buy(self, item: str, quantity: int, amount: float) -> None:
        if quantity < 1 or not (amount < 0) or item is None:
            raise ValueError("Provision buy must be a negative paid quantity")
        unit = abs(float(amount)) / quantity
        self.queues.setdefault(item, []).append([float(quantity), unit])

    def take(self, item: str, quantity: int = 1) -> float | None:
        queue = self.queues.get(item, [])
        if sum(row[0] for row in queue) < quantity:
            return None
        cost = 0.0
        remaining = float(quantity)
        while remaining:
            row = queue[0]
            n = min(remaining, row[0])
            cost += n * row[1]
            row[0] -= n
            remaining -= n
            if row[0] == 0:
                queue.pop(0)
        return cost


def default_quota(item: str) -> tuple[tuple[str, int], ...]:
    product = item if item in game.CROPS else game.ANIMALS[item]["product"]
    return ((product, 1),)


def input_list_price(item: str) -> float:
    """Catalog cost of one production input; used when an issue never bought it."""
    if item in game.CROPS:
        return float(game.CROPS[item]["seed"])
    if item in game.ANIMALS:
        return float(game.ANIMALS[item]["cost"])
    raise KeyError(item)


def issue_close_gold(status: TaskStatus, provision_cost: float, delivered: dict,
                     prices: dict, *, item: str | None = None) -> float:
    delivered_value = sum(float(prices.get(product, 0)) * int(quantity)
                          for product, quantity in delivered.items())
    cost = float(provision_cost)
    if (status in (TaskStatus.FAILED, TaskStatus.EXPIRED) and cost <= 0
            and item is not None
            and not any(int(quantity) > 0 for quantity in delivered.values())):
        cost = input_list_price(item)
    if status is TaskStatus.COMPLETE:
        return delivered_value - float(provision_cost)
    if status is TaskStatus.FAILED:
        return -cost
    if status is TaskStatus.EXPIRED:
        return delivered_value - cost
    raise ValueError(f"Reward is only defined for closed issues: {status}")


class IssueBook:
    def __init__(self, *, realized_sales=False):
        self.realized_sales = realized_sales
        self.issues: dict[str, Issue] = {}
        self.task_book = TaskBook()
        self.pool = ProvisionPool()
        self._counter = 0

    def open_issues(self) -> list[Issue]:
        return [issue for issue in self.issues.values() if issue.open]

    def occupied_tiles(self) -> set[tuple[int, int]]:
        return {r.tile_xy for r in self.task_book.records.values() if r.reserves_tile}

    def next_id(self) -> str:
        self._counter += 1
        return f"issue:{self._counter}"

    def add(self, issue: Issue, observation) -> Issue:
        if issue.task_id in self.issues:
            raise ValueError("Issue identity cannot be reused")
        if issue.tile_xy is not None and issue.tile_xy in self.occupied_tiles():
            raise ValueError("Tile already has an open issue")
        if issue.tile_xy is not None and issue.tile_xy not in legal_issue_tiles(observation, self.occupied_tiles()):
            raise ValueError("Issue tile is not an empty unlocked cell")
        issue.realized_sales = self.realized_sales
        self.issues[issue.task_id] = issue
        self.task_book.add(issue.production_task())
        self._fill_provisions()
        return issue

    def issue(self, observation, *, item, tile_xy=None, decision_id, step, day,
              quotas=None) -> Issue:
        created = Issue(
            self.next_id(), tile_xy, item, quotas or default_quota(item),
            decision_id, step, day,
        )
        return self.add(created, observation)

    def _fill_provisions(self) -> None:
        if self.realized_sales:
            return  # Actual PLANTED/PLACED lot allocations in the DAG own cost basis.
        for issue in self.open_issues():
            if issue.provision_cost > 0:
                continue
            cost = self.pool.take(issue.item, 1)
            if cost is None:
                continue
            issue.provision_cost = cost
            issue.provision_lots.append({"item": issue.item, "cost": cost})

    def observe_cash(self, entries, player: int) -> None:
        for entry in entries:
            if entry.player != player:
                continue
            if entry.operation in {"BUY_SEED", "BUY_ANIMAL"} and entry.quantity > 0:
                self.pool.buy(entry.item, entry.quantity, entry.amount)
        self._fill_provisions()

    def close(self, task_id: str, status: TaskStatus, *, step: int, prices: dict,
              delivered=None) -> float:
        issue = self.issues[task_id]
        if not issue.open:
            raise ValueError(f"Issue already closed: {task_id}")
        if status not in CLOSED:
            raise ValueError("Manager issues close only as COMPLETE/FAILED/EXPIRED")
        if delivered is not None:
            issue.delivered = {key: int(value) for key, value in delivered.items()}
        issue.status = status
        issue.closed_step = step
        issue.close_gold = (0.0 if self.realized_sales else
                            issue_close_gold(status, issue.provision_cost, issue.delivered, prices,
                                             item=issue.item))
        if issue.task_id in self.task_book.records:
            self.task_book.records[issue.task_id].status = status
            self.task_book.records[issue.task_id].delivered = dict(issue.delivered)
        return issue.close_gold

    def sync_from_tasks(self, tasks: TaskBook, *, step: int, prices: dict) -> list[Issue]:
        closed = []
        for task_id, record in tasks.records.items():
            if task_id not in self.issues:
                continue
            issue = self.issues[task_id]
            issue.tile_xy = record.tile_xy
            issue.bound_step = record.bound_step
            issue.bound_decision_id = record.bound_decision_id
            if record.producer_id:
                issue.producer_id = record.producer_id
            issue.delivered = dict(record.delivered)
            issue.production_closed = record.production_closed
            issue.consumed = dict(record.consumed)
            if record.status in CLOSED and issue.open:
                self.close(task_id, record.status, step=step, prices=prices,
                           delivered=record.delivered)
                closed.append(issue)
            elif record.status in OPEN:
                issue.status = record.status
        return closed

    def expire_open(self, *, step: int, prices: dict) -> list[Issue]:
        closed = []
        for issue in list(self.open_issues()):
            self.close(issue.task_id, TaskStatus.EXPIRED, step=step, prices=prices)
            closed.append(issue)
        return closed

    def to_task_book(self) -> TaskBook:
        return self.task_book
