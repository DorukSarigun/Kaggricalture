"""Frozen demand features; admission checks physical sites, NOT crew workload.

The manager learns how much work to request. No worker-count queue budget or
estimated route-capacity mask is applied. Missing seeds do not forbid planning.
"""
import math

from kaggle_environments.envs.kaggriculture import kaggriculture as game
from worker_grow.contracts import TASK_ITEMS
from worker_grow.issues import SHED_DOORS, crew_size

ADMISSION_VERSION = "physical-sites-only-v1"
ITEM_FEATURES = ("planned", "active", "delivering", "unbound", "quota_left", "missing_inputs")
WORKLOAD_NAMES = tuple(f"{item}.{name}" for item in TASK_ITEMS for name in ITEM_FEATURES) + (
    "workers", "open_jobs", "unstarted_jobs", "unreserved_sites",
)
WORKLOAD_DIM = len(WORKLOAD_NAMES)


def _sites(observation, book):
    farm = observation["farms"][int(observation["player"])]
    occupied = book.occupied_tiles()
    return [tile for y, row in enumerate(farm["tiles"]) for x, tile in enumerate(row)
            if (x, y) not in occupied and (x, y) not in SHED_DOORS
            and (tile is None or isinstance(tile, dict) and (
                tile.get("kind") == "WEED" or
                tile.get("kind") in ("COOP", "PASTURE") and not tile.get("animal")))]


def admission_items(observation, book, issue_items=None):
    records = [r for r in book.task_book.records.values() if r.needs_production]
    sites = _sites(observation, book)
    pending = [r.task.item for r in records if r.tile_xy is None]
    general = sum(tile is None or tile.get("kind") == "WEED" for tile in sites)
    coops = sum(tile is not None and tile.get("kind") == "COOP" for tile in sites)
    pastures = sum(tile is not None and tile.get("kind") == "PASTURE" for tile in sites)
    def fits(item):
        jobs = pending + [item]
        # Match dedicated structures first; only remaining jobs need free soil.
        soil_needed = (sum(i in game.CROPS for i in jobs) +
                       max(0, jobs.count("GOOSE") - coops) +
                       max(0, sum(i in ("COW", "SHEEP") for i in jobs) - pastures))
        return soil_needed <= general
    # A seed is not required yet: contracts precede this turn's purchases.
    items = tuple(item for item in TASK_ITEMS if fits(item))
    if issue_items is None:
        return items
    allowed = frozenset(issue_items)
    return tuple(item for item in items if item in allowed)


def workload_features(observation, book):
    """Immutable log1p counts; recomputed before EVERY manager decision."""
    records = [r for r in book.task_book.records.values() if r.open]
    private = observation["private"]
    counts = []
    for item in TASK_ITEMS:
        matching = [r for r in records if r.task.item == item]
        available = (private["seeds"].get(item, 0) if item in game.CROPS else
                     private["shed"].get(item, 0) +
                     sum(bag.get(item, 0) for bag in private["inventories"]))
        counts.extend([
            *(sum(r.status.value == status for r in matching)
              for status in ("PLANNED", "ACTIVE", "DELIVERING")),
            sum(r.tile_xy is None and r.needs_production for r in matching),
            sum(max(0, q - r.delivered.get(p, 0) - r.consumed.get(p, 0))
                for r in matching for p, q in r.task.quotas),
            max(0, sum(r.needs_production and r.producer_id is None for r in matching) - available),
        ])
    workers = crew_size(observation)
    counts.extend((workers, len(records),
                   sum(r.needs_production and r.producer_id is None for r in records),
                   len(_sites(observation, book))))
    return tuple(math.log1p(n) for n in counts)
