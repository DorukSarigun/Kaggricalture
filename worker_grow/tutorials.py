"""Mini crew tutorials on the live IssueBook contract.

Gifted plants bind through bootstrap PLANTED (unbound ISSUE + existing tile).
The plant slice leaves the ISSUE unbound so crew chooses the tile.
"""
from copy import deepcopy

from kaggle_environments.envs.kaggriculture import kaggriculture as game

from worker_grow.contracts import SPATIAL_CHANNELS, TASK_ITEMS, TaskStatus
from worker_grow.issues import IssueBook, default_quota

SLICES = ("drop", "harvest", "walk_harvest", "water", "plant")
SLICE_STEPS = {
    "drop": 8,
    "harvest": 16,
    "walk_harvest": 24,
    "water": 8,
    "plant": 16,
}
SHED_XY = (4, 4)
WALK_PLANT_XY = (2, 2)


def parse_slices(text):
    names = tuple(part.strip() for part in text.split(",") if part.strip())
    if not names or any(name not in SLICES for name in names):
        raise ValueError(f"Tutorial slices must be from {SLICES}")
    return names


def slice_for_episode(episode, slices, slice_episodes):
    if episode < 1 or slice_episodes < 1 or not slices:
        raise ValueError("Invalid tutorial schedule")
    index = min((episode - 1) // slice_episodes, len(slices) - 1)
    return slices[index]


def horizon(name, override=None):
    if override is not None:
        return override
    if name not in SLICE_STEPS:
        raise ValueError(f"Unknown tutorial slice: {name}")
    return SLICE_STEPS[name]


def _observation(env):
    return env.state[0].observation


def _commit(env, farm, private):
    for state in env.state:
        state.observation.farms[0] = deepcopy(farm)
    env.state[0].observation.private = deepcopy(private)
    return _observation(env)


def _solo_farm(env):
    farm = deepcopy(_observation(env).farms[0])
    private = deepcopy(_observation(env).private)
    farm["hands"] = []
    farm["farmer"] = list(SHED_XY)
    private["inventories"] = [{}]
    private["seeds"] = {crop: 0 for crop in game.CROPS}
    return farm, private


def _mature_plant(item, quantity):
    plant = game._new_plant(item, 0, 24)
    plant["planted_day"] = -game.CROPS[item]["first_yield_day"]
    plant["yield_units"] = quantity
    plant["max_lifespan_step"] = -1
    plant["watered_today"] = True
    plant["consecutive_unwatered"] = 0
    return plant


def _growing_plant(item):
    plant = game._new_plant(item, 0, 24)
    plant["watered_today"] = False
    plant["consecutive_unwatered"] = 1
    return plant


def _open_issue(observation, item, quantity):
    book = IssueBook()
    book.issue(
        observation, item=item, decision_id=0, step=0, day=int(observation["day"]),
        quotas=default_quota(item) if quantity == 1 else ((item if item in game.CROPS
            else game.ANIMALS[item]["product"], quantity),),
    )
    return book


def setup(env, name, *, item="WHEAT", quantity=1):
    if item not in game.CROPS:
        raise ValueError("Tutorials currently gift crop scenes only")
    farm, private = _solo_farm(env)
    x = y = None
    if name == "drop":
        farm["farmer"] = list(SHED_XY)
        private["inventories"][0][item] = quantity
    elif name == "harvest":
        x, y = SHED_XY
        farm["farmer"] = [x, y]
        farm["tiles"][y][x] = _mature_plant(item, quantity)
    elif name == "walk_harvest":
        x, y = WALK_PLANT_XY
        farm["farmer"] = list(SHED_XY)
        farm["tiles"][y][x] = _mature_plant(item, quantity)
    elif name == "water":
        x, y = SHED_XY
        farm["farmer"] = [x, y]
        farm["tiles"][y][x] = _growing_plant(item)
    elif name == "plant":
        farm["farmer"] = list(SHED_XY)
        private["seeds"][item] = 1
    else:
        raise ValueError(f"Unknown tutorial slice: {name}")
    observation = _commit(env, farm, private)
    return _open_issue(observation, item, quantity)


def attach_gifted_inventory(tracker, book, product):
    """Bootstrap inventory has no harvest ancestry; mark it as the open quota."""
    issue = book.open_issues()[0]
    record = book.task_book.records[issue.task_id]
    record.status = TaskStatus.DELIVERING
    for key, parts in tracker.ledger.lots.items():
        for part in parts:
            if part.product == product and part.task_id is None:
                part.task_id = issue.task_id
                if part.batch_id is None:
                    part.batch_id = key
                tracker.ledger.created[part.batch_id] += part.quantity


def demand_is_bound(book, item, xy):
    planes = book.task_book.planes()
    x, y = xy
    return bool(planes[TASK_ITEMS.index(item), y, x]) and not planes[SPATIAL_CHANNELS:].any()


def demand_is_unbound(book, item):
    planes = book.task_book.planes()
    return planes[SPATIAL_CHANNELS + TASK_ITEMS.index(item)].sum() >= 1 and not planes[:SPATIAL_CHANNELS].any()
