"""Explicit, interruptible crew subjobs. No nearest-product reward inference."""
from dataclasses import dataclass, asdict
import numpy as np
from kaggle_environments.envs.kaggriculture import kaggriculture as game
from worker_grow.contracts import TASK_ITEMS
from worker_grow.policy import UNIT_OPS

MOVES = ('NORTH', 'SOUTH', 'EAST', 'WEST')
WORK = ('DIG', 'PLANT', 'BUILD_COOP', 'BUILD_PASTURE', 'PLACE', 'WATER',
        'FERTILIZE', 'HARVEST', 'FEED', 'CARE', 'COLLECT_FERTILIZER')
FEATURE_DIM = len(UNIT_OPS) + len(TASK_ITEMS) + 10
ACCESS = ((4, 4), (5, 4), (4, 5), (5, 5))
UNPINNED_SITES = 8


@dataclass(frozen=True)
class Subtask:
    issue_id: str | None
    operation: str
    tile_xy: tuple[int, int] | None
    item: str | None = None
    producer_id: str | None = None
    day: int = 0

    @property
    def key(self):
        return (self.issue_id, self.operation, self.tile_xy, self.item, self.producer_id, self.day)

    @property
    def reservation_key(self):
        if self.issue_id is None:return None  # each worker delivers its own cargo
        target=None if self.operation=='PICKUP' else self.tile_xy
        return (self.issue_id,self.operation,target,self.item,self.producer_id,self.day)

    def as_dict(self):
        return asdict(self)


IDLE = Subtask(None, 'PASS', None)


def _unpinned_sites(farm, here, occupied=(), item=None):
    hx, hy = int(here[0]), int(here[1])
    empty = []
    for y, row in enumerate(farm['tiles']):
        for x, tile in enumerate(row):
            if (x, y) in ACCESS or (x, y) in occupied:
                continue
            if tile is None or isinstance(tile, dict) and (
                    tile.get('kind') == 'WEED' or item in game.ANIMALS and
                    tile.get('kind') == game.ANIMALS[item]['structure'] and not tile.get('animal')):
                empty.append((abs(x - hx) + abs(y - hy), x, y))
    empty.sort()
    return tuple((x, y) for _, x, y in empty[:UNPINNED_SITES])


def candidates(cursor):
    """Check readiness at target positions using the real action-mask adapter.

    Only a temporary shadow position is changed; no action or real state is
    executed. Missing feed/animals creates a PICKUP subjob, not impossible FEED.
    Unpinned issues only consider the nearest empty tiles; FIFO still binds the
    oldest compatible demand.
    """
    farm, index = cursor.rules.farm, cursor.rules.unit_index
    original = list(farm['farmer'] if index == 0 else farm['hands'][index-1])
    def position(xy):
        if index == 0: farm['farmer'] = list(xy)
        else: farm['hands'][index-1] = list(xy)
    result = [IDLE]
    day = int(cursor.original['day'])
    try:
        for record in cursor.tasks.records.values():
            if not record.needs_production:
                continue
            item = record.task.item
            if record.tile_xy is not None:
                targets = (record.tile_xy,)
            else:
                if cursor.tasks.unassigned_candidate((item,)) is not record:
                    continue
                targets = _unpinned_sites(farm, original, {
                    r.tile_xy for r in cursor.tasks.records.values() if r.reserves_tile
                }, item)
            for xy in targets:
                position(xy)
                # Same-item pending issues are allocated in FIFO order.
                selected = cursor.tasks.at(xy) or cursor.tasks.candidate(xy, (item,))
                if selected is not record:
                    continue
                for op in cursor.operations():
                    if op not in WORK: continue
                    if op.startswith('BUILD_') and (item not in game.ANIMALS or
                            op != 'BUILD_'+game.ANIMALS[item]['structure']):continue
                    if op.startswith('BUILD_') and cursor.candidate(tuple(a for a,spec in game.ANIMALS.items()
                            if op=='BUILD_'+spec['structure'])) is not record:continue
                    allowed = cursor.items(op)
                    arg = item if op in ('PLANT', 'PLACE') else None
                    if allowed and arg not in allowed: continue
                    if op == 'PLACE' and (item not in game.ANIMALS or record.producer_id): continue
                    result.append(Subtask(record.task.task_id,op,xy,arg,record.producer_id,day))
            # Fetch only inputs for a live job, and only if not already carried.
            needed = (['WHEAT'] + ([item] if not record.producer_id else [])
                      if item in game.ANIMALS else ['FERTILIZER'])
            if record.status.value == 'DELIVERING': needed = []
            for xy in ACCESS:
                position(xy)
                if 'PICKUP' not in cursor.operations(): continue
                allowed = cursor.items('PICKUP')
                for resource in needed:
                    if resource in allowed and not cursor.rules._inventory().get(resource,0):
                        result.append(Subtask(record.task.task_id,'PICKUP',xy,resource,record.producer_id,day))
        for xy in ACCESS:
            position(xy)
            if 'DROP' in cursor.operations() and any(cursor.rules._inventory().values()):
                result.append(Subtask(None,'DROP',xy,day=day))
    finally:
        position(original)
    return tuple(dict.fromkeys(result))


def available(options, leases, unit, step, day):
    # Hands are day-scoped. Stale/retired worker reservations expire in one turn.
    for owner, lease in list(leases.items()):
        if lease['day'] != day or step-lease['step'] > 1:
            del leases[owner]
    claimed = {lease['subtask'].reservation_key for owner,lease in leases.items() if owner != unit}
    return tuple(s for s in options if s.reservation_key is None or s.reservation_key not in claimed)


def features(options, cursor, previous=None):
    x,y = cursor.rules._position()
    rows=[]
    for s in options:
        row=np.zeros(FEATURE_DIM,np.float32)
        row[UNIT_OPS.index(s.operation)]=1
        record=cursor.tasks.records.get(s.issue_id)
        if record:row[len(UNIT_OPS)+TASK_ITEMS.index(record.task.item)]=1
        tx,ty=s.tile_xy if s.tile_xy is not None else (x,y)
        ix,iy=record.tile_xy if record and record.tile_xy is not None else (-1,-1)
        row[-10:]=[tx/9,ty/9,(tx-x)/9,(ty-y)/9,(abs(tx-x)+abs(ty-y))/18,
                  float(previous is not None and previous.key==s.key),
                  float(s.producer_id is not None),float(s.operation=='PASS'),ix/9,iy/9]
        rows.append(row)
    return np.stack(rows)


def operation_mask(cursor, selected):
    allowed=cursor.operations()
    if selected.operation=='PASS': return ('PASS',)
    at_target=tuple(cursor.rules._position())==selected.tile_xy
    return tuple(op for op in allowed if op in (*MOVES,'PASS') or
                 at_target and op==selected.operation)


def decision_metadata(row):
    selected=getattr(row,'subtask',None)
    return {'assignment_id':getattr(row,'assignment_id',None),
            'assignment_change':getattr(row,'assignment_change',None),
            'subtask':selected.as_dict() if selected is not None else None,
            'assignment_choice':getattr(row,'assignment_choice',None)}
