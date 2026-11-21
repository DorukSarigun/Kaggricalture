"""Daily-budget crew assignment. Learned walking stays optional.

Strategies pick a legal Subtask from `assignments.candidates`. They do not
emit NORTH/SOUTH themselves; greedy walk is a separate execution policy.
"""
from __future__ import annotations
from dataclasses import dataclass, field, replace
from itertools import product
import numpy as np
from scipy.optimize import linear_sum_assignment

from worker_grow.assignments import IDLE, available, candidates

STRATEGIES = ('learned', 'greedy', 'hungarian', 'tour', 'cbba', 'cpsat')
WALK_POLICIES = ('learned', 'greedy')
STEP = {'NORTH': (0, -1), 'SOUTH': (0, 1), 'EAST': (1, 0), 'WEST': (-1, 0)}
ACCESS = frozenset(((4, 4), (5, 4), (4, 5), (5, 5)))
ANIMALS = frozenset({'COW', 'GOOSE', 'SHEEP'})
MUST = frozenset({'WATER', 'FEED', 'HARVEST', 'DROP'})
GROW = frozenset({'PLANT', 'BUILD_COOP', 'BUILD_PASTURE', 'PLACE', 'DIG'})
INF = 1e6


def resolve_walk_policy(assignment, walk=None):
    if walk is None:
        return 'learned' if assignment == 'learned' else 'greedy'
    if walk not in WALK_POLICIES:
        raise ValueError(f'Unknown walk policy: {walk}')
    return walk


def manhattan(a, b):
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def hours_left(observation):
    return max(1, 24 - int(observation['hour']))


def crew_xy(observation):
    farm = observation['farms'][int(observation['player'])]
    return (tuple(farm['farmer']), *[tuple(hand) for hand in farm['hands']])


def tile_at(farm, xy):
    if xy is None:
        return None
    x, y = int(xy[0]), int(xy[1])
    return farm['tiles'][y][x]


def thirsty(tile):
    if not isinstance(tile, dict):
        return False
    if tile.get('kind') == 'PLANT':
        return not bool(tile.get('watered_today'))
    if tile.get('animal'):
        return not bool(tile.get('fed_today'))
    return False


def harvestable(tile):
    return isinstance(tile, dict) and int(tile.get('yield_units') or 0) > 0


def option_index(options, selected):
    if selected in options:
        return options.index(selected)
    for i, option in enumerate(options):
        if option.reservation_key == selected.reservation_key and option.operation == selected.operation:
            return i
    for i, option in enumerate(options):
        if option.operation == 'PASS':
            return i
    return 0


@dataclass
class WorkerView:
    index: int
    xy: tuple[int, int]
    inventory: dict
    options: tuple


def worker_views(observation, tasks, leases, step, day, config=None):
    from worker_grow.crew import CrewCursor
    cursor = CrewCursor(observation, tasks, config)
    owned = dict(leases)
    views = []
    while cursor.section == 'unit':
        index = cursor.rules.unit_index
        options = available(candidates(cursor), owned, index, step, day)
        views.append(WorkerView(index, tuple(cursor.rules._position()),
                                dict(cursor.rules._inventory()), options))
        cursor.apply(['PASS'])
    return tuple(views)


def unique_jobs(views):
    jobs = {}
    for view in views:
        for option in view.options:
            if option.operation == 'PASS':
                continue
            key = option.reservation_key or ('drop', view.index, option.tile_xy)
            jobs.setdefault(key, option)
    return jobs


def travel(view, option):
    target = option.tile_xy or view.xy
    return manhattan(view.xy, target)


def capacity_ok(views, observation, extra=0):
    farm = observation['farms'][int(observation['player'])]
    need = extra
    for y, row in enumerate(farm['tiles']):
        for x, tile in enumerate(row):
            if thirsty(tile):
                need += 1
    budget = len(views) * hours_left(observation)
    return need <= budget


def needed_items(views):
    items = set()
    for view in views:
        for option in view.options:
            if option.item:
                items.add(option.item)
            if option.operation in ('FEED', 'PLACE', 'BUILD_COOP', 'BUILD_PASTURE'):
                items.update(ANIMALS)
                items.add('WHEAT')
    return items


def carried(view):
    return {item for item, count in view.inventory.items() if count}


def live_animals(observation):
    farm = observation['farms'][int(observation['player'])]
    found = []
    for y, row in enumerate(farm['tiles']):
        for x, tile in enumerate(row):
            if isinstance(tile, dict) and tile.get('animal'):
                found.append(tile)
    return found


def job_cost(view, option, observation, views):
    """Lower is more urgent. Do not dump inputs that an open issue still needs."""
    if option.operation == 'PASS':
        work = [o for v in views for o in v.options if o.operation != 'PASS']
        return 0.0 if not work else 80.0
    dist = travel(view, option)
    tile = tile_at(observation['farms'][int(observation['player'])], option.tile_xy)
    need, held = needed_items(views), carried(view)
    animals = live_animals(observation)
    hungry = any(thirsty(animal) for animal in animals)
    holding_wheat = any('WHEAT' in carried(other) for other in views)
    if option.operation == 'WATER' and thirsty(tile):
        return dist
    if option.operation == 'FEED' and thirsty(tile):
        return dist if 'WHEAT' in held else INF
    if option.operation == 'HARVEST' and harvestable(tile):
        return 10 + dist
    if option.operation == 'DROP':
        if held & ANIMALS:
            return INF
        if 'WHEAT' in held and animals:
            return INF
        if held & need:
            return INF
        return 20 + dist if held else INF
    if option.operation == 'PICKUP':
        if option.item in held:
            return INF
        if option.item == 'WHEAT':
            if holding_wheat or not hungry:
                return INF
            return 8 + dist
        if option.item in need:
            return 45 + dist
        return 70 + dist
    if option.operation == 'PLACE' and option.item in held:
        if option.tile_xy in ACCESS:
            return INF
        return 5 + dist
    if option.operation in GROW:
        if option.tile_xy in ACCESS:
            return INF
        if not capacity_ok(views, observation, extra=1):
            return INF / 2
        return 30 + dist
    if option.operation in ('CARE', 'FERTILIZE', 'COLLECT_FERTILIZER'):
        return 90 + dist
    return 70 + dist


def option_rank(option):
    xy = option.tile_xy or (-1, -1)
    return (option.operation, option.item or '', option.issue_id or '',
            xy, option.producer_id or '', option.day)


def greedy_pick(view, observation, views):
    return min(view.options, key=lambda option: (job_cost(view, option, observation, views),
                                                 option.operation != 'PASS', option_rank(option)))


def hungarian_plan(views, observation):
    jobs = list(unique_jobs(views).items())
    if not jobs:
        return {view.index: IDLE for view in views}
    columns = [job for _, job in jobs] + [IDLE] * len(views)
    cost = np.zeros((len(views), len(columns)))
    for i, view in enumerate(views):
        keys = {option.reservation_key for option in view.options}
        for j, job in enumerate(columns):
            if job.operation == 'PASS':
                cost[i, j] = job_cost(view, IDLE, observation, views)
                continue
            if job.reservation_key not in keys and job not in view.options:
                cost[i, j] = INF
            else:
                match = next((o for o in view.options if o.reservation_key == job.reservation_key
                              or o == job), job)
                cost[i, j] = job_cost(view, match, observation, views)
    rows, cols = linear_sum_assignment(cost)
    plan = {}
    for i, j in zip(rows, cols):
        job = columns[j]
        view = views[i]
        if job.operation == 'PASS':
            plan[view.index] = IDLE
            continue
        match = next((o for o in view.options if o.reservation_key == job.reservation_key or o == job), None)
        plan[view.index] = match if match is not None else greedy_pick(view, observation, views)
    for view in views:
        plan.setdefault(view.index, greedy_pick(view, observation, views))
    return plan


def _tour_length(start, stops):
    points = [start, *stops]
    return sum(manhattan(points[i], points[i + 1]) + 1 for i in range(len(points) - 1))


def _two_opt(start, stops):
    stops = list(stops)
    changed = True
    while changed and len(stops) >= 2:
        changed = False
        best = _tour_length(start, stops)
        for i in range(len(stops) - 1):
            for j in range(i + 2, len(stops) + 1):
                candidate = stops[:i] + stops[i:j][::-1] + stops[j:]
                length = _tour_length(start, candidate)
                if length + 1e-9 < best:
                    stops, best, changed = candidate, length, True
    return stops


def _must_stops(views, observation):
    farm = observation['farms'][int(observation['player'])]
    stops = {}
    for view in views:
        for option in view.options:
            tile = tile_at(farm, option.tile_xy)
            if option.operation in ('WATER', 'FEED') and thirsty(tile) and option.tile_xy is not None:
                if job_cost(view, option, observation, views) < INF / 2:
                    stops.setdefault(option.tile_xy, option)
            elif option.operation == 'HARVEST' and harvestable(tile) and option.tile_xy is not None:
                stops.setdefault(('harvest', option.tile_xy), option)
            elif option.operation == 'DROP' and job_cost(view, option, observation, views) < INF / 2:
                stops.setdefault(('drop', view.index), option)
    return stops


def _seed_routes(views, must):
    routes = {view.index: [] for view in views}
    unused = list(must.values())
    unused.sort(key=lambda job: job.operation not in ('WATER', 'FEED'))
    for job in unused:
        target = job.tile_xy
        owner = min(views, key=lambda view: (
            _tour_length(view.xy, [s.tile_xy for s in routes[view.index]] + [target]),
            view.index))
        routes[owner.index].append(job)
    for view in views:
        xy = [job.tile_xy for job in routes[view.index]]
        order = _two_opt(view.xy, xy)
        lookup = {job.tile_xy: job for job in routes[view.index]}
        routes[view.index] = [lookup[point] for point in order if point in lookup]
    return routes


def tour_plan(views, observation, cache, day):
    if cache.get('tour_day') != day or cache.get('tour_crew') != len(views):
        routes = _seed_routes(views, _must_stops(views, observation))
        cache['tour_day'] = day
        cache['tour_crew'] = len(views)
        cache['tour_routes'] = routes
    routes = cache['tour_routes']
    plan = {}
    for view in views:
        route = [job for job in routes.get(view.index, ()) if job in view.options
                 or any(o.reservation_key == job.reservation_key for o in view.options)]
        if route:
            job = route[0]
            plan[view.index] = next((o for o in view.options if o.reservation_key == job.reservation_key or o == job),
                                    greedy_pick(view, observation, views))
        else:
            leftover = [o for o in view.options if o.operation in MUST or o.operation in GROW]
            plan[view.index] = greedy_pick(view, observation, views) if leftover or view.options else IDLE
    return plan


def cbba_plan(views, observation):
    """Centralized CBBA: bundle remaining-hour jobs, then conflict on bids."""
    remaining = {view.index: hours_left(observation) for view in views}
    bundles = {view.index: [] for view in views}

    def bid(view, option, bundle):
        last = view.xy if not bundle else (bundle[-1].tile_xy or view.xy)
        insert = manhattan(last, option.tile_xy or last) + 1
        urgency = max(0.1, INF / 10 - job_cost(view, option, observation, views))
        return urgency / insert, insert

    changed = True
    rounds = 0
    while changed and rounds < 12:
        changed = False
        rounds += 1
        claimed = {}
        for view in views:
            used = {job.reservation_key for owner, jobs in bundles.items() if owner != view.index for job in jobs}
            bundle, hours = [], remaining[view.index]
            while True:
                best = None
                for option in view.options:
                    if option.operation == 'PASS':
                        continue
                    if option.reservation_key in used:
                        continue
                    if any(job.reservation_key == option.reservation_key for job in bundle):
                        continue
                    score, insert = bid(view, option, bundle)
                    if insert > hours:
                        continue
                    if best is None or score > best[0]:
                        best = (score, insert, option)
                if best is None:
                    break
                _, insert, option = best
                bundle.append(option)
                hours -= insert
            if [job.reservation_key for job in bundles[view.index]] != [job.reservation_key for job in bundle]:
                changed = True
            bundles[view.index] = bundle
            for option in bundle:
                score, _ = bid(view, option, [])
                key = option.reservation_key
                winner = claimed.get(key)
                if winner is None or score > winner[0]:
                    claimed[key] = (score, view.index)
        for view in views:
            bundles[view.index] = [option for option in bundles[view.index]
                                   if claimed.get(option.reservation_key, (0, view.index))[1] == view.index]
    return {view.index: (bundles[view.index][0] if bundles[view.index]
                         else greedy_pick(view, observation, views)) for view in views}


def _min_makespan_routes(views, must):
    jobs = list(must.values())
    if not jobs:
        return {view.index: [] for view in views}
    if len(jobs) > 12 or len(views) > 6:
        return _seed_routes(views, must)
    best = None
    owners = range(len(views))
    # Assign each must-job to a worker; score 2-opt load.
    # Cap search: 3**8 = 6561, 4**8=65536; keep jobs*workers modest.
    if len(views) ** len(jobs) > 20000:
        return _seed_routes(views, must)
    for assignment in product(owners, repeat=len(jobs)):
        grouped = {view.index: [] for view in views}
        for job, owner in zip(jobs, assignment):
            grouped[views[owner].index].append(job)
        loads = []
        routes = {}
        for view in views:
            xy = [job.tile_xy for job in grouped[view.index]]
            order = _two_opt(view.xy, xy)
            lookup = {job.tile_xy: job for job in grouped[view.index]}
            routes[view.index] = [lookup[point] for point in order if point in lookup]
            loads.append(_tour_length(view.xy, [job.tile_xy for job in routes[view.index]]))
        peak = max(loads) if loads else 0
        slack = -sum(loads)
        key = (peak, slack)
        if best is None or key < best[0]:
            best = (key, routes)
    return best[1]


def cpsat_plan(views, observation, cache, day):
    """Exact-enough daily covering: min-makespan watering tours, then leftover."""
    if cache.get('cpsat_day') != day or cache.get('cpsat_crew') != len(views):
        must = {key: job for key, job in _must_stops(views, observation).items()
                if job.operation in ('WATER', 'FEED')}
        cache['cpsat_day'] = day
        cache['cpsat_crew'] = len(views)
        cache['cpsat_routes'] = _min_makespan_routes(views, must)
    routes = cache['cpsat_routes']
    plan = {}
    budget = hours_left(observation)
    for view in views:
        route = routes.get(view.index, [])
        pending = [job for job in route if any(o.reservation_key == job.reservation_key for o in view.options)]
        if pending and _tour_length(view.xy, [pending[0].tile_xy]) <= budget:
            job = pending[0]
            plan[view.index] = next(o for o in view.options if o.reservation_key == job.reservation_key)
            continue
        extra = [o for o in view.options if o.operation in ('HARVEST', 'DROP', 'PICKUP')
                 or (o.operation in GROW and capacity_ok(views, observation, extra=1))]
        plan[view.index] = greedy_pick(view, observation, views) if extra or view.options else IDLE
    return plan


@dataclass
class TurnPlan:
    selected: dict = field(default_factory=dict)


def plan_turn(strategy, observation, tasks, state, step, day, config=None):
    if strategy == 'learned':
        return TurnPlan()
    views = worker_views(observation, tasks, state.leases, step, day, config)
    if strategy == 'greedy':
        selected, taken = {}, set()
        for view in views:
            opts = tuple(option for option in view.options
                         if option.reservation_key is None or option.reservation_key not in taken)
            pick = greedy_pick(replace(view, options=opts or view.options), observation, views)
            selected[view.index] = pick
            if pick.reservation_key is not None:
                taken.add(pick.reservation_key)
    elif strategy == 'hungarian':
        selected = hungarian_plan(views, observation)
    elif strategy == 'tour':
        selected = tour_plan(views, observation, state.cache, day)
    elif strategy == 'cbba':
        selected = cbba_plan(views, observation)
    elif strategy == 'cpsat':
        selected = cpsat_plan(views, observation, state.cache, day)
    else:
        raise ValueError(f'Unknown assignment strategy: {strategy}')
    return TurnPlan(exclusive(selected, views, observation))


def exclusive(plan, views, observation):
    taken = set()
    ordered = {}
    by_index = {view.index: view for view in views}
    for index in sorted(plan):
        job = plan[index]
        view = by_index[index]
        if job.reservation_key is not None and job.reservation_key in taken:
            opts = tuple(option for option in view.options
                         if option.reservation_key is None or option.reservation_key not in taken)
            job = greedy_pick(replace(view, options=opts or (IDLE,)), observation, views)
        ordered[index] = job
        if job.reservation_key is not None:
            taken.add(job.reservation_key)
    return ordered


def greedy_action(cursor, selected, allowed):
    if not allowed:
        raise ValueError('Empty crew action mask')
    if selected.operation == 'PASS' or selected.tile_xy is None:
        return ['PASS'] if 'PASS' in allowed else [allowed[0]]
    here = tuple(cursor.rules._position())
    if here == selected.tile_xy and selected.operation in allowed:
        action = [selected.operation]
        items = cursor.items(selected.operation)
        if items:
            item = selected.item if selected.item in items else items[0]
            action.append(item)
            limit = cursor.quantity_limit(selected.operation, item)
            if limit:
                action.append(1)
        return action
    x, y = here
    tx, ty = selected.tile_xy
    ranked = []
    for op in allowed:
        if op not in STEP:
            continue
        dx, dy = STEP[op]
        ranked.append((abs(x + dx - tx) + abs(y + dy - ty), op))
    if ranked:
        return [min(ranked)[1]]
    if selected.operation in allowed:
        return greedy_action(cursor, selected, (selected.operation, *allowed))
    return ['PASS'] if 'PASS' in allowed else [allowed[0]]
