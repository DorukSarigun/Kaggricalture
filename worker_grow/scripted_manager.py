"""Translate a scripted full-farm agent into ISSUE + market for crew training.

Growing a plant is a crew job (farmer + hands). This module opens production
issues at the *ghost tape* tile (where the chassis would stand if it owned
units) and copies market orders from the live observation. Live crew feet are
not the pin. Learned manager tile heads are a later change.
"""
from copy import deepcopy
import importlib.util
from pathlib import Path

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as game

from worker_grow.issues import IssueBook, default_quota, production_pin
from worker_grow.schedule import live_animals

_CHASSIS = None
_MODULE = None
PASS = {"farmer": ["PASS"], "hands": [], "market": []}
PLANT_OPS = ("PLANT", "PLACE", "BUILD_COOP", "BUILD_PASTURE")


def _agent_module():
    load_best_agent_chassis()
    return _MODULE


def load_best_agent_chassis():
    global _CHASSIS, _MODULE
    if _CHASSIS is not None:
        return _CHASSIS
    path = Path(__file__).resolve().parents[1] / "third_party" / "kaggriculture_best_agent.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("kaggriculture_best_agent", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    chassis = getattr(module, "_IMPL", None)
    if chassis is None:
        raise RuntimeError("Best-agent chassis `_IMPL` is missing")
    _MODULE, _CHASSIS = module, chassis
    return _CHASSIS


def new_best_agent_chassis():
    module = _agent_module()
    return module.make_agent(module._ROUTES, router=module._router, **module._SETTINGS)


def best_agent_opponent(configuration=None):
    """Independent chassis per match; do not share the global `_IMPL` tape state."""
    chassis = new_best_agent_chassis()

    def opponent(observation):
        return deepcopy(chassis(observation, configuration))

    return opponent


def actor_positions(observation):
    farm = observation["farms"][int(observation["player"])]
    return [tuple(farm["farmer"]), *[tuple(hand) for hand in farm["hands"]]]


def actor_commands(action, count):
    commands = [list(action.get("farmer") or ["PASS"])]
    commands.extend(list(command) for command in (action.get("hands") or []))
    while len(commands) < count:
        commands.append(["PASS"])
    return commands[:count]


def _place_item(commands, actor):
    if actor >= len(commands):
        return None
    command = commands[actor]
    if command and command[0] == "PLACE" and len(command) > 1 and command[1] in game.ANIMALS:
        return command[1]
    return None


def _lookahead_place(action, actor, lookahead):
    now = actor_commands(action, max(actor + 1, 1 + len(action.get("hands") or ())))
    return _place_item(now, actor) or _place_item(lookahead or (), actor)


def _intent_pair(intent):
    if isinstance(intent, str):
        return intent, None
    item, tile = intent
    if tile is None:
        return item, None
    return item, (int(tile[0]), int(tile[1]))


def production_intents(observation, action, *, lookahead=None):
    """Product + the planning actor's tile. Caller chooses live vs ghost obs."""
    positions = actor_positions(observation)
    commands = actor_commands(action, len(positions))
    intents = []
    for index, command in enumerate(commands):
        if not command:
            continue
        op = command[0]
        item = None
        if op == "PLANT" and len(command) > 1:
            item = command[1]
        elif op == "PLACE" and len(command) > 1:
            item = command[1]
        elif op == "BUILD_COOP":
            item = "GOOSE"
        elif op == "BUILD_PASTURE":
            item = _lookahead_place(action, index, lookahead) or "SHEEP"
        if item not in (*game.CROPS, *game.ANIMALS):
            continue
        xy = positions[index]
        intents.append((item, (int(xy[0]), int(xy[1]))))
    return tuple(intents)


def apply_production_intents(observation, book: IssueBook, intents, *, step, day,
                             limit=None):
    opened = []
    for intent in intents:
        if limit is not None and len(book.open_issues()) >= limit:
            break
        item, tile_xy = _intent_pair(intent)
        if tile_xy is not None:
            occupied = book.occupied_tiles()
            tile_xy = production_pin(observation, occupied, tile_xy)
            if tile_xy is None:
                continue
        issue = book.issue(
            observation, item=item, tile_xy=tile_xy, decision_id=len(book.issues),
            step=step, day=day, quotas=default_quota(item),
        )
        opened.append(issue)
    return opened


def _wheat_stock(observation):
    private = observation["private"]
    shed = int((private.get("shed") or {}).get("WHEAT") or 0)
    carried = sum(int(inv.get("WHEAT") or 0) for inv in (private.get("inventories") or ()))
    return shed + carried


def protect_crew_market(observation, orders):
    """Keep enough shed wheat for live animals; BestAgent otherwise sells the feed."""
    animals = live_animals(observation)
    hungry = sum(1 for animal in animals if not animal.get("fed_today"))
    stock = _wheat_stock(observation)
    reserve = len(animals)
    protected = []
    selling = 0
    buying = 0
    for order in orders or ():
        if not order:
            continue
        command = list(order)
        op = command[0]
        item = command[1] if len(command) > 1 else None
        qty = int(command[2]) if len(command) > 2 else 1
        if op == "SELL" and item == "WHEAT" and reserve:
            qty = min(qty, max(0, stock - reserve - selling))
            if qty <= 0:
                continue
            command = ["SELL", "WHEAT", qty]
            selling += qty
        elif op == "BUY_PRODUCT" and item == "WHEAT":
            buying += qty
        protected.append(command)
    need = max(0, hungry - (stock - selling) - buying)
    if need:
        protected.insert(0, ["BUY_PRODUCT", "WHEAT", need])
    return protected


class BestAgentTeacher:
    """Chassis on a ghost tape for ISSUE tiles; live chassis for market only."""

    def __init__(self, chassis=None):
        self.chassis = chassis if chassis is not None else load_best_agent_chassis()
        self._key = None
        self._action = None
        self._ghost = None
        self._ghost_step = 0
        self._ghost_chassis = None

    def attach_world(self, configuration):
        self._ghost = make("kaggriculture", configuration=dict(configuration))
        self._ghost_step = 0
        self._key = None
        self._action = None
        self._ghost_chassis = (
            new_best_agent_chassis() if getattr(self.chassis, "chassis", None) is not None else None
        )

    def _planner(self):
        return self._ghost_chassis if self._ghost_chassis is not None else self.chassis

    def _planning_observation(self, live):
        if self._ghost is None:
            plan = deepcopy(live)
        else:
            plan = deepcopy(self._ghost.state[0].observation)
        plan["step"] = int(live.get("step", self._ghost_step))
        return plan

    def _lookahead(self, plan, step):
        chassis = getattr(self._planner(), "chassis", None)
        if chassis is None:
            return None
        player = int(plan["player"])
        route = chassis.players.get(player, {}).get("route", 0)
        tape = chassis.routes.get(route) or ()
        if 0 <= step + 1 < len(tape):
            return actor_commands(tape[step + 1], len(actor_positions(plan)))
        return None

    def observe(self, observation):
        key = (int(observation["player"]), int(observation.get("step", 0)))
        if self._key != key:
            self._action = deepcopy(self.chassis(observation))
            self._key = key
        return self._action

    def propose_contracts(self, observation, book: IssueBook, *, step, day, limit=None):
        plan = self._planning_observation(observation)
        action = deepcopy(self._planner()(plan))
        intents = production_intents(plan, action, lookahead=self._lookahead(plan, step))
        return apply_production_intents(observation, book, intents, step=step, day=day,
                                       limit=limit)

    def propose_market(self, observation):
        return protect_crew_market(observation, self.observe(observation).get("market") or [])

    def advance(self):
        if self._ghost is None or self._ghost.done:
            return
        obs = deepcopy(self._ghost.state[0].observation)
        obs["step"] = self._ghost_step
        action = deepcopy(self._planner()(obs))
        self._ghost.step([action, deepcopy(PASS)])
        if any(str(s.status) not in ("ACTIVE", "DONE") for s in self._ghost.state):
            raise RuntimeError("Ghost tape environment failure")
        self._ghost_step += 1
