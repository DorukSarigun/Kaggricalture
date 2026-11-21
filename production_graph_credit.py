"""Quantity-conserving material DAG and backwards, branch-split action credit.

Cash basis and action eligibility are separate graphs. Costs travel forward in
actual inventory quantities; discounted action weights travel BACKWARD from
each realized sale/loss. Bookkeeping nodes cause no discount. No potential,
notional sale, or future revenue estimate is used.

The default `tree` loss rule follows the plant/animal path we actually track:
BUY → PLANT/PLACE → WATER/FERTILIZE/FEED/CARE → HARVEST → move → SELL.
A realized sale is newest-heavy on that path. A planted/placed loss or leftover
harvest is oldest-heavy on the same path, so skipped care is not silent. Seed
or stock that never entered a producer stays on the BUY. HIRE/BUY_LAND stay
on their own cash decision. PASS, no-ops, and night auto-drop are not invented.

Joint-output convention: an input's cost is divided over the actual harvested /
collected units of that producer AFTER the input was consumed. Different output
types get equal per-unit basis. This is explicit accounting, not a claim about
causal marginal contributions. Unharvested output is not valued as inventory.
"""
from collections import defaultdict
from dataclasses import dataclass, field
import math

from lifecycle_tracking import LifecycleEventType as E

VERSION = "production-dag-v2"


@dataclass(frozen=True)
class GraphCreditConfig:
    decay: float = 0.8
    reward_scale: float = 3000.0
    branch_split: str = "equal_roles_quantity_within_role"
    production_cost_basis: str = "subsequent_harvested_units"
    unsold_weighting: str = "tree"

    def __post_init__(self):
        if not math.isfinite(self.decay) or not 0 < self.decay <= 1:
            raise ValueError("decay must be finite and in (0,1]")
        if not math.isfinite(self.reward_scale) or self.reward_scale <= 0:
            raise ValueError("reward_scale must be positive and finite")
        if self.branch_split != "equal_roles_quantity_within_role":
            raise ValueError("Unsupported branch split")
        if self.production_cost_basis != "subsequent_harvested_units":
            raise ValueError("Unsupported joint-output cost basis")
        if self.unsold_weighting not in ("tree", "oldest", "purchase_only"):
            raise ValueError("Unsupported unsold policy")


UNUSED_PURCHASE_ORIGINS = frozenset({"seed", "animal_stock", "purchase", "initial_seed"})


def logadd(a, b):
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    high, low = max(a, b), min(a, b)
    return high + math.log1p(math.exp(low-high))


class ActionDAG:
    """Contextual operation nodes; reward is finally merged by decision_id.

    Different items carried by one DROP must not acquire each other's history.
    Hence contextual nodes can share a decision ID. Equal contextual nodes are
    interned, branch messages merge before propagation, and fractions sum to 1.
    """
    def __init__(self):
        self.nodes = []
        self.cache = {}

    def node(self, decision_id, parents=(), event_id=None):
        weights = defaultdict(float)
        for parent, weight in parents:
            if parent is None:
                continue
            if not 0 <= parent < len(self.nodes) or not math.isfinite(weight) or weight <= 0:
                raise ValueError("Invalid action ancestry")
            weights[parent] += weight
        total = math.fsum(weights.values())
        normalized = tuple((p, w/total) for p, w in sorted(weights.items()))
        if decision_id is None and len(normalized) == 1:
            return normalized[0][0]  # a lot split/rename does not add a layer
        if decision_id is None and not normalized:
            return None
        if decision_id is not None and (not isinstance(decision_id, int) or decision_id < 0):
            raise ValueError("Invalid decision ID")
        key = (decision_id, normalized)
        if key in self.cache:
            return self.cache[key]
        index = len(self.nodes)
        self.nodes.append({"node_id": index, "decision_id": decision_id,
                           "parents": normalized, "event_id": event_id})
        self.cache[key] = index
        return index

    def shares(self, root, profit, decay, *, oldest=False):
        if root is None:
            if abs(profit) > 1e-9:
                raise ValueError("Nonzero credit without an action ancestor")
            return {}
        if not math.isfinite(profit):
            raise ValueError("Nonfinite realized profit")
        incoming, scores = {root: 0.0}, {}
        log_decay = -math.log(decay) if oldest else math.log(decay)
        # Parents always precede children; siblings cannot be discounted merely
        # because one decision happened earlier in the global episode ordering.
        for i in range(root, -1, -1):
            mass = incoming.get(i)
            if mass is None:
                continue
            node = self.nodes[i]
            decision = node["decision_id"]
            if decision is not None:
                scores[decision] = logadd(scores.get(decision, -math.inf), mass)
                mass += log_decay
            for parent, fraction in node["parents"]:
                incoming[parent] = logadd(incoming.get(parent, -math.inf), mass+math.log(fraction))
        if not scores:
            raise ValueError("No actionable ancestor")
        peak = max(scores.values())
        weights = {d: math.exp(w-peak) for d, w in scores.items()}
        total = math.fsum(weights.values())
        result = {d: profit*w/total for d, w in sorted(weights.items())}
        largest = max(weights, key=weights.get)
        result[largest] += profit-math.fsum(result.values())
        return result


@dataclass
class Stock:
    node: int
    quantity: int
    remaining: int
    head: int | None
    item: str
    origin: str | None = None


@dataclass
class Producer:
    head: int | None
    kind: str
    # (event sequence, consumed material node)
    inputs: list = field(default_factory=list)
    # (event sequence, harvested material node, quantity)
    outputs: list = field(default_factory=list)


class ProductionGraph:
    def __init__(self, config=None):
        self.config = config or GraphCreditConfig()
        self.actions = ActionDAG()
        self.nodes, self.edges = [], defaultdict(list)
        self.stocks, self.producers, self.sinks = {}, {}, []
        self.purchases, self.sale_totals = {}, defaultdict(float)
        self.pending_walk, self.operation_walk = {}, {}
        self.prepared, self.structures = {}, {}
        self.sequence = 0
        self.unknown_source_sale_revenue = 0.0
        self.finished = False

    def issue_action_parent(self, identity):
        """Manager ISSUE node on the producer path. Default: production DAG has none."""
        return None

    def cost_node(self, kind, sources=None, **metadata):
        index = len(self.nodes)
        self.nodes.append({"node_id": index, "kind": kind, "sources": sources or {}, **metadata})
        return index

    def edge(self, source, target, fraction, **metadata):
        if source >= target or not math.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError("Noncausal or invalid material edge")
        self.edges[source].append({"target": target, "fraction": fraction, **metadata})

    def stock(self, key, quantity, head, item, *, sources=None, node=None, origin=None):
        if key in self.stocks or not isinstance(quantity, int) or quantity <= 0:
            raise ValueError(f"Duplicate stock or invalid quantity: {key}, {quantity}")
        node = self.cost_node("material", sources, lot_id=key, item=item, quantity=quantity) if node is None else node
        self.stocks[key] = Stock(node, quantity, quantity, head, item, origin)
        return node

    def take(self, key, quantity):
        if key not in self.stocks:
            raise ValueError(f"Missing material provenance: {key}")
        stock = self.stocks[key]
        if not isinstance(quantity, int) or not 0 < quantity <= stock.remaining:
            raise ValueError(f"Material over-consumption: {key}, {quantity}/{stock.remaining}")
        node = self.cost_node("portion", lot_id=key, item=stock.item, quantity=quantity)
        self.edge(stock.node, node, quantity/stock.quantity, quantity=quantity,
                  reason="actual_quantity_split")
        stock.remaining -= quantity
        return node, stock.head, stock.item, stock.origin

    def walk(self, e, unit=None):
        if e.decision_id is None:
            return None
        unit = e.data.get("unit_index") if unit is None else unit
        if unit is None:
            return None
        if e.decision_id not in self.operation_walk:
            self.operation_walk[e.decision_id] = self.pending_walk.pop((e.day, unit), None)
        return self.operation_walk[e.decision_id]

    def operation(self, e, *roles):
        # Equal known input roles; each material role itself is quantity-weighted.
        parents = [(head, 1.0) for head in roles if head is not None]
        return self.actions.node(e.decision_id, parents, e.event_id)

    def material_role(self, pieces):
        return self.actions.node(None, [(head, quantity) for _, head, quantity in pieces])

    def register_purchase(self, e):
        if e.decision_id is None or e.event_id in self.purchases:
            raise ValueError("Purchase must have a unique event and decision")
        cost = float(e.data["total_cost"])
        if not math.isfinite(cost) or cost < 0:
            raise ValueError("Invalid cash cost")
        self.purchases[e.event_id] = {"cost": cost, "decision_id": e.decision_id}
        return self.operation(e)

    def ensure_unknown(self, allocation):
        key = allocation["product_lot_id"]
        if key not in self.stocks:
            if not allocation.get("reconciled"):
                raise ValueError(f"Missing tracked product: {key}")
            self.stock(key, int(allocation["quantity"]), None, "UNKNOWN", origin=None)

    def consume_allocations(self, allocations):
        pieces = []
        for a in allocations:
            self.ensure_unknown(a)
            quantity = int(a["quantity"])
            node, head, _, _ = self.take(a["product_lot_id"], quantity)
            pieces.append((node, head, quantity))
        return pieces

    def producer(self, e):
        key = e.plant_id or e.animal_id
        if key not in self.producers:
            raise ValueError(f"Missing producer: {key}")
        return self.producers[key]

    def loss(self, node, head, kind, **metadata):
        sink = self.cost_node("loss", **metadata)
        self.edge(node, sink, 1.0, reason=kind)
        self.sinks.append({"kind": kind, "node_id": sink, "action_root": head,
                           "revenue": 0.0, **metadata})

    def observe(self, e):
        if self.finished:
            raise ValueError("Graph is already settled")
        self.sequence += 1
        seq, kind = self.sequence, e.event_type
        if kind == E.SEED_PURCHASED:
            head = self.register_purchase(e)
            self.stock(e.seed_lot_id, int(e.quantity), head, e.crop,
                       sources={e.event_id: float(e.data["total_cost"])}, origin="seed")
        elif kind == E.INITIAL_SEED_LOT:
            self.stock(e.seed_lot_id, int(e.quantity), None, e.crop, origin="initial_seed")
        elif kind == E.ANIMAL_PURCHASED:
            head = self.register_purchase(e)
            lots = e.data["animal_lot_ids"]
            if not lots or len(lots) != e.quantity:
                raise ValueError("Animal purchase quantities do not match")
            for lot in lots:
                self.stock(lot, 1, head, e.data["animal"],
                           sources={e.event_id: e.data["total_cost"]/len(lots)}, origin="animal_stock")
        elif kind == E.PRODUCT_PURCHASED:
            head = self.register_purchase(e)
            lots, prices = e.data["product_lot_ids"], e.data["unit_prices"]
            if len(lots) != len(prices) or len(lots) != e.quantity or not math.isclose(
                    math.fsum(prices), e.data["total_cost"], rel_tol=0, abs_tol=1e-7):
                raise ValueError("Product purchase quantities/prices do not match")
            for lot, price in zip(lots, prices):
                self.stock(lot, 1, head, e.data["item"], sources={e.event_id: float(price)}, origin="purchase")
        elif kind == E.UNIT_MOVED:
            if e.decision_id is None:
                raise ValueError("Movement without decision")
            key = (e.day, e.data["unit_index"])
            self.pending_walk[key] = self.operation(e, self.pending_walk.get(key))
            for lot in (*e.data.get("carried_product_lot_ids", ()), *e.data.get("carried_animal_lot_ids", ())):
                stock = self.stocks[lot]
                if stock.remaining <= 0:
                    raise ValueError("Moving empty lot")
                stock.head = self.operation(e, stock.head)
        elif kind == E.TILE_CLEARED:
            self.prepared[e.tile_xy] = self.operation(e, self.walk(e))
        elif kind == E.STRUCTURE_BUILT:
            self.structures[e.tile_xy] = self.operation(e, self.prepared.pop(e.tile_xy, None), self.walk(e))
        elif kind in (E.PLANTED, E.ANIMAL_PLACED):
            key = e.plant_id or e.animal_id
            if key in self.producers:
                raise ValueError("Producer identity reused")
            lot = e.seed_lot_id if kind == E.PLANTED else e.animal_lot_id
            preparation = (self.prepared.pop(e.tile_xy, None) if kind == E.PLANTED
                           else self.structures.get(e.tile_xy))
            inputs = []
            material_head = None
            if lot is not None:
                node, material_head, _, _ = self.take(lot, 1)
                inputs = [(seq, node)]
            elif e.data.get("source") != "observed_existing_tile":
                raise ValueError("New producer without input identity")
            head = self.operation(e, material_head, preparation, self.walk(e),
                                  self.issue_action_parent(key))
            self.producers[key] = Producer(head, "plant" if kind == E.PLANTED else "animal", inputs)
        elif kind in (E.WATERED, E.FERTILIZED, E.ANIMAL_FED, E.ANIMAL_CARED):
            producer = self.producer(e)
            pieces = self.consume_allocations(e.data.get("consumed_allocations", ()))
            producer.head = self.operation(e, producer.head, self.material_role(pieces), self.walk(e))
            producer.inputs.extend((seq, node) for node, _, _ in pieces)
        elif kind in (E.HARVESTED, E.FERTILIZER_COLLECTED):
            producer = self.producer(e)
            head = self.operation(e, producer.head, self.walk(e))
            item = ("FERTILIZER" if kind == E.FERTILIZER_COLLECTED
                    else e.crop or e.data["product"])
            node = self.stock(e.product_lot_id, int(e.quantity), head, item,
                              origin=producer.kind+":"+(e.plant_id or e.animal_id))
            producer.outputs.append((seq, node, int(e.quantity)))
            # HARVEST frees tile capacity for later yields; retain that operation
            # in the producer's next state. Existing product heads stay frozen.
            producer.head = head
        elif kind in (E.PLANT_LOST, E.ANIMAL_LOST):
            producer = self.producer(e)
            producer.head = self.operation(e, producer.head, self.walk(e))
        elif kind == E.ANIMAL_STOCK_MOVED:
            quantity = int(e.quantity)
            source, target = e.data["source_lot_id"], e.data["destination_lot_id"]
            node, head, item, origin = self.take(source, quantity)
            location = next((e.data[k] for k in ("source", "destination")
                             if e.data[k].startswith("unit:")), None)
            walk = self.walk(e, int(location.split(":")[1])) if location else None
            self.stock(target, quantity, self.operation(e, head, walk), item, node=node, origin=origin)
        elif kind in (E.PRODUCT_MOVED, E.INVENTORY_RECONCILED):
            location = next((e.data.get(k, "") for k in ("source", "destination")
                             if e.data.get(k, "").startswith("unit:")), None)
            walk = self.walk(e, int(location.split(":")[1])) if location else None
            if e.data.get("source") == "external_or_unobserved" and e.product_lot_id:
                self.stock(e.product_lot_id, int(e.quantity), None, e.crop or "UNKNOWN")
            if e.data.get("source") == "external_or_unobserved" and e.animal_lot_id:
                self.stock(e.animal_lot_id, int(e.quantity), None, e.data["animal"])
            for a in e.data.get("allocations", ()):
                self.ensure_unknown(a)
                quantity = int(a["quantity"])
                node, head, item, origin = self.take(a["product_lot_id"], quantity)
                head = self.operation(e, head, walk)
                target = a.get("destination_product_lot_id")
                if target:
                    self.stock(target, quantity, head, item, node=node, origin=origin)
                else:
                    self.loss(node, head, "lost_or_consumed", item=item, quantity=quantity)
            for a in e.data.get("animal_allocations", ()):
                quantity = int(a["quantity"])
                node, head, item, _ = self.take(a["animal_lot_id"], quantity)
                self.loss(node, self.operation(e, head, walk), "lost_animal_stock",
                          item=item, quantity=quantity)
        elif kind == E.PRODUCT_SOLD:
            if e.decision_id is None or e.data.get("revenue_is_estimate", True):
                raise ValueError("Only actual executed sales can produce graph reward")
            total, quantity = 0.0, 0
            # A single market order may consume independently profitable and
            # loss-making lots. Do not smear the order's average profit over
            # unrelated histories. Merge unit executions only within a lot.
            groups = {}
            for a in e.data["allocations"]:
                self.ensure_unknown(a)
                q, revenue = int(a["quantity"]), float(a["realized_revenue"])
                if q <= 0 or not math.isfinite(revenue) or revenue < 0:
                    raise ValueError("Invalid executed revenue")
                group = groups.setdefault(a["product_lot_id"], [0, 0.0])
                group[0] += q
                group[1] += revenue
                total += revenue
                quantity += q
            if quantity != e.quantity or not math.isclose(total, e.attributed_value, rel_tol=0, abs_tol=1e-7):
                raise ValueError("Executed sale allocation mismatch")
            for lot, (q, revenue) in groups.items():
                node, head, item, origin = self.take(lot, q)
                head = self.operation(e, head)
                sink = self.cost_node("sale", event_id=e.event_id, lot_id=lot, quantity=q)
                self.edge(node, sink, 1.0, reason="sold")
                self.sinks.append({"kind": "sale", "node_id": sink, "action_root": head,
                    "revenue": revenue, "quantity": q, "event_id": e.event_id, "lot_id": lot,
                    "origins": [origin or "unknown"]})
                if origin is None:
                    self.unknown_source_sale_revenue += revenue
            self.sale_totals[e.decision_id] += total

    def _purchase_loss_shares(self, basis):
        shares = defaultdict(float)
        for purchase, amount in basis.items():
            shares[self.purchases[purchase]["decision_id"]] -= amount
        return dict(shares)

    def _uses_purchase_only_loss(self, sink):
        if sink["kind"] == "sale":
            return False
        if self.config.unsold_weighting == "purchase_only":
            return True
        if self.config.unsold_weighting == "oldest":
            return False
        # tree: unused bought stock stays on BUY; once it is planted, placed,
        # harvested, fed, or otherwise produced, the loss follows the path.
        return sink["kind"] == "unsold_inventory" and sink.get("origin") in UNUSED_PURCHASE_ORIGINS

    def _sink_shares(self, sink, profit, basis):
        if self._uses_purchase_only_loss(sink):
            return self._purchase_loss_shares(basis)
        return self.actions.shares(sink["action_root"], profit, self.config.decay,
                                   oldest=sink["kind"] != "sale")

    def finish(self, cash_entries, decision_refs, player):
        if self.finished:
            raise ValueError("Cannot settle the same graph twice")
        self.finished = True
        # Each material input routes its basis only into subsequent acquisition
        # batches. Time-stamped outputs avoid entity-identity cycles (e.g. wheat
        # -> animal fertilizer -> wheat -> same animal, on later dates).
        for identity, producer in self.producers.items():
            for seq, node in producer.inputs:
                outputs = [(target, q) for at, target, q in producer.outputs if at > seq]
                units = sum(q for _, q in outputs)
                if units:
                    for target, q in outputs:
                        self.edge(node, target, q/units, reason="joint_output_basis", producer_id=identity)
                else:
                    self.loss(node, producer.head, "unrealized_input", producer_id=identity)
        # Follow final portions, not only the original harvest lot. Every unsold
        # DROP/PICKUP descendant retains its complete contextual action history.
        for key, stock in list(self.stocks.items()):
            if stock.remaining:
                q = stock.remaining
                node, head, item, origin = self.take(key, q)
                self.loss(node, head, "unsold_inventory", lot_id=key, item=item,
                          quantity=q, origin=origin)

        own = [e for e in cash_entries if e.player == player]
        actual_purchases, actual_sales = defaultdict(float), defaultdict(float)
        for entry in own:
            if not entry.amount:
                continue
            decision = decision_refs[(entry.step, "market", entry.order_index)]
            if entry.operation == "SELL":
                actual_sales[decision] += entry.amount
            elif entry.operation in {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL"}:
                actual_purchases[decision] -= entry.amount
            elif entry.operation in {"HIRE", "BUY_LAND"}:
                key = f"cash:{entry.step}:{entry.order_index}"
                cost = -entry.amount
                self.purchases[key] = {"cost": cost, "decision_id": decision}
                node = self.cost_node("capital_or_labor", {key: cost}, operation=entry.operation)
                self.loss(node, self.actions.node(decision), "direct_cost", operation=entry.operation)
            else:
                raise ValueError(f"Unaccounted cash operation: {entry.operation}")
        expected_purchases = defaultdict(float)
        for key, purchase in self.purchases.items():
            if not key.startswith("cash:"):
                expected_purchases[purchase["decision_id"]] += purchase["cost"]
        for actual, expected in ((actual_purchases, expected_purchases), (actual_sales, self.sale_totals)):
            if set(actual) != set(expected) or any(abs(actual[k]-expected[k]) > 1e-7 for k in actual):
                raise ValueError("Per-decision journal does not match actual cash executions")

        balances = [defaultdict(float, n["sources"]) for n in self.nodes]
        sink_ids = {s["node_id"] for s in self.sinks}
        for i, outgoing in ((i, self.edges.get(i, [])) for i in range(len(self.nodes))):
            fraction = math.fsum(e["fraction"] for e in outgoing)
            if i not in sink_ids and not math.isclose(fraction, 1.0, rel_tol=0, abs_tol=1e-9):
                raise ValueError(f"Incomplete material accounting at node {i}: {fraction}")
            if i in sink_ids and outgoing:
                raise ValueError("Terminal material node has outgoing edges")
            for edge in outgoing:
                for purchase, amount in balances[i].items():
                    balances[edge["target"]][purchase] += amount*edge["fraction"]
        credits, claimed = defaultdict(float), defaultdict(float)
        allocations = []
        for sink in self.sinks:
            basis = balances[sink["node_id"]]
            cost = math.fsum(basis.values())
            for purchase, amount in basis.items():
                claimed[purchase] += amount
            profit = sink["revenue"]-cost
            shares = self._sink_shares(sink, profit, basis)
            for decision, amount in shares.items():
                credits[decision] += amount
            allocations.append({**sink, "cost": cost, "cost_sources": dict(basis), "net_profit": profit,
                "shares": [{"decision_id": d, "gold": amount} for d, amount in sorted(shares.items())]})
        for purchase, entry in self.purchases.items():
            if abs(claimed[purchase]-entry["cost"]) > 1e-7:
                raise ValueError(f"Purchase cost not conserved: {purchase}")
        net_cash = math.fsum(e.amount for e in own)
        total = math.fsum(credits.values())
        if abs(total-net_cash) > 1e-7:
            raise ValueError(f"Graph credit cash mismatch: {total} != {net_cash}")
        return {"version": VERSION, "reward_scale": self.config.reward_scale,
            "decay": self.config.decay, "decay_unit": "backward_action_edge",
            "branch_split": self.config.branch_split, "production_cost_basis": self.config.production_cost_basis,
            "unsold_weighting": self.config.unsold_weighting,
            "allocated_gold": total, "net_cash": net_cash, "reconciliation_error": total-net_cash,
            "unknown_source_sale_revenue": self.unknown_source_sale_revenue,
            "decision_credit": {d: amount/self.config.reward_scale for d, amount in credits.items()},
            "allocations": allocations,
            "graph": {"version": VERSION, "action_nodes": self.actions.nodes,
                      "material_nodes": self.nodes, "material_edges": dict(self.edges),
                      "purchase_cost_audit": {p: {**v, "allocated_cost": claimed[p]} for p, v in self.purchases.items()}}}


def assign_graph_credit(events, cash_entries, decision_refs, player, config=None):
    graph = ProductionGraph(config)
    previous = -1
    seen = set()
    for e in events:
        if e.player != player or e.step < previous or e.event_id in seen:
            raise ValueError("Journal is not chronological or belongs to another player")
        seen.add(e.event_id)
        previous = e.step
        graph.observe(e)
    return graph.finish(cash_entries, decision_refs, player)
