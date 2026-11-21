"""Realized net profit follows the action DAG to manager ancestors.

Money/quantity conservation is delegated to ProductionGraph. Each sink's
`revenue - carried cost` is gold. That gold is shared along the lot's action
path among manager decisions (BUY, the ISSUE that bound the producer, SELL).
Crew nodes are not paid. ISSUE is attached at plant/place; leftover cost stays
on the same path. No market price is used to value unsold products.
"""
from collections import defaultdict
import math

from lifecycle_tracking import LifecycleEventType as E
from production_graph_credit import ProductionGraph, GraphCreditConfig

VERSION = "issue-sale-dag-v2"


class IssueDAG:
    def __init__(self):
        self.nodes = []
        self.cache = {}

    def leaf(self, issue_id):
        key = ("issue", issue_id)
        if key not in self.cache:
            self.cache[key] = len(self.nodes)
            self.nodes.append({"node_id": len(self.nodes), "issue_id": issue_id, "children": []})
        return self.cache[key]

    def fork(self, children, *, lot_id=None):
        # Splitting the same material into two lots must not increase its vote.
        children = tuple(sorted(set(children)))
        if not children or any(type(i) is not int or not 0 <= i < len(self.nodes) for i in children):
            raise ValueError("Invalid/noncausal issue ancestry")
        if len(children) == 1 and lot_id is None:
            return children[0]
        # Distinct harvest batches are distinct intermediate products even if
        # they originate from the same issue. Only splits of ONE batch merge.
        key = ("fork", children, lot_id)
        if key not in self.cache:
            self.cache[key] = len(self.nodes)
            self.nodes.append({"node_id": len(self.nodes), "issue_id": None,
                               "children": list(children), "lot_id": lot_id})
        return self.cache[key]

    def shares(self, root, gold):
        if type(root) is not int or not 0 <= root < len(self.nodes) or not math.isfinite(gold):
            raise ValueError("Invalid issue reward root/budget")
        incoming, shares = {root: gold}, defaultdict(float)
        # Children predate parents, including fertilizer -> crop -> same animal
        # on a later harvest. Merge shared ancestors before propagating.
        for i in range(root, -1, -1):
            if i not in incoming:
                continue
            node, amount = self.nodes[i], incoming[i]
            if node["issue_id"] is not None:
                shares[node["issue_id"]] += amount
            else:
                for child in node["children"]:
                    incoming[child] = incoming.get(child, 0.) + amount / len(node["children"])
        if not shares:
            raise ValueError("Issue tree has no leaves")
        largest = max(shares, key=lambda k: abs(shares[k]))
        shares[largest] += gold - math.fsum(shares.values())
        return dict(shares)


class IssueProductionGraph(ProductionGraph):
    def __init__(self, issues, *, trade_decay=1.):
        # Gold stays gold. Net profit at each sink follows the action path,
        # including the ISSUE that bound the producer. Crew nodes are not paid.
        super().__init__(GraphCreditConfig(decay=trade_decay, reward_scale=1.0, unsold_weighting="tree"))
        self.issues = issues
        self.issue_dag = IssueDAG()
        self.producer_issues = {}
        self.input_roots = {}
        self.lot_roots = {}
        self.issue_gold = defaultdict(float)
        self._seen = set()
        self._last_step = -1
        self._player = None
        self._issue_decisions = {}
        for issue in issues.values():
            if issue.producer_id:
                if issue.producer_id in self.producer_issues:
                    raise ValueError("Producer assigned to multiple issues")
                self.producer_issues[issue.producer_id] = issue.task_id
            self._issue_decisions[2 * issue.decision_id] = issue.task_id

    def issue_action_parent(self, identity):
        if identity not in self.producer_issues:
            raise ValueError(f"Producer without a verified ISSUE: {identity}")
        return self.actions.node(2 * self.issues[self.producer_issues[identity]].decision_id)

    def observe(self, e):
        if e.event_id in self._seen or e.step < self._last_step:
            raise ValueError("Duplicate or unordered issue lifecycle event")
        if self._player is not None and self._player != e.player:
            raise ValueError("Mixed players in issue DAG")
        self._seen.add(e.event_id)
        self._last_step, self._player = e.step, e.player
        kind = e.event_type
        identity = e.plant_id or e.animal_id
        if kind in (E.PLANTED, E.ANIMAL_PLACED):
            if identity not in self.producer_issues:
                raise ValueError(f"Producer without a verified ISSUE: {identity}")
            issue = self.issues[self.producer_issues[identity]]
            if issue.tile_xy is None or tuple(e.tile_xy) != tuple(issue.tile_xy) or e.step < issue.opened_step:
                raise ValueError("Producer/ISSUE tile or time mismatch")
            if e.crop and e.crop != issue.item:
                raise ValueError("Producer/ISSUE crop mismatch")
            if e.data.get("animal") and e.data["animal"] != issue.item:
                raise ValueError("Producer/ISSUE animal mismatch")
            self.input_roots[identity] = set()
        elif kind in (E.FERTILIZED, E.ANIMAL_FED):
            for allocation in e.data.get("consumed_allocations", ()):
                root = self.lot_roots.get(allocation["product_lot_id"])
                if root is not None:
                    self.input_roots[identity].add(root)
        # Existing quantity/cash checks remain authoritative. Missing provenance
        # is never replaced with a guessed profit or a guessed issue.
        super().observe(e)
        if kind in (E.HARVESTED, E.FERTILIZER_COLLECTED):
            leaf = self.issue_dag.leaf(self.producer_issues[identity])
            self.lot_roots[e.product_lot_id] = self.issue_dag.fork(
                [leaf, *self.input_roots[identity]], lot_id=e.product_lot_id)
        elif kind in (E.PRODUCT_MOVED, E.INVENTORY_RECONCILED):
            for allocation in e.data.get("allocations", ()):
                target = allocation.get("destination_product_lot_id")
                if target:
                    self.lot_roots[target] = self.lot_roots.get(allocation["product_lot_id"])

    def _manager_path_shares(self, sink, profit):
        raw = self.actions.shares(sink["action_root"], 1., self.config.decay)
        weights = {d: w for d, w in raw.items() if d % 2 == 0}
        total = math.fsum(weights.values())
        if total <= 0:
            raise ValueError("Path has no manager ancestor")
        shares = {d: profit * w / total for d, w in weights.items()}
        largest = max(shares, key=lambda d: abs(shares[d]))
        shares[largest] += profit - math.fsum(shares.values())
        return shares

    def _sink_shares(self, sink, profit, basis):
        shares = self._manager_path_shares(sink, profit)
        produced = any(str(origin).startswith(("plant:", "animal:")) for origin in sink.get("origins", ()))
        if produced and not any(d in self._issue_decisions for d in shares):
            raise ValueError("Produced sink lost its ISSUE ancestor")
        if sink["kind"] == "sale" and "unknown" in sink.get("origins", ()):
            raise ValueError("Cannot reward sale with unknown material provenance")
        sink["credit_route"] = "action_path"
        issue_shares = defaultdict(float)
        for decision, amount in shares.items():
            if decision in self._issue_decisions:
                key = self._issue_decisions[decision]
                self.issue_gold[key] += amount
                issue_shares[key] += amount
        if issue_shares:
            sink["issue_shares"] = dict(issue_shares)
        return shares

    def finish(self, cash_entries, decision_refs, player):
        report = super().finish(cash_entries, decision_refs, player)
        if any(d % 2 for d in report["decision_credit"]):
            raise ValueError("Economic reward leaked into crew namespace")
        report["material_engine_version"] = report["version"]
        report["version"] = VERSION
        report["decision_credit"] = {d // 2: v for d, v in report["decision_credit"].items()}
        report["issue_gold"] = {key: self.issue_gold.get(key, 0.) for key in self.issues}
        report["issue_rule"] = "net_profit_follows_action_path"
        report["graph"]["issue_nodes"] = self.issue_dag.nodes
        report["graph"]["lot_issue_roots"] = self.lot_roots
        report["graph"]["producer_issues"] = self.producer_issues
        report["graph"]["producer_outputs"] = {
            identity: [{"sequence": at, "material_node": node,
                        "lot_id": self.nodes[node]["lot_id"], "quantity": quantity}
                       for at, node, quantity in producer.outputs]
            for identity, producer in self.producers.items()
        }
        report["graph"]["decision_namespace"] = "manager=2*decision_id; crew=2*decision_id+1"
        return report


def assign_issue_credit(events, cash_entries, decision_refs, player, issues):
    graph = IssueProductionGraph(issues)
    for event in events:
        if event.player != player:
            raise ValueError("Wrong player in issue journal")
        graph.observe(event)
    return graph.finish(cash_entries, decision_refs, player)
