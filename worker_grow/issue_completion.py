"""Actual issue completion and non-cash failure sinks.

Material quantities follow verified lot IDs, never inferred sale prices or
counterfactual yields. Cash credit remains exclusively in issue_credit.py.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import math

from kaggle_environments.envs.kaggriculture import kaggriculture as game
from lifecycle_tracking import LifecycleEventType as E
from production_graph_credit import ActionDAG

VERSION = 'issue-completion-dag-v1'
THROUGHPUT_VERSION = 'issue-throughput-v1'
SETTLEMENT_RULES = ('unfulfilled_fraction', 'preserve')
DEFAULT_FULFILL_REWARD = 1.0
DEFAULT_UNMET_REWARD = 1.0


@dataclass
class OutputPortion:
    item: str
    issue_id: str | None
    quantity: int
    fulfilled: bool = False


class IssueMaterialAudit:
    """Quantity-only parallel projection of the already validated lifecycle."""
    def __init__(self, issues):
        self.issues = issues
        self.owners = {i.producer_id: i.task_id for i in issues.values() if i.producer_id}
        if len(self.owners) != sum(i.producer_id is not None for i in issues.values()):
            raise ValueError('Producer assigned to multiple issues')
        self.lots = {}
        self.counts = defaultdict(lambda: defaultdict(Counter))
        self.inputs = defaultdict(Counter)
        self.events = defaultdict(list)
        self.seen = set()
        self.last_step = -1

    def create(self, lot, portion):
        if not lot or lot in self.lots or portion.quantity <= 0:
            raise ValueError('Invalid or reused completion lot')
        self.lots[lot] = portion

    def take(self, allocation):
        lot, quantity = allocation['product_lot_id'], int(allocation['quantity'])
        if lot not in self.lots:
            if not allocation.get('reconciled'):
                raise ValueError(f'Completion provenance missing: {lot}')
            self.create(lot, OutputPortion('UNKNOWN', None, quantity))
        portion = self.lots[lot]
        if not 0 < quantity <= portion.quantity:
            raise ValueError('Completion material over-consumption')
        portion.quantity -= quantity
        return replace(portion, quantity=quantity)

    def count(self, portion, stage):
        if portion.issue_id is not None:
            self.counts[portion.issue_id][portion.item][stage] += portion.quantity

    def fulfill(self, portion):
        if not portion.fulfilled:
            self.count(portion, 'fulfilled_unique')
            portion.fulfilled = True

    def observe(self, event):
        if event.event_id in self.seen or event.step < self.last_step:
            raise ValueError('Duplicate or unordered completion event')
        self.seen.add(event.event_id); self.last_step = event.step
        kind = event.event_type
        identity = event.plant_id or event.animal_id
        owner = self.owners.get(identity)
        if owner is not None:
            self.events[owner].append(event.event_id)
        if kind == E.PRODUCT_PURCHASED:
            for lot in event.data['product_lot_ids']:
                self.create(lot, OutputPortion(event.data['item'], None, 1))
        elif kind in (E.PLANTED, E.ANIMAL_PLACED):
            if owner is None:
                raise ValueError('Producer without completion ISSUE')
            if event.data.get('source') != 'observed_existing_tile':
                resource = 'seed:' + event.crop if kind == E.PLANTED else 'animal:' + self.issues[owner].item
                self.inputs[owner][resource] += 1
        elif kind in (E.HARVESTED, E.FERTILIZER_COLLECTED):
            if owner is None:
                raise ValueError('Output without completion ISSUE')
            item = ('FERTILIZER' if kind == E.FERTILIZER_COLLECTED
                    else event.crop or event.data['product'])
            portion = OutputPortion(item, owner, int(event.quantity))
            self.create(event.product_lot_id, portion)
            self.count(portion, 'harvested')
        elif kind in (E.ANIMAL_FED, E.FERTILIZED):
            if owner is None:
                raise ValueError('Consumption without completion ISSUE')
            resource = 'WHEAT' if kind == E.ANIMAL_FED else 'FERTILIZER'
            for allocation in event.data.get('consumed_allocations', ()):
                portion = self.take(allocation)
                self.inputs[owner][resource] += portion.quantity
                self.count(portion, 'consumed')
                self.fulfill(portion)
        elif kind in (E.PRODUCT_MOVED, E.INVENTORY_RECONCILED):
            if event.data.get('source') == 'external_or_unobserved' and event.product_lot_id:
                self.create(event.product_lot_id, OutputPortion(event.crop or 'UNKNOWN', None,
                                                               int(event.quantity)))
            for allocation in event.data.get('allocations', ()):
                portion = self.take(allocation)
                target = allocation.get('destination_product_lot_id')
                if target:
                    if event.data.get('destination') == 'shed':
                        self.fulfill(portion)
                    self.create(target, portion)
                else:
                    self.count(portion, 'lost')
        elif kind == E.PRODUCT_SOLD:
            for allocation in event.data['allocations']:
                portion = self.take(allocation)
                self.count(portion, 'sold')
                self.fulfill(portion)

    def report(self, book, observation):
        counts = {issue: {item: Counter(values) for item, values in products.items()}
                  for issue, products in self.counts.items()}
        for portion in self.lots.values():
            if portion.issue_id is not None:
                counts[portion.issue_id][portion.item]['held'] += portion.quantity
        farm = observation['farms'][int(observation['player'])]
        result = {}
        for key, issue in self.issues.items():
            record = book.task_book.records[key]
            products = counts.setdefault(key, {})
            ready = {}
            if record.producer_id and not record.production_closed and record.tile_xy is not None:
                x, y = record.tile_xy; tile = farm['tiles'][y][x]
                if isinstance(tile, dict):
                    if issue.item in game.CROPS and tile.get('crop') == issue.item:
                        # yield_units alone is not proof of harvest readiness.
                        if int(observation['day']) - int(tile['planted_day']) >= game.CROPS[issue.item]['first_yield_day']:
                            ready[issue.item] = int(tile.get('yield_units', 0))
                    elif tile.get('animal') == issue.item:
                        ready[game.ANIMALS[issue.item]['product']] = int(tile.get('yield_units', 0))
                        ready['FERTILIZER'] = int(bool(tile.get('fertilizer_available')))
            for item in set(products) | set(dict(issue.quotas)) | set(ready):
                values = products.setdefault(item, Counter())
                for stage in ('harvested', 'sold', 'consumed', 'held', 'lost', 'fulfilled_unique'):
                    values.setdefault(stage, 0)
                if values['harvested'] != sum(values[k] for k in ('sold', 'consumed', 'held', 'lost')):
                    raise ValueError('Issue output quantities do not conserve')
                values['ready_unharvested'] = ready.get(item, 0)
                quota = dict(issue.quotas).get(item, 0)
                values['requested'] = quota
                values['fulfilled'] = min(quota, values['fulfilled_unique'])
                values['unmet'] = quota - values['fulfilled']
                values['not_harvested'] = max(0, quota - values['harvested'])
            requested = sum(q for _, q in issue.quotas)
            unmet = sum(products[item]['unmet'] for item, _ in issue.quotas)
            result[key] = {'status': issue.status.value, 'demand_fulfilled': unmet == 0,
                           'requested': requested, 'unmet': unmet,
                           'unfulfilled_fraction': unmet / requested,
                           'products': {p: dict(v) for p, v in products.items()},
                           'actual_inputs_consumed': dict(self.inputs[key]),
                           'producer_event_ids': self.events[key],
                           'producer_id': record.producer_id, 'tile_xy': record.tile_xy}
        return result


def settle_task_dag(raw_report, completion, issues, *, rule='unfulfilled_fraction'):
    """Redistribute ONE non-cash budget, never add a second expiry/purchase loss.

ISSUE is the fallback responsibility. A proven resource-withdrawal role, when
present, receives half the budget; its sales split that half by removed units.
This explicit attribution convention is not a marginal causal-effect estimate.
"""
    if rule not in SETTLEMENT_RULES:
        raise ValueError('Unknown task settlement rule')
    if not math.isclose(math.fsum(e['reward'] for e in raw_report['events']),
                        raw_report['task_credit'], abs_tol=1e-10):
        raise ValueError('Task budget does not reconcile before DAG settlement')
    if len({i.decision_id for i in issues.values()}) != len(issues):
        raise ValueError('Duplicate ISSUE decision identity')
    graph = ActionDAG()
    roles, decisions, issue_credit, sinks = {}, defaultdict(float), defaultdict(float), []
    nodes = {}
    for key, issue in issues.items():
        nodes[key] = graph.node(issue.decision_id)
        roles[issue.decision_id] = ['ISSUE', issue.item]
    for index, event in enumerate(raw_report['events']):
        key = event['issue_id']; issue = issues[key]
        if event['decision_id'] != issue.decision_id:
            raise ValueError('Task event/ISSUE mismatch')
        budget = float(event['reward'])
        if not math.isfinite(budget) or budget > 0:
            raise ValueError('Invalid task cost budget')
        ratio = completion[key]['unfulfilled_fraction'] if rule == 'unfulfilled_fraction' else 1.
        if not math.isfinite(ratio) or not 0 <= ratio <= 1:
            raise ValueError('Invalid unmet demand fraction')
        reward = budget * ratio
        if ratio == 0 or budget == 0:
            continue
        parents = [(nodes[key], 1.)]
        withdrawal = []
        for evidence in event.get('withdrawal_evidence', ()):
            decision = evidence['decision_id']
            if (type(decision) is not int or decision < 0 or
                    type(evidence['quantity']) is not int or evidence['quantity'] <= 0):
                raise ValueError('Invalid resource withdrawal evidence')
            if decision in roles and roles[decision] != ['SELL', 'WHEAT']:
                raise ValueError('Conflicting manager action identity')
            roles[decision] = ['SELL', 'WHEAT']
            leaf = graph.node(decision, event_id=evidence['event_id'])
            withdrawal.append((leaf, evidence['quantity']))
        if withdrawal:
            parents.append((graph.node(None, withdrawal), 1.))
        root = graph.node(None, parents)
        shares = graph.shares(root, reward, 1.)
        for decision, credit in shares.items():
            decisions[decision] += credit
        issue_credit[key] += reward
        sinks.append({'sink_id': index, 'kind': 'unmet_demand' if ratio < 1 or rule == 'unfulfilled_fraction'
                      else 'observed_task_cost', 'issue_id': key, 'source_event': event,
                      'raw_budget': budget, 'unfulfilled_fraction': completion[key]['unfulfilled_fraction'],
                      'reward': reward, 'action_root': root, 'decision_credit': shares})
    total = math.fsum(s['reward'] for s in sinks)
    if not math.isclose(total, math.fsum(decisions.values()), abs_tol=1e-10):
        raise ValueError('Task DAG credit conservation failed')
    return {**raw_report, 'version': VERSION, 'settlement_rule': rule,
            'raw_task_credit': raw_report['task_credit'], 'task_credit': total,
            'decision_credit': dict(decisions), 'issue_credit': dict(issue_credit),
            'decision_roles': roles, 'completion': completion,
            'graph': {'version': VERSION, 'action_nodes': graph.nodes, 'sinks': sinks,
                      'economic_cost_added': 0., 'estimated_revenue_added': 0.}}


def assign_issue_throughput(completion, issues, *, fulfill_reward=DEFAULT_FULFILL_REWARD,
                            unmet_reward=DEFAULT_UNMET_REWARD):
    """Count-based ISSUE credit: crew-fulfilled quota minus unmet demand.

Not cash. Does not invent sale prices or write into the sale DAG.
"""
    for name, value in (('fulfill_reward', fulfill_reward), ('unmet_reward', unmet_reward)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if len({i.decision_id for i in issues.values()}) != len(issues):
        raise ValueError('Duplicate ISSUE decision identity')
    if set(completion) != set(issues):
        raise ValueError('Throughput completion does not match issues')
    decisions, issue_credit, sinks = {}, {}, []
    for key, issue in issues.items():
        row = completion[key]
        requested, unmet = row['requested'], row['unmet']
        if not all(type(n) is int and n >= 0 for n in (requested, unmet)):
            raise ValueError('Invalid throughput counts')
        if unmet > requested:
            raise ValueError('Unmet demand exceeds requested quota')
        fulfilled = requested - unmet
        credit = fulfill_reward * fulfilled - unmet_reward * unmet
        if not math.isfinite(credit):
            raise ValueError('Invalid throughput credit')
        issue_credit[key] = credit
        if credit:
            decisions[issue.decision_id] = decisions.get(issue.decision_id, 0.) + credit
        sinks.append({'issue_id': key, 'decision_id': issue.decision_id, 'fulfilled': fulfilled,
                      'unmet': unmet, 'requested': requested, 'reward': credit})
    total = math.fsum(issue_credit.values())
    if not math.isclose(total, math.fsum(decisions.values()), abs_tol=1e-10):
        raise ValueError('Throughput credit conservation failed')
    return {'version': THROUGHPUT_VERSION, 'fulfill_reward': fulfill_reward,
            'unmet_reward': unmet_reward, 'throughput_credit': total,
            'decision_credit': decisions, 'issue_credit': issue_credit,
            'fulfilled_units': sum(s['fulfilled'] for s in sinks),
            'unmet_units': sum(s['unmet'] for s in sinks),
            'sinks': sinks, 'economic_cost_added': 0., 'estimated_revenue_added': 0.}
