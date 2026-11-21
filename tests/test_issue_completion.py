from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import unittest

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as game
from lifecycle_tracking import LifecycleEvent, LifecycleEventType as E
from tests.test_manager import observation
from tests.test_production_graph import Journal
from worker_grow.contracts import TaskStatus
from worker_grow.issues import IssueBook
from worker_grow.issue_completion import IssueMaterialAudit, settle_task_dag, assign_issue_throughput
from worker_grow.issue_credit import assign_issue_credit
from worker_grow.issue_outcomes import IssueOutcomeLedger
from worker_grow.resource_evidence import FeedWithdrawalEvidence
from worker_grow.manager import ManagerDecision
from worker_grow.manager_training import finish_manager_episode
from worker_grow.delivery import CrewDeliveryTracker
from cash_accounting import CashLedger
from market_execution import trace_market_executions


def book_with_producers(quota=10):
    obs = observation(); book = IssueBook(realized_sales=True)
    for item, xy, identity, decision in [('WHEAT', (1, 1), 'p', 100), ('GOOSE', (2, 2), 'a', 101)]:
        product = item if item in game.CROPS else game.ANIMALS[item]['product']
        issue = book.issue(obs, item=item, tile_xy=xy, decision_id=decision, step=0, day=0,
                           quotas=((product, quota),))
        issue.producer_id = identity; issue.status = TaskStatus.EXPIRED
        record = book.task_book.records[issue.task_id]
        record.producer_id = identity; record.status = TaskStatus.EXPIRED
    return obs, book


class CompletionTests(unittest.TestCase):
    def test_partial_production_sales_feed_and_holding_are_disjoint(self):
        obs, book = book_with_producers()
        j = Journal()
        seed = j.buy('WHEAT', 1, 10)[0]; j.plant(seed)
        animal = j.buy('GOOSE', 1, 300, 'animal')[0]; j.animal(animal)
        j.harvest('p', 'h', 6)
        j.move('h', 'shed', 4); j.sale('shed', 3, 75)
        j.feed('a', 'h')  # One unit consumed BEFORE depot delivery.
        j.harvest('a', 'eggs', 1, animal=True)
        audit = IssueMaterialAudit(book.issues)
        for event in j.events: audit.observe(event)
        report = audit.report(book, obs)
        wheat = report['issue:1']['products']['WHEAT']
        self.assertEqual([wheat[k] for k in ('harvested', 'sold', 'consumed', 'held', 'lost')],
                         [6, 3, 1, 2, 0])
        self.assertEqual(wheat['fulfilled'], 5)  # Four delivered + one directly fed.
        self.assertEqual(wheat['unmet'], 5)
        self.assertEqual(report['issue:2']['actual_inputs_consumed']['WHEAT'], 1)
        self.assertEqual(report['issue:2']['products']['EGG']['harvested'], 1)

    def test_depot_then_feed_or_resale_never_double_counts_completion(self):
        obs, book = book_with_producers(quota=3)
        j = Journal(); seed = j.buy('WHEAT', 1, 10)[0]; j.plant(seed)
        animal = j.buy('GOOSE', 1, 300, 'animal')[0]; j.animal(animal)
        j.harvest('p', 'h', 3); j.move('h', 'shed', 3)
        j.feed('a', 'shed'); j.sale('shed', 1, 25)
        audit = IssueMaterialAudit(book.issues)
        for event in j.events: audit.observe(event)
        result = audit.report(book, obs)['issue:1']
        self.assertTrue(result['demand_fulfilled'])
        self.assertEqual(result['products']['WHEAT']['fulfilled_unique'], 3)
        # A still-live producer marked EXPIRED is not an unmet demand.
        raw = {'events': [{'issue_id': 'issue:1', 'decision_id': 100, 'step': 0,
                           'reason': 'start_backlog', 'reward': -.1}], 'task_credit': -.1}
        settled = settle_task_dag(raw, audit.report(book, obs), book.issues)
        self.assertEqual(settled['task_credit'], 0)
        self.assertEqual(settled['graph']['sinks'], [])
        throughput = assign_issue_throughput(audit.report(book, obs), book.issues)
        self.assertEqual(throughput['issue_credit']['issue:1'], 3.)
        self.assertEqual(throughput['issue_credit']['issue:2'], -3.)
        self.assertEqual(throughput['decision_credit'][100], 3.)
        self.assertEqual(throughput['throughput_credit'], 0.)
        self.assertEqual(throughput['economic_cost_added'], 0.)

    def test_ready_unharvested_is_not_future_yield_or_profit(self):
        obs, book = book_with_producers(quota=5)
        obs['farms'][0]['tiles'][1][1] = game._new_plant('WHEAT', 0, 24)
        obs['farms'][0]['tiles'][1][1]['yield_units'] = 3
        obs['farms'][0]['tiles'][2][2] = game._new_animal('GOOSE', 0)
        obs['farms'][0]['tiles'][2][2].update(yield_units=2, fertilizer_available=True)
        audit = IssueMaterialAudit(book.issues)
        self.assertEqual(audit.report(book, obs)['issue:1']['products']['WHEAT']['ready_unharvested'], 0)
        obs['day'] = 2
        report = audit.report(book, obs)
        self.assertEqual(report['issue:1']['products']['WHEAT']['ready_unharvested'], 3)
        self.assertEqual(report['issue:1']['unmet'], 5)  # No delivery/use yet.
        self.assertEqual(report['issue:2']['products']['EGG']['ready_unharvested'], 2)
        self.assertEqual(report['issue:2']['products']['FERTILIZER']['ready_unharvested'], 1)
        self.assertNotIn('profit', report['issue:1'])

    def test_unharvested_old_tile_is_not_reassigned_to_closed_issue(self):
        obs, book = book_with_producers()
        book.task_book.records['issue:1'].production_closed = True
        obs['day'] = 4
        obs['farms'][0]['tiles'][1][1] = game._new_plant('WHEAT', 0, 24)
        obs['farms'][0]['tiles'][1][1]['yield_units'] = 6
        self.assertEqual(IssueMaterialAudit(book.issues).report(book, obs)['issue:1']
                         ['products']['WHEAT']['ready_unharvested'], 0)

    def test_partial_cost_dag_conserves_budget_and_does_not_touch_cash(self):
        obs, book = book_with_producers(quota=10)
        j = Journal(); seed = j.buy('WHEAT', 1, 10)[0]; j.plant(seed)
        j.harvest('p', 'h', 6); j.move('h', 'shed', 6); j.sale('shed', 6, 150)
        audit = IssueMaterialAudit(book.issues)
        for event in j.events: audit.observe(event)
        raw = {'events': [{'issue_id': 'issue:1', 'decision_id': 100, 'step': 5,
                           'reason': 'missed_water', 'reward': -.2}], 'task_credit': -.2}
        task = settle_task_dag(raw, audit.report(book, obs), book.issues)
        self.assertAlmostEqual(task['task_credit'], -.08)
        self.assertAlmostEqual(sum(task['decision_credit'].values()), -.08)
        self.assertEqual(task['graph']['economic_cost_added'], 0)
        refs = {k: 2*v for k, v in j.refs.items()}
        events = [replace(e, decision_id=2*e.decision_id+(0 if e.decision_id in j.refs.values() else 1))
                  for e in j.events]
        before = assign_issue_credit(events, j.cash, refs, 0, book.issues)
        after = assign_issue_credit(events, j.cash, refs, 0, book.issues)
        self.assertEqual(before, after)
        self.assertEqual(after['net_cash'], 140)
        self.assertGreater(after['issue_gold']['issue:1'], 0)
        self.assertLess(after['issue_gold']['issue:1'], 140)
        self.assertAlmostEqual(sum(after['decision_credit'].values()), 140)
        full = settle_task_dag(raw, audit.report(book, obs), book.issues, rule='preserve')
        self.assertEqual(full['task_credit'], -.2)
        throughput = assign_issue_throughput(audit.report(book, obs), book.issues)
        self.assertEqual(throughput['decision_credit'][100], 2.)  # 6 fulfilled, 4 unmet
        self.assertEqual(throughput['decision_credit'][101], -10.)
        self.assertEqual(after['net_cash'], 140)
        self.assertEqual(throughput['economic_cost_added'], 0.)

    def test_proven_sale_role_and_issue_role_split_one_cost(self):
        obs, book = book_with_producers(quota=10)
        completion = IssueMaterialAudit(book.issues).report(book, obs)
        raw = {'task_credit': -.2, 'events': [{'issue_id': 'issue:2', 'decision_id': 101,
                  'reward': -.2, 'reason': 'missed_feed', 'step': 23,
                  'withdrawal_evidence': [{'event_id': 'sale', 'decision_id': 9, 'quantity': 1}]}]}
        task = settle_task_dag(raw, completion, book.issues)
        self.assertEqual(task['decision_credit'], {9: -.1, 101: -.1})
        rows = [ManagerDecision(None, None, 0, kind='contract', action=['ISSUE','GOOSE'], decision_id=101),
                ManagerDecision(None, None, 0, kind='market', action=['SELL','WHEAT',1], decision_id=9),
                ManagerDecision(None, None, 0, kind='boundary', boundary=True, decision_id=999)]
        finish_manager_episode(rows, 0, [], task_report=task,
                               credit_report={'decision_credit': {}, 'net_cash': 0})
        self.assertEqual([r.task_credit for r in rows], [-.1, -.1, 0.])
        self.assertEqual([r.local_credit for r in rows], [0., 0., 0.])
        self.assertEqual([r.throughput_credit for r in rows], [0., 0., 0.])

    def test_throughput_credits_issue_not_sell_and_stays_off_cash(self):
        obs, book = book_with_producers(quota=2)
        completion = IssueMaterialAudit(book.issues).report(book, obs)
        completion['issue:1']['unmet'] = 0
        throughput = assign_issue_throughput(completion, book.issues)
        self.assertEqual(throughput['decision_credit'][100], 2.)
        self.assertEqual(throughput['decision_credit'][101], -2.)
        rows = [ManagerDecision(None, None, 0, kind='contract', action=['ISSUE','WHEAT'],
                                decision_id=100, throughput_value=0.5),
                ManagerDecision(None, None, 0, kind='contract', action=['ISSUE','GOOSE'],
                                decision_id=101),
                ManagerDecision(None, None, 0, kind='market', action=['SELL','WHEAT',1], decision_id=9),
                ManagerDecision(None, None, 0, kind='boundary', boundary=True, decision_id=999)]
        finish_manager_episode(rows, 0, [],
                               credit_report={'decision_credit': {}, 'net_cash': 0},
                               throughput_report=throughput)
        self.assertEqual([r.throughput_credit for r in rows], [2., -2., 0., 0.])
        self.assertEqual([r.local_credit for r in rows], [0., 0., 0., 0.])
        self.assertAlmostEqual(rows[0].throughput_advantage, 1.5)
        with self.assertRaises(ValueError):
            finish_manager_episode(
                [ManagerDecision(None, None, 0, kind='market', action=['SELL','WHEAT',1], decision_id=100),
                 ManagerDecision(None, None, 0, kind='boundary', boundary=True, decision_id=1)],
                0, [], credit_report={'decision_credit': {}, 'net_cash': 0},
                throughput_report={'version': 'issue-throughput-v1', 'decision_credit': {100: 1.},
                                   'throughput_credit': 1.})

    def test_duplicate_unknown_and_overconsumed_material_fails(self):
        obs, book = book_with_producers()
        j = Journal(); seed = j.buy('WHEAT', 1, 10)[0]; j.plant(seed); j.harvest('p', 'h', 1)
        audit = IssueMaterialAudit(book.issues)
        for event in j.events: audit.observe(event)
        with self.assertRaises(ValueError): audit.observe(j.events[-1])
        sale = j.sale('h', 2, 50)
        with self.assertRaises(ValueError): audit.observe(sale)

    def test_real_engine_partial_episode_links_sales_without_double_cash(self):
        cfg = {'episodeSteps': 76, 'seed': 11, 'weedSpawnChance': 0.}
        env = make('kaggriculture', configuration=cfg)
        env.state[0].observation['farms'][0]['farmer'] = [0, 0]
        env.state[0].observation['farms'][0]['tiles'][0][0] = None
        env.state[0].observation['private']['seeds']['WHEAT'] = 1
        initial = deepcopy(env.state[0].observation)
        book = IssueBook(realized_sales=True)
        book.issue(initial, item='WHEAT', tile_xy=(0, 0), decision_id=0, step=0, day=0,
                   quotas=(('WHEAT', 10),))
        tracker = CrewDeliveryTracker(initial, book.task_book)
        costs = IssueOutcomeLedger(); cash = CashLedger([3000., 3000.]); refs = {}
        for step in range(74):
            observations = [deepcopy(s.observation) for s in env.state]
            for obs in observations: obs['step'] = step
            farm = observations[0]['farms'][0]
            x, y = farm['farmer']; tile = farm['tiles'][0][0]
            op = ['PASS']
            if step == 2:
                op = ['PLANT', 'WHEAT']
            elif step > 2 and isinstance(tile, dict) and tile.get('kind') == 'PLANT':
                if x: op = ['WEST']
                elif y: op = ['NORTH']
                elif not tile['watered_today']: op = ['WATER']
                elif observations[0]['day'] - tile['planted_day'] >= 2: op = ['HARVEST']
            market = [['SELL', 'WHEAT', 100]] if step == 72 else []
            action = {'farmer': op, 'hands': [], 'market': market}
            actions = [action, {'farmer': ['PASS'], 'hands': [], 'market': []}]
            executions = trace_market_executions(observations, actions, cfg)
            env.step(actions)
            after = deepcopy(env.state[0].observation); after['step'] = step + 1
            orders = [SimpleNamespace(decision_id=3*step+1, action=m, boundary=False) for m in market]
            start = len(tracker.lifecycle.events)
            tracker.observe_transition(observations[0], action, after,
                market_executions=executions[0], decisions=[SimpleNamespace(unit_index=0, decision_id=3*step)],
                market_decisions=orders)
            book.sync_from_tasks(book.task_book, step=step, prices=observations[0]['market']['prices'])
            costs.observe(observations[0], after, book, tracker.lifecycle.events[start:])
            cash.observe(step, executions, [f['money'] for f in observations[0]['farms']],
                         [f['money'] for f in after['farms']])
            if market: refs[(step, 'market', 0)] = 2*(3*step+1)
        tracker.ledger.finish()
        book.sync_from_tasks(book.task_book, step=74, prices=after['market']['prices'])
        economic = assign_issue_credit(tracker.lifecycle.events, cash.entries, refs, 0, book.issues)
        task = costs.finish(book, after, tracker.lifecycle.events)
        result = task['completion']['issue:1']
        harvested = sum(e.quantity for e in tracker.lifecycle.events if e.event_type == E.HARVESTED)
        self.assertGreater(harvested, 0)
        self.assertEqual(result['products']['WHEAT']['sold'], harvested)
        self.assertEqual(result['unmet'], 10-harvested)
        self.assertAlmostEqual(task['task_credit'], task['raw_task_credit']*(10-harvested)/10)
        self.assertGreater(economic['net_cash'], 0)
        self.assertAlmostEqual(sum(economic['decision_credit'].values()), economic['net_cash'])
        with self.assertRaises(ValueError): costs.finish(book, after, tracker.lifecycle.events)


class ResourceEvidenceTests(unittest.TestCase):
    def source(self, stock=1):
        obs, book = book_with_producers()
        record = book.task_book.records['issue:2']; record.status = TaskStatus.ACTIVE
        obs['farms'][0]['tiles'][2][2] = game._new_animal('GOOSE', 0)
        obs['private']['shed']['WHEAT'] = stock
        return obs, book

    def sale(self, *, step=0, quantity=1):
        return LifecycleEvent('sale', E.PRODUCT_SOLD, 1, step, 0, step, 0, crop='WHEAT',
                              decision_id=18, quantity=quantity,
                              data={'revenue_is_estimate': False,
                                    'allocations': [{'product_lot_id': 'stock', 'quantity': quantity}]})

    def test_same_turn_sale_not_blamed_but_persistent_next_turn_shortage_is(self):
        obs, book = self.source(); after = deepcopy(obs)
        after['private']['shed']['WHEAT'] = 0
        evidence = FeedWithdrawalEvidence(); evidence.observe(obs, after, book, [self.sale()])
        cost = {'issue_id': 'issue:2', 'reason': 'missed_feed'}
        evidence.allocate([cost], 0); self.assertNotIn('withdrawal_evidence', cost)
        after['step'] = 1
        evidence.observe(after, after, book, [])
        evidence.allocate([cost], 1)
        self.assertEqual(cost['withdrawal_evidence'][0]['decision_id'], 9)

    def test_repurchase_retires_sale_evidence_on_next_crew_phase(self):
        obs, book = self.source(); empty = deepcopy(obs); empty['private']['shed']['WHEAT'] = 0
        evidence = FeedWithdrawalEvidence(); evidence.observe(obs, empty, book, [self.sale()])
        buy = LifecycleEvent('buy', E.PRODUCT_PURCHASED, 1, 1, 0, 1, 0, quantity=1,
                             data={'item':'WHEAT'})
        empty['step'] = 1
        evidence.observe(empty, obs, book, [buy])
        self.assertEqual(evidence.withdrawals[0]['quantity'], 1)
        cost = {'issue_id': 'issue:2', 'reason': 'missed_feed'}
        evidence.allocate([cost], 1)
        self.assertEqual(cost['withdrawal_evidence'][0]['decision_id'], 9)
        obs['step'] = 2
        evidence.observe(obs, obs, book, [])
        self.assertEqual(evidence.withdrawals, [])

    def test_harvested_wheat_retires_sale_evidence_this_crew_phase(self):
        obs, book = self.source(); empty = deepcopy(obs); empty['private']['shed']['WHEAT'] = 0
        evidence = FeedWithdrawalEvidence(); evidence.observe(obs, empty, book, [self.sale()])
        harvest = LifecycleEvent('h', E.HARVESTED, 1, 1, 0, 1, 0, crop='WHEAT', quantity=1)
        empty['step'] = 1
        evidence.observe(empty, obs, book, [harvest])
        self.assertEqual(evidence.withdrawals, [])

    def test_same_turn_resale_does_not_restore_on_empty_crew_stock(self):
        obs, book = self.source(); empty = deepcopy(obs); empty['private']['shed']['WHEAT'] = 0
        evidence = FeedWithdrawalEvidence(); evidence.observe(obs, empty, book, [self.sale()])
        buy = LifecycleEvent('buy', E.PRODUCT_PURCHASED, 1, 1, 0, 1, 0, quantity=1,
                             data={'item':'WHEAT'})
        empty['step'] = 1
        evidence.observe(empty, empty, book, [buy, self.sale(step=1)])
        self.assertEqual(sum(w['quantity'] for w in evidence.withdrawals if w['step'] == 0), 1)
        empty['step'] = 2
        evidence.observe(empty, empty, book, [])
        self.assertEqual(sum(w['quantity'] for w in evidence.withdrawals if w['step'] == 0), 1)

    def test_surplus_sale_and_new_issue_do_not_create_blame(self):
        obs, book = self.source(stock=10); after = deepcopy(obs); after['private']['shed']['WHEAT'] = 9
        evidence = FeedWithdrawalEvidence(); evidence.observe(obs, after, book, [self.sale()])
        self.assertEqual(evidence.withdrawals, [])
        obs, book = self.source(); book.task_book.records['issue:2'].status = TaskStatus.COMPLETE
        after = deepcopy(obs); after['private']['shed']['WHEAT'] = 0
        evidence.observe(obs, after, book, [self.sale()])
        self.assertEqual(evidence.withdrawals, [])

    def test_unknown_stock_change_discards_evidence(self):
        obs, book = self.source(); after = deepcopy(obs); after['private']['shed']['WHEAT'] = 0
        evidence = FeedWithdrawalEvidence(); evidence.observe(obs, after, book, [])
        self.assertEqual(evidence.withdrawals, [])
        self.assertEqual(evidence.history[-1]['kind'], 'unexplained_stock_change')

    def test_one_sold_unit_cannot_explain_two_missing_feeds(self):
        obs, book = self.source(stock=2)
        issue = book.issue(obs, item='GOOSE', tile_xy=(3, 2), decision_id=102, step=0, day=0)
        issue.producer_id = 'b'
        record = book.task_book.records[issue.task_id]
        record.producer_id = 'b'; record.status = TaskStatus.ACTIVE
        obs['farms'][0]['tiles'][2][3] = game._new_animal('GOOSE', 0)
        after = deepcopy(obs); after['private']['shed']['WHEAT'] = 1
        evidence = FeedWithdrawalEvidence(); evidence.observe(obs, after, book, [self.sale()])
        costs = [{'issue_id': k, 'reason': 'missed_feed'} for k in ('issue:2', issue.task_id)]
        evidence.allocate(costs, 1)
        self.assertEqual(sum('withdrawal_evidence' in c for c in costs), 1)


if __name__ == '__main__': unittest.main()
