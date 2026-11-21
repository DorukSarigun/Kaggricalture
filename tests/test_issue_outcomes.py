from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch
from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as game

from lifecycle_tracking import LifecycleEvent, LifecycleEventType as E
from market_execution import trace_market_executions
from tests.test_manager import observation
from worker_grow.contracts import TaskStatus
from worker_grow.delivery import CrewDeliveryTracker
from worker_grow.issue_outcomes import IssueOutcomeLedger
from worker_grow.issues import IssueBook
from worker_grow.manager import ManagerDecision, ManagerActor, propose_manager_contracts
from worker_grow.manager_training import finish_manager_episode, decision_record
from worker_grow.manager_workload import admission_items, WORKLOAD_NAMES, workload_features
from worker_grow.train_manager import parser, collect, _validate
from worker_grow.train_joint import parser as joint_parser, validate as joint_validate


class IssueOutcomeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setup_book(self, *items):
        obs, book = observation(), IssueBook(realized_sales=True)
        for i, item in enumerate(items):
            book.issue(obs, item=item, decision_id=10+i, step=0, day=0)
        return obs, book

    def observe(self, ledger, obs, book, events=(), *, next_day=False):
        after = deepcopy(obs)
        after['step'] += 1
        if next_day:
            after['day'] += 1
            after['hour'] = 0
        rows = ledger.observe(obs, after, book, events)
        return after, rows

    def test_shared_stock_is_counted_once_and_waiting_targets_original_issue(self):
        obs, book = self.setup_book('WHEAT', 'WHEAT')
        obs['private']['seeds']['WHEAT'] = 1
        ledger = IssueOutcomeLedger(wait_cost=.1)
        obs, events = self.observe(ledger, obs, book)
        self.assertEqual([(e['decision_id'], e['reason']) for e in events], [(11, 'missing_input')])
        obs, events = self.observe(ledger, obs, book)
        self.assertEqual([e['reason'] for e in events], ['start_backlog', 'missing_input'])
        report = ledger.report()
        self.assertAlmostEqual(report['decision_credit'][10], -.1)
        self.assertAlmostEqual(report['decision_credit'][11], -.2)
        self.assertAlmostEqual(report['task_credit'], -.3)

    def test_funding_in_opening_turn_no_penalty_and_then_start_stops_cost(self):
        obs, book = self.setup_book('CARROT')
        after = deepcopy(obs); after['step'] = 1
        after['private']['seeds']['CARROT'] = 1
        ledger = IssueOutcomeLedger()
        self.assertEqual(ledger.observe(obs, after, book, []), [])
        record = next(iter(book.task_book.records.values()))
        record.producer_id = 'plant:1'; record.status = TaskStatus.ACTIVE
        _, events = self.observe(ledger, after, book)
        self.assertEqual(events, [])

    def test_carried_animal_stock_counts_but_feed_is_not_an_animal(self):
        obs, book = self.setup_book('COW', 'COW', 'GOOSE')
        obs['private']['inventories'][0]['COW'] = 1
        obs['private']['shed']['WHEAT'] = 100
        _, events = self.observe(IssueOutcomeLedger(), obs, book)
        self.assertEqual([e['decision_id'] for e in events], [11, 12])

    def test_completed_cancelled_and_production_closed_do_not_wait(self):
        obs, book = self.setup_book('WHEAT', 'CARROT', 'COW')
        records = list(book.task_book.records.values())
        records[0].status = TaskStatus.COMPLETE
        records[1].status = TaskStatus.CANCELLED
        records[2].close_production()
        _, events = self.observe(IssueOutcomeLedger(), obs, book)
        self.assertEqual(events, [])

    def test_normal_growth_is_free_daily_missed_care_is_not(self):
        obs, book = self.setup_book('WHEAT', 'COW')
        for i, record in enumerate(book.task_book.records.values()):
            record.producer_id = f'p:{i}'; record.assigned_tile = (i, 0)
            record.status = TaskStatus.ACTIVE
        obs['farms'][0]['tiles'][0][0] = {'kind': 'PLANT', 'consecutive_unwatered': 1}
        obs['farms'][0]['tiles'][0][1] = {'kind': 'PASTURE', 'animal': 'COW', 'consecutive_unfed': 1}
        ledger = IssueOutcomeLedger(missed_care_cost=.3)
        obs, events = self.observe(ledger, obs, book)
        self.assertEqual(events, [])
        obs, events = self.observe(ledger, obs, book, next_day=True)
        self.assertEqual([e['reason'] for e in events], ['missed_water', 'missed_feed'])
        self.assertAlmostEqual(ledger.report()['task_credit'], -.6)
        for tile in obs['farms'][0]['tiles'][0][:2]:
            tile['consecutive_unwatered'] = tile['consecutive_unfed'] = 0
        _, events = self.observe(ledger, obs, book, next_day=True)
        self.assertEqual(events, [])

    def test_late_market_wheat_keeps_prior_sale_blame_on_missed_feed(self):
        obs, book = observation(), IssueBook(realized_sales=True)
        issue = book.issue(obs, item='GOOSE', tile_xy=(2, 2), decision_id=10, step=0, day=0)
        issue.producer_id = 'a'
        record = book.task_book.records[issue.task_id]
        record.producer_id = 'a'
        record.status = TaskStatus.ACTIVE
        obs['farms'][0]['tiles'][2][2] = game._new_animal('GOOSE', 0)
        obs['private']['shed']['WHEAT'] = 1
        sale = LifecycleEvent('sale', E.PRODUCT_SOLD, 1, 0, 0, 0, 0, crop='WHEAT',
                              decision_id=18, quantity=1,
                              data={'revenue_is_estimate': False,
                                    'allocations': [{'product_lot_id': 'stock', 'quantity': 1}]})
        ledger = IssueOutcomeLedger()
        empty = deepcopy(obs)
        empty['step'] = 1
        empty['private']['shed']['WHEAT'] = 0
        ledger.observe(obs, empty, book, [sale])
        empty['farms'][0]['tiles'][2][2]['consecutive_unfed'] = 1
        bought = deepcopy(empty)
        bought['step'] = 2
        bought['day'] = empty['day'] + 1
        bought['private']['shed']['WHEAT'] = 1
        buy = LifecycleEvent('buy', E.PRODUCT_PURCHASED, 1, 1, 0, 1, 0, quantity=1,
                             data={'item': 'WHEAT'})
        charges = ledger.observe(empty, bought, book, [buy])
        missed = [e for e in charges if e['reason'] == 'missed_feed']
        self.assertEqual(len(missed), 1)
        self.assertEqual(missed[0]['withdrawal_evidence'][0]['decision_id'], 9)

    def test_dead_producer_is_charged_but_harvested_producer_is_not(self):
        obs, book = self.setup_book('WHEAT', 'WHEAT', 'COW')
        records = list(book.task_book.records.values())
        for i, record in enumerate(records):
            record.producer_id = f'p:{i}'; record.assigned_tile = (i, 0)
            record.close_production()
        obs['farms'][0]['tiles'][0][0] = {'kind': 'PLANT', 'consecutive_unwatered': 1,
                                                      'max_lifespan_step': 200}
        events = [LifecycleEvent('lost-plant', E.PLANT_LOST, 1, 0, 0, 23, 0,
                                 plant_id='p:0', data={'reason': 'BECAME_WEED'}),
                  LifecycleEvent('lost-animal', E.ANIMAL_LOST, 1, 0, 0, 23, 0,
                                 animal_id='p:2', data={'reason': 'ESCAPED'})]
        _, charges = self.observe(IssueOutcomeLedger(), obs, book, events, next_day=True)
        self.assertEqual([e['decision_id'] for e in charges], [10, 12])
        # Expiry/decay is not classified as a missed watering.
        obs['farms'][0]['tiles'][0][0]['max_lifespan_step'] = 0
        _, charges = self.observe(IssueOutcomeLedger(), obs, book, events, next_day=True)
        self.assertEqual([e['decision_id'] for e in charges], [12])

    def test_repeated_turn_and_invalid_cost_fail(self):
        obs, book = self.setup_book('WHEAT')
        ledger = IssueOutcomeLedger()
        self.observe(ledger, obs, book)
        with self.assertRaises(ValueError): self.observe(ledger, obs, book)
        for cost in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError): IssueOutcomeLedger(wait_cost=cost)
            with self.assertRaises(ValueError): IssueOutcomeLedger(missed_care_cost=cost)

    def test_task_cost_actor_and_critic_target_keep_real_gold_separate(self):
        issue = ManagerDecision(None, None, 0., kind='contract', action=['ISSUE', 'WHEAT'],
                                decision_id=0, chain_value=.2)
        buy = ManagerDecision(None, None, 0., action=['BUY_SEED', 'WHEAT', 1], decision_id=1)
        boundary = ManagerDecision(None, None, 0., boundary=True, decision_id=2)
        rows = [issue, buy, boundary]
        finish_manager_episode(rows, 100., [],
                               credit_report={'decision_credit': {1: -10}, 'net_cash': -10},
                               task_report={'decision_credit': {0: -.3}, 'task_credit': -.3})
        self.assertAlmostEqual(sum(r.local_credit for r in rows), -10)
        self.assertEqual(issue.local_credit, 0.)
        self.assertAlmostEqual(issue.chain_advantage, -.5)
        self.assertEqual(decision_record(issue)['local_target'], -.3)
        self.assertEqual(issue.throughput_credit, 0.)
        self.assertEqual(buy.task_credit, 0.)
        self.assertAlmostEqual(boundary.reward, 100)
        with self.assertRaises(ValueError):
            finish_manager_episode(rows, 100., [], credit_report={'decision_credit': {}, 'net_cash': 0})

    def test_task_cost_cannot_target_market_stop_or_boundary(self):
        for action, kind, boundary in [(['STOP'], 'contract', False),
                                       (['BUY_SEED', 'WHEAT', 1], 'market', False),
                                       ([], 'boundary', True)]:
            row = ManagerDecision(None, None, 0., action=action, kind=kind,
                                  decision_id=0, boundary=boundary)
            last = ManagerDecision(None, None, 0., boundary=True, decision_id=1)
            with self.assertRaises(ValueError):
                finish_manager_episode([row, last], 0, [],
                    credit_report={'decision_credit': {}, 'net_cash': 0},
                    task_report={'decision_credit': {0: -.1}, 'task_credit': -.1})

    def test_no_worker_or_32_issue_limit(self):
        obs, book = self.setup_book(*(['WHEAT'] * 40))
        obs['farms'][0]['tiles'] = [[None]*10 for _ in range(10)]
        self.assertIn('WHEAT', admission_items(obs, book))
        self.assertNotIn('queue_capacity', WORKLOAD_NAMES)
        i = WORKLOAD_NAMES.index('unstarted_jobs')
        self.assertGreater(workload_features(obs, book)[i], 0)
        manager = ManagerActor()
        with torch.no_grad():
            manager.contract_op.weight.zero_()
            manager.contract_op.bias.copy_(torch.tensor([20., -20.]))
        rows, _ = propose_manager_contracts(manager, obs, book, step=0, day=0,
                                           next_id=100, deterministic=True)
        self.assertGreater(sum(r.action[0] == 'ISSUE' for r in rows), 32)
        self.assertFalse(admission_items(obs, book))
        self.assertEqual(rows[-1].action, ['STOP'])

    def test_cli_has_costs_but_no_worker_cap(self):
        for p, validate in ((parser(), _validate), (joint_parser(), joint_validate)):
            args = p.parse_args(['--run-dir', 'unused', '--device', 'cpu'])
            self.assertFalse(hasattr(args, 'max_open_per_worker'))
            self.assertGreater(args.issue_wait_cost, 0)
            self.assertEqual(args.fulfill_reward, 1)
            self.assertEqual(args.unmet_reward, 1)
            validate(args)
            args.missed_care_cost = -1
            with self.assertRaises(ValueError): validate(args)

    def test_real_day_refresh_respects_last_turn_water(self):
        self.check_real_day('WHEAT')

    def test_real_day_refresh_respects_last_turn_feed(self):
        self.check_real_day('COW')

    def test_real_day_death_does_not_hide_missed_care(self):
        for item in ('WHEAT', 'COW'):
            with self.subTest(item=item):
                self.check_real_day(item, second_day=True)

    def test_real_decay_is_not_mislabelled_as_missed_care(self):
        self.check_real_day('WHEAT', second_day=True, decay=True)

    def check_real_day(self, item, *, second_day=False, decay=False):
        cfg = {'episodeSteps': 50, 'seed': 9, 'weedSpawnChance': 0.}
        env = make('kaggriculture', configuration=cfg)
        live = env.state[0].observation
        live['step'] = 23; live['day'] = 0; live['hour'] = 23
        live['farms'][0]['farmer'] = [0, 0]
        if item == 'COW':
            live['private']['inventories'][0]['WHEAT'] = 1
        book = IssueBook(realized_sales=True)
        for x in (0, 1):
            live['farms'][0]['tiles'][0][x] = None
            book.issue(live, item=item, tile_xy=(x, 0), decision_id=x, step=0, day=0)
            live['farms'][0]['tiles'][0][x] = (game._new_plant(item, 0, 24) if item in game.CROPS
                                               else game._new_animal(item, 0))
        if second_day:
            field = 'consecutive_unwatered' if item in game.CROPS else 'consecutive_unfed'
            live['farms'][0]['tiles'][0][1][field] = 1
        if decay:
            live['farms'][0]['tiles'][0][1].update(max_lifespan_step=23, yield_units=1)
        obs = deepcopy(live)
        tracker = CrewDeliveryTracker(obs, book.task_book)
        action = {'farmer': ['WATER' if item in game.CROPS else 'FEED'], 'hands': [], 'market': []}
        actions = [action, {'farmer': ['PASS'], 'hands': [], 'market': []}]
        before = [deepcopy(s.observation) for s in env.state]
        market = trace_market_executions(before, actions, cfg)
        start = len(tracker.lifecycle.events)
        env.step(actions)
        after = deepcopy(env.state[0].observation); after['step'] = 24
        tracker.observe_transition(obs, action, after, market_executions=market[0], decisions=[])
        self.assertEqual(after['day'], 1)
        ledger = IssueOutcomeLedger(); ledger.last_step = 22
        charges = ledger.observe(obs, after, book, tracker.lifecycle.events[start:])
        expected = [] if decay else [(1, 'missed_water' if item in game.CROPS else 'missed_feed')]
        self.assertEqual([(e['decision_id'], e['reason']) for e in charges], expected)

    def test_costs_can_be_disabled_without_affecting_accounting(self):
        obs, book = self.setup_book('WHEAT')
        ledger = IssueOutcomeLedger(wait_cost=0., missed_care_cost=0.)
        self.observe(ledger, obs, book)
        self.assertEqual(ledger.report()['task_credit'], 0.)

    def test_real_collector_logs_task_cost_without_changing_cash(self):
        torch.manual_seed(11)
        manager = ManagerActor()
        with torch.no_grad():
            manager.contract_op.weight.zero_()
            manager.contract_op.bias.copy_(torch.tensor([20., -20.]))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = parser().parse_args(['--run-dir', tmp, '--device', 'cpu', '--opponent', 'pass',
                                        '--episode-steps', '4', '--save-every', '0'])
            rows, summary = collect(manager, None, args, 1, root/'episode')
            self.assertLess(summary['task_credit_sum'], 0)
            self.assertAlmostEqual(sum(r.local_credit for r in rows), summary['economic_net_cash'])
            self.assertAlmostEqual(sum(r.task_credit for r in rows), summary['task_credit_sum'])
            self.assertAlmostEqual(sum(r.throughput_credit for r in rows), summary['throughput_credit_sum'])
            self.assertLess(summary['throughput_credit_sum'], 0)
            self.assertEqual(summary['crew_reward'], 0)
            self.assertTrue((root/'episode'/'issue_task_cost.json').exists())
            self.assertTrue((root/'episode'/'issue_throughput.json').exists())
            self.assertTrue(all(r.action[0] == 'ISSUE' for r in rows if r.task_credit))
            self.assertTrue(all(r.action[0] == 'ISSUE' for r in rows if r.throughput_credit))


if __name__ == '__main__':
    unittest.main()
