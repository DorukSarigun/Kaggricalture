from dataclasses import replace
from copy import deepcopy
import math
import unittest
from types import SimpleNamespace

from kaggle_environments import make
from cash_accounting import CashLedger
from market_execution import trace_market_executions
from worker_grow.delivery import CrewDeliveryTracker
from worker_grow.crew import CrewCursor
from worker_grow.train_crew import curriculum

from tests.test_production_graph import Journal
from tests.test_feudal_crew import Events
from tests.test_manager import observation
from lifecycle_tracking import LifecycleEventType as E
from worker_grow.contracts import ProductionTask, TaskBook, TaskStatus
from worker_grow.issues import Issue, IssueBook
from worker_grow.issue_credit import IssueDAG, assign_issue_credit
from worker_grow.manager import ManagerDecision
from worker_grow.manager_training import finish_manager_episode


def settle(journal):
    market_ids = set(journal.refs.values())
    events = [replace(e, decision_id=None if e.decision_id is None else
                      2*e.decision_id+(0 if e.decision_id in market_ids else 1))
              for e in journal.events]
    refs = {k: 2*v for k, v in journal.refs.items()}
    issues = {}
    for e in events:
        if e.event_type in (E.PLANTED, E.ANIMAL_PLACED):
            identity = e.plant_id or e.animal_id
            item = e.crop or e.data.get('animal','GOOSE')
            issues[identity] = Issue(identity, e.tile_xy, item,
                ((item if e.crop else 'EGG', 1),), 10000+len(issues), 0, 0,
                producer_id=identity, realized_sales=True)
    return assign_issue_credit(events, journal.cash, refs, 0, issues), issues


class IssueDAGTests(unittest.TestCase):
    def test_each_fork_equal_own_issue_gets_share_and_shared_leaves_merge(self):
        g = IssueDAG()
        w, x, y, z = [g.leaf(k) for k in ('w', 'x', 'y', 'z')]
        xr = g.fork([x, w])
        yr = g.fork([y, w])
        root = g.fork([z, xr, yr])
        self.assertEqual(g.shares(root, 90), {'w': 30., 'x': 15., 'y': 15., 'z': 30.})
        self.assertEqual(g.fork([z, xr, xr, yr]), root)
        with self.assertRaises(ValueError): g.fork([len(g.nodes)])

    def assert_gold(self, report):
        self.assertEqual(report['version'], 'issue-sale-dag-v2')
        self.assertAlmostEqual(sum(report['decision_credit'].values()), report['net_cash'])
        self.assertAlmostEqual(report['allocated_gold'], report['net_cash'])
        self.assertEqual(report['reconciliation_error'], 0.)

    def test_sale_net_profit_follows_buy_issue_sell_path(self):
        j = Journal()
        seed = j.buy('WHEAT', 1, 10)[0]
        j.plant(seed); j.harvest('p', 'w', 1); j.move('w', 'shed', 1)
        before, _ = settle(j)
        self.assert_gold(before)
        self.assertEqual(before['net_cash'], -10)
        self.assertIn(0, before['decision_credit'])
        self.assertIn(10000, before['decision_credit'])
        j.sale('shed', 1, 25)
        after, issues = settle(j)
        self.assert_gold(after)
        self.assertEqual(after['net_cash'], 15)
        sale_id = j.events[-1].decision_id
        self.assertIn(0, after['decision_credit'])
        self.assertIn(issues['p'].decision_id, after['decision_credit'])
        self.assertIn(sale_id, after['decision_credit'])
        self.assertGreater(after['issue_gold']['p'], 0)
        self.assertLess(after['issue_gold']['p'], 15)

    def test_x_plus_y_to_z_net_profit_stays_on_the_path(self):
        j = Journal()
        x = j.buy('WHEAT', 1, 10)[0]; j.plant(x, 'x'); j.harvest('x', 'hx', 1)
        y = j.buy('WHEAT', 1, 20)[0]; j.plant(y, 'y'); j.harvest('y', 'hy', 1)
        z = j.buy('GOOSE', 1, 300, 'animal')[0]; j.animal(z, 'z')
        j.feed('z', 'hx'); j.feed('z', 'hy'); j.harvest('z', 'hz', 1, animal=True)
        j.sale('hz', 1, 390)
        r, issues = settle(j)
        self.assert_gold(r)
        self.assertEqual(r['net_cash'], 60.)
        for key in ('x', 'y', 'z'):
            self.assertIn(issues[key].decision_id, r['decision_credit'])
        self.assertAlmostEqual(sum(r['issue_gold'].values()),
                               sum(r['decision_credit'][issues[k].decision_id] for k in issues))
        for purchase in r['graph']['purchase_cost_audit'].values():
            self.assertEqual(purchase['cost'], purchase['allocated_cost'])

    def test_partial_sale_partial_feed_and_no_double_counted_seed_cost(self):
        j = Journal()
        seed = j.buy('WHEAT', 1, 10)[0]; j.plant(seed); j.harvest('p', 'w', 2)
        j.sale('w', 1, 25)
        animal = j.buy('GOOSE', 1, 300, 'animal')[0]; j.animal(animal)
        j.feed('a', 'w'); j.harvest('a', 'egg', 1, animal=True); j.sale('egg', 1, 400)
        r, _ = settle(j)
        self.assert_gold(r)
        self.assertEqual(r['net_cash'], 115)
        self.assertEqual([s['cost'] for s in r['allocations'] if s['kind']=='sale'], [5., 305.])

    def test_future_feed_does_not_change_older_output_ancestry(self):
        j = Journal()
        stock = j.buy('GOOSE', 1, 300, 'animal')[0]; j.animal(stock)
        j.harvest('a', 'early', 1, animal=True)
        seed = j.buy('WHEAT', 1, 10)[0]; j.plant(seed); j.harvest('p', 'w', 1)
        j.feed('a', 'w'); j.harvest('a', 'late', 1, animal=True)
        j.sale('early', 1, 200); j.sale('late', 1, 200)
        r, issues = settle(j)
        self.assert_gold(r)
        sales = [s for s in r['allocations'] if s['kind']=='sale']
        self.assertNotIn('p', sales[0].get('issue_shares', {}))
        self.assertIn('a', sales[0].get('issue_shares', {}))
        self.assertIn('p', sales[1].get('issue_shares', {}))
        self.assertIn('a', sales[1].get('issue_shares', {}))

    def test_repeated_feed_from_split_same_ancestry_does_not_add_votes(self):
        j = Journal()
        seed = j.buy('WHEAT', 1, 10)[0]; j.plant(seed); j.harvest('p', 'w', 2)
        j.move('w', 'split1', 1); j.move('w', 'split2', 1)
        stock = j.buy('GOOSE', 1, 300, 'animal')[0]; j.animal(stock)
        j.feed('a', 'split1'); j.feed('a', 'split2')
        j.harvest('a', 'egg', 1, animal=True); j.sale('egg', 1, 370)
        r, _ = settle(j)
        self.assert_gold(r)
        self.assertEqual(r['net_cash'], 60)

    def test_distinct_intermediates_from_same_issue_keep_branches_then_merge_leaf_credit(self):
        g=IssueDAG()
        source, final=g.leaf('source'),g.leaf('final')
        x=g.fork([source],lot_id='x')
        y=g.fork([source],lot_id='y')
        z=g.fork([final,x,y],lot_id='z')
        self.assertNotEqual(x,y)
        self.assertEqual(g.shares(z,90),{'source':60.,'final':30.})

    def test_fertilizer_crop_feed_same_animal_is_temporal_dag_not_identity_cycle(self):
        j = Journal()
        a = j.buy('GOOSE', 1, 300, 'animal')[0]; j.animal(a)
        j.harvest('a', 'f', 1, animal=True, fertilizer=True)
        p = j.buy('WHEAT', 1, 10)[0]; j.plant(p)
        j.fertilize('p', 'f'); j.harvest('p', 'w', 1)
        j.feed('a', 'w'); j.harvest('a', 'egg', 1, animal=True); j.sale('egg', 1, 400)
        r, _ = settle(j)
        self.assert_gold(r)
        self.assertAlmostEqual(r['allocated_gold'], 90)
        for node in r['graph']['issue_nodes']:
            self.assertTrue(all(child < node['node_id'] for child in node['children']))

    def test_unsold_unused_and_lost_costs_follow_the_path(self):
        j = Journal()
        seed = j.buy('WHEAT', 2, 20)[0]; j.plant(seed); j.lose_plant()
        r, issues = settle(j)
        self.assert_gold(r)
        self.assertEqual(r['net_cash'], -20)
        self.assertIn(0, r['decision_credit'])
        self.assertIn(issues['p'].decision_id, r['decision_credit'])
        self.assertLess(r['issue_gold']['p'], 0)

    def test_negative_sale_profit_is_distributed_with_same_sign(self):
        j = Journal()
        seed = j.buy('WHEAT', 1, 10)[0]; j.plant(seed); j.harvest('p', 'w', 1); j.sale('w', 1, 4)
        r, _ = settle(j)
        self.assert_gold(r)
        self.assertEqual(r['net_cash'], -6.)
        self.assertTrue(all(v < 0 for v in r['decision_credit'].values()))

    def test_direct_trade_is_separate_and_no_crew_economic_credit(self):
        j = Journal()
        lot = j.buy('WHEAT', 1, 25, 'product')[0]; j.move(lot, 'shed', 1)
        sale = j.sale('shed', 1, 30)
        r, _ = settle(j)
        self.assert_gold(r)
        self.assertEqual(r['issue_gold'], {})
        self.assertEqual(set(r['decision_credit']), {0, sale.decision_id})
        self.assertAlmostEqual(sum(r['decision_credit'].values()), 5)

    def test_manager_ppo_targets_path_in_gold_and_keeps_terminal_separate(self):
        j = Journal()
        seed = j.buy('WHEAT', 1, 10)[0]; j.plant(seed); j.harvest('p', 'w', 1)
        sale = j.sale('w', 1, 25)
        report, issues = settle(j)
        rows = [ManagerDecision(None,None,0.,kind='market',action=['BUY_SEED','WHEAT',1],decision_id=0),
                ManagerDecision(None,None,0.,kind='contract',action=['ISSUE','WHEAT',1,1],
                                decision_id=issues['p'].decision_id),
                ManagerDecision(None,None,0.,kind='market',action=['SELL','WHEAT',1],decision_id=sale.decision_id),
                ManagerDecision(None,None,0.,kind='boundary',boundary=True,decision_id=99999)]
        finish_manager_episode(rows,15,[],credit_report=report)
        self.assertAlmostEqual(sum(r.local_credit for r in rows), 15)
        self.assertTrue(rows[0].local_credit)
        self.assertTrue(rows[1].local_credit)
        self.assertTrue(rows[2].local_credit)
        self.assertEqual(rows[-1].local_credit, 0.)
        self.assertEqual(rows[-1].reward, 15)
        with self.assertRaises(ValueError): finish_manager_episode(rows,15,[],credit_report=report)

    def test_missing_issue_and_reused_material_fail_closed(self):
        j = Journal(); seed=j.buy('WHEAT',1,10)[0]; j.plant(seed)
        with self.assertRaisesRegex(ValueError,'verified ISSUE'):
            assign_issue_credit([replace(e,decision_id=2*e.decision_id) for e in j.events],
                               j.cash,{k:2*v for k,v in j.refs.items()},0,{})
        j.harvest('p','w',1); j.sale('w',1,25); j.sale('w',1,25)
        with self.assertRaisesRegex(ValueError,'over-consumption'): settle(j)


class LifetimeIssueTests(unittest.TestCase):
    def test_real_repeating_crops_and_animals_remain_assigned_after_delivery(self):
        for item in ('TOMATO','STRAWBERRY','GOOSE','COW','SHEEP'):
            with self.subTest(item=item):
                config={'seed':77,'episodeSteps':30,'weedSpawnChance':0.}
                env=make('kaggriculture',configuration=config)
                tasks=curriculum(env,item,'mature',0,2)
                record=tasks.records['lesson:0']
                record.task=replace(record.task,producer_lifetime=True)
                tracker=CrewDeliveryTracker(deepcopy(env.state[0].observation),tasks)
                # Curriculum places the unit on a field tile, not the shed.
                # Keep the harvest until the engine's actual night deposit.
                care=['WATER'] if item in ('TOMATO','STRAWBERRY') else ['PASS']
                for step,op in enumerate([['HARVEST'],care]+[['PASS']]*22):
                    before=[deepcopy(s.observation) for s in env.state]
                    for obs in before:obs['step']=step
                    action={'farmer':op,'hands':[],'market':[]}
                    actions=[action,{'farmer':['PASS'],'hands':[],'market':[]}]
                    executions=trace_market_executions(before,actions,config)
                    env.step(actions)
                    after=deepcopy(env.state[0].observation);after['step']=step+1
                    tracker.observe_transition(before[0],action,after,market_executions=executions[0],
                                               decisions=[SimpleNamespace(unit_index=0,decision_id=step)])
                self.assertTrue(record.open)
                self.assertFalse(record.production_closed)
                self.assertIs(tasks.at((0,0)),record)
                self.assertIsNotNone(record.producer_id)
                self.assertEqual(sum(record.total_delivered.values()),2)
                self.assertNotEqual(CrewCursor(after,tasks).operations(),('PASS',))

    def test_delivery_does_not_close_living_producer_or_generate_manager_gold(self):
        b = IssueBook(realized_sales=True)
        i = b.issue(observation(),item='WHEAT',tile_xy=(0,0),decision_id=100,step=0,day=0)
        events = Events(b.task_book)
        events.plant(); events.harvest(q=1); events.move('h','shed',1)
        r = b.task_book.records[i.task_id]
        self.assertEqual(r.status, TaskStatus.ACTIVE)
        self.assertEqual(events.ledger.report()['reward'],1.)
        b.sync_from_tasks(b.task_book,step=4,prices={'WHEAT':10000})
        self.assertTrue(i.open)
        self.assertIsNone(i.close_gold)
        # The same producer keeps its issue and can create a new harvest batch.
        events.harvest(lot='h2',q=1); events.move('h2','shed2',1)
        self.assertEqual(events.ledger.producers['p'][1],i.task_id)
        self.assertTrue(r.open)
        self.assertEqual(r.total_delivered, {'WHEAT':2})
        # Crew quota has deliberately NOT been changed by manager reward redesign.
        self.assertEqual(events.ledger.report()['reward'],1.)
        b.close(i.task_id,TaskStatus.EXPIRED,step=8,prices={'WHEAT':10000})
        self.assertEqual(i.close_gold,0.)

    def test_real_engine_purchase_to_sale_uses_two_agent_namespaces_and_real_revenue(self):
        self._engine_purchase_to_sale(crew_site=False)

    def test_crew_selected_site_retains_issue_sale_credit(self):
        self._engine_purchase_to_sale(crew_site=True)

    def _engine_purchase_to_sale(self, *, crew_site):
        config={'seed':77,'episodeSteps':16,'turnsPerDay':4,'weedSpawnChance':0.}
        env=make('kaggriculture',configuration=config)
        book=IssueBook(realized_sales=True)
        initial=deepcopy(env.state[0].observation)
        xy=tuple(initial['farms'][0]['farmer'])
        issue=book.issue(initial,item='WHEAT',tile_xy=None if crew_site else xy,decision_id=999,step=0,day=0)
        if crew_site:
            self.assertIsNone(issue.tile_xy)
        tracker=CrewDeliveryTracker(initial,book.task_book,turns_per_day=4)
        cash=CashLedger([f['money'] for f in initial['farms']])
        refs={}
        script=[(['PASS'],[['BUY_SEED','WHEAT',1]]),(['PLANT','WHEAT'],[]),(['WATER'],[]),
                (['PASS'],[]),(['WATER'],[]),(['PASS'],[]),(['PASS'],[]),(['PASS'],[]),
                (['WATER'],[]),(['HARVEST'],[]),(['DROP'],[]),(['PASS'],[['SELL','WHEAT',1]])]
        for step,(unit,market) in enumerate(script):
            before=[deepcopy(s.observation) for s in env.state]
            for obs in before:obs['step']=step
            if step == 11:
                market=[['SELL','WHEAT',int(before[0]['private']['shed']['WHEAT'])]]
            action={'farmer':unit,'hands':[],'market':market}
            actions=[action,{'farmer':['PASS'],'hands':[],'market':[]}]
            executions=trace_market_executions(before,actions,config)
            env.step(actions)
            after=deepcopy(env.state[0].observation);after['step']=step+1
            # Deliberately reuse bare IDs across agents: they must not collide.
            crew=[SimpleNamespace(unit_index=0,decision_id=step)]
            manager=[SimpleNamespace(decision_id=step,boundary=False,action=a) for a in market]
            tracker.observe_transition(before[0],action,after,market_executions=executions[0],
                                       decisions=crew,market_decisions=manager)
            refs.update({(step,'market',i):2*step for i in range(len(market))})
            cash.observe(step,executions,[f['money'] for f in before[0]['farms']],
                         [f['money'] for f in after['farms']])
            book.sync_from_tasks(book.task_book,step=step,prices=before[0]['market']['prices'])
            if crew_site and step == 1:
                self.assertEqual(issue.tile_xy, xy)
                self.assertEqual(issue.bound_step, 1)
                self.assertEqual(issue.bound_decision_id, 1)
            if step==10:
                r=assign_issue_credit(tracker.lifecycle.events,cash.entries,refs,0,book.issues)
                self.assertEqual(r['allocated_gold'],-10.)
                self.assertLess(r['issue_gold'][issue.task_id],0.)
                self.assertIn(0, r['decision_credit'])
                self.assertIn(999, r['decision_credit'])
        delivery=tracker.ledger.finish()
        self.assertEqual(delivery['reward'],1.)
        self.assertNotIn(0,delivery['decision_credit'])  # manager BUY is not crew production
        self.assertNotIn(11,delivery['decision_credit']) # SELL is not a crew reward
        self.assertIn(1,delivery['decision_credit'])
        report=assign_issue_credit(tracker.lifecycle.events,cash.entries,refs,0,book.issues)
        actual=sum(e.amount for e in cash.entries if e.player==0)
        self.assertGreater(actual,0.)
        self.assertAlmostEqual(sum(report['decision_credit'].values()), actual)
        self.assertGreater(report['issue_gold'][issue.task_id], 0.)
        self.assertLess(report['issue_gold'][issue.task_id], actual)
        self.assertEqual(set(report['decision_credit']), {0, 999, 11})
        self.assertEqual(report['reconciliation_error'],0.)


if __name__ == '__main__': unittest.main()
