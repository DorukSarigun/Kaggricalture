"""Production occupancy is not the lifetime of harvested goods or their credit."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import unittest

import torch
from kaggle_environments import make

from tests.test_manager import observation
from tests.test_feudal_crew import Events
from worker_grow.contracts import TaskStatus
from worker_grow.issues import IssueBook
from worker_grow.crew import CrewCursor
from worker_grow.delivery import CrewDeliveryTracker
from worker_grow.assignments import candidates
from worker_grow.manager_workload import admission_items
from worker_grow.train_crew import curriculum
from lifecycle_tracking import LifecycleEventType as E
from market_execution import trace_market_executions


class TaskReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def source(self, quantity=1, *, close=True):
        obs=observation();b=IssueBook(realized_sales=True)
        i=b.issue(obs,item='WHEAT',tile_xy=(0,0),decision_id=100,step=0,day=0,
                  quotas=(('WHEAT',quantity),))
        j=Events(b.task_book);j.plant()
        j.emit(E.HARVESTED,plant_id='p',crop='WHEAT',product_lot_id='h',quantity=quantity,
               data={'unit_index':0,'plant_closed':close})
        return obs,b,i,j

    def consumer(self,b,j,item='GOOSE',xy=(1,0)):
        i=b.issue(observation(),item=item,tile_xy=xy,decision_id=101,step=0,day=0)
        if item=='GOOSE':
            j.emit(E.ANIMAL_PLACED,animal_id='a',tile_xy=xy,data={'animal':item,'unit_index':0})
        else:
            j.emit(E.PLANTED,plant_id='new',tile_xy=xy,crop=item,data={'unit_index':0})
        return i

    def test_replant_and_deliver_old_lot_without_overwriting_ancestry(self):
        obs,b,old,j=self.source()
        record=b.task_book.records[old.task_id]
        self.assertEqual(record.status,TaskStatus.DELIVERING)
        self.assertFalse(record.reserves_tile)
        self.assertIsNone(b.task_book.at((0,0)))
        self.assertEqual(record.tile_xy,(0,0))
        self.assertNotIn((0,0),b.occupied_tiles())
        self.assertFalse(b.task_book.planes().any())
        self.assertTrue(admission_items(obs,b))
        new=b.issue(obs,item='WHEAT',tile_xy=(0,0),decision_id=102,step=2,day=0)
        obs['farms'][0]['farmer']=[0,0];obs['private']['seeds']['WHEAT']=1
        c=CrewCursor(obs,b.task_book)
        self.assertIn('PLANT',c.operations())
        self.assertTrue(any(s.issue_id==new.task_id and s.operation=='PLANT' for s in candidates(c)))
        j.emit(E.PLANTED,plant_id='new',crop='WHEAT',tile_xy=(0,0),data={'unit_index':0})
        j.move('h','shed-old',1)
        self.assertEqual(record.status,TaskStatus.COMPLETE)
        self.assertEqual(j.ledger.deliveries[0]['task_id'],old.task_id)
        self.assertEqual(j.ledger.producers['p'][1],old.task_id)
        self.assertEqual(j.ledger.producers['new'][1],new.task_id)
        self.assertIs(b.task_book.at((0,0)),b.task_book.records[new.task_id])
        self.assertEqual(j.ledger.report()['reward'],1.)

    def test_feed_fulfills_closed_source_without_delivery_or_money_reward(self):
        _,b,old,j=self.source();self.consumer(b,j)
        j.emit(E.ANIMAL_FED,animal_id='a',data={'consumed_allocations':[{'product_lot_id':'h','quantity':1}]})
        record=b.task_book.records[old.task_id]
        self.assertEqual(record.status,TaskStatus.COMPLETE)
        self.assertEqual(record.consumed,{'WHEAT':1})
        self.assertEqual(record.delivered,{})
        self.assertEqual(j.ledger.report()['reward'],0.)
        b.sync_from_tasks(b.task_book,step=10,prices={})
        self.assertEqual(old.close_gold,0.)
        self.assertEqual(old.consumed,{'WHEAT':1})
        self.assertEqual(old.producer_id,'p')

    def test_partial_consumption_and_delivery_fulfill_distinct_units(self):
        _,b,old,j=self.source(2);self.consumer(b,j)
        j.emit(E.ANIMAL_FED,animal_id='a',data={'consumed_allocations':[{'product_lot_id':'h','quantity':1}]})
        record=b.task_book.records[old.task_id]
        self.assertEqual(record.status,TaskStatus.DELIVERING)
        self.assertFalse(record.reserves_tile)
        j.move('h','shed-old',1)
        self.assertEqual(record.status,TaskStatus.COMPLETE)
        self.assertEqual(record.consumed,{'WHEAT':1})
        self.assertEqual(record.delivered,{'WHEAT':1})
        self.assertEqual(j.ledger.report()['reward'],1.)
        # Feeding an already delivered unit does not fulfill another quota unit.
        j.emit(E.ANIMAL_FED,animal_id='a',data={'consumed_allocations':[{'product_lot_id':'shed-old','quantity':1}]})
        self.assertEqual(record.consumed,{'WHEAT':1})
        self.assertEqual(j.ledger.report()['reward'],1.)

    def test_consumption_does_not_steal_existing_delivery_reward_budget(self):
        _,b,old,j=self.source(2)
        r=b.task_book.records[old.task_id]
        r.task=replace(r.task,quotas=(('WHEAT',1),))
        self.consumer(b,j)
        j.emit(E.ANIMAL_FED,animal_id='a',data={'consumed_allocations':[{'product_lot_id':'h','quantity':1}]})
        j.move('h','shed-old',1)
        self.assertEqual(j.ledger.report()['reward'],1.)

    def test_living_repeating_producer_stays_reserved_after_consumption(self):
        _,b,old,j=self.source(close=False);self.consumer(b,j)
        j.emit(E.ANIMAL_FED,animal_id='a',data={'consumed_allocations':[{'product_lot_id':'h','quantity':1}]})
        record=b.task_book.records[old.task_id]
        self.assertEqual(record.status,TaskStatus.ACTIVE)
        self.assertTrue(record.reserves_tile)
        self.assertIs(b.task_book.at((0,0)),record)

    def test_actual_engine_harvest_then_other_worker_plants_same_turn(self):
        cfg={'seed':77,'episodeSteps':8,'weedSpawnChance':0.}
        env=make('kaggriculture',configuration=cfg)
        tasks=curriculum(env,'WHEAT','mature',1,1)
        obs=deepcopy(env.state[0].observation);obs['step']=0
        old=next(iter(tasks.records.values()));xy=old.tile_xy
        # Both workers at the producer, with a seed for the next crop.
        env.state[0].observation.farms[0]['farmer']=list(xy)
        env.state[0].observation.farms[0]['hands']=[list(xy)]
        env.state[0].observation.private['seeds']['WHEAT']=1
        from worker_grow.contracts import ProductionTask
        tasks.add(ProductionTask('next',None,'WHEAT',(('WHEAT',1),),True))
        before=[deepcopy(s.observation) for s in env.state]
        for o in before:o['step']=0
        tracker=CrewDeliveryTracker(before[0],tasks)
        cursor=CrewCursor(before[0],tasks)
        cursor.apply(['HARVEST'])
        self.assertFalse(cursor.tasks.records[old.task.task_id].reserves_tile)
        self.assertIn('PLANT',cursor.operations())
        cursor.apply(['PLANT','WHEAT'])
        self.assertTrue(old.reserves_tile)  # real ledger untouched until env success
        actions=[cursor.actions,{'farmer':['PASS'],'hands':[],'market':[]}]
        executions=trace_market_executions(before,actions,cfg);env.step(actions)
        after=deepcopy(env.state[0].observation);after['step']=1
        tracker.observe_transition(before[0],cursor.actions,after,market_executions=executions[0],
                                   decisions=[SimpleNamespace(unit_index=i,decision_id=i) for i in (0,1)])
        self.assertTrue(old.production_closed)
        self.assertEqual(tasks.at(xy).task.task_id,'next')
        self.assertNotEqual(old.producer_id,tasks.at(xy).producer_id)

    def test_fertilizer_consumption_fulfills_quota_but_preserves_live_animal(self):
        b=IssueBook(realized_sales=True)
        old=b.issue(observation(),item='GOOSE',tile_xy=(0,0),decision_id=100,step=0,day=0,
                    quotas=(('FERTILIZER',1),))
        j=Events(b.task_book)
        j.emit(E.ANIMAL_PLACED,animal_id='a',tile_xy=(0,0),data={'animal':'GOOSE','unit_index':0})
        j.emit(E.FERTILIZER_COLLECTED,animal_id='a',product_lot_id='f',quantity=1,
               data={'unit_index':0})
        self.consumer(b,j,item='CARROT')
        j.emit(E.FERTILIZED,plant_id='new',data={'consumed_allocations':[{'product_lot_id':'f','quantity':1}]})
        r=b.task_book.records[old.task_id]
        self.assertEqual(r.consumed,{'FERTILIZER':1})
        self.assertTrue(r.reserves_tile)
        self.assertTrue(r.fulfilled)
        self.assertEqual(r.status,TaskStatus.ACTIVE)
        self.assertEqual(j.ledger.report()['reward'],0.)


if __name__=='__main__':unittest.main()
