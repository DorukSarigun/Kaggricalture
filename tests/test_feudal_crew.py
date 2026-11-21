from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
import numpy as np
import torch
from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as game

from lifecycle_tracking import LifecycleEvent, LifecycleEventType as E
from market_execution import trace_market_executions
from worker_grow.contracts import ProductionTask, TaskBook, GOAL_DIM, TaskStatus
from worker_grow.crew import (FeudalCrew,CrewCursor,propose_crew_turn,statistics,attach_manager_market)
from worker_grow.delivery import DeliveryLedger,CrewDeliveryTracker
from worker_grow.training import finish_episode,update_crew
from worker_grow.train_crew import curriculum,parser,collect


def book(item='WHEAT',quota=4):
    product=item if item in game.CROPS else game.ANIMALS[item]['product']
    return TaskBook([ProductionTask('job',(0,0),item,((product,quota),))])


class Events:
    def __init__(self,tasks,**kwargs):
        self.ledger=DeliveryLedger(tasks,**kwargs);self.i=0
    def emit(self,kind,**kwargs):
        i=self.i;self.i+=1
        e=LifecycleEvent(event_id=f'e{i}',episode=1,step=i,day=0,hour=i,player=0,
            decision_id=kwargs.pop('decision_id',i),event_type=kind,**kwargs)
        self.ledger.observe(e)
        return e
    def plant(self):
        return self.emit(E.PLANTED,plant_id='p',crop='WHEAT',tile_xy=(0,0),data={'unit_index':0})
    def harvest(self,lot='h',q=4):
        return self.emit(E.HARVESTED,plant_id='p',crop='WHEAT',product_lot_id=lot,quantity=q,
                         data={'unit_index':1})
    def move(self,source,target,q,destination='shed',decision_id='auto'):
        kw={} if decision_id=='auto' else {'decision_id':decision_id}
        return self.emit(E.PRODUCT_MOVED,data={'source':'unit:0' if destination=='shed' else 'shed',
            'destination':destination,'allocations':[{'product_lot_id':source,
            'destination_product_lot_id':target,'quantity':q}]},**kw)


class ContractTests(unittest.TestCase):
    def test_identity_and_tile_reservation(self):
        b=book()
        with self.assertRaises(ValueError):b.add(ProductionTask('other',(0,0),'CARROT',(('CARROT',1),)))
        b.cancel('job');b.add(ProductionTask('other',(0,0),'CARROT',(('CARROT',1),)))
        with self.assertRaises(ValueError):b.add(ProductionTask('job',(1,0),'WHEAT',(('WHEAT',1),)))
        with self.assertRaises(ValueError):ProductionTask('bad',(0,0),'GOOSE',(('WOOL',1),))
        self.assertAlmostEqual(float(np.linalg.norm(b.goal())),1,places=6)

    def test_masks_respect_manager_plan_and_shadow_unit_order(self):
        env=make('kaggriculture',configuration={'seed':77})
        b=curriculum(env,'WHEAT','seeded',2,1)
        obs=deepcopy(env.state[0].observation);old=deepcopy(obs)
        obs['private']['seeds']['CARROT']=1
        cursor=CrewCursor(obs,b)
        self.assertEqual(cursor.items('PLANT'),('WHEAT',))
        self.assertNotIn('BUILD_COOP',cursor.operations())
        cursor.apply(['PLANT','WHEAT'])
        self.assertNotIn('PLANT',cursor.operations());self.assertIn('WATER',cursor.operations())
        cursor.apply(['WATER'])
        self.assertNotIn('WATER',cursor.operations())
        self.assertNotIn('DIG',cursor.operations())
        self.assertEqual(obs['farms'],old['farms'])

    def test_manager_market_runs_after_all_units_and_does_not_add_new_hand(self):
        env=make('kaggriculture',configuration={'seed':77})
        curriculum(env,'WHEAT','seeded',1,1)
        obs=deepcopy(env.state[0].observation)
        crew={'farmer':['PLANT','WHEAT'],'hands':[['WATER']],'market':[]}
        def manager(shadow):
            self.assertTrue(shadow['farms'][0]['tiles'][0][0]['watered_today'])
            return [['HIRE']]
        action=attach_manager_market(obs,crew,manager)
        self.assertEqual(len(action['hands']),1)
        self.assertEqual(action['market'],[['HIRE']])
        self.assertEqual(obs['private']['seeds']['WHEAT'],1)


class FeudalPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_whole_crew_recurrent_replay_and_gradient_boundary(self):
        torch.manual_seed(4)
        env=make('kaggriculture',configuration={'seed':77})
        b=curriculum(env,'WHEAT','seeded',2,1)
        model=FeudalCrew();obs=deepcopy(env.state[0].observation)
        action,rows,state=propose_crew_turn(model,obs,b)
        self.assertEqual(len(rows),3);self.assertEqual(len(action['hands']),2)
        self.assertEqual(action['market'],[])
        encoded=model.encode([r.inputs for r in rows])
        h=torch.zeros(model.hidden_dim)
        goal=torch.tensor(rows[0].inputs.goal_sum,requires_grad=True)
        loss=0
        for i,r in enumerate(rows):
            h,w,v=model.step(encoded[i],h,goal)
            logp,_=statistics(model,h,w,r)
            self.assertAlmostEqual(float(logp.detach()),r.log_prob,places=5)
            loss=loss-logp+v.square()
            h=model.after_action(h,r.action)
        loss.backward()
        self.assertIsNone(goal.grad)
        self.assertIsNotNone(model.phi.weight.grad)
        self.assertGreater(float(model.recurrent.weight_hh.grad.abs().sum()),0)
        np.testing.assert_allclose(h.detach().numpy(),state.hidden.numpy(),atol=1e-6)
        with self.assertRaises(ValueError):propose_crew_turn(model,obs,b,state)

    def test_goal_changes_logits_and_no_opponent_input(self):
        model=FeudalCrew();h=torch.randn(model.hidden_dim)
        g=torch.zeros(GOAL_DIM);g[0]=1
        w=model.phi(g);z=model.phi(-g)
        self.assertFalse(torch.allclose(model.op_U(h).view(-1,model.goal_rank)@w,
                                       model.op_U(h).view(-1,model.goal_rank)@z))
        env=make('kaggriculture',configuration={'seed':77});obs=deepcopy(env.state[0].observation)
        b=book();a,rows,_=propose_crew_turn(model,obs,b,deterministic=True)
        obs['farms'][1]['money']=999999;obs['farms'][1]['farmer']=[9,9]
        a2,rows2,_=propose_crew_turn(model,obs,b,deterministic=True)
        self.assertEqual(a,a2);self.assertEqual(rows[0].log_prob,rows2[0].log_prob)

    def test_quantity_has_full_support(self):
        m=FeudalCrew();h=torch.zeros(m.hidden_dim);w=m.phi(torch.ones(GOAL_DIM))
        d=m.quantity_distribution(h,w,'PICKUP','WHEAT',17)
        self.assertEqual(d.probs.numel(),17);self.assertTrue(torch.all(d.probs>0))


class DeliveryTests(unittest.TestCase):
    @staticmethod
    def play(env,tracker,script,config):
        decision=0
        for step,(unit,market) in enumerate(script):
            before=[deepcopy(s.observation) for s in env.state]
            for obs in before:obs['step']=step
            action={'farmer':unit,'hands':[],'market':market}
            actions=[action,{'farmer':['PASS'],'hands':[],'market':[]}]
            executions=trace_market_executions(before,actions,config)
            env.step(actions);after=deepcopy(env.state[0].observation);after['step']=step+1
            rows=[SimpleNamespace(unit_index=0,decision_id=decision)];decision+=1
            tracker.observe_transition(before[0],action,after,market_executions=executions[0],decisions=rows)

    def test_split_delivery_no_double_pay_and_only_crew_ancestors(self):
        b=book();j=Events(b)
        plant=j.plant();harvest=j.harvest()
        drop=j.move('h','s',2)
        j.move('s','u',1,'unit:0');j.move('u','again',1)
        j.move('h','rest',2,decision_id=None)
        report=j.ledger.report()
        self.assertEqual(report['reward'],4)
        self.assertEqual(len(report['deliveries']),2)
        self.assertEqual(report['paid_units'],{'h':4})
        self.assertEqual(b.records['job'].status,TaskStatus.COMPLETE)
        self.assertIn(plant.decision_id,report['decision_credit'])
        self.assertIn(harvest.decision_id,report['decision_credit'])
        self.assertIn(drop.decision_id,report['decision_credit'])
        self.assertNotIn(3,report['decision_credit']) # pickup already rewarded inventory
        self.assertAlmostEqual(sum(report['decision_credit'].values()),4)

    def test_delivery_bonus_scales_first_shed_quota(self):
        with self.assertRaises(ValueError):
            DeliveryLedger(book(), delivery_bonus=0)
        j=Events(book(), delivery_bonus=2.)
        j.plant();j.harvest();j.move('h','s',4)
        report=j.ledger.report()
        self.assertEqual(report['delivery_reward'],8)
        self.assertEqual(report['grow_reward'],0)
        self.assertEqual(report['reward'],8)
        self.assertAlmostEqual(sum(report['decision_credit'].values()),8)

    def test_no_bonus_for_build_dig_purchase_resale_or_undelivered_harvest(self):
        j=Events(book())
        for _ in range(4):
            j.emit(E.STRUCTURE_BUILT,tile_xy=(1,1),data={'unit_index':0,'kind':'COOP'})
            j.emit(E.TILE_CLEARED,tile_xy=(1,1),data={'unit_index':0})
        j.emit(E.PRODUCT_PURCHASED,data={'product_lot_ids':['b'],'item':'WHEAT'})
        j.move('b','b2',1)
        j.emit(E.PRODUCT_SOLD,decision_id=999,data={'allocations':[{'product_lot_id':'b2','quantity':1}]})
        j.plant();j.harvest()
        self.assertEqual(j.ledger.report()['reward'],0)

    def test_quota_and_cancellation_do_not_create_more_reward(self):
        b=book(quota=2);j=Events(b);j.plant();j.harvest(q=5);j.move('h','s',5)
        self.assertEqual(j.ledger.report()['reward'],2)
        b2=book();j2=Events(b2);j2.plant();j2.harvest();b2.cancel('job');j2.move('h','s',4)
        self.assertEqual(j2.ledger.report()['reward'],0)

    def test_new_harvest_snapshot_excludes_future_care(self):
        j=Events(book(quota=8));j.plant();j.harvest('h1',4)
        care=j.emit(E.WATERED,plant_id='p',data={'unit_index':0})
        j.harvest('h2',4);j.move('h1','s1',4);j.move('h2','s2',4)
        first,second=j.ledger.report()['deliveries']
        self.assertNotIn(care.decision_id,first['shares'])
        self.assertIn(care.decision_id,second['shares'])

    def test_missing_lot_and_duplicate_event_fail_closed(self):
        j=Events(book())
        with self.assertRaises(ValueError):j.move('missing','new',1)
        j=Events(book());e=j.plant()
        with self.assertRaises(ValueError):j.ledger.observe(e)

    def test_actual_engine_all_crops_animals_night_delivery_and_sales(self):
        for item in (*game.CROPS,*game.ANIMALS):
            with self.subTest(item=item):
                config={'seed':77,'episodeSteps':30,'turnsPerDay':24,'weedSpawnChance':0.}
                env=make('kaggriculture',configuration=config)
                tasks=curriculum(env,item,'mature',1,1)
                tracker=CrewDeliveryTracker(deepcopy(env.state[0].observation),tasks)
                product=item if item in game.CROPS else game.ANIMALS[item]['product']
                for step in range(25):
                    before=[deepcopy(s.observation) for s in env.state]
                    for o in before:o['step']=step
                    # Deliberately leave the harvest in a hand until night.
                    hands=[['HARVEST']] if step==0 else [['PASS']]
                    if step>=24:hands=[]
                    action={'farmer':['PASS'],'hands':hands,'market':[['SELL',product,1]] if step==24 else []}
                    actions=[action,{'farmer':['PASS'],'hands':[],'market':[]}]
                    executions=trace_market_executions(before,actions,config)
                    env.step(actions);after=deepcopy(env.state[0].observation);after['step']=step+1
                    rows=[SimpleNamespace(unit_index=i,decision_id=step*2+i) for i in range(1+len(hands))]
                    tracker.observe_transition(before[0],action,after,market_executions=executions[0],decisions=rows)
                report=tracker.ledger.report()
                self.assertEqual(report['reward'],1)
                self.assertTrue(report['deliveries'][0]['automatic'])
                self.assertIn(1,report['decision_credit'])
                self.assertNotIn(48,report['decision_credit'])

    def test_seed_purchase_to_plant_water_harvest_delivery(self):
        config={'seed':77,'episodeSteps':40,'turnsPerDay':4,'weedSpawnChance':0.}
        env=make('kaggriculture',configuration=config)
        xy=tuple(env.state[0].observation.farms[0].farmer)
        tasks=TaskBook([ProductionTask('job',xy,'WHEAT',(('WHEAT',1),))])
        tracker=CrewDeliveryTracker(deepcopy(env.state[0].observation),tasks,turns_per_day=4)
        script=[(['PASS'],[['BUY_SEED','WHEAT',1]]),(['PLANT','WHEAT'],[]),(['WATER'],[]),(['PASS'],[]),
                (['WATER'],[]),(['PASS'],[]),(['PASS'],[]),(['PASS'],[]),
                (['WATER'],[]),(['HARVEST'],[]),(['DROP'],[])]
        self.play(env,tracker,script,config)
        r=tracker.ledger.finish()
        self.assertEqual(r['reward'],1)
        self.assertNotIn(0,r['decision_credit']) # PASS + manager purchase: not crew production credit
        for decision in (1,2,4,8,9,10):self.assertIn(decision,r['decision_credit'])

    def test_animal_purchase_build_place_feed_and_two_products(self):
        config={'seed':77,'episodeSteps':80,'turnsPerDay':8,'weedSpawnChance':0.}
        env=make('kaggriculture',configuration=config)
        xy=tuple(env.state[0].observation.farms[0].farmer)
        tasks=TaskBook([ProductionTask('job',xy,'GOOSE',(('EGG',2),('FERTILIZER',2)))])
        tracker=CrewDeliveryTracker(deepcopy(env.state[0].observation),tasks,turns_per_day=8)
        script=[(['BUILD_COOP'],[['BUY_ANIMAL','GOOSE',1],['BUY_PRODUCT','WHEAT',10]]),
                (['PICKUP','GOOSE'],[]),(['PLACE','GOOSE'],[]),(['PICKUP','WHEAT',1],[]),
                (['FEED'],[]),(['CARE'],[]),(['PASS'],[]),(['PASS'],[])]
        for _ in range(6):
            script += [(['PICKUP','WHEAT',1],[]),(['FEED'],[]),(['CARE'],[]),(['HARVEST'],[]),
                       (['COLLECT_FERTILIZER'],[]),(['DROP'],[]),(['PASS'],[]),(['PASS'],[])]
        self.play(env,tracker,script,config)
        r=tracker.ledger.finish()
        self.assertEqual(r['reward'],4)
        self.assertEqual(r['tasks']['job']['delivered'],{'FERTILIZER':2,'EGG':2})
        for decision in (0,1,2,3,4):self.assertIn(decision,r['decision_credit'])

    def test_full_shed_overflow_is_not_delivery(self):
        config={'seed':77,'episodeSteps':26,'turnsPerDay':24,'weedSpawnChance':0.}
        env=make('kaggriculture',configuration=config)
        tasks=curriculum(env,'WHEAT','mature',0,1)
        env.state[0].observation.private['shed']['WHEAT']=100
        tracker=CrewDeliveryTracker(deepcopy(env.state[0].observation),tasks)
        self.play(env,tracker,[(['HARVEST'],[])]+[(['PASS'],[])]*23,config)
        r=tracker.ledger.finish()
        self.assertEqual(r['reward'],0)
        self.assertEqual(sum(r['created_units'].values()),1)
        self.assertEqual(r['tasks']['lesson:0']['status'],'EXPIRED')

    def test_held_product_failure_has_no_fictitious_penalty_or_reward(self):
        j=Events(book());j.plant();j.harvest()
        j.emit(E.PLANT_LOST,plant_id='p',data={'reason':'DIED'})
        self.assertEqual(j.ledger.report()['reward'],0)
        j.move('h','s',4) # the already harvested output is still real
        self.assertEqual(j.ledger.finish()['reward'],4)

    def test_bound_grow_events_pay_the_acting_decision_even_if_later_lost(self):
        j=Events(book());j.ledger.grow_reward=1.
        plant=j.plant()
        water=j.emit(E.WATERED,plant_id='p',data={'unit_index':0})
        j.emit(E.PLANT_LOST,plant_id='p',data={'reason':'DIED'})
        report=j.ledger.report()
        self.assertEqual(report['delivery_reward'],0)
        self.assertEqual(report['grow_reward'],2)
        self.assertEqual(report['reward'],2)
        self.assertEqual(report['decision_credit'][plant.decision_id],1)
        self.assertEqual(report['decision_credit'][water.decision_id],1)
        empty=Events(book());empty.ledger.grow_reward=1.
        empty.emit(E.STRUCTURE_BUILT,tile_xy=(1,1),data={'unit_index':0,'kind':'COOP'})
        empty.emit(E.TILE_CLEARED,tile_xy=(1,1),data={'unit_index':0})
        self.assertEqual(empty.ledger.report()['grow_reward'],0)


class TrainingTests(unittest.TestCase):
    def test_collection_and_ppo_are_separate_from_manager_cash(self):
        torch.set_num_threads(1);torch.manual_seed(2)
        with tempfile.TemporaryDirectory() as directory:
            args=parser().parse_args(['--run-dir',directory,'--episode-steps','25','--save-every','1'])
            model=FeudalCrew()
            rows,summary=collect(model,args,1,Path(directory)/'episode')
            self.assertTrue(summary['gifted_curriculum'])
            self.assertAlmostEqual(sum(r.credit for r in rows),summary['crew_reward'])
            old=model.phi.weight.detach().clone()
            metrics=update_crew(model,torch.optim.Adam(model.parameters(),lr=1e-4),[rows],epochs=1,target_kl=1.)
            self.assertGreater(metrics['chunks'],0)
            self.assertFalse(torch.equal(old,model.phi.weight))
            self.assertTrue((Path(directory)/'episode'/'replay.json.gz').exists())
            self.assertTrue((Path(directory)/'episode'/'delivery_credit.json').exists())


if __name__=='__main__':unittest.main()
