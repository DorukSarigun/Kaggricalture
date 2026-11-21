from copy import deepcopy
from types import SimpleNamespace
import unittest
import numpy as np
import torch

from tests.test_manager import observation
from tests.test_feudal_crew import Events
from worker_grow.contracts import ProductionTask,TaskBook,TaskStatus
from worker_grow.crew import FeudalCrew,CrewCursor,propose_crew_turn,statistics
from worker_grow.assignments import Subtask,candidates,available,operation_mask,IDLE
from lifecycle_tracking import LifecycleEventType as E
from kaggle_environments import make
from market_execution import trace_market_executions
from worker_grow.delivery import CrewDeliveryTracker
from worker_grow.issues import IssueBook


def task(key,xy,item='WHEAT'):
    return ProductionTask(key,xy,item,((item if item in ('WHEAT','CARROT') else 'MILK',1),))

def decision(i,action,subtask,segment,step=None):
    return SimpleNamespace(decision_id=i,step=i if step is None else step,unit_index=0,
        action=action,subtask=subtask,assignment_id=segment,assignment_change='assign',assignment_choice=0)


class AssignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_ready_target_uses_manager_tile_and_preserves_observation(self):
        obs=observation();obs['private']['seeds']['WHEAT']=1
        b=TaskBook([task('a',(1,2))]);old=deepcopy(obs)
        c=CrewCursor(obs,b);ops=candidates(c)
        planting=[s for s in ops if s.operation=='PLANT']
        self.assertEqual([s.tile_xy for s in planting],[(1,2)])
        self.assertEqual(obs,old);self.assertEqual(c.observation(),old)
        selected=planting[0]
        self.assertNotIn('PLANT',operation_mask(c,selected))
        self.assertIn('WEST',operation_mask(c,selected))

    def test_unpinned_issues_only_offer_nearest_sites(self):
        from worker_grow.assignments import UNPINNED_SITES
        obs=observation();obs['private']['seeds']['WHEAT']=8
        obs['farms'][0]['farmer']=[0,0]
        book=IssueBook()
        for _ in range(8):
            book.issue(obs,item='WHEAT',decision_id=len(book.issues),step=0,day=0)
        ops=candidates(CrewCursor(obs,book.task_book))
        sites={s.tile_xy for s in ops if s.operation=='PLANT'}
        self.assertLessEqual(len(sites),UNPINNED_SITES)
        self.assertIn((0,0),sites)
        self.assertNotIn((4,4),sites)
        oldest=next(iter(book.task_book.records))
        self.assertTrue(all(s.issue_id==oldest for s in ops if s.operation=='PLANT'))

    def test_missing_seed_waits_and_missing_feed_offers_fetch(self):
        obs=observation();b=TaskBook([task('a',(1,2))])
        self.assertFalse(any(s.operation=='PLANT' for s in candidates(CrewCursor(obs,b))))
        b=TaskBook([task('cow',(1,2),'COW')]);obs['private']['shed']['COW']=1
        picks=[s for s in candidates(CrewCursor(obs,b)) if s.operation=='PICKUP' and s.item=='COW']
        self.assertEqual(len(picks),4);self.assertTrue(all(s.issue_id=='cow' for s in picks))

    def test_bound_unfunded_issue_under_worker_does_not_hide_other_jobs(self):
        for blocked_item in ('GOOSE', 'WHEAT'):
            with self.subTest(blocked_item=blocked_item):
                obs=observation();obs['farms'][0]['farmer']=[3,4]
                obs['private']['seeds']['WHEAT']=1
                b=IssueBook()
                pinned=b.issue(obs,item=blocked_item,tile_xy=(3,4),decision_id=0,step=0,day=0)
                pending=b.issue(obs,item='WHEAT',decision_id=1,step=0,day=0)
                if blocked_item=='GOOSE':
                    obs['farms'][0]['tiles'][4][3]={'kind':'COOP'}
                snapshot=deepcopy(obs)
                c=CrewCursor(obs,b.task_book)
                plants=[s for s in candidates(c) if s.operation=='PLANT' and s.issue_id==pending.task_id]
                self.assertTrue(plants)
                self.assertTrue(all(s.tile_xy!=(3,4) for s in plants))
                self.assertEqual(c.observation(),snapshot)
                self.assertIsNone(b.task_book.records[pending.task_id].tile_xy)
                self.assertEqual(b.task_book.records[pinned.task_id].tile_xy,(3,4))

    def test_reserved_nearest_sites_do_not_hide_free_distant_site(self):
        from worker_grow.assignments import _unpinned_sites
        obs=observation();farm=obs['farms'][0];farm['farmer']=[0,0]
        obs['private']['seeds']['WHEAT']=1
        b=IssueBook()
        for xy in _unpinned_sites(farm,(0,0)):
            b.issue(obs,item='GOOSE',tile_xy=xy,decision_id=len(b.issues),step=0,day=0)
        pending=b.issue(obs,item='WHEAT',decision_id=len(b.issues),step=0,day=0)
        plants=[s for s in candidates(CrewCursor(obs,b.task_book)) if s.operation=='PLANT']
        self.assertTrue(plants)
        self.assertTrue(all(s.issue_id==pending.task_id for s in plants))

    def test_unbound_animal_can_reuse_unowned_empty_structure(self):
        obs=observation();obs['farms'][0]['tiles']=[['LOCKED']*10 for _ in range(10)]
        obs['farms'][0]['tiles'][0][0]={'kind':'COOP'}
        obs['private']['inventories'][0]['GOOSE']=1
        b=IssueBook();i=b.issue(obs,item='GOOSE',decision_id=0,step=0,day=0)
        places=[s for s in candidates(CrewCursor(obs,b.task_book)) if s.operation=='PLACE']
        self.assertEqual([(s.issue_id,s.tile_xy) for s in places],[(i.task_id,(0,0))])

    def test_unbound_build_is_never_assigned_to_crop_or_wrong_animal(self):
        obs=observation();b=TaskBook([task('w',None),task('c',None,'COW'),
            ProductionTask('g',None,'GOOSE',(('EGG',1),))])
        jobs=candidates(CrewCursor(obs,b))
        for s in jobs:
            if s.operation=='BUILD_COOP':self.assertEqual(s.issue_id,'g')
            if s.operation=='BUILD_PASTURE':self.assertEqual(s.issue_id,'c')

    def test_reservations_are_subjob_scoped_and_can_be_released(self):
        a=Subtask('a','WATER',(1,1),day=0)
        b=Subtask('a','HARVEST',(1,1),day=0)
        leases={0:{'subtask':a,'day':0,'step':4,'id':'a'}}
        self.assertNotIn(a,available((IDLE,a,b),leases,1,4,0))
        self.assertIn(b,available((IDLE,a,b),leases,1,4,0))
        self.assertIn(a,available((IDLE,a,b),leases,0,4,0))
        leases.pop(0)
        self.assertIn(a,available((IDLE,a,b),leases,1,4,0))
        leases[0]={'subtask':a,'day':0,'step':4,'id':'a'}
        available((IDLE,a,b),leases,1,24,1)
        self.assertFalse(leases)

    def test_cancelled_jobs_disappear(self):
        obs=observation();obs['private']['seeds']['WHEAT']=1
        b=TaskBook([task('a',(1,2))]);b.cancel('a')
        self.assertEqual(candidates(CrewCursor(obs,b)),(IDLE,))

    def test_assignment_and_primitive_logprob_replay_and_gradients(self):
        torch.manual_seed(17);m=FeudalCrew();obs=observation(hands=1)
        obs['private']['seeds']['WHEAT']=2
        b=TaskBook([task('a',(1,2)),task('b',(3,3))])
        _,rows,state=propose_crew_turn(m,obs,b)
        h=torch.zeros(m.hidden_dim);loss=0
        for row,enc in zip(rows,m.encode([r.inputs for r in rows])):
            h,w,_=m.step(enc,h,row.inputs.goal_sum)
            logp,_=statistics(m,h,w,row)
            self.assertAlmostEqual(float(logp.detach()),row.log_prob,places=5)
            loss-=logp;h=m.after_action(h,row.action)
        loss.backward()
        self.assertGreater(float(m.assignment_score[-1].weight.grad.abs().sum()),0)
        np.testing.assert_allclose(h.detach().numpy(),state.hidden.numpy(),atol=1e-6)

    def test_reassignment_does_not_give_abandoned_path_to_new_product(self):
        j=Events(TaskBook([task('a',(0,0)),task('b',(1,1))]))
        a=Subtask('a','PLANT',(0,0),'WHEAT');b=Subtask('b','PLANT',(1,1),'WHEAT')
        rows=[decision(0,['EAST'],a,'A'),decision(1,['SOUTH'],b,'B'),
              decision(2,['PLANT','WHEAT'],b,'B')]
        j.ledger.register_decisions(rows)
        j.emit(E.UNIT_MOVED,data={'unit_index':0})
        j.emit(E.UNIT_MOVED,data={'unit_index':0})
        j.emit(E.PLANTED,plant_id='p',crop='WHEAT',tile_xy=(1,1),data={'unit_index':0})
        j.harvest(q=1);j.move('h','s',1)
        r=j.ledger.report()
        self.assertNotIn(0,r['decision_credit'])
        self.assertGreater(r['decision_credit'][1],0)
        self.assertIn('A',r['uncompleted_paths'])
        self.assertNotIn('B',r['uncompleted_paths'])
        self.assertEqual(r['navigation'][0]['issue_id'],'b')

    def test_walk_cost_once_even_noop_no_positive_completion(self):
        j=Events(TaskBook([task('a',(0,0))]));j.ledger.walk_cost=.01
        row=decision(0,['WEST'],Subtask('a','PLANT',(0,0),'WHEAT'),'A')
        j.ledger.register_decisions([row]);j.ledger.charge_walks([row],[])
        r=j.ledger.report()
        self.assertEqual(r['reward'],-.01);self.assertEqual(r['decision_credit'],{0:-.01})
        self.assertFalse(r['walk_penalties'][0]['moved'])
        self.assertFalse(r['navigation'])
        with self.assertRaisesRegex(ValueError,'twice'):j.ledger.charge_walks([row],[])

    def test_wrong_issue_completion_does_not_consume_path(self):
        j=Events(TaskBook([task('a',(0,0)),task('b',(1,1))]))
        a=Subtask('a','PLANT',(0,0),'WHEAT')
        j.ledger.register_decisions([decision(0,['EAST'],a,'A'),decision(1,['PLANT','WHEAT'],a,'A')])
        j.emit(E.UNIT_MOVED,data={'unit_index':0})
        j.emit(E.PLANTED,plant_id='p',crop='WHEAT',tile_xy=(1,1),data={'unit_index':0})
        self.assertIn('A',j.ledger.assignment_paths)
        self.assertFalse(j.ledger.navigation)

    def test_legacy_transfer_preserves_base_and_rejects_missing_parameters(self):
        src=FeudalCrew();dest=FeudalCrew()
        legacy={k:v for k,v in src.state_dict().items()
                if not k.startswith(('assignment_score.','assignment_condition.'))}
        dest.load_initial_weights({'version':'feudal-contract-crew-site-v2','model':legacy})
        for k,v in legacy.items():self.assertTrue(torch.equal(dest.state_dict()[k],v),k)
        self.assertFalse(dest.assignment_condition.weight.any())
        legacy.pop('phi.weight')
        with self.assertRaises(ValueError):dest.load_initial_weights({'version':'feudal-contract-crew-site-v2','model':legacy})

    def test_real_engine_walk_to_pinned_plant_and_deliver_has_verified_credit(self):
        cfg={'seed':77,'episodeSteps':16,'turnsPerDay':4,'weedSpawnChance':0.}
        env=make('kaggriculture',configuration=cfg);initial=deepcopy(env.state[0].observation)
        b=IssueBook();issue=b.issue(initial,item='WHEAT',tile_xy=(3,4),decision_id=999,step=0,day=0)
        tr=CrewDeliveryTracker(initial,b.task_book,turns_per_day=4,walk_cost=.001)
        script=[(['WEST'],[['BUY_SEED','WHEAT',1]]),(['PLANT','WHEAT'],[]),(['WATER'],[]),
            (['PASS'],[]),(['WEST'],[]),(['WATER'],[]),(['PASS'],[]),(['PASS'],[]),
            (['WEST'],[]),(['WATER'],[]),(['HARVEST'],[]),(['EAST'],[])]
        for step,(unit,market) in enumerate(script):
            before=[deepcopy(s.observation) for s in env.state]
            for o in before:o['step']=step
            op='PLANT' if step in (0,1) else 'DROP' if step==11 else 'WATER' if step in (4,8) else unit[0]
            sub=Subtask(None if op=='DROP' else issue.task_id,op,
                (4,4) if op=='DROP' else (3,4),'WHEAT' if op=='PLANT' else None,
                None if op in ('PLANT','DROP') else issue.producer_id,int(before[0]['day']))
            segment=('plant' if step in (0,1) else 'drop' if step==11 else
                     'water1' if step in (4,5) else 'water2' if step in (8,9) else str(step))
            row=decision(step,unit,sub,segment)
            actions=[{'farmer':unit,'hands':[],'market':market},
                     {'farmer':['PASS'],'hands':[],'market':[]}]
            executions=trace_market_executions(before,actions,cfg);env.step(actions)
            after=deepcopy(env.state[0].observation);after['step']=step+1
            tr.observe_transition(before[0],actions[0],after,market_executions=executions[0],decisions=[row])
            b.sync_from_tasks(b.task_book,step=step,prices=before[0]['market']['prices'])
        r=tr.ledger.finish()
        self.assertEqual(r['walk_attempts'],4);self.assertAlmostEqual(r['walk_reward'],-.004)
        self.assertGreater(r['decision_credit'][0],0)
        self.assertTrue(any(n['assignment_id']=='plant' and n['issue_id']==issue.task_id for n in r['navigation']))
        self.assertAlmostEqual(sum(r['decision_credit'].values()),r['reward'])


if __name__=='__main__':unittest.main()
