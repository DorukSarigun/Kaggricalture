from copy import deepcopy
from collections import Counter
from dataclasses import replace, asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import random
import unittest

import numpy as np
import torch
from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as game

from lifecycle_tracking import LifecycleEventType as E
from market_execution import trace_market_executions
from worker_grow.assignments import Subtask, candidates
from worker_grow.contracts import TaskBook, ProductionTask
from worker_grow.crew import FeudalCrew, CrewCursor, propose_crew_turn
from worker_grow.delivery import CrewDeliveryTracker
from worker_grow.mastery import LESSONS, MasteryGate, MasteryLedger, segment_returns
from worker_grow.train_mastery import evaluate, parser
from worker_grow.training import finish_episode, update_crew
from tests.test_feudal_crew import Events
from tests.test_crew_assignments import decision, task


class MasteryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_fixtures_are_varied_reproducible_and_have_only_exact_seed_supply(self):
        starts=set();targets=set()
        for seed in range(12):
            lesson=LESSONS[4]
            env=make('kaggriculture',configuration={'seed':1})
            tasks=lesson.setup(env,seed)
            obs=deepcopy(env.state[0].observation)
            tile=next(iter(tasks.records.values())).tile_xy
            starts.add(tuple(obs['farms'][0]['farmer']));targets.add(tile)
            self.assertEqual(obs['private']['seeds'],{'WHEAT':1})
            self.assertNotIn('money',obs['private'])
            self.assertIsNone(obs['farms'][0]['tiles'][tile[1]][tile[0]])
            self.assertEqual(sum(abs(a-b) for a,b in zip(tile,obs['farms'][0]['farmer'])),2)
            env2=make('kaggriculture',configuration={'seed':1});lesson.setup(env2,seed)
            self.assertEqual(obs,env2.state[0].observation)
        self.assertGreater(len(starts),3);self.assertGreater(len(targets),3)

    def test_gate_requires_consecutive_heldout_success_and_speed(self):
        g=MasteryGate()
        good=[{'success':True,'turns':3}]*18+[{'success':False,'turns':24}]*2
        self.assertFalse(g.assess(good)['promoted'])
        g.assess([{'success':False,'turns':24}]*20)
        self.assertEqual(g.passes,0)
        g.assess(good)
        self.assertTrue(g.assess(good)['promoted']);self.assertEqual(g.level,1)
        slow=[{'success':True,'turns':24}]*20
        self.assertFalse(g.assess(slow)['passed'])
        with self.assertRaises(ValueError):g.assess([])
        restored=MasteryGate(**asdict(g));self.assertEqual(asdict(restored),asdict(g))

    def test_segment_returns_include_future_costs_but_not_replacement_or_other_worker(self):
        rows=[SimpleNamespace(unit_index=u,assignment_id=s,credit=c,target_return=None)
              for u,s,c in [(0,'A',-.01),(1,'B',1.),(0,'A',-.01),(0,'C',2.)]]
        segment_returns(rows)
        self.assertEqual([r.target_return for r in rows],[-.02,1.,-.01,2.])
        self.assertEqual([r.credit for r in rows],[-.01,1.,-.01,2.])

    def test_bounded_support_shares_real_approach_once_and_keeps_accounting(self):
        j=Events(TaskBook([task('a',(0,0))]))
        j.ledger=MasteryLedger(j.ledger.tasks,grow_reward=.1,progress_cap=.15,decision_cost=.001,walk_cost=.002)
        sub=Subtask('a','PLANT',(0,0),'WHEAT')
        rows=[decision(0,['EAST'],sub,'route'),decision(1,['PLANT','WHEAT'],sub,'route')]
        for r in rows:r.stochastic=True
        j.ledger.register_decisions(rows)
        movement=j.emit(E.UNIT_MOVED,data={'unit_index':0})
        plant=j.plant()
        j.ledger.charge_walks(rows,[movement,plant])
        r=j.ledger.report()
        self.assertGreater(r['decision_credit'][0],0)
        self.assertAlmostEqual(r['grow_reward'],.1)
        self.assertAlmostEqual(r['reward'],.096)
        # Repeated report must not change credits; same subjob cannot pay twice.
        self.assertEqual(r,j.ledger.report())
        j.ledger.award_progress(plant,'a')
        self.assertEqual(r,j.ledger.report())
        water=Subtask('a','WATER',(0,0),producer_id='p')
        row=decision(2,['WATER'],water,'water');row.stochastic=True
        j.ledger.register_decisions([row])
        j.emit(E.WATERED,plant_id='p',crop='WHEAT',tile_xy=(0,0),data={'unit_index':0})
        r=j.ledger.report()
        self.assertAlmostEqual(r['grow_reward'],.15)
        self.assertAlmostEqual(r['reward'],sum(r['decision_credit'].values()))

    def test_abandoned_route_and_unverified_operation_get_no_support(self):
        j=Events(TaskBook([task('a',(0,0)),task('b',(1,1))]))
        j.ledger=MasteryLedger(j.ledger.tasks,grow_reward=.1)
        a=Subtask('a','PLANT',(0,0),'WHEAT')
        j.ledger.register_decisions([decision(0,['EAST'],a,'old'),decision(1,['PLANT','WHEAT'],a,'old')])
        j.emit(E.UNIT_MOVED,data={'unit_index':0})
        j.emit(E.PLANTED,plant_id='p',crop='WHEAT',tile_xy=(1,1),data={'unit_index':0})
        self.assertEqual(j.ledger.report()['grow_reward'],0)
        self.assertIn('old',j.ledger.assignment_paths)

    def test_evaluation_preserves_weights_and_rng_and_writes_separate_artifacts(self):
        m=FeudalCrew();m.train()
        weights={k:v.clone() for k,v in m.state_dict().items()}
        rng=torch.get_rng_state().clone();py=random.getstate();nrng=np.random.get_state()
        with TemporaryDirectory() as d:
            args=parser().parse_args(['--run-dir',d,'--device','cpu','--eval-episodes','2'])
            args.stage='mastery'
            result=evaluate(m,args,MasteryGate(),Path(d),1)
            self.assertEqual(len(result['seeds']),2)
            self.assertGreater(min(result['seeds']),10_000_000)
            self.assertTrue((Path(d)/'evaluations/update_000001/episode_0001/decisions.json.gz').exists())
            self.assertFalse((Path(d)/'episodes').exists())
        self.assertTrue(m.training);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(py,random.getstate());np.testing.assert_array_equal(nrng[1],np.random.get_state()[1])
        for k,v in m.state_dict().items():self.assertTrue(torch.equal(v,weights[k]))

    def test_ppo_uses_return_target_not_ledger_credit(self):
        env=make('kaggriculture',configuration={'seed':5})
        tasks=LESSONS[0].setup(env,5);model=FeudalCrew()
        _,rows,_=propose_crew_turn(model,deepcopy(env.state[0].observation),tasks)
        for row in rows:row.credit=0.;row.target_return=2.
        with TemporaryDirectory() as d:
            path=Path(d)/'batches.jsonl'
            result=update_crew(model,torch.optim.Adam(model.parameters(),lr=1e-5),[rows],epochs=1,batch_log=path)
            self.assertGreater(result['value_loss'],.1)
            self.assertIn('"target_return": 2.0',path.read_text())
        self.assertEqual(rows[0].credit,0.)

    def test_every_lesson_is_solvable_in_real_engine_with_test_only_controller(self):
        # This scripted controller is TEST ONLY, never used to collect train/eval.
        lessons=(*LESSONS,*(replace(LESSONS[-1],name='single_'+item,items=(item,),tasks=1,hands=0)
                           for item in game.CROPS))
        for index,lesson in enumerate(lessons):
            with self.subTest(lesson=lesson.name):
                cfg={'seed':80+index,'episodeSteps':lesson.steps,'turnsPerDay':24,'weedSpawnChance':0.}
                env=make('kaggriculture',configuration=cfg);tasks=lesson.setup(env,90+index)
                tracker=CrewDeliveryTracker(deepcopy(env.state[0].observation),tasks,
                    ledger_class=MasteryLedger,grow_reward=lesson.progress)
                serial=0;leases={};attempts=Counter()
                for step in range(lesson.steps-1):
                    obs=[deepcopy(s.observation) for s in env.state]
                    for o in obs:o['step']=step
                    cursor=CrewCursor(obs[0],tasks);rows=[]
                    while cursor.section=='unit':
                        unit=cursor.rules.unit_index;x,y=cursor.rules._position()
                        options=candidates(cursor)
                        priority={'HARVEST':0,'WATER':1,'FEED':1,'DROP':9,'PLACE':3,
                                  'PLANT':3,'BUILD_COOP':4,'BUILD_PASTURE':4,'PICKUP':5,'CARE':6}
                        options=[s for s in options if s.operation in priority]
                        s=min(options,key=lambda s:(priority[s.operation],abs(s.tile_xy[0]-x)+abs(s.tile_xy[1]-y))) if options else Subtask(None,'PASS',None)
                        old=leases.get(unit)
                        segment=old[1] if old and old[0].key==s.key else str(serial)
                        leases[unit]=(s,segment)
                        if s.tile_xy and (x,y)!=s.tile_xy:
                            tx,ty=s.tile_xy
                            op='EAST' if tx>x else 'WEST' if tx<x else 'SOUTH' if ty>y else 'NORTH'
                            action=[op]
                        else:
                            action=[s.operation]
                            items=cursor.items(s.operation)
                            if items:action.append(s.item if s.item in items else items[0])
                            limit=cursor.quantity_limit(s.operation,action[1] if items else None)
                            if limit:action.append(min(4,limit))
                        row=decision(serial,action,s,segment,step);row.unit_index=unit;row.stochastic=True
                        rows.append(row);serial+=1;attempts[tuple(action)]+=1;cursor.apply(action)
                    actions=[deepcopy(cursor.actions),{'farmer':['PASS'],'hands':[],'market':[]}]
                    actions[0]['market']=lesson.market(obs[0])
                    market=trace_market_executions(obs,actions,cfg);env.step(actions)
                    after=deepcopy(env.state[0].observation);after['step']=step+1
                    tracker.observe_transition(obs[0],actions[0],after,market_executions=market[0],decisions=rows)
                    if not any(r.open for r in tasks.records.values()):break
                report=tracker.ledger.finish()
                self.assertTrue(all(r['status']=='COMPLETE' for r in report['tasks'].values()),
                    ({k:(r['item'],r['status']) for k,r in report['tasks'].items()},attempts))
                self.assertAlmostEqual(report['reward'],report['allocated'])


if __name__=='__main__':unittest.main()
