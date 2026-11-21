"""Site selection belongs to crew; real events alone commit reservations."""
from copy import deepcopy
from types import SimpleNamespace
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch
from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as game

from tests.test_manager import observation
from worker_grow.contracts import TASK_ITEMS, SPATIAL_CHANNELS, TaskStatus
from worker_grow.issues import IssueBook
from worker_grow.crew import CrewCursor, FeudalCrew, VERSION as CREW_VERSION
from worker_grow.manager import (
    ManagerActor, propose_manager_contracts, manager_action_statistics,
)
from worker_grow.manager_workload import admission_items
from worker_grow.scripted_manager import BestAgentTeacher
from worker_grow.train_crew import curriculum
from worker_grow.delivery import CrewDeliveryTracker
from worker_grow.train_joint import _initialize
from market_execution import trace_market_executions


class CrewSiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def issue(self, book, obs, item='WHEAT'):
        return book.issue(obs, item=item, decision_id=len(book.issues), step=0, day=0)

    def test_demand_has_no_fake_location_and_counts_duplicate_issues(self):
        obs=observation(hands=1);book=IssueBook()
        a=self.issue(book,obs);self.issue(book,obs)
        self.assertIsNone(a.tile_xy)
        self.assertIsNone(a.as_dict()['tile_xy'])
        planes=book.task_book.planes()
        self.assertFalse(planes[:SPATIAL_CHANNELS].any())
        np.testing.assert_array_equal(planes[SPATIAL_CHANNELS+TASK_ITEMS.index('WHEAT')],2)
        self.assertAlmostEqual(float(np.linalg.norm(book.task_book.goal())),1.,places=6)

    def test_two_units_choose_distinct_sites_fifo_and_shadow_does_not_leak(self):
        obs=observation(hands=1);obs['farms'][0]['farmer']=[1,2]
        obs['farms'][0]['hands']=[[3,1]];obs['private']['seeds']['WHEAT']=2
        book=IssueBook();a=self.issue(book,obs);b=self.issue(book,obs)
        cursor=CrewCursor(obs,book.task_book)
        cursor.apply(['PLANT','WHEAT']);cursor.apply(['PLANT','WHEAT'])
        self.assertEqual(cursor.tasks.records[a.task_id].tile_xy,(1,2))
        self.assertEqual(cursor.tasks.records[b.task_id].tile_xy,(3,1))
        self.assertIsNone(book.task_book.records[a.task_id].tile_xy)
        self.assertIsNone(book.task_book.records[b.task_id].tile_xy)
        self.assertEqual(obs['private']['seeds']['WHEAT'],2)

    def test_one_issue_cannot_be_planted_twice_in_same_turn(self):
        obs=observation(hands=1);obs['farms'][0]['farmer']=[1,2]
        obs['farms'][0]['hands']=[[3,1]];obs['private']['seeds']['WHEAT']=2
        book=IssueBook();self.issue(book,obs)
        c=CrewCursor(obs,book.task_book);c.apply(['PLANT','WHEAT'])
        self.assertNotIn('PLANT',c.operations())

    def test_same_tile_next_unit_can_water_new_plant(self):
        obs=observation(hands=1);obs['farms'][0]['farmer']=[0,0]
        obs['private']['seeds']['WHEAT']=1
        b=IssueBook();self.issue(b,obs)
        c=CrewCursor(obs,b.task_book);c.apply(['PLANT','WHEAT'])
        self.assertIn('WATER',c.operations());c.apply(['WATER'])
        self.assertTrue(c.rules.farm['tiles'][0][0]['watered_today'])

    def test_no_resources_locked_tiles_and_other_products_do_not_bind(self):
        obs=observation();b=IssueBook();i=self.issue(b,obs)
        obs['private']['seeds']={'CARROT':1}
        c=CrewCursor(obs,b.task_book)
        with self.assertRaises(ValueError):c.apply(['PLANT','CARROT'])
        self.assertIsNone(b.task_book.records[i.task_id].tile_xy)
        obs['private']['seeds']['WHEAT']=1;obs['farms'][0]['farmer']=[9,9]
        c=CrewCursor(obs,b.task_book);self.assertNotIn('PLANT',c.operations())

    def test_real_animal_build_then_place_commits_same_issue(self):
        for animal in game.ANIMALS:
            with self.subTest(animal=animal):
                cfg={'seed':77,'episodeSteps':6,'weedSpawnChance':0.}
                env=make('kaggriculture',configuration=cfg)
                curriculum(env,animal,'seeded',1,1)
                env.state[0].observation.private['shed'].pop(animal)
                env.state[0].observation.private['inventories'][1][animal]=1
                obs=deepcopy(env.state[0].observation);obs['step']=0
                b=IssueBook(realized_sales=True);issue=self.issue(b,obs,animal)
                tracker=CrewDeliveryTracker(obs,b.task_book)
                c=CrewCursor(obs,b.task_book)
                c.apply(['BUILD_'+game.ANIMALS[animal]['structure']])
                c.apply(['PLACE',animal,1])
                self.assertIsNone(b.task_book.records[issue.task_id].tile_xy)
                actions=[c.actions,{'farmer':['PASS'],'hands':[],'market':[]}]
                before=[deepcopy(s.observation) for s in env.state]
                executions=trace_market_executions(before,actions,cfg);env.step(actions)
                after=deepcopy(env.state[0].observation);after['step']=1
                tracker.observe_transition(obs,c.actions,after,market_executions=executions[0],
                    decisions=[SimpleNamespace(unit_index=u,decision_id=u) for u in (0,1)])
                b.sync_from_tasks(b.task_book,step=0,prices=obs['market']['prices'])
                self.assertEqual(issue.tile_xy,(0,0))
                self.assertIsNotNone(issue.producer_id)
                self.assertEqual(issue.bound_decision_id,0)
                self.assertEqual(issue.status,TaskStatus.ACTIVE)

    def test_manager_samples_only_op_and_item_and_replays_logprob(self):
        m=ManagerActor();self.assertFalse(hasattr(m,'tile_score'))
        with torch.no_grad():
            m.contract_op.weight.zero_();m.contract_op.bias.copy_(torch.tensor([20.,-20.]))
        b=IssueBook();rows,_=propose_manager_contracts(m,observation(),b,step=0,day=0)
        self.assertEqual(len(rows[0].action),2)
        self.assertIsNone(b.open_issues()[0].tile_xy)
        self.assertGreater(len(b.open_issues()), 4)
        self.assertFalse(admission_items(observation(), b))
        self.assertEqual(rows[-1].action, ['STOP'])
        self.assertFalse(rows[-1].stochastic)
        self.assertGreater(len(b.open_issues()), 1 + len(observation()['farms'][0]['hands']))
        logp,_=manager_action_statistics(m,rows[0])
        self.assertAlmostEqual(float(logp.detach()),rows[0].log_prob,places=5)
        (-logp).backward();self.assertIsNotNone(m.item_score.weight.grad)

    def test_scripted_issue_uses_ghost_tile_not_live_farmer(self):
        env=make('kaggriculture',configuration={'seed':77,'episodeSteps':8,'weedSpawnChance':0.})
        start=tuple(env.state[0].observation['farms'][0]['farmer'])
        self.assertNotEqual(start,(9,9))
        teacher=BestAgentTeacher(lambda obs:{'farmer':['PLANT','MELON'],'hands':[],'market':[]})
        cfg={'seed':77,'episodeSteps':8,'weedSpawnChance':0.}
        teacher.attach_world(cfg)
        live=deepcopy(env.state[0].observation);live['step']=0;live['farms'][0]['farmer']=[9,9]
        b=IssueBook()
        opened=teacher.propose_contracts(live,b,step=0,day=0)
        self.assertEqual(len(opened),1)
        self.assertNotEqual(opened[0].tile_xy,(9,9))
        self.assertNotIn(opened[0].tile_xy,{(4,4),(5,4),(4,5),(5,5)})
        pin=opened[0].tile_xy
        planes=b.task_book.planes()
        self.assertEqual(float(planes[TASK_ITEMS.index('MELON')][pin[1], pin[0]]),1)

    def test_old_crew_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'old.pt';torch.save({'version':'feudal-contract-crew-v1'},p)
            with self.assertRaisesRegex(ValueError,'Incompatible'):
                _initialize(FeudalCrew(),p,CREW_VERSION,'cpu')


if __name__=='__main__':unittest.main()
