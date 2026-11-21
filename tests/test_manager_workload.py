from copy import copy, deepcopy
import unittest

import torch

from tests.test_manager import observation
from worker_grow.issues import IssueBook
from worker_grow.contracts import TaskStatus
from worker_grow.crew import CrewCursor
from worker_grow.manager import ManagerActor, propose_manager_contracts, propose_manager_market, manager_action_statistics
from worker_grow.manager_workload import admission_items, workload_features, WORKLOAD_DIM, WORKLOAD_NAMES


class ManagerWorkloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_worker_count_does_not_limit_issue_admission(self):
        obs=observation(hands=1);b=IssueBook(realized_sales=True)
        for n in range(8):
            b.issue(obs,item='WHEAT',decision_id=n,step=0,day=0)
        self.assertTrue(admission_items(obs,b))
        b.task_book.records['issue:1'].status=TaskStatus.COMPLETE
        self.assertTrue(admission_items(obs,b))
        obs['farms'][0]['hands']=[]
        self.assertTrue(admission_items(obs,b))
        self.assertEqual(sum(r.open for r in b.task_book.records.values()),7)

    def test_physical_sites_limit_admission_but_seed_money_is_not_required(self):
        obs=observation();b=IssueBook()
        obs['farms'][0]['money']=0
        self.assertIn('WHEAT',admission_items(obs,b))
        obs['farms'][0]['tiles']=[['LOCKED']*10 for _ in range(10)]
        self.assertFalse(admission_items(obs,b))
        obs['farms'][0]['tiles'][0][0]=None
        b.issue(obs,item='WHEAT',decision_id=0,step=0,day=0)
        self.assertFalse(admission_items(obs,b))

    def test_each_issue_changes_snapshot_and_ppo_replays_frozen_inputs(self):
        torch.manual_seed(40);m=ManagerActor();obs=observation();b=IssueBook(realized_sales=True)
        obs['farms'][0]['tiles']=[['LOCKED']*10 for _ in range(10)]
        obs['farms'][0]['tiles'][0][:2]=[None,None]
        with torch.no_grad():
            m.contract_op.weight.zero_();m.contract_op.bias.copy_(torch.tensor([20.,-20.]))
        rows,_=propose_manager_contracts(m,obs,b,step=0,day=0)
        self.assertEqual(len(b.issues),2)
        self.assertEqual(rows[-1].allowed_ops,('STOP',))
        self.assertFalse(rows[-1].stochastic)
        self.assertEqual(rows[-1].log_prob,0.)
        self.assertTrue(all(r.inputs.section=='contract' for r in rows))
        self.assertNotEqual(rows[0].inputs.workload,rows[1].inputs.workload)
        frozen=rows[0].inputs.workload
        b.issue(obs,item='MELON',decision_id=99,step=1,day=0)
        self.assertEqual(frozen,rows[0].inputs.workload)
        for r in rows:
            logp,_=manager_action_statistics(m,r)
            self.assertAlmostEqual(float(logp.detach()),r.log_prob,places=5)
            _,v,local,throughput=m.body([r.inputs])
            self.assertAlmostEqual(float(v.detach()[0]),r.value,places=5)
            self.assertAlmostEqual(float(local.detach()[0]),r.chain_value,places=5)
            self.assertAlmostEqual(float(throughput.detach()[0]),r.throughput_value,places=5)
        logp,_=manager_action_statistics(m,rows[0]);(-logp).backward()
        self.assertGreater(float(m.body.demand_context[0].weight.grad.abs().sum()),0.)

    def test_pending_crops_cannot_reserve_animal_only_capacity(self):
        obs=observation();b=IssueBook()
        obs['farms'][0]['tiles']=[['LOCKED']*10 for _ in range(10)]
        obs['farms'][0]['tiles'][0][:3]=[None,{'kind':'COOP'},{'kind':'PASTURE'}]
        b.issue(obs,item='WHEAT',decision_id=0,step=0,day=0)
        allowed=admission_items(obs,b)
        self.assertNotIn('CARROT',allowed)
        self.assertEqual(set(allowed),{'GOOSE','COW','SHEEP'})
        b.issue(obs,item='COW',decision_id=1,step=0,day=0)
        self.assertEqual(admission_items(obs,b),('GOOSE',))

    def test_demand_and_supply_change_missing_input_feature(self):
        obs=observation();b=IssueBook(realized_sales=True)
        empty=workload_features(obs,b)
        b.issue(obs,item='WHEAT',decision_id=0,step=0,day=0)
        need=workload_features(obs,b)
        stocked=deepcopy(obs);stocked['private']['seeds']['WHEAT']=1
        bought=workload_features(stocked,b)
        i=WORKLOAD_NAMES.index('WHEAT.missing_inputs')
        self.assertEqual(len(empty),WORKLOAD_DIM)
        self.assertEqual(empty[i],0.)
        self.assertGreater(need[i],0.)
        self.assertEqual(bought[i],0.)

    def test_market_uses_demand_and_replays_same_logprob(self):
        torch.manual_seed(42);m=ManagerActor();obs=observation();b=IssueBook(realized_sales=True)
        b.issue(obs,item='WHEAT',decision_id=0,step=0,day=0)
        _,rows,_=propose_manager_market(m,obs,step=0,book=b)
        self.assertEqual(rows[-1].inputs.section,'boundary')
        self.assertEqual(rows[0].inputs.workload,workload_features(obs,b))
        for r in rows:
            if r.boundary:continue
            logp,_=manager_action_statistics(m,r)
            self.assertAlmostEqual(float(logp.detach()),r.log_prob,places=5)

    def test_crew_shadow_binding_changes_market_demand_without_touching_real_book(self):
        obs=observation();obs['farms'][0]['farmer']=[0,0]
        obs['private']['seeds']['WHEAT']=1
        b=IssueBook(realized_sales=True)
        issue=b.issue(obs,item='WHEAT',decision_id=0,step=0,day=0)
        cursor=CrewCursor(obs,b.task_book);cursor.apply(['PLANT','WHEAT'])
        market_book=copy(b);market_book.task_book=cursor.tasks
        encoded=workload_features(cursor.observation(),market_book)
        self.assertEqual(encoded[WORKLOAD_NAMES.index('WHEAT.missing_inputs')],0.)
        self.assertGreater(encoded[WORKLOAD_NAMES.index('WHEAT.active')],0.)
        self.assertIsNone(b.task_book.records[issue.task_id].producer_id)
        self.assertIsNone(b.task_book.records[issue.task_id].tile_xy)


if __name__=='__main__':unittest.main()
