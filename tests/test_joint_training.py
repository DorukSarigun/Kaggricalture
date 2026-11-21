from copy import deepcopy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from worker_grow.crew import FeudalCrew
from worker_grow.manager import ManagerActor, CONTRACT_OPS
from worker_grow.train_manager import collect
from worker_grow.train_joint import parser, run, update_both, validate, VERSION
from worker_grow.manager import ManagerDecision
from worker_grow.manager_training import apply_market_cash, finish_manager_episode
from cash_accounting import CashEntry


class JointTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def args(self, root, *extra):
        return parser().parse_args([
            '--run-dir', str(root), '--device', 'cpu', '--opponent', 'pass',
            '--episodes', '2', '--episode-steps', '5', '--epochs', '1',
            '--episodes-per-update', '2', '--save-every', '1', *extra,
        ])

    def test_defaults_both_trainable_and_configuration_validation(self):
        args = parser().parse_args(['--run-dir', 'unused'])
        self.assertEqual(args.device, 'cuda')
        self.assertEqual(args.episode_steps, 720)
        self.assertFalse(hasattr(args, 'crew'))  # No pass/frozen choice in joint CLI.
        args.device = 'cpu'
        for key, bad in [('credit_decay', 0), ('gae_lambda', float('nan')),
                         ('crew_learning_rate', 0), ('sequence_length', 0)]:
            invalid = deepcopy(args)
            setattr(invalid, key, bad)
            with self.assertRaises(ValueError): validate(invalid)

    def test_collect_update_both_agents_logging_and_no_cross_agent_data(self):
        torch.manual_seed(7)
        manager, crew = ManagerActor(), FeudalCrew()
        # Encourage actual tasks so the crew has stochastic decisions, not forced PASS only.
        with torch.no_grad():
            manager.contract_op.weight.zero_()
            manager.contract_op.bias[CONTRACT_OPS.index('ISSUE')] = 20
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.args(root)
            episodes = [collect(manager, crew, args, ep,
                        root / 'episodes' / f'episode_{ep:06d}', joint=True) for ep in (1, 2)]
            for ep, (mr, cr, summary) in enumerate(episodes, 1):
                self.assertTrue(all(r.episode == ep for r in mr))
                self.assertEqual([r.decision_id for r in cr], list(range(len(cr))))
                self.assertGreater(sum(r.stochastic for r in cr), 0)
                folder = root / 'episodes' / f'episode_{ep:06d}'
                with gzip.open(folder / 'trace.jsonl.gz', 'rt') as file:
                    trace = [json.loads(line) for line in file]
                self.assertEqual(len(trace), summary['turns'])
                self.assertEqual(sum(len(t['crew_decision_ids']) for t in trace), len(cr))
                self.assertEqual(sum(r.credit for r in cr), summary['crew_reward'])
                self.assertTrue((folder / 'replay.json.gz').exists())
                self.assertFalse(summary['gifted_curriculum'])
                credit=json.loads((folder/'issue_sale_credit.json').read_text())
                self.assertAlmostEqual(sum(r.local_credit for r in mr),credit['net_cash'])
                self.assertEqual(credit['reconciliation_error'],0.)
                self.assertTrue((folder/'issue_production_graph.json').exists())
                self.assertTrue(all(i['close_gold']==0 for i in summary['issues'].values()))
                for t in trace:
                    farm = t['observation']['farms'][t['observation']['player']]
                    self.assertEqual(len(t['crew_decision_ids']), 1 + len(farm['hands']))
            m_before = {k: v.clone() for k, v in manager.state_dict().items()}
            c_before = {k: v.clone() for k, v in crew.state_dict().items()}
            mo, co = torch.optim.Adam(manager.parameters(), lr=3e-5), torch.optim.Adam(crew.parameters(), lr=3e-5)
            from worker_grow.train_joint import update_manager as real_update_manager
            def isolated(*pos, **kw):
                result = real_update_manager(*pos, **kw)
                self.assertTrue(all(torch.equal(v, crew.state_dict()[k]) for k, v in c_before.items()))
                return result
            with patch('worker_grow.train_joint.update_manager', side_effect=isolated):
                metrics = update_both(manager, crew, mo, co, episodes, args, root / 'update', 1)
            self.assertGreater(metrics['manager']['minibatches'], 0)
            self.assertGreater(metrics['crew']['chunks'], 0)
            self.assertTrue(any(not torch.equal(v, manager.state_dict()[k]) for k, v in m_before.items()))
            self.assertFalse(torch.equal(c_before['op_U.weight'], crew.op_U.weight))
            self.assertTrue(mo.state and co.state)
            manager_log = [json.loads(l) for l in (root/'update'/'manager_batches.jsonl').read_text().splitlines()]
            self.assertEqual({d['episode'] for b in manager_log for d in b['decisions']}, {1, 2})
            crew_log = [json.loads(l) for l in (root/'update'/'crew_batches.jsonl').read_text().splitlines()]
            self.assertEqual({b['episode_in_update'] for b in crew_log}, {0, 1})

    def test_checkpoint_resume_both_models_and_protect_partial_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'run'
            run(self.args(root))
            saved = torch.load(root/'latest.pt', weights_only=False)
            self.assertEqual(saved['version'], VERSION)
            self.assertEqual(saved['episode'], 2)
            self.assertTrue(saved['manager_optimizer']['state'])
            self.assertTrue(saved['crew_optimizer']['state'])
            # A no-op resume restores both checkpoints but must not update either.
            run(self.args(root, '--resume'))
            same = torch.load(root/'latest.pt', weights_only=False)
            for agent in ('manager', 'crew'):
                self.assertTrue(all(torch.equal(v, same[agent][k]) for k, v in saved[agent].items()))
            run(self.args(root, '--resume', '--episodes', '3'))
            resumed = torch.load(root/'latest.pt', weights_only=False)
            self.assertEqual((resumed['episode'], resumed['update']), (3, 2))
            for agent in ('manager', 'crew'):
                self.assertTrue(any(not torch.equal(v, resumed[agent][k]) for k, v in saved[agent].items()))
            with self.assertRaises(FileExistsError): run(self.args(root))
            (root/'episodes'/'episode_000004').mkdir()
            with self.assertRaisesRegex(RuntimeError, 'Uncheckpointed'):
                run(self.args(root, '--resume', '--episodes', '4'))

    def test_joint_rejects_absent_or_frozen_crew(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager, crew = ManagerActor(), FeudalCrew()
            with self.assertRaises(ValueError):
                collect(manager, None, self.args(root), 1, root/'a', joint=True)
            crew.requires_grad_(False)
            with self.assertRaises(ValueError):
                collect(manager, crew, self.args(root), 1, root/'a', joint=True)

    def test_purchase_without_any_issue_still_has_full_negative_manager_credit(self):
        buy = ManagerDecision(None, None, 0., kind='market', action=['BUY_SEED', 'WHEAT', 10])
        boundary = ManagerDecision(None, None, 0., kind='boundary', boundary=True, decision_id=1)
        apply_market_cash([buy, boundary], [CashEntry(0, 0, 0, 'BUY_SEED', 'WHEAT', 10, -100.)], 0, 0)
        finish_manager_episode([buy, boundary], -100., [])
        self.assertAlmostEqual(buy.local_credit, -100)
        self.assertAlmostEqual(boundary.reward, -100)

    def test_resume_matches_uninterrupted_cpu_training_for_both_agents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run(self.args(root/'full', '--episodes-per-update', '1'))
            run(self.args(root/'split', '--episodes-per-update', '1', '--episodes', '1'))
            run(self.args(root/'split', '--episodes-per-update', '1', '--resume'))
            full = torch.load(root/'full'/'latest.pt', weights_only=False)
            split = torch.load(root/'split'/'latest.pt', weights_only=False)
            for agent in ('manager', 'crew'):
                self.assertTrue(all(torch.equal(v, split[agent][k]) for k, v in full[agent].items()))
            self.assertTrue(torch.equal(full['torch_rng'], split['torch_rng']))

    def test_old_joint_reward_checkpoint_is_not_silently_resumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            torch.save({'version':'feudal-joint-v1'},root/'latest.pt')
            with self.assertRaisesRegex(ValueError,'compatible joint'):
                run(self.args(root,'--resume'))
            self.assertFalse((root/'config.json').exists())


if __name__ == '__main__':
    unittest.main()
