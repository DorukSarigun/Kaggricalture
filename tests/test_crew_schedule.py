from copy import deepcopy
import tempfile
from pathlib import Path
import unittest
import torch

from tests.test_manager import observation
from worker_grow.contracts import ProductionTask, TaskBook
from worker_grow.crew import CrewCursor, CrewState, FeudalCrew, propose_crew_turn
from worker_grow.schedule import STRATEGIES, greedy_pick, plan_turn, worker_views
from worker_grow.train_crew import collect, parser


def wheat(key, xy):
    return ProductionTask(key, xy, 'WHEAT', (('WHEAT', 1),))


class CrewScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_parser_lists_every_planner(self):
        args = parser().parse_args(['--run-dir', 'unused', '--assignment-strategy', 'cpsat'])
        self.assertEqual(args.assignment_strategy, 'cpsat')
        self.assertEqual(set(STRATEGIES),
                         {'learned', 'greedy', 'hungarian', 'tour', 'cbba', 'cpsat'})

    def test_greedy_waters_live_plant_before_new_sowing(self):
        from kaggle_environments.envs.kaggriculture import kaggriculture as game
        obs = observation()
        obs['farms'][0]['farmer'] = [1, 1]
        obs['farms'][0]['tiles'][1][1] = game._new_plant('WHEAT', 0, 24)
        obs['farms'][0]['tiles'][1][1]['watered_today'] = False
        obs['private']['seeds']['WHEAT'] = 1
        book = TaskBook([wheat('live', (1, 1)), wheat('next', (8, 8))])
        book.records['live'].producer_id = 'p'
        from worker_grow.contracts import TaskStatus
        book.records['live'].status = TaskStatus.ACTIVE
        views = worker_views(obs, book, {}, 0, 0)
        pick = greedy_pick(views[0], obs, views)
        self.assertEqual(pick.operation, 'WATER')
        self.assertEqual(pick.issue_id, 'live')

    def test_hungarian_gives_two_workers_distinct_jobs(self):
        from kaggle_environments.envs.kaggriculture import kaggriculture as game
        from worker_grow.contracts import TaskStatus
        obs = observation(hands=1)
        obs['farms'][0]['farmer'] = [1, 1]
        obs['farms'][0]['hands'] = [[3, 1]]
        obs['farms'][0]['tiles'][1][1] = game._new_plant('WHEAT', 0, 24)
        obs['farms'][0]['tiles'][1][1]['watered_today'] = False
        obs['private']['seeds']['WHEAT'] = 1
        book = TaskBook([wheat('live', (1, 1)), wheat('next', (3, 1))])
        book.records['live'].producer_id = 'p'
        book.records['live'].status = TaskStatus.ACTIVE
        planned = plan_turn('hungarian', obs, book, CrewState(), 0, 0)
        ops = {index: job.operation for index, job in planned.selected.items()}
        self.assertEqual(ops[0], 'WATER')
        self.assertEqual(ops[1], 'PLANT')

    def test_planners_do_not_share_a_water_reservation(self):
        from kaggle_environments.envs.kaggriculture import kaggriculture as game
        from worker_grow.contracts import TaskStatus
        obs = observation(hands=1)
        obs['farms'][0]['farmer'] = [1, 1]
        obs['farms'][0]['hands'] = [[2, 1]]
        obs['farms'][0]['tiles'][1][1] = game._new_plant('WHEAT', 0, 24)
        obs['private']['seeds']['WHEAT'] = 0
        book = TaskBook([wheat('live', (1, 1))])
        book.records['live'].producer_id = 'p'
        book.records['live'].status = TaskStatus.ACTIVE
        for name in ('greedy', 'hungarian', 'tour', 'cbba', 'cpsat'):
            planned = plan_turn(name, deepcopy(obs), deepcopy(book), CrewState(), 0, 0)
            waters = [job for job in planned.selected.values() if job.operation == 'WATER']
            self.assertEqual(len(waters), 1, name)

    def test_does_not_dump_animal_input_when_place_is_ready(self):
        obs = observation()
        obs['farms'][0]['farmer'] = [4, 4]
        obs['private']['inventories'][0]['COW'] = 1
        book = TaskBook([ProductionTask('cow', (2, 2), 'COW', (('MILK', 1),))])
        views = worker_views(obs, book, {}, 0, 0)
        pick = greedy_pick(views[0], obs, views)
        self.assertNotEqual(pick.operation, 'DROP')
        self.assertIn(pick.operation, ('PLACE', 'BUILD_PASTURE'))
        from kaggle_environments.envs.kaggriculture import kaggriculture as game
        from worker_grow.contracts import TaskStatus
        obs = observation()
        obs['farms'][0]['farmer'] = [1, 1]
        obs['farms'][0]['tiles'][1][1] = game._new_plant('WHEAT', 0, 24)
        book = TaskBook([wheat('live', (1, 1))])
        book.records['live'].producer_id = 'p'
        book.records['live'].status = TaskStatus.ACTIVE
        model = FeudalCrew()
        action, rows, _ = propose_crew_turn(
            model, obs, book, assignment_strategy='greedy', walk_policy='greedy')
        self.assertEqual(action['farmer'][0], 'WATER')
        self.assertEqual(rows[0].subtask.operation, 'WATER')

    def test_greedy_tie_does_not_compare_none_to_str(self):
        from types import SimpleNamespace
        from kaggle_environments.envs.kaggriculture import kaggriculture as game
        from worker_grow.assignments import Subtask
        obs = observation()
        obs['farms'][0]['tiles'][3][4] = game._new_animal('COW', 0)
        obs['farms'][0]['tiles'][3][4]['fed_today'] = False
        options = (
            Subtask(None, 'DROP', (4, 4)),
            Subtask('issue:1', 'FEED', (4, 3), None, 'p', 0),
        )
        view = SimpleNamespace(index=0, xy=(4, 4), inventory={'COW': 1}, options=options)
        pick = greedy_pick(view, obs, (view,))
        self.assertIn(pick.operation, ('DROP', 'FEED'))

    def test_hungry_animal_picks_up_wheat_before_feed(self):
        from kaggle_environments.envs.kaggriculture import kaggriculture as game
        from worker_grow.contracts import TaskStatus
        obs = observation()
        obs['farms'][0]['farmer'] = [4, 4]
        obs['farms'][0]['tiles'][3][4] = game._new_animal('COW', 0)
        obs['farms'][0]['tiles'][3][4]['fed_today'] = False
        obs['private']['shed']['WHEAT'] = 2
        book = TaskBook([ProductionTask('cow', (4, 3), 'COW', (('MILK', 1),))])
        book.records['cow'].producer_id = 'p'
        book.records['cow'].status = TaskStatus.ACTIVE
        views = worker_views(obs, book, {}, 0, 0)
        pick = greedy_pick(views[0], obs, views)
        self.assertEqual(pick.operation, 'PICKUP')
        self.assertEqual(pick.item, 'WHEAT')

    def test_each_planner_closes_a_mature_quota(self):
        for name in ('greedy', 'hungarian', 'tour', 'cbba', 'cpsat'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                args = parser().parse_args([
                    '--run-dir', directory, '--episodes', '1', '--stage', 'mature',
                    '--hands', '0', '--quantity', '1', '--save-every', '0',
                    '--assignment-strategy', name, '--walk-policy', 'greedy',
                    '--no-update', '--episode-steps', '48',
                ])
                _, summary = collect(FeudalCrew(), args, 1, Path(directory) / 'ep')
                self.assertEqual(summary['tasks']['lesson:0'], 'COMPLETE', name)
                self.assertGreaterEqual(summary['delivered_units'], 1, name)
                self.assertEqual(summary['assignment_strategy'], name)
