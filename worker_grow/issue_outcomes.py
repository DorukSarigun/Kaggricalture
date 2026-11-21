"""Observed task costs for the manager, separate from realized gold.

No capacity prediction, work quota, deadline action head or crew penalty.
An unstarted issue incurs a small holding cost once per physical turn. Normal
growth is free; missed daily WATER/FEED is charged only after the engine's
daily refresh. These are behavioral objectives, NOT fictional cash expenses.
"""
from collections import Counter, defaultdict
import math

from kaggle_environments.envs.kaggriculture import kaggriculture as game
from lifecycle_tracking import LifecycleEventType as E
from worker_grow.resource_evidence import FeedWithdrawalEvidence

VERSION = "manager-completion-dag-cost-v2"
DEFAULT_WAIT_COST = 0.0001
DEFAULT_MISSED_CARE_COST = 0.001


class IssueOutcomeLedger:
    def __init__(self, *, wait_cost=DEFAULT_WAIT_COST, missed_care_cost=DEFAULT_MISSED_CARE_COST):
        for value in (wait_cost, missed_care_cost):
            if not math.isfinite(value) or value < 0:
                raise ValueError("Issue costs must be finite and nonnegative")
        self.wait_cost, self.missed_care_cost = wait_cost, missed_care_cost
        self.events = []
        self.last_step = -1
        self.resources = FeedWithdrawalEvidence()
        self.resource_windows = {}
        self.finished = False

    def observe(self, before, after, book, events):
        step, player = int(before['step']), int(before['player'])
        if self.finished or step != self.last_step + 1 or int(after['step']) != step + 1:
            raise ValueError("Issue outcomes require consecutive, unique physical turns")
        if int(after['player']) != player:
            raise ValueError("Mixed players in issue outcomes")
        if any(e.player != player or e.step != step for e in events):
            raise ValueError("Wrong lifecycle transition for issue outcomes")
        start = len(self.events)
        self.resources.observe(before, after, book, events)
        private = after['private']
        supply = {item: int(private['seeds'].get(item, 0)) for item in game.CROPS}
        supply.update({item: int(private['shed'].get(item, 0)) +
                       sum(int(bag.get(item, 0)) for bag in private['inventories'])
                       for item in game.ANIMALS})
        # FIFO accounting only: this does not lock stock or constrain actions.
        for issue in sorted(book.issues.values(), key=lambda i: (i.opened_step, i.decision_id)):
            record = book.task_book.records[issue.task_id]
            if not record.needs_production or record.producer_id is not None:
                continue
            funded = supply[issue.item] > 0
            if funded:
                supply[issue.item] -= 1
            windows = self.resource_windows.setdefault(issue.task_id, [])
            if windows and windows[-1]['funded'] == funded and windows[-1]['end_step'] == step - 1:
                windows[-1]['end_step'] = step
            else:
                windows.append({'start_step': step, 'end_step': step, 'funded': funded,
                                'resource': ('seed:' if issue.item in game.CROPS else 'animal:') + issue.item})
            # A new, funded issue cannot consume this turn's market purchase:
            # units execute before market. Start charging on subsequent turns.
            if issue.opened_step == step and funded:
                continue
            self._charge(issue, step, 'start_backlog' if funded else 'missing_input',
                         self.wait_cost)

        if int(after['day']) > int(before['day']):
            lost = {e.plant_id or e.animal_id: e for e in events
                    if e.event_type in (E.PLANT_LOST, E.ANIMAL_LOST)}
            watered = {e.plant_id for e in events if e.event_type == E.WATERED}
            for issue in book.issues.values():
                record = book.task_book.records[issue.task_id]
                identity, xy = record.producer_id, record.tile_xy
                if identity is None or xy is None:
                    continue
                x, y = xy
                tile = after['farms'][player]['tiles'][y][x]
                missing = False
                if record.needs_production and isinstance(tile, dict):
                    field = 'consecutive_unwatered' if issue.item in game.CROPS else 'consecutive_unfed'
                    missing = int(tile.get(field, 0)) > 0
                elif identity in lost:
                    event = lost[identity]
                    if event.event_type == E.ANIMAL_LOST:
                        missing = event.data.get('reason') == 'ESCAPED'
                    else:
                        old = before['farms'][player]['tiles'][y][x]
                        # BECAME_WEED can also mean normal lifespan decay.
                        # Only attribute a verified second dry day, excluding
                        # a simultaneously decaying plant and successful WATER.
                        missing = (event.data.get('reason') == 'BECAME_WEED'
                                   and isinstance(old, dict) and old.get('kind') == 'PLANT'
                                   and int(old.get('consecutive_unwatered', 0)) >= 1
                                   and not old.get('watered_today') and identity not in watered
                                   and (int(old.get('max_lifespan_step', -1)) < 0
                                        or step < int(old['max_lifespan_step'])))
                if missing:
                    self._charge(issue, step, 'missed_water' if issue.item in game.CROPS
                                 else 'missed_feed', self.missed_care_cost)
        self.last_step = step
        self.resources.allocate(self.events[start:], step)
        return self.events[start:]

    def finish(self, book, observation, lifecycle_events, *, rule='unfulfilled_fraction'):
        from worker_grow.issue_completion import IssueMaterialAudit, settle_task_dag
        if self.finished:
            raise ValueError('Task outcomes already settled')
        audit = IssueMaterialAudit(book.issues)
        for event in lifecycle_events:
            if event.player != int(observation['player']):
                raise ValueError('Wrong player in completion history')
            audit.observe(event)
        completion = audit.report(book, observation)
        for key, value in completion.items():
            value['startup_resource_windows'] = self.resource_windows.get(key, [])
        result = settle_task_dag(self.report(), completion, book.issues, rule=rule)
        result['resource_evidence'] = list(self.resources.history)
        self.finished = True
        return result

    def _charge(self, issue, step, reason, cost):
        self.events.append({'issue_id': issue.task_id, 'decision_id': issue.decision_id,
                            'step': step, 'reason': reason, 'reward': -cost})

    def report(self):
        decisions, issues = defaultdict(float), defaultdict(float)
        for event in self.events:
            decisions[event['decision_id']] += event['reward']
            issues[event['issue_id']] += event['reward']
        return {'version': VERSION, 'wait_cost': self.wait_cost,
                'missed_care_cost': self.missed_care_cost,
                'task_credit': math.fsum(e['reward'] for e in self.events),
                'decision_credit': dict(decisions), 'issue_credit': dict(issues),
                'counts': dict(Counter(e['reason'] for e in self.events)),
                'events': list(self.events)}
