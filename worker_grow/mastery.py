"""Isolated crew skill curriculum. No market-profit or manager reward changes."""
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, asdict
import math
import random

from kaggle_environments.envs.kaggriculture import kaggriculture as game
from worker_grow.contracts import ProductionTask, TaskBook, TaskStatus
from worker_grow.delivery import DeliveryLedger
from worker_grow.assignments import MOVES

SEMANTICS = 'crew-mastery-segment-return-v1'
ACCESS = ((4,4),(4,5),(5,4),(5,5))


@dataclass(frozen=True)
class Lesson:
    name: str
    stage: str
    items: tuple
    tasks: int = 1
    hands: int = 0
    radius: int = 2
    start_distance: int = 0
    steps: int = 72
    progress: float = .05
    success_steps_limit: int | None = None

    def setup(self, env, seed):
        rng = random.Random(seed)
        own = deepcopy(env.state[0].observation.farms[0])
        private = deepcopy(env.state[0].observation.private)
        # Explicitly gifted, finite supplies. The teacher never picks unit actions.
        private['seeds'] = {}
        private['shed'] = {}
        private['inventories'] = [{} for _ in range(1+self.hands)]
        own['money'] = max(3000, self.hands * math.ceil(self.steps/24) * 200)
        tiles = [(x,y) for y,row in enumerate(own['tiles']) for x,t in enumerate(row)
                 if t is None and min(abs(x-a)+abs(y-b) for a,b in ACCESS) <= self.radius]
        if len(tiles) < self.tasks:
            raise ValueError('Not enough legal lesson tiles')
        targets = rng.sample(tiles, self.tasks)
        tasks = []
        for i,(x,y) in enumerate(targets):
            item = rng.choice(self.items)
            product = item if item in game.CROPS else game.ANIMALS[item]['product']
            tasks.append(ProductionTask(f'lesson:{i}',(x,y),item,((product,1),)))
            if self.stage == 'seeded':
                stock = private['seeds'] if item in game.CROPS else private['shed']
                stock[item] = stock.get(item,0)+1
            else:
                producer = game._new_plant(item,0,24) if item in game.CROPS else game._new_animal(item,0)
                if self.stage == 'mature':
                    spec = game.CROPS[item] if item in game.CROPS else game.ANIMALS[item]
                    producer['planted_day' if item in game.CROPS else 'placed_day'] = -spec['first_yield_day']
                    producer['yield_units'] = 1
                    if item in game.CROPS:producer['max_lifespan_step'] = -1
                own['tiles'][y][x] = producer
            if item in game.ANIMALS:
                private['shed']['WHEAT'] = private['shed'].get('WHEAT',0)+math.ceil(self.steps/24)+2
        tx,ty = targets[0]
        # Exact distance where possible; legal starts, never a hidden path oracle.
        starts = [(x,y) for y,row in enumerate(own['tiles']) for x,t in enumerate(row)
                  if t != 'LOCKED' and abs(x-tx)+abs(y-ty) == self.start_distance]
        if not starts:raise ValueError('No legal lesson start')
        own['farmer'] = list(rng.choice(starts))
        own['hands'] = [list(rng.choice(starts)) for _ in range(self.hands)]
        own['hires_today'] = self.hands
        for state in env.state:state.observation.farms[0] = deepcopy(own)
        env.state[0].observation.private = private
        return TaskBook(tasks)

    def market(self, observation):
        # The engine retires hands each night. Rehire through real market orders.
        farm = observation['farms'][observation['player']]
        return [['HIRE'] for _ in range(max(0,self.hands-len(farm['hands'])))]


LESSONS = (
    Lesson('harvest_here','mature',('WHEAT',),radius=1,success_steps_limit=12),
    Lesson('harvest_near','mature',('WHEAT',),start_distance=2,success_steps_limit=18),
    Lesson('harvest_far','mature',('WHEAT',),radius=6,start_distance=4,steps=96,success_steps_limit=28),
    Lesson('care_wheat','growing',('WHEAT',),start_distance=1,steps=144,progress=.03),
    Lesson('seed_wheat','seeded',('WHEAT',),start_distance=2,steps=144,progress=.02),
    Lesson('team_harvest','mature',('WHEAT','CARROT'),tasks=2,hands=1,start_distance=2,steps=96,progress=.02),
    Lesson('team_seed','seeded',('WHEAT','CARROT'),tasks=2,hands=1,start_distance=2,steps=168,progress=.01),
    Lesson('goose','seeded',('GOOSE',),hands=1,start_distance=2,steps=240,progress=.01),
    Lesson('cow','seeded',('COW',),hands=1,start_distance=2,steps=360,progress=.01),
    Lesson('sheep','seeded',('SHEEP',),hands=1,start_distance=2,steps=288,progress=.01),
    Lesson('mixed','seeded',tuple(game.CROPS)+tuple(game.ANIMALS),tasks=3,hands=2,
           radius=6,start_distance=3,steps=720,progress=0.),
)


@dataclass
class MasteryGate:
    level: int = 0
    passes: int = 0
    threshold: float = .9
    required_passes: int = 2
    mastered: bool = False

    def assess(self, summaries):
        if not summaries:raise ValueError('Empty mastery evaluation')
        rate = sum(s['success'] for s in summaries)/len(summaries)
        steps = [s['turns'] for s in summaries if s['success']]
        mean_steps = sum(steps)/len(steps) if steps else None
        limit = LESSONS[self.level].success_steps_limit
        passed = rate >= self.threshold and (limit is None or mean_steps is not None and mean_steps <= limit)
        self.passes = self.passes+1 if passed else 0
        old = self.level
        promoted = self.passes >= self.required_passes
        if promoted:
            self.passes = 0
            if self.level < len(LESSONS)-1:self.level += 1
            else:self.mastered = True
        return {'level':old,'lesson':LESSONS[old].name,'success_rate':rate,
                'mean_success_steps':mean_steps,'passed':passed,'promoted':promoted,
                'next_level':self.level,'gate':asdict(self)}


class MasteryLedger(DeliveryLedger):
    """Bounded skill support, once per actual daily subjob, plus execution cost.

    Delivery keeps the existing material DAG. Support uses ONLY the verified
    short approach, never all earlier producer actions. Accounting sums to the
    emitted reward; value targets below are returns, not additional payments.
    """
    def __init__(self,*args,progress_cap=.25,decision_cost=.001,**kwargs):
        super().__init__(*args,**kwargs)
        if any(not math.isfinite(x) or x < 0 for x in (progress_cap,decision_cost)):
            raise ValueError('Invalid mastery cost/cap')
        self.progress_cap=progress_cap;self.decision_cost=decision_cost
        self.progress_paid=defaultdict(float);self.progress_keys=set()
        self.decision_penalties=[];self._charged_decisions=set()

    def award_progress(self,e,task_id):
        meta=self.assignments.get(e.decision_id)
        if not meta or not task_id or not self.matches_assignment(e,meta):return
        record=self.tasks.records.get(task_id)
        if record is None or record.status==TaskStatus.CANCELLED:return
        # Producer identity and day prevent repeated operations on the same need.
        key=(task_id,e.plant_id or e.animal_id,meta['subtask']['operation'],e.day)
        if key in self.progress_keys:return
        self.progress_keys.add(key)
        amount=min(self.grow_reward,max(0.,self.progress_cap-self.progress_paid[task_id]))
        if amount<=0:return
        self.progress_paid[task_id]+=amount
        root=self.action(e,self.approach(e))
        shares=self.graph.shares(root,amount,self.decay)
        for d,value in shares.items():self.credits[d]+=value
        self.progress.append({'event_id':e.event_id,'step':e.step,'task_id':task_id,
            'decision_id':e.decision_id,'event_type':e.event_type.value,'reward':amount,
            'assignment_id':meta['assignment_id'],'shares':shares})

    def charge_walks(self,decisions,events):
        super().charge_walks(decisions,events)
        for row in decisions:
            if row.decision_id in self._charged_decisions:raise ValueError('Decision charged twice')
            self._charged_decisions.add(row.decision_id)
            # Forced biological waits do not cost; choosing IDLE over available work does.
            if not row.stochastic or not self.decision_cost:continue
            self.credits[row.decision_id]-=self.decision_cost
            self.decision_penalties.append({'decision_id':row.decision_id,'reward':-self.decision_cost})

    def report(self):
        # Base report checks its own conservation; add the separately audited costs.
        costs=math.fsum(p['reward'] for p in self.decision_penalties)
        for p in self.decision_penalties:self.credits[p['decision_id']]-=p['reward']
        try:report=super().report()
        finally:
            for p in self.decision_penalties:self.credits[p['decision_id']]+=p['reward']
        report.update(version=SEMANTICS,decision_reward=costs,
                      decision_penalties=self.decision_penalties,decision_credit=dict(self.credits))
        report['reward']+=costs;report['allocated']=math.fsum(self.credits.values())
        if not math.isclose(report['reward'],report['allocated'],abs_tol=1e-8):
            raise ValueError('Mastery credit mismatch')
        return report


def segment_returns(rows):
    """Undiscounted remaining assigned credit, ONLY inside the same unit subjob.

    An abandoned segment inherits its own costs, not its replacement's success.
    Initial selection now sees subsequent approach costs/completion. These value
    targets need not sum to the ledger reward (ordinary return-to-go semantics).
    """
    remaining=defaultdict(float)
    for row in reversed(rows):
        if row.assignment_id is None:raise ValueError('Missing mastery assignment')
        key=(row.unit_index,row.assignment_id)
        remaining[key]+=row.credit
        row.target_return=remaining[key]


def episode_metrics(rows,report):
    tasks=list(report['tasks'].values())
    complete=sum(t['status']=='COMPLETE' for t in tasks)
    segments={r.assignment_id for r in rows if r.subtask and r.subtask.operation!='PASS'}
    verified={x['assignment_id'] for x in report['navigation']}
    ops=Counter(r.action[0] for r in rows)
    walking=sum(ops[m] for m in MOVES)
    return {'success':bool(tasks) and complete==len(tasks),'task_success_rate':complete/max(1,len(tasks)),
            'completed_tasks':complete,'operations':dict(ops),
            'walks_per_completed_task':walking/complete if complete else None,
            'segments':len(segments),'completed_segments':len(segments & verified),
            'uncompleted_segments':len(segments-verified),
            'reassignments':sum(r.assignment_change=='reassign' for r in rows)}
