"""FuN-style goal modulation for ONE recurrent, autoregressive field crew.

pi(a)=softmax(U_a(h) @ phi(sum recent detached manager goals)). Explicit
production contracts replace the paper's learned directional latent goals.
The full FuN manager/transition policy gradient is NOT implemented here.
"""
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from action_encoding import ActionGrammarConfig, UNIT_ITEM_OPS
from structured_actions import OperationCursor
from worker_grow.contracts import GOAL_DIM, TASK_CHANNELS, TaskStatus
from worker_grow.encoding import encode_own_farm, OwnFarmObservation
from worker_grow.policy import WorkerGrowActor, UNIT_OPS, ITEMS
from kaggle_environments.envs.kaggriculture import kaggriculture as game
from worker_grow.assignments import (FEATURE_DIM, candidates, available, features,
                                    operation_mask, Subtask)

VERSION = 'feudal-crew-subtasks-v4'
INIT_VERSIONS = frozenset({
    'feudal-contract-crew-site-v2',
    'feudal-contract-crew-site-v3',
    VERSION,
})


@dataclass
class CrewInput:
    farm: OwnFarmObservation
    tasks: np.ndarray
    goal_sum: np.ndarray


@dataclass
class CrewDecision:
    inputs: CrewInput
    hidden_before: np.ndarray
    action: list
    allowed_ops: tuple
    allowed_items: tuple
    quantity_limit: int
    log_prob: float
    value: float
    decision_id: int
    step: int
    unit_index: int
    stochastic: bool
    credit: float = 0.0
    assignment_features: np.ndarray | None = None
    assignment_choice: int = 0
    subtask: Subtask | None = None
    assignment_id: str | None = None
    assignment_change: str | None = None
    # Optional mastery target; ledger credit remains independently auditable.
    target_return: float | None = None


@dataclass
class CrewState:
    hidden: torch.Tensor | None = None
    goals: deque = field(default_factory=deque)
    next_decision: int = 0
    last_step: int = -1
    leases: dict = field(default_factory=dict)
    assignment_strategy: str = 'learned'
    walk_policy: str | None = None
    cache: dict = field(default_factory=dict)


class CrewCursor(OperationCursor):
    """Task masks intersect the engine's legal actions; shadow follows unit order."""
    def __init__(self, observation, tasks, config=None):
        super().__init__(observation, config)
        # Shadow reservations must not mutate real issues before engine success.
        self.tasks = deepcopy(tasks)

    def candidate(self, items):
        return self.tasks.candidate(self.rules._position(), items)

    def apply(self, action):
        xy = tuple(self.rules._position())
        before = deepcopy(self.rules.farm['tiles'][xy[1]][xy[0]])
        super().apply(action)
        after = self.rules.farm['tiles'][xy[1]][xy[0]]
        if (action[0] == 'HARVEST' and isinstance(before, dict) and before.get('kind') == 'PLANT'
                and after is None):
            record = self.tasks.at(xy)
            if record is not None:
                record.close_production()
        if not isinstance(after, dict) or before == after:
            return
        op = action[0]
        items = ()
        if op == 'PLANT' and after.get('kind') == 'PLANT':
            items = (after['crop'],)
        elif op in ('BUILD_COOP', 'BUILD_PASTURE'):
            items = tuple(a for a, spec in game.ANIMALS.items() if spec['structure'] == after.get('kind'))
        elif op == 'PLACE' and after.get('animal'):
            items = (after['animal'],)
        if items:
            r = self.tasks.bind(xy, items)
            if r is None:
                raise ValueError('Crew production has no matching issue')
            if op in ('PLANT', 'PLACE'):
                r.producer_id = 'shadow-produced'
                r.status = TaskStatus.ACTIVE

    def items(self, operation):
        allowed = super().items(operation)
        r = self.tasks.at(self.rules._position())
        if operation == 'PLANT':
            return tuple(i for i in allowed if self.candidate((i,)) is not None)
        if operation == 'PLACE':
            # At an empty animal structure PLACE takes precedence over shed drop.
            x, y = self.rules._position()
            tile = self.rules.farm['tiles'][y][x]
            return tuple(i for i in allowed if not (i in game.ANIMALS and isinstance(tile, dict)
                and tile.get('kind') == game.ANIMALS[i]['structure'] and 'animal' not in tile)
                or self.candidate((i,)) is not None)
        if operation == 'PICKUP':
            needed = set()
            for record in self.tasks.records.values():
                if not record.needs_production:
                    continue
                if record.task.item in game.ANIMALS:
                    needed.add('WHEAT')
                    if record.producer_id is None:
                        needed.add(record.task.item)
                else:
                    needed.add('FERTILIZER')
            return tuple(i for i in allowed if i in needed)
        return allowed

    def operations(self):
        if self.section != 'unit':
            return ()  # market belongs exclusively to manager
        allowed = super().operations()
        if not any(r.open for r in self.tasks.records.values()) and not any(self.rules._inventory().values()):
            return ('PASS',)
        r = self.tasks.at(self.rules._position())
        x, y = self.rules._position()
        tile = self.rules.farm['tiles'][y][x]
        matches = r and isinstance(tile, dict) and (tile.get('crop') == r.task.item
                                                    or tile.get('animal') == r.task.item)
        keep = []
        for op in allowed:
            if op in ('WATER','FERTILIZE','HARVEST','FEED','CARE','COLLECT_FERTILIZER') and not matches:
                continue
            if op == 'DIG' and not (self.candidate(tuple(game.CROPS)+tuple(game.ANIMALS)) and isinstance(tile, dict)
                                     and tile.get('kind') == 'WEED'):
                continue  # demolition/cancellation needs a separate manager order
            if op.startswith('BUILD_') and self.candidate(tuple(a for a, spec in game.ANIMALS.items()
                    if op == 'BUILD_'+spec['structure'])) is None:
                continue
            if op in UNIT_ITEM_OPS and not self.items(op):
                continue
            keep.append(op)
        return tuple(keep)


class FeudalCrew(nn.Module):
    def __init__(self, hidden_dim=128, goal_rank=16, goal_horizon=10):
        super().__init__()
        if min(hidden_dim, goal_rank, goal_horizon) <= 0:
            raise ValueError('Invalid crew dimensions')
        self.hidden_dim, self.goal_rank, self.goal_horizon = hidden_dim, goal_rank, goal_horizon
        # Reuse the own-farm encoder, not the legacy independent worker heads.
        encoder = WorkerGrowActor()
        self.farm_encoder, self.farm_pooling = encoder.farm_encoder, encoder.farm_pooling
        self.base_context = encoder.context
        self.task_encoder = nn.Sequential(nn.Conv2d(TASK_CHANNELS, 8, 3, padding=1), nn.Tanh(),
                                          nn.Flatten(), nn.Linear(800, 64), nn.Tanh())
        self.recurrent = nn.GRUCell(256+64+2, hidden_dim)
        self.command_op = nn.Embedding(len(UNIT_OPS), 16)
        self.command_item = nn.Embedding(len(ITEMS)+1, 8)
        self.command_memory = nn.GRUCell(25, hidden_dim)
        self.phi = nn.Linear(GOAL_DIM, goal_rank, bias=False)
        self.op_U = nn.Linear(hidden_dim, len(UNIT_OPS)*goal_rank)
        self.item_U = nn.Linear(hidden_dim+16, len(ITEMS)*goal_rank)
        self.quantity_U = nn.Sequential(nn.Linear(hidden_dim+24+4,64), nn.Tanh(), nn.Linear(64,goal_rank))
        self.value_head = nn.Sequential(nn.Linear(hidden_dim+goal_rank,64), nn.Tanh(), nn.Linear(64,1))
        self.assignment_score = nn.Sequential(nn.Linear(hidden_dim+FEATURE_DIM,64), nn.Tanh(), nn.Linear(64,1))
        self.assignment_condition = nn.Linear(FEATURE_DIM,hidden_dim,bias=False)
        # Old movement/action features survive a weights-only warm start.
        nn.init.zeros_(self.assignment_condition.weight)

    def load_initial_weights(self, saved):
        if saved.get('version') not in INIT_VERSIONS:
            raise ValueError('Incompatible crew checkpoint')
        weights=saved['model']
        if saved['version']==VERSION:
            self.load_state_dict(weights,strict=True)
            return
        expected=self.state_dict()
        new={k for k in expected if k.startswith(('assignment_score.','assignment_condition.'))}
        if set(weights)!=set(expected)-new:
            raise ValueError('Unexpected legacy crew weights; migration refused')
        self.load_state_dict({**expected,**weights},strict=True)

    def assignment_distribution(self,h,option_features):
        f=torch.as_tensor(option_features,device=self.device,dtype=h.dtype)
        logits=self.assignment_score(torch.cat([h.expand(len(f),-1),f],-1)).squeeze(-1)
        return Categorical(logits=logits)

    def assigned_hidden(self,h,option_features,choice):
        f=torch.as_tensor(option_features[choice],device=self.device,dtype=h.dtype)
        return h+self.assignment_condition(f)

    @property
    def device(self):
        return next(self.parameters()).device

    def encode(self, rows):
        farms = [r.farm for r in rows]
        def tensor(values, dtype=torch.float32):
            return torch.as_tensor(np.stack(values), device=self.device, dtype=dtype)
        m = self.farm_encoder(tensor([r.categorical_ids for r in farms],torch.long),
                              tensor([r.numeric_features for r in farms]),
                              tensor([r.worker_positions for r in farms]))
        local = torch.stack([m[i,:,int(r.unit_position_yx[0]),int(r.unit_position_yx[1])]
                             for i,r in enumerate(farms)])
        extra = tensor([np.concatenate([r.unit_features,r.time_features]) for r in farms])
        base = self.base_context(torch.cat([self.farm_pooling(m),local,extra],-1))
        xy = tensor([r.unit_position_yx/9 for r in farms])
        return torch.cat([base,self.task_encoder(tensor([r.tasks for r in rows])),xy],-1)

    def step(self, encoded, hidden, goal_sum):
        h = self.recurrent(encoded, hidden)
        # Explicit stop-gradient at manager/crew boundary. phi belongs to crew.
        g = torch.as_tensor(goal_sum, dtype=torch.float32, device=self.device).detach()
        w = self.phi(g)
        value = self.value_head(torch.cat([h,w],-1)).squeeze(-1)
        return h, w, value

    def after_action(self, h, action):
        op = self.command_op(torch.tensor(UNIT_OPS.index(action[0]),device=self.device))
        item = self.command_item(torch.tensor(ITEMS.index(action[1])+1 if len(action)>1 else 0,device=self.device))
        q = torch.tensor([np.log1p(action[2]) if len(action)>2 else 0],device=self.device,dtype=h.dtype)
        return self.command_memory(torch.cat([op,item,q]),h)

    @staticmethod
    def masked(logits, names, allowed):
        mask = torch.tensor([n in allowed for n in names],device=logits.device)
        if not mask.any():
            raise ValueError('Empty crew action mask')
        return Categorical(logits=logits.masked_fill(~mask,-torch.inf))

    def operation_distribution(self,h,w,allowed):
        return self.masked(self.op_U(h).view(len(UNIT_OPS),self.goal_rank)@w,UNIT_OPS,allowed)

    def item_distribution(self,h,w,op,allowed):
        emb = self.command_op(torch.tensor(UNIT_OPS.index(op),device=self.device))
        logits = self.item_U(torch.cat([h,emb])).view(len(ITEMS),self.goal_rank)@w
        return self.masked(logits,ITEMS,allowed)

    def quantity_distribution(self,h,w,op,item,limit):
        if limit < 1:
            raise ValueError('Invalid quantity limit')
        emb = torch.cat([h,self.command_op(torch.tensor(UNIT_OPS.index(op),device=self.device)),
            self.command_item(torch.tensor(ITEMS.index(item)+1,device=self.device))])
        q = torch.arange(1,limit+1,device=self.device,dtype=h.dtype)
        f = q/limit
        features = torch.stack([f,f.square(),f.sqrt(),torch.log1p(q)/np.log1p(limit)],-1)
        return Categorical(logits=self.quantity_U(torch.cat([emb.expand(limit,-1),features],-1))@w)


def statistics(model,h,w,row):
    assignment_logp=h.sum()*0
    assignment_entropy=[]
    if row.assignment_features is not None:
        d=model.assignment_distribution(h,row.assignment_features)
        assignment_logp=d.log_prob(torch.tensor(row.assignment_choice,device=model.device))
        if len(row.assignment_features)>1: assignment_entropy.append(d.entropy())
        h=model.assigned_hidden(h,row.assignment_features,row.assignment_choice)
    op = row.action[0]
    d = model.operation_distribution(h,w,row.allowed_ops)
    logp = assignment_logp+d.log_prob(torch.tensor(UNIT_OPS.index(op),device=model.device))
    entropy = assignment_entropy+([d.entropy()] if len(row.allowed_ops)>1 else [])
    if row.allowed_items:
        d = model.item_distribution(h,w,op,row.allowed_items)
        logp = logp+d.log_prob(torch.tensor(ITEMS.index(row.action[1]),device=model.device))
        if len(row.allowed_items)>1: entropy.append(d.entropy())
    if row.quantity_limit:
        d = model.quantity_distribution(h,w,op,row.action[1],row.quantity_limit)
        logp = logp+d.log_prob(torch.tensor(row.action[2]-1,device=model.device))
        if row.quantity_limit>1: entropy.append(d.entropy())
    return logp, torch.stack(entropy).mean() if entropy else h.sum()*0


@torch.no_grad()
def propose_crew_turn(model, observation, tasks, state=None, *, goal=None, deterministic=False, config=None,
                      assignment_strategy=None, walk_policy=None):
    from worker_grow.schedule import (STRATEGIES, greedy_action, option_index, plan_turn,
                                      resolve_walk_policy)
    state = state or CrewState()
    if assignment_strategy is None:
        assignment_strategy = state.assignment_strategy
    if assignment_strategy not in STRATEGIES:
        raise ValueError(f'Unknown assignment strategy: {assignment_strategy}')
    if walk_policy is None:
        walk_policy = state.walk_policy
    walk_policy = resolve_walk_policy(assignment_strategy, walk_policy)
    state.assignment_strategy, state.walk_policy = assignment_strategy, walk_policy
    step = int(observation.get('step', observation['day']*24+observation['hour']))
    if step <= state.last_step:
        raise ValueError('Crew state must reset between episodes; turns must increase')
    state.last_step = step
    goal = tasks.goal() if goal is None else np.asarray(goal.detach().cpu() if torch.is_tensor(goal) else goal,dtype=np.float32)
    if goal.shape != (GOAL_DIM,) or not np.isfinite(goal).all() or np.linalg.norm(goal)==0:
        raise ValueError('Invalid manager goal')
    state.goals.append(goal.copy()/np.linalg.norm(goal))
    while len(state.goals)>model.goal_horizon: state.goals.popleft()
    goal_sum = np.sum(state.goals,axis=0)
    day=int(observation['day'])
    planned = plan_turn(assignment_strategy, observation, tasks, state, step, day, config)
    cursor = CrewCursor(observation,tasks,config)
    rows=[]
    hidden = torch.zeros(model.hidden_dim,device=model.device) if state.hidden is None else state.hidden.to(model.device)
    def choose(d): return int(d.logits.argmax() if deterministic else d.sample())
    while cursor.section == 'unit':
        index = cursor.rules.unit_index
        inputs = CrewInput(encode_own_farm(cursor.observation(),index),cursor.tasks.planes(),goal_sum.copy())
        before = hidden.cpu().numpy().copy()
        h,w,value = model.step(model.encode([inputs])[0],hidden,goal_sum)
        options=available(candidates(cursor),state.leases,index,step,day)
        prior=state.leases.get(index)
        f=features(options,cursor,prior['subtask'] if prior else None)
        if assignment_strategy == 'learned':
            choice=choose(model.assignment_distribution(h,f));selected=options[choice]
        else:
            selected=planned.selected.get(index, options[0])
            if selected not in options:
                selected=options[option_index(options, selected)]
            choice=option_index(options, selected)
        same=prior is not None and prior['subtask'].key==selected.key
        segment=prior['id'] if same else f'{day}:{index}:{state.next_decision}'
        changed='continue' if same else 'reassign' if prior else 'assign'
        if selected.operation=='PASS':
            state.leases.pop(index,None);changed='release' if prior else 'idle'
        else:
            state.leases[index]={'id':segment,'subtask':selected,'day':day,'step':step}
        assigned=model.assigned_hidden(h,f,choice)
        allowed = operation_mask(cursor,selected)
        if walk_policy == 'greedy':
            action = greedy_action(cursor, selected, allowed)
            op = action[0]
            items = cursor.items(op)
            if items and selected.item is not None:
                items = tuple(i for i in items if i == selected.item) or items
        else:
            op = UNIT_OPS[choose(model.operation_distribution(assigned,w,allowed))]
            action=[op]; items=cursor.items(op)
            if items and selected.item is not None:items=tuple(i for i in items if i==selected.item)
            if items: action.append(ITEMS[choose(model.item_distribution(assigned,w,op,items))])
            limit = cursor.quantity_limit(op,action[1] if items else None)
            if limit: action.append(choose(model.quantity_distribution(assigned,w,op,action[1],limit))+1)
        limit = action[2] if len(action) > 2 else 0
        row = CrewDecision(inputs,before,action,allowed,items,limit,0.,float(value),state.next_decision,
                           step,index,len(options)>1 or len(allowed)>1 or len(items)>1 or limit>1,
                           assignment_features=f,assignment_choice=choice,subtask=selected,
                           assignment_id=segment,assignment_change=changed)
        row.log_prob = float(statistics(model,h,w,row)[0])
        cursor.apply(action)
        if op==selected.operation and op!='PASS':state.leases.pop(index,None)
        hidden = model.after_action(h,action)
        rows.append(row); state.next_decision += 1
    state.hidden = hidden.detach()
    return deepcopy(cursor.actions),rows,state


def attach_manager_market(observation, crew_action, market_policy, config=None):
    """Manager sees the post-crew shadow; it can append only market orders.

market_policy(shadow_observation) -> list of market commands. Its purchases
cannot retroactively provision this turn's crew or activate a new hand.
"""
    cursor=OperationCursor(observation,config)
    expected=1+len(observation['farms'][observation['player']]['hands'])
    units=[crew_action['farmer'],*crew_action['hands']]
    if len(units)!=expected or crew_action.get('market'):
        raise ValueError('Crew must provide exactly the current units and no market')
    for unit in units:cursor.apply(unit)
    orders=market_policy(cursor.observation())
    for order in orders:cursor.apply(order)
    return deepcopy(cursor.actions)
