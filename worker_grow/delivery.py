"""First-delivery DAG plus optional per-event grow progress for crew.

Parts retain their paid flag when split, moved, fed or resold. A new harvest
creates new parts. This ledger consumes verified lifecycle events, not action
names or guessed shadow inventory differences. Grow points are issued only to
the crew decision that actually advanced a bound producer.
"""
from collections import defaultdict, deque
from dataclasses import dataclass, replace
import math

from lifecycle_tracking import FarmLifecycleTracker, LifecycleEventType as E
from production_graph_credit import ActionDAG
from worker_grow.contracts import TaskStatus
from kaggle_environments.envs.kaggriculture import kaggriculture as game
from worker_grow.assignments import MOVES, decision_metadata


@dataclass
class Part:
    quantity: int
    batch_id: str | None
    task_id: str | None
    product: str
    head: int | None
    delivered: bool = False


class DeliveryLedger:
    def __init__(self, tasks, *, decay=.8, unit_reward=1., grow_reward=0., delivery_bonus=1., walk_cost=0.):
        if not 0 < decay <= 1 or not math.isfinite(unit_reward) or unit_reward <= 0:
            raise ValueError('Invalid delivery reward configuration')
        if not math.isfinite(grow_reward) or grow_reward < 0:
            raise ValueError('Invalid grow progress reward')
        if not math.isfinite(delivery_bonus) or delivery_bonus <= 0:
            raise ValueError('Invalid delivery bonus')
        if not math.isfinite(walk_cost) or walk_cost < 0:
            raise ValueError('Invalid walk cost')
        self.walk_cost=walk_cost
        self.assignments={}
        self.assignment_paths={}
        self.navigation=[]
        self.walk_penalties=[]
        self._charged_walks=set()
        self.tasks, self.decay, self.unit_reward, self.grow_reward = (
            tasks, decay, unit_reward, grow_reward)
        self.delivery_bonus = delivery_bonus
        self.graph = ActionDAG()
        self.lots, self.producers, self.prepared = {}, {}, {}
        self.walks, self.operation_walk = {}, {}
        self.credits = defaultdict(float)
        self.deliveries, self.progress, self.events = [], [], set()
        self.created, self.paid = defaultdict(int), defaultdict(int)
        self.last_step = -1
        self.closed = False

    def register_decisions(self, decisions):
        for row in decisions:
            meta=decision_metadata(row)
            if meta['subtask'] is None:continue  # legacy/manual fixtures
            if row.decision_id in self.assignments:raise ValueError('Duplicate assignment decision')
            self.assignments[row.decision_id]={**meta,'decision_id':row.decision_id,
                'step':row.step,'unit_index':row.unit_index,'action':list(row.action)}

    def charge_walks(self, decisions, events):
        moved={e.decision_id for e in events if e.event_type==E.UNIT_MOVED}
        for row in decisions:
            action=getattr(row,'action',())
            if not action or action[0] not in MOVES:continue
            if row.decision_id in self._charged_walks:raise ValueError('Walk charged twice')
            self._charged_walks.add(row.decision_id)
            if self.walk_cost:self.credits[row.decision_id]-=self.walk_cost
            self.walk_penalties.append({'decision_id':row.decision_id,'step':row.step,
                'unit_index':row.unit_index,'action':row.action,'moved':row.decision_id in moved,
                'reward':-self.walk_cost,**decision_metadata(row)})

    def matches_assignment(self,e,meta):
        s=meta['subtask'];op=s['operation']
        if 'unit_index' in e.data and e.data['unit_index']!=meta['unit_index']:return False
        mapping={E.PLANTED:'PLANT',E.WATERED:'WATER',E.FERTILIZED:'FERTILIZE',
            E.HARVESTED:'HARVEST',E.ANIMAL_PLACED:'PLACE',E.ANIMAL_FED:'FEED',
            E.ANIMAL_CARED:'CARE',E.FERTILIZER_COLLECTED:'COLLECT_FERTILIZER',E.TILE_CLEARED:'DIG'}
        actual=mapping.get(e.event_type)
        if e.event_type==E.STRUCTURE_BUILT:actual='BUILD_'+e.data['kind']
        if e.event_type==E.PRODUCT_MOVED or e.event_type==E.ANIMAL_STOCK_MOVED:
            if e.data.get('source')=='shed':actual='PICKUP'
            elif e.data.get('destination')=='shed':actual='DROP'
        if actual!=op or meta['action'][0]!=op:return False
        if e.tile_xy is not None and tuple(s['tile_xy'] or ())!=tuple(e.tile_xy):return False
        if s['issue_id']:
            p=self.producers.get(e.plant_id or e.animal_id)
            record=self.tasks.at(e.tile_xy) if e.tile_xy is not None else None
            if record is None and op=='DIG' and e.tile_xy is not None:
                wanted=self.tasks.records.get(s['issue_id'])
                if wanted:record=self.tasks.candidate(e.tile_xy,(wanted.task.item,))
            owner=p[1] if p else record.task.task_id if record else None
            if op not in ('PICKUP','DROP') and owner!=s['issue_id']:return False
            known=s['producer_id']
            if known and known!='shadow-produced' and (e.plant_id or e.animal_id):
                if known != (e.plant_id or e.animal_id):return False
        return True

    def award_progress(self, e, task_id):
        if self.grow_reward == 0 or e.decision_id is None or not task_id:
            return
        record = self.tasks.records.get(task_id)
        if record is None or record.status == TaskStatus.CANCELLED:
            return
        self.credits[e.decision_id] += self.grow_reward
        self.progress.append({
            'event_id': e.event_id, 'step': e.step, 'task_id': task_id,
            'decision_id': e.decision_id, 'event_type': e.event_type.value,
            'reward': self.grow_reward,
        })

    def create(self, key, parts):
        if key in self.lots:
            raise ValueError(f'Duplicate delivery lot: {key}')
        if any(type(p.quantity) is not int or p.quantity <= 0 for p in parts):
            raise ValueError('Invalid material quantity')
        self.lots[key] = deque(parts)

    def take(self, key, quantity):
        if key not in self.lots or type(quantity) is not int or quantity <= 0:
            raise ValueError(f'Missing delivery provenance: {key}')
        queue = self.lots[key]
        if sum(p.quantity for p in queue) < quantity:
            raise ValueError('Delivery material over-consumption')
        parts=[]
        while quantity:
            p=queue[0]; n=min(quantity,p.quantity)
            parts.append(replace(p,quantity=n))
            p.quantity-=n; quantity-=n
            if p.quantity==0: queue.popleft()
        return parts

    def approach(self,e,unit=None):
        if e.decision_id is None: return None
        meta=self.assignments.get(e.decision_id)
        if meta is not None:
            if not self.matches_assignment(e,meta):return None
            if e.decision_id not in self.operation_walk:
                segment=meta['assignment_id']
                self.operation_walk[e.decision_id]=self.assignment_paths.pop(segment,None)
                self.navigation.append({'event_id':e.event_id,'decision_id':e.decision_id,
                    'assignment_id':segment,'issue_id':meta['subtask']['issue_id'],
                    'operation':meta['subtask']['operation'],'path_root':self.operation_walk[e.decision_id],
                    'status':'verified'})
            return self.operation_walk[e.decision_id]
        unit=e.data.get('unit_index') if unit is None else unit
        if unit is None: return None
        if e.decision_id not in self.operation_walk:
            self.operation_walk[e.decision_id]=self.walks.pop((e.day,unit),None)
        return self.operation_walk[e.decision_id]

    def action(self,e,*parents):
        return self.graph.node(e.decision_id,[(p,1.) for p in parents if p is not None],e.event_id)

    def material(self,parts):
        return self.graph.node(None,[(p.head,p.quantity) for p in parts if p.head is not None])

    def allocations(self,allocations):
        parts=[]
        for a in allocations:
            key=a['product_lot_id']
            if key not in self.lots and a.get('reconciled'):
                self.create(key,[Part(int(a['quantity']),None,None,'UNKNOWN',None)])
            parts.extend(self.take(key,int(a['quantity'])))
        return parts

    def deliver(self,e,part):
        if part.delivered: return
        part.delivered=True  # even unrequested outputs cannot be redeemed again
        if part.task_id is None: return
        r=self.tasks.records[part.task_id]
        if r.status==TaskStatus.CANCELLED: return
        r.total_delivered[part.product]=r.total_delivered.get(part.product,0)+part.quantity
        quota=dict(r.task.quotas).get(part.product,0)
        # Delivery reward remains delivery-only, with its existing quota cap.
        quantity=min(part.quantity,max(0,quota-r.delivered.get(part.product,0)))
        if not quantity: return
        r.delivered[part.product]=r.delivered.get(part.product,0)+quantity
        self.paid[part.batch_id]+=quantity
        if self.paid[part.batch_id]>self.created[part.batch_id]:
            raise ValueError('Harvest batch rewarded twice')
        reward=quantity*self.unit_reward*self.delivery_bonus
        shares=self.graph.shares(part.head,reward,self.decay)
        for decision,credit in shares.items(): self.credits[decision]+=credit
        self.deliveries.append({'event_id':e.event_id,'step':e.step,'task_id':part.task_id,
            'batch_id':part.batch_id,'product':part.product,'quantity':quantity,'reward':reward,
            'action_root':part.head,'shares':shares,'automatic':e.decision_id is None})
        r.complete_if_fulfilled()

    def consume(self, parts):
        """Intermediate use can fulfill a job, but never mints delivery/gold reward.

        A part that already reached the depot fulfilled its quota there; feeding
        it later must not count it a second time.
        """
        for part in parts:
            if part.task_id is None or part.delivered:
                continue
            record = self.tasks.records[part.task_id]
            if not record.open:
                continue
            quota = dict(record.task.quotas).get(part.product, 0)
            remaining = max(0, quota - record.delivered.get(part.product, 0)
                            - record.consumed.get(part.product, 0))
            quantity = min(part.quantity, remaining)
            if quantity:
                record.consumed[part.product] = record.consumed.get(part.product, 0) + quantity
            record.complete_if_fulfilled()

    def observe(self,e):
        if self.closed or e.event_id in self.events or e.step<self.last_step:
            raise ValueError('Duplicate or unordered lifecycle event')
        self.events.add(e.event_id);self.last_step=e.step
        kind=e.event_type
        if kind==E.PRODUCT_PURCHASED:
            for key in e.data['product_lot_ids']:
                self.create(key,[Part(1,None,None,e.data['item'],None)])
        elif kind==E.ANIMAL_PURCHASED:
            for key in e.data['animal_lot_ids']:
                self.create(key,[Part(1,None,None,e.data['animal'],None)])
        elif kind==E.UNIT_MOVED:
            meta=self.assignments.get(e.decision_id)
            if meta is not None:
                key=meta['assignment_id']
                self.assignment_paths[key]=self.action(e,self.assignment_paths.get(key))
            else:
                key=(e.day,e.data['unit_index'])
                self.walks[key]=self.action(e,self.walks.get(key))
            for lot in (*e.data.get('carried_product_lot_ids',()),*e.data.get('carried_animal_lot_ids',())):
                if lot not in self.lots: raise ValueError('Untracked carried lot')
                for p in self.lots[lot]: p.head=self.action(e,p.head)
        elif kind in (E.TILE_CLEARED,E.STRUCTURE_BUILT):
            if kind == E.STRUCTURE_BUILT:
                items = tuple(a for a, spec in game.ANIMALS.items() if spec['structure'] == e.data['kind'])
                built=self.tasks.bind(e.tile_xy, items, step=e.step, decision_id=e.decision_id)
                if built: self.award_progress(e, built.task.task_id)
            self.prepared[e.tile_xy]=self.action(e,self.prepared.get(e.tile_xy),self.approach(e))
        elif kind in (E.PLANTED,E.ANIMAL_PLACED):
            identity=e.plant_id or e.animal_id
            if identity in self.producers: raise ValueError('Producer reused')
            item=e.crop or e.data.get('animal')
            r=self.tasks.bind(e.tile_xy, (item,), step=e.step, decision_id=e.decision_id)
            if r and (r.task.item!=item or r.producer_id is not None): r=None
            parts=self.take(e.animal_lot_id,1) if e.animal_lot_id else []
            head=self.action(e,self.material(parts),self.prepared.pop(e.tile_xy,None),self.approach(e))
            self.producers[identity]=[head,r.task.task_id if r else None]
            if r:
                r.producer_id=identity;r.status=TaskStatus.ACTIVE
                self.award_progress(e, r.task.task_id)
        elif kind in (E.WATERED,E.FERTILIZED,E.ANIMAL_FED,E.ANIMAL_CARED):
            p=self.producers[e.plant_id or e.animal_id]
            parts=self.allocations(e.data.get('consumed_allocations',()))
            self.consume(parts)
            p[0]=self.action(e,p[0],self.material(parts),self.approach(e))
            self.award_progress(e, p[1])
        elif kind in (E.HARVESTED,E.FERTILIZER_COLLECTED):
            p=self.producers[e.plant_id or e.animal_id]
            p[0]=self.action(e,p[0],self.approach(e))
            item='FERTILIZER' if kind==E.FERTILIZER_COLLECTED else e.crop or e.data['product']
            self.create(e.product_lot_id,[Part(int(e.quantity),e.product_lot_id,p[1],item,p[0])])
            self.created[e.product_lot_id]=int(e.quantity)
            self.award_progress(e, p[1])
            if p[1] and e.data.get('plant_closed'):
                r=self.tasks.records[p[1]]
                r.close_production()
        elif kind in (E.PLANT_LOST,E.ANIMAL_LOST):
            p=self.producers[e.plant_id or e.animal_id]
            if p[1]: self.tasks.records[p[1]].production_closed=True
            if p[1] and self.tasks.records[p[1]].open:
                self.tasks.records[p[1]].status=TaskStatus.FAILED
        elif kind==E.ANIMAL_STOCK_MOVED:
            parts=self.take(e.data['source_lot_id'],int(e.quantity))
            location=next((e.data[k] for k in ('source','destination') if e.data[k].startswith('unit:')),None)
            walk=self.approach(e,int(location.split(':')[1])) if location else None
            for p in parts:p.head=self.action(e,p.head,walk)
            self.create(e.data['destination_lot_id'],parts)
        elif kind in (E.PRODUCT_MOVED,E.INVENTORY_RECONCILED):
            if e.data.get('source')=='external_or_unobserved':
                key=e.product_lot_id or e.animal_lot_id
                if key:self.create(key,[Part(int(e.quantity),None,None,e.crop or e.data.get('animal','UNKNOWN'),None)])
            location=next((e.data.get(k,'') for k in ('source','destination')
                           if e.data.get(k,'').startswith('unit:')),None)
            walk=self.approach(e,int(location.split(':')[1])) if location else None
            for a in e.data.get('allocations',()):
                parts=self.allocations([a])
                target=a.get('destination_product_lot_id')
                for p in parts:
                    p.head=self.action(e,p.head,walk)
                    if target and e.data.get('destination')=='shed':self.deliver(e,p)
                if target:self.create(target,parts)
            for a in e.data.get('animal_allocations',()):self.take(a['animal_lot_id'],int(a['quantity']))
        elif kind==E.PRODUCT_SOLD:
            # Only remove the sold stock. SELL neither receives nor generates crew reward.
            self.allocations(e.data['allocations'])

    def finish(self):
        self.closed=True
        for r in self.tasks.records.values():
            if r.open:r.status=TaskStatus.EXPIRED
        return self.report()

    def report(self):
        delivery=math.fsum(d['reward'] for d in self.deliveries)
        grow=math.fsum(p['reward'] for p in self.progress)
        walking=math.fsum(p['reward'] for p in self.walk_penalties)
        total=delivery+grow+walking
        allocated=math.fsum(self.credits.values())
        if abs(total-allocated)>1e-8:raise ValueError('Delivery credit mismatch')
        return {'version':'crew-subtasks-walk-v2','reward':total,'allocated':allocated,
            'walk_reward':walking,'walk_attempts':len(self.walk_penalties),
            'walk_penalties':self.walk_penalties,'assignments':list(self.assignments.values()),
            'navigation':self.navigation,'uncompleted_paths':dict(self.assignment_paths),
            'delivery_reward':delivery,'grow_reward':grow,'progress':self.progress,
            'decision_credit':dict(self.credits),'deliveries':self.deliveries,
            'action_nodes':self.graph.nodes,'created_units':dict(self.created),'paid_units':dict(self.paid),
            'tasks':{k:{'status':r.status.value,'producer_id':r.producer_id,'delivered':r.delivered,
                         'consumed':r.consumed,'reserves_tile':r.reserves_tile,
                         'total_delivered':r.total_delivered,'production_closed':r.production_closed,
                         'item':r.task.item,'tile_xy':r.tile_xy,'requested_tile_xy':r.task.tile_xy,
                         'bound_step':r.bound_step,'bound_decision_id':r.bound_decision_id,
                         'quotas':dict(r.task.quotas)}
                     for k,r in self.tasks.records.items()}}


class CrewDeliveryTracker:
    """Online adapter: manager IDs are deliberately absent from crew graph."""
    def __init__(self,observation,tasks,*,episode=1,turns_per_day=24,decay=.8,grow_reward=0.,
                 delivery_bonus=1.,walk_cost=0.,ledger_class=DeliveryLedger,**ledger_options):
        self.lifecycle=FarmLifecycleTracker(observation,episode_number=episode,
            player_index=observation['player'],turns_per_day=turns_per_day,graph_tracking=True)
        self.ledger=ledger_class(tasks,decay=decay,grow_reward=grow_reward,
                                  delivery_bonus=delivery_bonus,walk_cost=walk_cost,**ledger_options)
        for e in self.lifecycle.events:self.ledger.observe(e)

    def observe_transition(self,before,action,after,*,market_executions,decisions,market_decisions=None):
        self.ledger.register_decisions(decisions)
        economic = market_decisions is not None
        refs={('unit',r.unit_index):(2*r.decision_id+1 if economic else r.decision_id) for r in decisions}
        if economic:
            orders=[r for r in market_decisions if not r.boundary and r.action and r.action[0]!='STOP']
            refs.update({('market',i):2*r.decision_id for i,r in enumerate(orders)})
        events=self.lifecycle.observe_transition(before,action,after,
                           market_executions=market_executions,decision_refs=refs)
        start=len(self.ledger.deliveries)
        crew_events=[]
        for e in events:
            # The stored journal uses disjoint agent IDs. Delivery credit still
            # targets only bare crew IDs; BUY/SELL must not enter its graph.
            if economic:
                d=e.decision_id
                e=replace(e,decision_id=d//2 if d is not None and d%2 else None)
            crew_events.append(e)
            self.ledger.observe(e)
        self.ledger.charge_walks(decisions,crew_events)
        return self.ledger.deliveries[start:]
