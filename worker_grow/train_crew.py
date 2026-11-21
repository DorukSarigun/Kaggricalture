"""Train ONLY the FuN-style crew on fixed manager contracts (explicit curriculum).

This is not a two-agent full-game benchmark. Curriculum assets are gifted.
No economic manager is learned, and no previous training run is touched.
"""
import argparse
from copy import deepcopy
import gzip
import json
import logging
from pathlib import Path
import random
import shutil
import numpy as np
import torch
from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as game

from market_execution import trace_market_executions
from worker_grow.contracts import ProductionTask, TaskBook, TASK_ITEMS
from worker_grow.crew import FeudalCrew, VERSION, INIT_VERSIONS, propose_crew_turn
from worker_grow.delivery import CrewDeliveryTracker
from worker_grow.issues import IssueBook, issue_unlock_limit
from worker_grow.scripted_manager import BestAgentTeacher
from worker_grow.training import finish_episode, update_crew
from worker_grow.assignments import decision_metadata
from worker_grow.schedule import STRATEGIES, WALK_POLICIES, resolve_walk_policy

def initialize_weights(model, path, version, device):
    if path is None:
        return
    saved = torch.load(path, map_location=device, weights_only=False)
    allowed = INIT_VERSIONS if version == VERSION else {version}
    if saved.get("version") not in allowed:
        raise ValueError("Incompatible crew checkpoint")
    model.load_initial_weights(saved)


PASS={'farmer':['PASS'],'hands':[],'market':[]}


def write_json(path,data):
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w',encoding='utf8') as f:json.dump(data,f,ensure_ascii=False,indent=2)
    temporary.replace(path)


def prepare_run_dir(root, *, resume=False, overwrite=False):
    root = Path(root).resolve()
    if overwrite and resume:
        raise ValueError('Cannot overwrite and resume')
    if overwrite and root.exists():
        shutil.rmtree(root)
    if root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError('Use a NEW crew run directory or --overwrite')
    if resume and not (root / 'latest.pt').exists():
        raise FileNotFoundError('Missing crew checkpoint')
    root.mkdir(parents=True, exist_ok=True)
    return root


def curriculum(env,item,stage,hands,quantity):
    """Clearly labelled starting assets, confined to this training fixture."""
    own=deepcopy(env.state[0].observation.farms[0])
    own['farmer']=[0,0];own['hands']=[[0,0] for _ in range(hands)]
    private=deepcopy(env.state[0].observation.private)
    private['inventories']=[{} for _ in range(hands+1)]
    if item in game.CROPS:
        product=item
        if stage=='seeded':private['seeds'][item]=1
        else:
            plant=game._new_plant(item,0,24)
            if stage=='mature':
                plant['planted_day']=-game.CROPS[item]['first_yield_day']
                plant['yield_units']=quantity
                plant['max_lifespan_step']=-1
            own['tiles'][0][0]=plant
    else:
        product=game.ANIMALS[item]['product']
        private['shed']['WHEAT']=30
        if stage=='seeded':private['shed'][item]=1
        else:
            animal=game._new_animal(item,0)
            if stage=='mature':
                animal['placed_day']=-game.ANIMALS[item]['first_yield_day']
                animal['yield_units']=quantity
            own['tiles'][0][0]=animal
    for state in env.state:state.observation.farms[0]=deepcopy(own)
    env.state[0].observation.private=private
    return TaskBook([ProductionTask('lesson:0',(0,0),item,((product,quantity),))])


def collect(model,args,episode,directory,teacher=None,*,lesson=None,evaluation=False):
    # Mastery callers predate the optional scripted assignment/walk controls.
    assignment_strategy = getattr(args, 'assignment_strategy', 'learned')
    walk_policy = resolve_walk_policy(assignment_strategy, getattr(args, 'walk_policy', None))
    directory.mkdir(parents=True,exist_ok=False)
    configuration={'episodeSteps':lesson.steps if lesson else args.episode_steps,'turnsPerDay':24,'seed':args.seed+episode,
                   'weedSpawnChance':0.}
    env=make('kaggriculture',configuration=configuration)
    gifted=lesson is not None or args.stage!='best_agent'
    if gifted:
        tasks=lesson.setup(env,args.seed+episode) if lesson else curriculum(env,args.item,args.stage,args.hands,args.quantity)
        book=None
    else:
        book=IssueBook()
        tasks=book.task_book
        teacher=teacher or BestAgentTeacher()
        teacher.attach_world(configuration)
    ledger_options={}
    if lesson is not None:
        from worker_grow.mastery import MasteryLedger
        ledger_options={'ledger_class':MasteryLedger,'progress_cap':args.progress_cap,
                        'decision_cost':args.decision_cost}
    tracker=CrewDeliveryTracker(deepcopy(env.state[0].observation),tasks,episode=episode,
                                decay=args.credit_decay,
                                grow_reward=lesson.progress if lesson else getattr(args,'grow_event_reward',0.),
                                delivery_bonus=getattr(args,'delivery_bonus',1.),
                                walk_cost=getattr(args,'walk_cost',0.),**ledger_options)
    rows=[];state=None;turn=0
    save=args.save_every>0 and episode%args.save_every==0
    trace=gzip.open(directory/'trace.jsonl.gz','wt',encoding='utf8')
    model.eval()
    try:
        while not env.done:
            observations=[deepcopy(s.observation) for s in env.state]
            for obs in observations:obs['step']=turn
            before=observations[0]
            opened=[]
            if book is not None:
                opened=teacher.propose_contracts(
                    before,book,step=turn,day=int(before['day']),
                    limit=issue_unlock_limit(
                        episode, interval=args.issue_unlock_interval,
                        initial=args.issue_unlock_initial,
                    ),
                )
                if any(issue.tile_xy is None for issue in opened):
                    raise ValueError('Scripted teacher must pin every ISSUE to a tile')
            action,new_rows,state=propose_crew_turn(
                model,before,tasks,state,
                assignment_strategy=assignment_strategy,
                walk_policy=walk_policy,
            )
            if lesson is not None:
                action=dict(action);action['market']=lesson.market(before)
            if book is not None:
                action=dict(action)
                action['market']=teacher.propose_market(before)
            actions=[action,deepcopy(PASS)]
            executions=trace_market_executions(observations,actions,configuration)
            env.step(actions)
            if any(str(s.status) not in ('ACTIVE','DONE') for s in env.state):
                raise RuntimeError('Environment failure')
            if book is not None:
                teacher.advance()
            after=deepcopy(env.state[0].observation);after['step']=turn+1
            deliveries=tracker.observe_transition(before,action,after,market_executions=executions[0],decisions=new_rows)
            if book is not None:
                book.sync_from_tasks(book.task_book,step=turn,prices=before['market']['prices'])
            if trace:
                trace.write(json.dumps({'observation':before,'action':action,'next_observation':after,
                    'decision_ids':[r.decision_id for r in new_rows],'deliveries':deliveries,
                    'assignments':[decision_metadata(r) for r in new_rows],
                    'opened_issues':[issue.task_id for issue in opened]})+'\n')
            rows.extend(new_rows);turn+=1
            if lesson is not None and not any(r.open for r in tasks.records.values()):break
    finally:
        if trace:trace.close()
    report=tracker.ledger.finish();finish_episode(rows,report)
    if lesson is not None:
        from worker_grow.mastery import segment_returns
        segment_returns(rows)
    if book is not None:
        book.sync_from_tasks(book.task_book,step=turn,prices=env.state[0].observation['market']['prices'])
        book.expire_open(step=turn,prices=env.state[0].observation['market']['prices'])
    summary={'event':'episode','episode':episode,'turns':turn,'decisions':len(rows),
             'crew_reward':report['reward'],'delivery_reward':report.get('delivery_reward',report['reward']),
             'grow_reward':report.get('grow_reward',0.),'grow_events':len(report.get('progress') or ()),
             'walk_reward':report.get('walk_reward',0.),'walk_attempts':report.get('walk_attempts',0),
             'deliveries':len(report['deliveries']),
             'delivered_units':sum(d['quantity'] for d in report['deliveries']),
             'stage':args.stage,'gifted_curriculum':gifted,
             'issue_limit':(None if book is None else issue_unlock_limit(
                 episode, interval=args.issue_unlock_interval,
                 initial=args.issue_unlock_initial,
             )),
             'assignment_strategy':assignment_strategy,
             'walk_policy':walk_policy,
             'tasks':{k:v['status'] for k,v in report['tasks'].items()}}
    if lesson is not None:
        from worker_grow.mastery import episode_metrics
        summary.update(episode_metrics(rows,report),lesson=lesson.name,evaluation=evaluation,
                       decision_reward=report['decision_reward'])
    write_json(directory/'delivery_credit.json',report)
    write_json(directory/'summary.json',summary)
    decision_log=[{'decision_id':r.decision_id,'step':r.step,'unit_index':r.unit_index,
        'action':r.action,'log_prob':r.log_prob,'value':r.value,'credit':r.credit,
        'target_return':r.target_return,
        'advantage':(r.credit if r.target_return is None else r.target_return)-r.value,'allowed_ops':r.allowed_ops,
        'assignment_features':r.assignment_features.tolist() if r.assignment_features is not None else None,
        **decision_metadata(r)} for r in rows]
    if lesson is not None:
        with gzip.open(directory/'decisions.json.gz','wt',encoding='utf8') as f:json.dump(decision_log,f)
    else:write_json(directory/'decisions.json',decision_log)
    tracker.lifecycle.write_artifacts(directory)
    if save:
        with gzip.open(directory/'replay.json.gz','wt',encoding='utf8') as f:json.dump(env.toJSON(),f)
    return rows,summary


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=100)
    p.add_argument('--episode-steps',type=int,default=None,
                   help='Gifted default 96; best_agent default 720')
    p.add_argument('--episodes-per-update',type=int,default=4)
    p.add_argument('--stage',choices=('mature','growing','seeded','best_agent'),default='mature')
    p.add_argument('--init-crew',type=Path,help='Optional crew checkpoint; weights only')
    p.add_argument('--item',choices=TASK_ITEMS,default='WHEAT')
    p.add_argument('--hands',type=int,default=2)
    p.add_argument('--quantity',type=int,default=1)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--sequence-length',type=int,default=16)
    p.add_argument('--learning-rate',type=float,default=3e-5)
    p.add_argument('--credit-decay',type=float,default=.8)
    p.add_argument('--walk-cost',type=float,default=.001,
                   help='Crew-local cost per attempted move; charged once, including no-ops')
    p.add_argument('--grow-event-reward',type=float,default=1.,
                   help='Positive crew credit for each verified bound grow event; 0 disables')
    p.add_argument('--delivery-bonus',type=float,default=2.,
                   help='Multiplier on first-quota shed delivery; grow events stay 1x')
    p.add_argument('--entropy-coef',type=float,default=.01)
    p.add_argument('--max-grad-norm',type=float,default=1.)
    p.add_argument('--target-kl',type=float,default=.02)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--save-every',type=int,default=10)
    p.add_argument('--issue-unlock-interval',type=int,default=10,
                   help='Episodes per extra concurrent issue; 0 disables')
    p.add_argument('--issue-unlock-initial',type=int,default=1)
    p.add_argument('--assignment-strategy',choices=STRATEGIES,default='learned',
                   help='Subtask chooser. learned is the PPO head; others are daily-budget planners')
    p.add_argument('--walk-policy',choices=WALK_POLICIES,default=None,
                   help='How to execute a chosen subtask. Default: learned with learned assignment, else greedy')
    p.add_argument('--no-update',action='store_true',
                   help='Roll out without PPO updates (for comparing planners)')
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--overwrite',action='store_true',
                   help='Delete an existing run directory and start from episode 1')
    return p


def main():
    args=parser().parse_args()
    if args.episode_steps is None:
        args.episode_steps=720 if args.stage=='best_agent' else 96
    for name in ('episodes','episodes_per_update','quantity','epochs','sequence_length'):
        if getattr(args,name)<1:raise ValueError(f'Invalid {name}')
    if args.episode_steps<2 or args.hands<0 or args.save_every<0 or not 0<args.credit_decay<=1:
        raise ValueError('Invalid crew run configuration')
    if not np.isfinite(args.grow_event_reward) or args.grow_event_reward<0:
        raise ValueError('Invalid grow event reward')
    if not np.isfinite(args.walk_cost) or args.walk_cost<0:raise ValueError('Invalid walk cost')
    if not np.isfinite(args.delivery_bonus) or args.delivery_bonus<=0:
        raise ValueError('Invalid delivery bonus')
    if args.issue_unlock_interval<0 or args.issue_unlock_initial<1:
        raise ValueError('Invalid issue unlock schedule')
    for name in ('learning_rate','max_grad_norm','target_kl','entropy_coef'):
        if not np.isfinite(getattr(args,name)) or getattr(args,name)<0:
            raise ValueError(f'Invalid {name}')
    if min(args.learning_rate,args.max_grad_norm,args.target_kl)<=0:raise ValueError('Zero optimizer parameter')
    args.walk_policy=resolve_walk_policy(args.assignment_strategy,args.walk_policy)
    if args.init_crew is not None and not args.init_crew.is_file():
        raise FileNotFoundError(args.init_crew)
    if args.resume and args.init_crew is not None:
        raise ValueError('Cannot init-crew and resume')
    root=prepare_run_dir(args.run_dir,resume=args.resume,overwrite=args.overwrite)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(message)s',
        handlers=[logging.FileHandler(root/'train.log'),logging.StreamHandler()])
    torch.set_num_threads(1);torch.manual_seed(args.seed);random.seed(args.seed);np.random.seed(args.seed)
    model=FeudalCrew().to(args.device);optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    from worker_grow.contracts import TASK_LIFECYCLE_VERSION
    config={**vars(args),'run_dir':str(root),'version':VERSION,'reward':'grow-delivery-walk-subtasks-v2',
            'task_lifecycle':TASK_LIFECYCLE_VERSION,
            'site_selection':'scripted_ghost_tile' if args.stage=='best_agent' else 'fixed_curriculum',
            'assignment_policy':args.assignment_strategy,
            'walk_policy':args.walk_policy,
            'issue_matching':'oldest_compatible',
            'gifted_curriculum':args.stage!='best_agent'}
    for key,value in list(config.items()):
        if isinstance(value,Path):config[key]=str(value)
    first=1;update=0
    if args.resume:
        saved=torch.load(root/'latest.pt',map_location=args.device,weights_only=False)
        if saved.get('version')!=VERSION:raise ValueError('Incompatible crew checkpoint')
        for k,v in config.items():
            if k in ('episodes','save_every','resume','overwrite','init_crew','no_update'): continue
            if saved['config'].get(k)!=v:raise ValueError(f'Resume mismatch: {k}')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng'].cpu());np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])
        if args.device=='cuda':torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
        first=saved['episode']+1;update=saved['update']
        if any(int(p.name.split('_')[-1])>=first for p in (root/'episodes').glob('episode_*')):
            raise RuntimeError('Uncheckpointed episodes exist; archive them before resume')
        write_json(root/'config.json',config)
    else:
        initialize_weights(model,args.init_crew,VERSION,args.device)
        write_json(root/'config.json',config)
    logging.info('Crew subtask training | site=%s | assignment=%s | walk=%s | walk_cost=%g | init=%s | device=%s',
                 config['site_selection'],args.assignment_strategy,args.walk_policy,
                 args.walk_cost,args.init_crew,args.device)
    pending=[]
    for episode in range(first,args.episodes+1):
        rows,summary=collect(model,args,episode,root/'episodes'/f'episode_{episode:06d}')
        pending.append(rows)
        with (root/'metrics.jsonl').open('a') as f:f.write(json.dumps(summary)+'\n')
        logging.info('Crew episode %d | delivery_reward=%.3f grow_reward=%.3f delivered_units=%d grow_events=%d tasks=%s',
                     episode,summary.get('delivery_reward',summary['crew_reward']),summary.get('grow_reward',0.),
                     summary['delivered_units'],summary.get('grow_events',0),summary['tasks'])
        logging.info('Crew navigation | episode=%d attempts=%d cost=%.4f',
                     episode,summary['walk_attempts'],summary['walk_reward'])
        if args.no_update:
            continue
        if len(pending)<args.episodes_per_update and episode!=args.episodes:continue
        metrics=update_crew(model,optimizer,pending,epochs=args.epochs,sequence_length=args.sequence_length,
            entropy_coef=args.entropy_coef,max_grad_norm=args.max_grad_norm,target_kl=args.target_kl,
            batch_log=root/'updates'/f'update_{update+1:06d}'/'crew_batches.jsonl',update_id=update+1)
        update+=1;pending.clear()
        with (root/'metrics.jsonl').open('a') as f:f.write(json.dumps({'event':'update','update':update,**metrics})+'\n')
        logging.info('Crew update %d | %s',update,metrics)
        torch.save({'version':VERSION,'config':config,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'episode':episode,'update':update,'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),
            'python_rng':random.getstate(),'cuda_rng':torch.cuda.get_rng_state_all() if args.device=='cuda' else []},root/'latest.pt.tmp')
        (root/'latest.pt.tmp').replace(root/'latest.pt')
    logging.info('Crew training completed (curriculum, not a competitive evaluation)')


if __name__=='__main__':main()
