"""Crew-only mastery training, held-out promotion gates, explicit gifted lessons.

No economic manager is trained. Never resume a legacy reward run here.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import logging
from pathlib import Path
import random
import subprocess

import numpy as np
import torch

from worker_grow.crew import FeudalCrew, VERSION
from worker_grow.mastery import LESSONS, SEMANTICS, MasteryGate
from worker_grow.train_crew import collect, initialize_weights, prepare_run_dir, write_json
from worker_grow.training import update_crew


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=5000,help='Maximum training games; evaluation games are separate')
    p.add_argument('--episodes-per-update',type=int,default=8)
    p.add_argument('--eval-every',type=int,default=10,help='Updates between held-out evaluations')
    p.add_argument('--eval-episodes',type=int,default=20)
    p.add_argument('--promotion-threshold',type=float,default=.9)
    p.add_argument('--promotion-passes',type=int,default=2)
    p.add_argument('--review-fraction',type=float,default=.15,help='Earlier mastered lessons retained in training')
    p.add_argument('--init-crew',type=Path)
    p.add_argument('--learning-rate',type=float,default=3e-5)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--sequence-length',type=int,default=24)
    p.add_argument('--credit-decay',type=float,default=.95)
    p.add_argument('--walk-cost',type=float,default=.002)
    p.add_argument('--decision-cost',type=float,default=.001)
    p.add_argument('--progress-cap',type=float,default=.25)
    p.add_argument('--delivery-bonus',type=float,default=2.)
    p.add_argument('--entropy-coef',type=float,default=.01)
    p.add_argument('--target-kl',type=float,default=.02)
    p.add_argument('--max-grad-norm',type=float,default=1.)
    p.add_argument('--save-every',type=int,default=20)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--resume',action='store_true')
    return p


def append(path,row):
    with path.open('a',encoding='utf8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')


def evaluate(model,args,gate,root,update):
    """Fresh disjoint seeds; sampling RNG restored, no optimizer calls or rows reused."""
    py=random.getstate();npstate=np.random.get_state();cpu=torch.get_rng_state()
    cuda=torch.cuda.get_rng_state_all() if args.device=='cuda' else []
    was_training=model.training
    summaries=[]
    try:
        seed=args.seed+10_000_000+update*args.eval_episodes
        random.seed(seed);np.random.seed(seed % (2**32));torch.manual_seed(seed)
        eval_args=argparse.Namespace(**vars(args));eval_args.seed=seed;eval_args.save_every=1
        for index in range(1,args.eval_episodes+1):
            _,summary=collect(model,eval_args,index,root/'evaluations'/f'update_{update:06d}'/f'episode_{index:04d}',
                              lesson=LESSONS[gate.level],evaluation=True)
            summaries.append(summary)
        result=gate.assess(summaries)
        result.update(event='evaluation',update=update,seeds=[seed+i for i in range(1,args.eval_episodes+1)],
                      tasks_completed=sum(s['completed_tasks'] for s in summaries),
                      mean_walks=sum(s['walk_attempts'] for s in summaries)/len(summaries))
        return result
    finally:
        random.setstate(py);np.random.set_state(npstate);torch.set_rng_state(cpu)
        if cuda:torch.cuda.set_rng_state_all(cuda)
        model.train(was_training)


def save_checkpoint(root,model,optimizer,args,config,gate,episode,update):
    saved={'version':VERSION,'semantics':SEMANTICS,'config':config,'gate':asdict(gate),
           'model':model.state_dict(),'optimizer':optimizer.state_dict(),'episode':episode,'update':update,
           'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
           'cuda_rng':torch.cuda.get_rng_state_all() if args.device=='cuda' else []}
    torch.save(saved,root/'latest.pt.tmp');(root/'latest.pt.tmp').replace(root/'latest.pt')


def main():
    args=parser().parse_args()
    for key in ('episodes','episodes_per_update','eval_every','eval_episodes','promotion_passes','epochs','sequence_length'):
        if getattr(args,key)<1:raise ValueError(f'Invalid {key}')
    if (not 0<args.promotion_threshold<=1 or not 0<args.credit_decay<=1
            or not 0<=args.review_fraction<1):raise ValueError('Invalid threshold/decay/review')
    if args.episodes>=10_000_000:raise ValueError('Training seeds would overlap the held-out seed range')
    for key in ('walk_cost','decision_cost','progress_cap','entropy_coef','learning_rate','delivery_bonus','target_kl','max_grad_norm'):
        x=getattr(args,key)
        if not np.isfinite(x) or x<0:raise ValueError(f'Invalid {key}')
    if min(args.learning_rate,args.delivery_bonus,args.target_kl,args.max_grad_norm)<=0 or args.save_every<0:
        raise ValueError('Invalid optimizer/replay configuration')
    if args.resume and args.init_crew:raise ValueError('Resume and init-crew are exclusive')
    if args.init_crew and not args.init_crew.is_file():raise FileNotFoundError(args.init_crew)
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    root=prepare_run_dir(args.run_dir,resume=args.resume)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(message)s',
        handlers=[logging.FileHandler(root/'train.log'),logging.StreamHandler()])
    torch.set_num_threads(1);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    # Compatibility values consumed only by collect; lesson overrides the fixture.
    args.stage='mastery';args.episode_steps=720
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(run_dir=str(root),semantics=SEMANTICS,lessons=[asdict(x) for x in LESSONS],
                  gifted_curriculum=True,manager='fixed_contracts_and_daily_rehire',version=VERSION)
    source_files=sorted(Path('worker_grow').glob('*.py'))
    config['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    # Roundtrip normalizes tuples for strict, reproducible resume comparisons.
    config=json.loads(json.dumps(config))
    model=FeudalCrew().to(args.device);optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    gate=MasteryGate(threshold=args.promotion_threshold,required_passes=args.promotion_passes)
    first=1;update=0
    if args.resume:
        saved=torch.load(root/'latest.pt',map_location=args.device,weights_only=False)
        if saved.get('semantics')!=SEMANTICS or saved.get('version')!=VERSION:raise ValueError('Incompatible mastery resume')
        for k,v in config.items():
            if k not in ('episodes','save_every','resume','init_crew') and saved['config'].get(k)!=v:
                raise ValueError(f'Resume mismatch: {k}')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        gate=MasteryGate(**saved['gate']);first=saved['episode']+1;update=saved['update']
        torch.set_rng_state(saved['torch_rng'].cpu());np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])
        if args.device=='cuda':torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
        if any(int(p.name.split('_')[-1])>=first for p in (root/'episodes').glob('episode_*')):
            raise RuntimeError('Uncheckpointed episodes exist; archive explicitly before resume')
        if any(int(p.name.split('_')[-1])>update for p in (root/'evaluations').glob('update_*')):
            raise RuntimeError('Uncheckpointed evaluations exist; archive explicitly before resume')
    else:
        initialize_weights(model,args.init_crew,VERSION,args.device)
        # Source snapshot does not touch or revert the working tree.
        snapshot=root/'source';snapshot.mkdir()
        for source in source_files:
            (snapshot/source.name).write_bytes(source.read_bytes())
        diff=subprocess.run(['git','diff','--stat'],capture_output=True,text=True,check=True).stdout
        write_json(root/'source_manifest.json',{'git_diff_stat':diff,'semantics':SEMANTICS})
    write_json(root/'config.json',config)
    logging.info('Mastery crew | max_episodes=%d | CUDA=%s | manager=fixed | init=%s',args.episodes,args.device,args.init_crew)
    pending=[]
    for episode in range(first,args.episodes+1):
        if gate.mastered:break
        lesson_level=gate.level
        if gate.level and random.random()<args.review_fraction:
            lesson_level=random.randrange(gate.level)
        rows,summary=collect(model,args,episode,root/'episodes'/f'episode_{episode:06d}',lesson=LESSONS[lesson_level])
        pending.append(rows);summary.update(level=lesson_level,gate_level=gate.level,review=lesson_level!=gate.level)
        append(root/'metrics.jsonl',summary)
        logging.info('Episode %d/%d | lesson=%s success=%s turns=%d walks=%d delivered=%d support=%.3f reward=%.3f',
            episode,args.episodes,summary['lesson'],summary['success'],summary['turns'],summary['walk_attempts'],
            summary['delivered_units'],summary['grow_reward'],summary['crew_reward'])
        if len(pending)<args.episodes_per_update and episode!=args.episodes:continue
        update+=1
        metrics=update_crew(model,optimizer,pending,epochs=args.epochs,sequence_length=args.sequence_length,
            entropy_coef=args.entropy_coef,max_grad_norm=args.max_grad_norm,target_kl=args.target_kl,
            batch_log=root/'updates'/f'update_{update:06d}'/'crew_batches.jsonl',update_id=update)
        pending.clear();append(root/'metrics.jsonl',{'event':'update','update':update,**metrics})
        logging.info('Update %d | %s',update,metrics)
        if update%args.eval_every==0:
            result=evaluate(model,args,gate,root,update)
            append(root/'metrics.jsonl',result)
            logging.info('Evaluation | %s',result)
            if result['promoted']:
                torch.save({'version':VERSION,'semantics':SEMANTICS,'model':model.state_dict(),
                            'evaluation':result,'config':config},root/f'mastered_{result["level"]:02d}.pt')
        save_checkpoint(root,model,optimizer,args,config,gate,episode,update)
    logging.info('Mastery training completed | mastered=%s level=%d (NOT competitive evaluation)',gate.mastered,gate.level)


if __name__=='__main__':main()
