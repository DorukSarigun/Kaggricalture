"""Crew-only recurrent PPO on explicit delivery credits, without cash/GAE.

Contiguous chunks use detached behavior hidden states at their boundary (TBPTT).
No manager parameter or terminal-gold target enters this optimizer.
"""
import math
import numpy as np
import torch
from worker_grow.crew import statistics
from worker_grow.assignments import decision_metadata


def finish_episode(rows,report):
    byid={r.decision_id:r for r in rows}
    if len(byid)!=len(rows):raise ValueError('Duplicate crew decision IDs')
    if any(d not in byid for d in report['decision_credit']):raise ValueError('Non-crew reward target')
    for row in rows:row.credit=report['decision_credit'].get(row.decision_id,0.)
    if not math.isclose(sum(r.credit for r in rows),report['reward'],abs_tol=1e-8):
        raise ValueError('Crew target mismatch')


def update_crew(model,optimizer,episodes,*,epochs=3,sequence_length=16,clip=.2,
                entropy_coef=.01,value_coef=.5,max_grad_norm=1.,target_kl=.02,
                batch_log=None,update_id=0):
    if epochs<1 or sequence_length<1 or max_grad_norm<=0 or target_kl<=0:
        raise ValueError('Invalid PPO configuration')
    chunks=[rows[i:i+sequence_length] for rows in episodes for i in range(0,len(rows),sequence_length)]
    chunk_episodes=[ep for ep,rows in enumerate(episodes)
                    for i in range(0,len(rows),sequence_length)]
    if not chunks:raise ValueError('No crew training data')
    metrics=[];skipped=0;stopped=False
    model.train()
    for epoch in range(epochs):
        attempted=[]
        for index in torch.randperm(len(chunks)).tolist():
            rows=chunks[index]
            encoded=model.encode([r.inputs for r in rows])
            h=torch.as_tensor(rows[0].hidden_before,device=model.device)
            policy=[];values=[];entropy=[];kls=[]
            for j,r in enumerate(rows):
                h,w,value=model.step(encoded[j],h,r.inputs.goal_sum)
                target = r.credit if r.target_return is None else r.target_return
                values.append((value-target).square())
                if r.stochastic:
                    logp,ent=statistics(model,h,w,r)
                    logratio=logp-r.log_prob
                    ratio=logratio.clamp(-20,20).exp()
                    advantage=target-r.value
                    policy.append(-torch.minimum(ratio*advantage,ratio.clamp(1-clip,1+clip)*advantage))
                    entropy.append(ent);kls.append((ratio-1)-logratio)
                h=model.after_action(h,r.action)
            zero=encoded.sum()*0
            mean=lambda xs:torch.stack(xs).mean() if xs else zero
            kl=float(mean(kls).detach());attempted.append(kl)
            def log_chunk(rejected, details=None):
                if batch_log is None:return
                from worker_grow.manager_training import append_jsonl
                append_jsonl(batch_log,{'event':'crew_batch','update':update_id,'epoch':epoch,
                    'chunk':index,'episode_in_update':chunk_episodes[index],
                    'size':len(rows),'kl':kl,'skipped':rejected,
                    'decisions':[{'decision_id':r.decision_id,'step':r.step,
                        'unit_index':r.unit_index,'action':r.action,'credit':r.credit,
                        'value':r.value,'target_return':r.target_return,
                        'advantage':(r.credit if r.target_return is None else r.target_return)-r.value,
                        'stochastic':r.stochastic,**decision_metadata(r)} for r in rows],**(details or {})})
            if kl>2*target_kl:
                skipped+=1;log_chunk(True);continue
            value_loss=.5*mean(values)
            loss=mean(policy)+value_coef*value_loss-entropy_coef*mean(entropy)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),max_grad_norm,error_if_nonfinite=True)
            optimizer.step()
            metrics.append({'loss':float(loss.detach()),'value_loss':float(value_loss.detach()),
                'entropy':float(mean(entropy).detach()),'kl':kl,'gradient_norm_preclip':float(norm)})
            log_chunk(False,metrics[-1])
        if np.mean(attempted)>target_kl:stopped=True;break
    if not metrics:raise RuntimeError('All crew PPO chunks rejected')
    return {**{k:float(np.mean([m[k] for m in metrics])) for k in metrics[0]},
            'chunks':len(metrics),'skipped_chunks':skipped,'kl_stopped':stopped,'epochs':epoch+1}
