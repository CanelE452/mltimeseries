"""Real end-to-end selected policies, including lookahead and rollback costs."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='2'
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch
from experiments.peft_contribution_freeze_v1.fit import Controlled,check,predict,shared,native,digest,snapshot,restore,freeze,sha,save
from experiments.peft_future_utility_v1.fit import cpu_copy,state_hash,rng_state,optimizer_for
from experiments.peft_capacity_probe_v1.model import Controlled as CapacityControlled
from experiments.peft_decision_transfer_v1.panel import Panel
from experiments.peft_decision_transfer_v1.policy import probe_choice,selected
from experiments.peft_decision_transfer_v1.analyse import option

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--index',required=True,type=int)
    p.add_argument('--method',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    start=time.perf_counter();plan=json.loads(Path(a.plan).read_text());entry=plan['jobs'][a.index];job=entry['job']
    run=ROOT/plan['run_dir'];choice=json.loads((run/'development_choice.json').read_text());spec=choice['methods'][a.method]
    assert job['episode']=='test' and job['family']==spec['family'] and job['recipe']==spec['recipe']
    for path,h in plan['source_hashes'].items():assert sha(ROOT/path)==h,path
    panel_spec=plan['data']['test'][job['dataset']];path=ROOT/panel_spec['fit_path'];assert sha(path)==panel_spec['fit_sha256']
    panel=Panel(path,'fit',False);panel.origins['train']=panel.origins['train'][np.asarray(job['training_rows'],dtype=int)]
    runtime=SimpleNamespace(checkpoint=plan['checkpoint'],device='cuda',min_free_ram_gib=5.)
    check(runtime);base=shared.load_base(runtime)
    model=(CapacityControlled if job['family']=='WIDE' else Controlled)(base,job,panel.count_channels)
    names={n for n,v in model.named_parameters() if v.requires_grad};adapters={n for n in names if 'lora_' in n}
    frozen={n for n,v in model.named_parameters() if not v.requires_grad};before=digest(model,frozen)
    params=[v for v in model.parameters() if v.requires_grad];optimizer=optimizer_for(model,adapters,job)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    schedule=list(plan['schedules'][job['condition']]);cap=schedule[-1];fork=cap//3;probe=cap//12;qstep=fork+probe
    schedule=sorted(set(schedule+[qstep]));samples=np.random.default_rng(job['seed']).integers(len(panel.origins['train']),size=(cap,8))
    q=torch.as_tensor(panel.quantiles,dtype=torch.float32,device='cuda');cache={};trial_steps=0;updates=0

    def evaluate(step,mode):
        pred,score=predict(model,panel,'val',runtime);key=f'{mode}_{step}'
        cache[key]=(snapshot(model,names),pred)
        return {'step':step,'V':score,'checkpoint':key}

    def capture(mode):
        return {'weights':snapshot(model,names),'optimizer':cpu_copy(optimizer.state_dict()),'rng':cpu_copy(rng_state()),'mode':mode}

    def reset(state):
        for n,v in model.named_parameters():v.requires_grad_(n in names);v.grad=None
        restore(model,state['weights']);opt=optimizer_for(model,adapters,job)
        if state['mode']=='HEAD':freeze(model,opt,adapters)
        opt.load_state_dict(cpu_copy(state['optimizer']))
        torch.set_rng_state(state['rng']['cpu']);torch.cuda.set_rng_state_all(state['rng']['cuda'])
        assert state_hash(opt.state_dict())==state_hash(state['optimizer'])
        return opt

    def update(step):
        nonlocal updates
        model.train();optimizer.zero_grad(set_to_none=True)
        for offset in (0,4):
            context,y,groups=panel.batch(panel.origins['train'][samples[step-1,offset:offset+4]],'cuda')
            with shared.legacy.precision(runtime):
                _,norm,loc,sc=model.encode(context,groups,48)
                loss=native.native_pinball(norm,y,loc,sc,q,model.use_arcsinh)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
            (loss/2).backward();check(runtime)
            del context,y,groups,norm,loc,sc,loss
        torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step();updates+=1

    history=[evaluate(0,'JOINT')];initial=history[0]['V'];mode='JOINT';action=spec['path'];last=0
    fixed_step=0 if action=='HEAD0' else fork if action=='HEAD1' else 2*fork if action=='HEAD2' else None
    patience=int(action[-1]) if action.startswith('ES') else None
    if fixed_step==0:freeze(model,optimizer,adapters);mode='HEAD0'
    step=1
    while step<=cap:
        update(step);last=step
        if step in schedule:
            history.append(evaluate(step,mode))
            if patience is not None:
                _,stop=selected(history,patience)
                if stop==step and len(history)>1:
                    best=history[0]['V'];stale=0
                    for h in history[1:]:
                        if h['V']<best:best=h['V'];stale=0
                        else:stale+=1
                    if stale>=patience:break
        if fixed_step==step:freeze(model,optimizer,adapters);mode=action
        if a.method=='PROBE' and step==fork:
            prefix=list(history);state=capture('JOINT');stop_score=history[-1]['V']
            for t in range(fork+1,qstep+1):update(t)
            jp=evaluate(qstep,'JOINT');joint_state=capture('JOINT')
            optimizer=reset(state);freeze(model,optimizer,adapters)
            for t in range(fork+1,qstep+1):update(t)
            hp=evaluate(qstep,'HEAD1');head_state=capture('HEAD')
            action=probe_choice({'STOP':stop_score,'HEAD':hp['V'],'JOINT':jp['V']},initial,spec['margin'])
            trial_steps=2*probe
            if action=='STOP':history=prefix;last=fork;break
            if action=='HEAD':optimizer=reset(head_state);history=prefix+[hp];mode='HEAD1'
            else:optimizer=reset(joint_state);history=prefix+[jp];mode='JOINT'
            step=qstep;last=qstep
        step+=1
    pick,_=selected(history);weights,pred=cache[pick['checkpoint']];restore(model,weights)
    assert digest(model,frozen)==before
    torch.cuda.synchronize();adaptation_seconds=time.perf_counter()-start
    # Verification is timed separately from the policy's adaptation execution.
    reference_dir=run/'fit'/entry['key']/'output';reference=json.loads((reference_dir/'result.json').read_text())
    expected=option(reference,spec)['selected'];assert pick['step']==expected['step']
    assert digest(model,names)==expected['model_hash']
    with np.load(reference_dir/(expected['checkpoint']+'.npz')) as z:np.testing.assert_array_equal(pred,z['prediction'])
    np.savez_compressed(out/'val_predictions.npz',prediction=pred,target=panel.targets('val'),scale=panel.fit_std[panel.target_indices],quantiles=panel.quantiles)
    save(out/'result.json',{'completed':True,'method':a.method,'job':job,'action':action,'selected_step':pick['step'],
         'selected_V':pick['V'],'selected_model_hash':expected['model_hash'],'reference_exact':True,
         'adaptation_seconds':adaptation_seconds,'total_updates_including_trials':updates,'trial_updates':trial_steps,
         'last_step':last,'history':history,'plan_sha256':sha(a.plan),'holdout_opened':False,
         'seconds_including_verification':time.perf_counter()-start})
    print(json.dumps({'completed':True,'method':a.method,'seconds':adaptation_seconds}),flush=True)


if __name__=='__main__':main()
