"""Guarded shared-prefix tree for prospective adaptation decisions."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='2'
import argparse
import copy
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch
from experiments.peft_contribution_freeze_v1.fit import (
    Controlled, check, predict, shared, native, digest, snapshot, restore,
    adapters_off, freeze, sha, save,
)
from experiments.peft_future_utility_v1.fit import cpu_copy, state_hash, rng_state, optimizer_for
from experiments.peft_capacity_probe_v1.model import Controlled as CapacityControlled
from experiments.peft_decision_transfer_v1.policy import paths, selected
from experiments.peft_decision_transfer_v1.panel import Panel

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',required=True);parser.add_argument('--index',type=int,required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--stage',choices=['fit','forecast'],default='fit')
    parser.add_argument('--fit-dir');parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();started=time.perf_counter()
    plan=json.loads(Path(args.plan).read_text());entry=plan['jobs'][args.index];job=entry['job']
    for p,h in plan['source_hashes'].items():assert sha(ROOT/p)==h,p
    spec=plan['data'][job['episode']][job['dataset']]
    forecast=args.stage=='forecast';role='holdout' if forecast else 'fit'
    if forecast:
        seal=ROOT/plan['run_dir']/f"{job['episode']}_fits_sealed.json"
        assert seal.exists()
        if job['episode']=='test':assert (ROOT/plan['run_dir']/'development_choice.json').exists()
    data=ROOT/spec[f'{role}_path'];assert sha(data)==spec[f'{role}_sha256']
    panel=Panel(data,'forecast' if forecast else 'fit',False)
    if forecast:panel.origins['eval']=panel.origins['eval'][::plan['eval_origin_subsample_stride']]
    if not forecast:panel.origins['train']=panel.origins['train'][np.asarray(job['training_rows'],dtype=int)]
    runtime=SimpleNamespace(checkpoint=plan['checkpoint'],device='cuda',min_free_ram_gib=5.)
    check(runtime);base=shared.load_base(runtime)
    model=(CapacityControlled if job['family']=='WIDE' else Controlled)(base,job,panel.count_channels)
    names={n for n,p in model.named_parameters() if p.requires_grad}
    native_names={n for n,p in model.named_parameters() if not p.requires_grad}
    adapters={n for n in names if 'lora_' in n};heads=names-adapters
    native_hash=digest(model,native_names);params=[p for p in model.parameters() if p.requires_grad]
    assert sum(p.numel() for p in params)==1768949
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    if forecast:
        source=Path(args.fit_dir);record=json.loads((source/'result.json').read_text())
        sealed=json.loads(seal.read_text())
        assert sealed['fit_result_hashes'][entry['key']]==sha(source/'result.json')
        assert record['job']==job and record['plan_sha256']==sha(args.plan)
        results={}
        for name,expected in record['kept_checkpoints'].items():
            p=source/(name+'.pt');assert sha(p)==expected
            restore(model,torch.load(p,map_location='cpu',weights_only=True))
            pred,score=predict(model,panel,'eval',runtime)
            np.savez_compressed(out/(name+'.npz'),prediction=pred,target=panel.targets('eval'),
                quantiles=panel.quantiles,scale=panel.fit_std[panel.target_indices],origins=panel.origins['eval'])
            results[name]=score
        assert digest(model,native_names)==native_hash
        save(out/'result.json',{'completed':True,'job':job,'scores':results,'plan_sha256':sha(args.plan),
             'fit_result_sha256':sha(source/'result.json'),'seconds':time.perf_counter()-started})
        return
    schedule=[0,1,2,3,4,6] if args.smoke else plan['schedules'][job['condition']]
    cap=schedule[-1];fork=cap//3;probe=1 if args.smoke else cap//12
    schedule=sorted(set(schedule+[fork+probe]));forks=(0,fork,2*fork)
    samples=np.random.default_rng(job['seed']).integers(len(panel.origins['train']),size=(cap,8))
    q=torch.as_tensor(panel.quantiles,dtype=torch.float32,device='cuda')
    target=panel.targets('val');scale=panel.fit_std[panel.target_indices]
    histories={};states={};saved={};current_C=0.;model_load_seconds=time.perf_counter()-started
    first_joint={};history=[]

    def evaluate(mode,step):
        torch.cuda.synchronize();begin=time.perf_counter()
        pred,score=predict(model,panel,'val',runtime)
        name=f'{mode}_{step}'
        np.savez_compressed(out/(name+'.npz'),prediction=pred,target=target,scale=scale,quantiles=panel.quantiles)
        weights=snapshot(model,names);states[name]=weights
        point={'step':step,'V':score,'checkpoint':name,'model_hash':digest(model,names)}
        torch.cuda.synchronize();point['evaluation_seconds']=time.perf_counter()-begin
        return point

    def capture(optimizer):
        value={'weights':snapshot(model,names),'optimizer':cpu_copy(optimizer.state_dict()),'rng':cpu_copy(rng_state())}
        value['model_hash']=digest(model,names);value['optimizer_hash']=state_hash(value['optimizer'])
        value['rng_hash']=state_hash(value['rng'])
        value['head_optimizer_hash']=state_hash({n:optimizer.state[p] for n,p in model.named_parameters() if n in heads})
        return value

    def reset(value):
        for n,p in model.named_parameters():p.requires_grad_(n in names);p.grad=None
        restore(model,value['weights']);optimizer=optimizer_for(model,adapters,job)
        optimizer.load_state_dict(cpu_copy(value['optimizer']))
        torch.set_rng_state(value['rng']['cpu']);torch.cuda.set_rng_state_all(value['rng']['cuda'])
        assert digest(model,names)==value['model_hash']
        assert state_hash(optimizer.state_dict())==value['optimizer_hash']
        assert state_hash(rng_state())==value['rng_hash']
        return optimizer

    def update(optimizer,step,mode):
        torch.cuda.synchronize();begin=time.perf_counter();model.train();optimizer.zero_grad(set_to_none=True)
        for offset in (0,4):
            context,y,groups=panel.batch(panel.origins['train'][samples[step-1,offset:offset+4]],'cuda')
            with shared.legacy.precision(runtime):
                _,norm,loc,sc=model.encode(context,groups,48)
                loss=native.native_pinball(norm,y,loc,sc,q,model.use_arcsinh)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
            (loss/2).backward();check(runtime)
            del context,y,groups,norm,loc,sc,loss
        grad=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
        if mode=='MASKED1':
            for n,p in model.named_parameters():
                if n in adapters:p.grad=None
        optimizer.step();torch.cuda.synchronize()
        return grad,time.perf_counter()-begin

    optimizer=optimizer_for(model,adapters,job)
    point=evaluate('JOINT',0);history=[point];saved[0]=capture(optimizer)
    assert point['V']>0
    cumulative=point['evaluation_seconds'];mode='JOINT'
    for step in range(1,cap+1):
        grad,seconds=update(optimizer,step,mode);cumulative+=seconds
        if step==fork+1:first_joint={'grad_norm':grad,'head_hash':digest(model,heads)}
        if step in schedule:
            point=evaluate(mode,step);cumulative+=point['evaluation_seconds'];point['trajectory_seconds']=cumulative;history.append(point)
            if step in forks:saved[step]=capture(optimizer)
            if step==fork and adapters:
                begin=time.perf_counter();before=digest(model,names)
                with adapters_off(model):off,offscore=predict(model,panel,'val',runtime)
                assert digest(model,names)==before
                current_C=100*(offscore-point['V'])/history[0]['V']
                np.savez_compressed(out/'current_off.npz',prediction=off,target=target,scale=scale,quantiles=panel.quantiles)
                off_seconds=time.perf_counter()-begin
            save(out/'progress.json',{'mode':mode,'step':step,'seconds':time.perf_counter()-started})
    history[0]['trajectory_seconds']=history[0]['evaluation_seconds'];histories['JOINT']=history
    checks=[]
    if job['family']!='WIDE':
        for mode,start in [('HEAD0',0),('HEAD1',fork),('HEAD2',2*fork),('MASKED1',fork)]+([('REPLAY',fork)] if args.smoke else []):
            begin=time.perf_counter();optimizer=reset(saved[start]);adapter_hash=digest(model,adapters)
            if mode.startswith('HEAD'):freeze(model,optimizer,adapters)
            assert state_hash({n:optimizer.state[p] for n,p in model.named_parameters() if n in heads})==saved[start]['head_optimizer_hash']
            restore_seconds=time.perf_counter()-begin
            branch=copy.deepcopy([p for p in history if p['step']<=start]);cumulative=branch[-1]['trajectory_seconds']
            restore_pred,_=predict(model,panel,'val',runtime)
            with np.load(out/(branch[-1]['checkpoint']+'.npz')) as z:np.testing.assert_array_equal(restore_pred,z['prediction'])
            for step in range(start+1,cap+1):
                grad,seconds=update(optimizer,step,mode);cumulative+=seconds
                if step==start+1 and mode in ('MASKED1','REPLAY'):
                    assert {'grad_norm':grad,'head_hash':digest(model,heads)}==first_joint
                if step in schedule:
                    point=evaluate(mode,step);cumulative+=point['evaluation_seconds'];point['trajectory_seconds']=cumulative
                    branch.append(point)
                    if mode=='REPLAY':
                        with np.load(out/(point['checkpoint']+'.npz')) as z,np.load(out/f'JOINT_{step}.npz') as ref:
                            np.testing.assert_array_equal(z['prediction'],ref['prediction'])
                    save(out/'progress.json',{'mode':mode,'step':step,'seconds':time.perf_counter()-started})
            if mode!='REPLAY':assert digest(model,adapters)==adapter_hash
            assert digest(model,native_names)==native_hash
            histories[mode]=branch;checks.append({'mode':mode,'start':start,'restore_verified':True,
                'head_adam_preserved':True,'adapter_unchanged':mode!='REPLAY','restore_seconds':restore_seconds})
    record={'completed':True,'job':job,'cap':cap,'fork':fork,'probe_steps':probe,'histories':histories,
        'current_C_pct_F0':current_C,'onoff_seconds':off_seconds if adapters else 0.,'branch_checks':checks,
        'model_load_seconds':model_load_seconds,'plan_sha256':sha(args.plan),'holdout_opened':False,
        'train_sample_sha256':sha_bytes(samples.tobytes()),'native_hash':native_hash,'smoke':args.smoke}
    keep=set()
    if job['family']=='WIDE':
        keep.update(p['checkpoint'] for p in [selected(history)[0],history[-1],history[0]])
    else:
        for margin in plan['margins_pct_F0']:
            for option in paths(record,margin).values():
                keep.update(option[k]['checkpoint'] for k in ('selected','final'))
        keep.add(history[0]['checkpoint'])
    kept={}
    for name in sorted(keep):
        torch.save(states[name],out/(name+'.pt'));kept[name]=sha(out/(name+'.pt'))
    assert digest(model,native_names)==native_hash
    record['kept_checkpoints']=kept;record['seconds']=time.perf_counter()-started
    save(out/'result.json',record)
    print(json.dumps({'completed':True,'job':job,'seconds':record['seconds']}),flush=True)


def sha_bytes(value):
    import hashlib
    return hashlib.sha256(value).hexdigest()


if __name__=='__main__':main()
