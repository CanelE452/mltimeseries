"""One guarded budget trajectory, component intervention, or sealed forecast."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import torch
from experiments.peft_contribution_freeze_v1.fit import Controlled, check, predict, shared, native, digest, snapshot, restore, sha, save
from experiments.peft_capacity_probe_v1.model import Controlled as WideControlled
from experiments.peft_decision_transfer_v1.panel import Panel

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',required=True); parser.add_argument('--index',type=int,required=True)
    parser.add_argument('--output',required=True); parser.add_argument('--stage',choices=['fit','forecast'],default='fit')
    parser.add_argument('--budget',choices=['S180','L720'],default='L720'); parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args(); started=time.perf_counter()
    plan=json.loads(Path(args.plan).read_text()); entry=plan['jobs'][args.index]; job=entry['job']
    for path,expected in plan['source_hashes'].items(): assert sha(ROOT/path)==expected,path
    run=ROOT/plan['run_dir']; phase=job['phase']; family=job['family']; forecast=args.stage=='forecast'
    spec=plan['data'][job['dataset']]; path=ROOT/spec['holdout_path' if forecast else 'fit_path']
    assert sha(path)==spec['holdout_sha256' if forecast else 'fit_sha256']
    seal=None
    if forecast:
        seal=json.loads((run/f'selection_{phase}.json').read_text())
        assert seal['plan_sha256']==sha(args.plan)
        assert [args.index,args.budget] in seal['forecasts']
    if phase=='B':
        gate=json.loads((ROOT/'results/peft_head_convergence_v1/summary_A.json').read_text())
        assert gate['plan_sha256']==sha(args.plan) and gate['gates']['G1']['passes']
    panel=Panel(path,'forecast' if forecast else 'fit',args.smoke and phase=='A')
    if forecast: panel.origins['eval']=panel.origins['eval'][::plan['eval_stride']]
    runtime=SimpleNamespace(checkpoint=plan['checkpoint'],device='cuda',min_free_ram_gib=5.)
    check(runtime)
    model=(WideControlled if family=='WIDE' else Controlled)(shared.load_base(runtime),job,panel.count_channels)
    adapters={n for n,p in model.named_parameters() if 'lora_' in n}
    heads={n for n,p in model.named_parameters() if n.startswith('probe.')}
    original_head_hash=digest(model,heads)
    dependency=None
    if phase=='B':
        stage_a=json.loads((run/'selection_A.json').read_text()); assert stage_a['plan_sha256']==sha(args.plan)
        source_family='JOINT' if family=='REFIT' else 'HEAD'
        selected=stage_a['cells'][f"{job['dataset']}/s{job['seed']}"]['L720'][source_family]
        source=run/'fit'/selected['key']/'output'
        result=json.loads((source/'result.json').read_text())
        assert sha(source/'result.json')==stage_a['fit_hashes'][selected['key']]
        assert sha(source/'L720.pt')==result['budgets']['L720']['checkpoint_sha256']
        weights=torch.load(source/'L720.pt',map_location='cpu',weights_only=True)
        owned=adapters if family=='REFIT' else heads
        partial={n:v for n,v in weights.items() if n in owned}; assert set(partial)==owned
        restore(model,partial)
        if family=='REFIT':
            assert digest(model,heads)==original_head_hash
            for n,p in model.named_parameters():
                if n in adapters: p.requires_grad_(False)
        dependency={'source_key':selected['key'],'source_family':source_family,'source_hash':sha(source/'L720.pt'),
                    'loaded_component_hash':digest(model,owned),'initial_head_hash':digest(model,heads)}
        del weights,partial
    names={n for n,p in model.named_parameters() if p.requires_grad}
    frozen={n for n,p in model.named_parameters() if not p.requires_grad}
    frozen_hash=digest(model,frozen)
    persisted=names|adapters
    count=sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert count==(1768949 if family in ('JOINT','WIDE','WARM_JOINT') else 589301)
    out=Path(args.output); out.mkdir(parents=True,exist_ok=False)

    def export(name,prediction,split):
        np.savez_compressed(out/f'{name}.npz',prediction=prediction,target=panel.targets(split),
            scale=panel.fit_std[panel.target_indices],quantiles=panel.quantiles,origins=panel.origins[split])

    if forecast:
        source=run/'fit'/entry['key']/'output'; record=json.loads((source/'result.json').read_text())
        assert record['job']==job and record['plan_sha256']==sha(args.plan)
        assert sha(source/'result.json')==seal['fit_hashes'][entry['key']]
        selected=record['budgets'][args.budget]
        assert sha(source/f'{args.budget}.pt')==selected['checkpoint_sha256']
        if phase=='A' and family=='HEAD':
            initial,initial_score=predict(model,panel,'eval',runtime); export('F0',initial,'eval')
        weights=torch.load(source/f'{args.budget}.pt',map_location='cpu',weights_only=True)
        assert set(weights)==persisted
        restore(model,weights); assert digest(model,persisted)==selected['selected_hash']
        prediction,score=predict(model,panel,'eval',runtime); export('selected',prediction,'eval')
        assert digest(model,frozen)==frozen_hash
        save(out/'result.json',{'completed':True,'job':job,'budget':args.budget,'D_score':score,
             'plan_sha256':sha(args.plan),'selected_hash':digest(model,persisted),'seconds':time.perf_counter()-started,
             'finished_utc':datetime.now(timezone.utc).isoformat()})
        return

    schedule=[0,1,2,3] if args.smoke else plan['schedule']
    stream_seed=job['seed']+(200000 if family.startswith('WARM_') else 0)
    samples=np.random.default_rng(stream_seed).integers(len(panel.origins['train']),size=(schedule[-1],8))
    params=[p for p in model.parameters() if p.requires_grad]
    groups=[{'params':list(model.probe.parameters()),'lr':job['head_lr']}]
    active_adapters=adapters&names
    if active_adapters: groups.append({'params':[p for n,p in model.named_parameters() if n in active_adapters],'lr':job['lora_lr']})
    optimizer=torch.optim.AdamW(groups,weight_decay=0.,foreach=False)
    assert {id(p) for g in groups for p in g['params']}=={id(p) for p in params}
    q=torch.as_tensor(panel.quantiles,dtype=torch.float32,device='cuda')
    initial,initial_v=predict(model,panel,'val',runtime); export('initial_val',initial,'val')
    if dependency and family.startswith('WARM_'):
        with np.load(source/'L720_val.npz') as z: np.testing.assert_array_equal(initial,z['prediction'])
        dependency['initial_prediction_exact']=True
    if phase=='A' and family=='HEAD' and job['recipe']==0:
        train_f0,_=predict(model,panel,'train',runtime); export('F0_train',train_f0,'train')
    budgets={'S180':min(180,schedule[-1]),'L720':schedule[-1]} if phase=='A' else {'L720':schedule[-1]}
    best={k:{'V':initial_v,'step':0,'state':snapshot(model,persisted),'prediction':initial.copy()} for k in budgets}
    history=[]; gradients=[]
    torch.cuda.synchronize(); trajectory_start=time.perf_counter()
    def observe(step,prediction,v):
        _,train_score=predict(model,panel,'train',runtime)
        export(f'point_{step}',prediction,'val')
        history.append({'step':step,'V':v,'train_score':train_score,'utc':datetime.now(timezone.utc).isoformat()})
        for budget,cap in budgets.items():
            if step<=cap and v<best[budget]['V']:
                best[budget]={'V':v,'step':step,'state':snapshot(model,persisted),'prediction':prediction.copy()}
    observe(0,initial,initial_v)
    for step in range(1,schedule[-1]+1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        for offset in (0,4):
            context,target,ids=panel.batch(panel.origins['train'][samples[step-1,offset:offset+4]],'cuda')
            with shared.legacy.precision(runtime):
                _,norm,loc,scale=model.encode(context,ids,48)
                loss=native.native_pinball(norm,target,loc,scale,q,model.use_arcsinh)
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
            (loss/2).backward(); check(runtime)
            del context,target,ids,norm,loc,scale,loss
        gradients.append(float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))); optimizer.step()
        if step in schedule:
            prediction,v=predict(model,panel,'val',runtime); observe(step,prediction,v)
            save(out/'progress.json',{'step':step,'cap':schedule[-1],'seconds':time.perf_counter()-started})
    torch.cuda.synchronize(); trajectory_seconds=time.perf_counter()-trajectory_start
    final_hash=digest(model,persisted); final_v=history[-1]['V']
    assert digest(model,frozen)==frozen_hash
    result_budgets={}
    for budget,b in best.items():
        restore(model,b['state']); replay,score=predict(model,panel,'val',runtime)
        np.testing.assert_array_equal(replay,b['prediction']); assert abs(score-b['V'])<1e-12
        torch.save(b['state'],out/f'{budget}.pt'); export(f'{budget}_val',replay,'val')
        result_budgets[budget]={'V':score,'step':b['step'],'selected_hash':digest(model,persisted),
                                'checkpoint_sha256':sha(out/f'{budget}.pt')}
    assert digest(model,frozen)==frozen_hash
    save(out/'result.json',{'completed':True,'job':job,'plan_sha256':sha(args.plan),'initial_V':initial_v,
         'initial_head_hash':original_head_hash,'dependency':dependency,'history':history,'budgets':result_budgets,
         'trainable':count,'frozen_hash':frozen_hash,'frozen_verified':True,'restore_exact':True,'D_opened':False,
         'sample_sha256':hashlib.sha256(samples.tobytes()).hexdigest(),'steps':schedule[-1],
         'gradient_max':max(gradients),'clipped_steps':sum(g>1 for g in gradients),
         'final_hash':final_hash,'final_V':final_v,'trajectory_seconds':trajectory_seconds,
         'seconds':time.perf_counter()-started,'smoke':args.smoke,'finished_utc':datetime.now(timezone.utc).isoformat()})
    print(json.dumps({'done':entry['key'],'seconds':time.perf_counter()-started}),flush=True)


if __name__=='__main__': main()
