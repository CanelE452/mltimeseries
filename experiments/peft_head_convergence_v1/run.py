"""Serial, resource-guarded Study35 with a prespecified conditional branch."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import sys
import time
import numpy as np
from experiments.hospital_shared_strength_v1.resume_monitored import admission
from experiments.peft_adaptation_scope_v1.guard import run_guarded
from experiments.peft_head_convergence_v1.analyse import read,write,sha,seal,analyse,finish

ROOT=Path(__file__).resolve().parents[2]; RUN=ROOT/'runs/peft_head_convergence_v1'; CODE=Path(__file__).parent


def verify(plan):
    for path,expected in {**plan['source_hashes'],**plan['input_hashes']}.items(): assert sha(ROOT/path)==expected,path


def prepare_plan():
    from experiments.peft_head_convergence_v1.prepare import prepare
    if not (RUN/'prepared/summary.json').exists(): prepare(RUN/'prepared')
    data=read(RUN/'prepared/summary.json'); assert data['all_qc_passed']
    audit=read(RUN/'prepared/data_audit.json')
    assert all(not item['exposure']['any_known_overlap'] for item in audit['audits'].values())
    window_qc={}
    for dataset,spec in data['data'].items():
        window_qc[dataset]={}
        for role,splits in [('fit',('train','val')),('holdout',('cal','eval'))]:
            with np.load(ROOT/spec[role+'_path']) as z:
                target=z['target_values']; indices=z['target_indices']; horizon=int(z['horizon'])
                for split in splits:
                    fractions=np.array([np.isfinite(target[o:o+horizon][:,indices]).mean(axis=0) for o in z[split+'_origins']])
                    assert fractions.mean(axis=1).min()>=.7,(dataset,split)
                    window_qc[dataset][split]={'min_pooled_window_fraction':float(fractions.mean(axis=1).min()),'min_each_target_fraction':fractions.min(axis=0).tolist()}
    write(RUN/'window_qc.json',window_qc)
    old=read(ROOT/'runs/peft_initial_headroom_v1/plan.json'); sources=dict(old['source_hashes'])
    for p in [*CODE.glob('*.py'),*CODE.glob('*.md')]:
        if not p.name.startswith(('independent_','completion_')): sources[p.relative_to(ROOT).as_posix()]=sha(p)
    inputs={f'runs/peft_head_convergence_v1/prepared/{name}':sha(RUN/'prepared'/name) for name in ('summary.json','data_audit.json')}
    inputs['runs/peft_head_convergence_v1/window_qc.json']=sha(RUN/'window_qc.json')
    for spec in data['data'].values():
        for role in ('fit','holdout'): inputs[spec[role+'_path']]=spec[role+'_sha256']
    jobs=[]; rates=[1e-5,3e-5,1e-4,3e-4]; joint=[(1e-5,1e-5),(3e-5,3e-5),(1e-4,3e-5),(1e-4,1e-4)]
    for phase,families in [('A',('HEAD','WIDE','JOINT')),('B',('REFIT','WARM_HEAD','WARM_JOINT'))]:
        for seed in (30000,30001):
            for dataset in sorted(data['data']):
                for family in families:
                    for recipe in range(4):
                        hl,ll=joint[recipe] if family in ('JOINT','WARM_JOINT') else (rates[recipe],0.)
                        blocks=list(range(12)) if family in ('JOINT','REFIT','WARM_JOINT') else []
                        j={'phase':phase,'dataset':dataset,'condition':'FULL90','seed':seed,'family':family,
                           'recipe':recipe,'arm':family,'head':'mlp','rank':8,'blocks':blocks,'head_lr':hl,'lora_lr':ll}
                        jobs.append({'key':f'{phase}/{dataset}/{family}/r{recipe}/s{seed}','job':j})
    plan={'study':'peft_head_convergence_v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'run_dir':RUN.relative_to(ROOT).as_posix(),'source_hashes':sources,'input_hashes':inputs,'data':data['data'],
        'checkpoint':old['checkpoint'],'jobs':jobs,'schedule':[0,1,2,4,8,15,30,60,120,180,240,360,540,720],
        'eval_stride':4,'gate_band_pct_F0':.25,'D_is_development':True,'no_final_test':True}
    verify(plan); write(RUN/'plan.json',plan)
    print(json.dumps({'prepared':True,'jobs_A':48,'conditional_jobs_B':48}),flush=True)


def execute(plan,index,stage='fit',budget='L720',smoke=False):
    e=plan['jobs'][index]; j=e['job']
    key=f"smoke/{j['phase']}/{j['family']}" if smoke else f"{stage}/{e['key']}"+(f'/{budget}' if stage=='forecast' else '')
    parent=RUN/key
    if not parent.exists():
        admission(RUN,key)
        cmd=[sys.executable,'-m','experiments.peft_head_convergence_v1.fit','--plan',str(RUN/'plan.json'),
             '--index',str(index),'--output',str(parent/'output'),'--stage',stage,'--budget',budget]
        if smoke: cmd+=['--smoke']
        print(json.dumps({'starting':key}),flush=True)
        status=run_guarded(cmd,parent/'guard',ROOT,900,require_gpu=True)
        assert status['completed'] and status['returncode']==0 and not status['reasons'],f'Failure preserved: {key}'
    status=read(parent/'guard/status.json'); result=read(parent/'output/result.json')
    assert status['completed'] and status['returncode']==0 and not status['reasons'],key
    assert result['completed'] and result['job']==j and result['plan_sha256']==sha(RUN/'plan.json'),key


def phase_run(plan,phase,smoke_only=False):
    indices=[i for i,e in enumerate(plan['jobs']) if e['job']['phase']==phase]
    families=sorted({plan['jobs'][i]['job']['family'] for i in indices})
    for family in families:
        i=next(i for i in indices if plan['jobs'][i]['job']['family']==family)
        execute(plan,i,smoke=True)
    if smoke_only: return
    started=time.perf_counter()
    for count,i in enumerate(indices,1):
        execute(plan,i)
        (RUN/'progress.json').write_text(json.dumps({'phase':phase,'stage':'fit','done':count,'total':len(indices),'seconds':time.perf_counter()-started}),encoding='utf-8')
    selected=seal(phase)
    for count,(i,budget) in enumerate(selected['forecasts'],1):
        execute(plan,i,'forecast',budget)
        (RUN/'progress.json').write_text(json.dumps({'phase':phase,'stage':'forecast','done':count,'total':len(selected['forecasts']),'seconds':time.perf_counter()-started}),encoding='utf-8')
    verify(plan)
    result=analyse(phase); print(json.dumps({'phase_completed':phase,'gates':result['gates']}),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--prepare',action='store_true'); parser.add_argument('--smoke-only',action='store_true')
    args=parser.parse_args(); RUN.mkdir(parents=True,exist_ok=True)
    if args.prepare: prepare_plan(); return
    assert not (RUN/'completed.json').exists()
    plan=read(RUN/'plan.json'); verify(plan); started=time.perf_counter()
    a=phase_run(plan,'A',args.smoke_only)
    if args.smoke_only: return
    if a['gates']['G1']['passes']: phase_run(plan,'B')
    finish(); verify(plan)
    write(RUN/'completed.json',{'completed':True,'seconds':time.perf_counter()-started,'finished_utc':datetime.now(timezone.utc).isoformat(),
                              'plan_sha256':sha(RUN/'plan.json'),'stage_B_executed':a['gates']['G1']['passes']})


if __name__=='__main__': main()
