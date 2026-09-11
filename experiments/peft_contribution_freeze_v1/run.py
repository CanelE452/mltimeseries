"""Sequential, resource-admitted training and selection-sealed evaluation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from experiments.peft_optimization_control_v1.run import sha, save
from experiments.hospital_shared_strength_v1.resume_monitored import admission
from experiments.peft_adaptation_scope_v1.guard import run_guarded

ROOT=Path(__file__).resolve().parents[2]
CODE=ROOT/'experiments/peft_contribution_freeze_v1'
RUN=ROOT/'runs/peft_contribution_freeze_v1'
ARMS=('FULL','ES2','FIXED_FREEZE','CONTRIB_FREEZE')


def verify(plan):
    for p,d in {**plan['source_hashes'],**plan['original_input_hashes'],**plan['prepared_hashes']}.items():
        assert sha(ROOT/p)==d,p


def prepare():
    assert not (RUN/'plan.json').exists()
    old=json.loads((ROOT/'runs/peft_optimization_control_v1/contract.json').read_text())
    prepared=json.loads((RUN/'prepared/summary.json').read_text())
    sources=dict(old['source_hashes'])
    for p in [*CODE.glob('*.py'),*CODE.glob('*.md'),ROOT/'experiments/peft_overlap_transfer_v1/f0.py',
              ROOT/'experiments/peft_fullft_reference_v3/data.py',ROOT/'experiments/peft_external_gap_v1/data.py']:
        sources[p.relative_to(ROOT).as_posix()]=sha(p)
    datasets={}; prepared_hashes={}
    for ds,spec in prepared['datasets'].items():
        datasets[ds]={}
        for role in ('fit','holdout'):
            info=spec[f'{role}_data']; p=Path(info['path'])
            if not p.is_absolute():p=ROOT/p
            assert sha(p)==info['sha256']
            datasets[ds][f'{role}_data_path']=p.relative_to(ROOT).as_posix()
            datasets[ds][f'{role}_data_sha256']=info['sha256']
            prepared_hashes[p.relative_to(ROOT).as_posix()]=info['sha256']
    assert set(datasets)=={'bdg2','jena'}
    for p in [RUN/'prepared/summary.json',RUN/'data_audit.json']:
        prepared_hashes[p.relative_to(ROOT).as_posix()]=sha(p)
    audit=json.loads((RUN/'data_audit.json').read_text())
    assert audit['decision']=='PASS'
    raw_paths={'bdg2_raw':'data_external/bdg2_coarse_supervision_v1/raw/electricity.csv',
               'bdg2_metadata':'data_external/bdg2_coarse_supervision_v1/raw/metadata.csv',
               'bdg2_qc':'data_external/bdg2_coarse_supervision_v1/qc.json',
               'jena_2023a':'data/jena_mpi_roof/mpi_roof_2023a.csv',
               'jena_2023b':'data/jena_mpi_roof/mpi_roof_2023b.csv'}
    for key,path in raw_paths.items():
        prepared_hashes[path]=audit['source_hashes'][key]
    rows={'FULL90':list(range(90)),'SPREAD30':[round(i*89/29) for i in range(30)],'RECENT30':list(range(60,90))}
    schedules={c:([0,15,30,60,120,180] if c=='FULL90' else [0,5,10,20,40,60]) for c in rows}
    jobs=[]
    for si,seed in enumerate((27000,27001)):
        for di,ds in enumerate(('bdg2','jena')):
            for ci,(condition,indices) in enumerate(rows.items()):
                offset=(si+di+ci)%4
                for arm in ARMS[offset:]+ARMS[:offset]:
                    job={'dataset':ds,'condition':condition,'arm':arm,'seed':seed,'head':'mlp','rank':8,
                         'blocks':list(range(12)),'head_lr':1e-4,'lora_lr':1e-4,'training_rows':indices}
                    jobs.append({'job':job,'key':f'fits/{ds}/{condition}/{arm}/s{seed}'})
    reference={'checkpoint':old['reference']['checkpoint'],'datasets':datasets,
               'provenance':'prepared/summary.json; old contract used only for checkpoint and protected input hashes'}
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'source_hashes':sources,
          'original_input_hashes':old['original_input_hashes'],'prepared_hashes':prepared_hashes,
          'reference':reference,'schedules':schedules,'jobs':jobs,'seeds':[27000,27001],
          'gates':{'macro_regret_max_pct_F0':.25,'source_regret_max_pct_F0':.5,'savings_min_percent':5.,'each_seed_required':True},
          'exposure':'Existing source families; different evaluation periods; no pretraining-unseen claim',
          'expected_fits':48,'expected_forecasts':50,'expected_smokes':1}
    verify(plan); save(RUN/'plan.json',plan)
    save(RUN/'contract.json',{'reference':reference,'source_hashes':sources,'schedules':schedules,'plan_sha256':sha(RUN/'plan.json')})
    print(json.dumps({'prepared':True,'fits':48,'forecasts':50,'smoke':1}),flush=True)


def execute(key, command):
    parent=RUN/key
    if not parent.exists():
        admission(RUN,key)
        print(json.dumps({'starting':key}),flush=True)
        status=run_guarded(command,parent/'guard',ROOT,900,require_gpu=True)
        assert status['completed'] and not status['reasons'],f'Failure preserved: {key}'
    status=json.loads((parent/'guard/status.json').read_text())
    assert status['completed'] and not status['reasons'],key
    result=json.loads((parent/'output/result.json').read_text())
    assert result['completed'] and result['contract_sha256']==sha(RUN/'contract.json')
    return result


def command(e,key,extra=()):
    return [sys.executable,'-m','experiments.peft_contribution_freeze_v1.fit','--contract',str(RUN/'contract.json'),
            '--job',json.dumps(e['job']),'--output',str(RUN/key/'output'),*extra]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--smoke-only',action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare();return
    assert not (RUN/'completed.json').exists()
    started=time.perf_counter();plan=json.loads((RUN/'plan.json').read_text());verify(plan)
    smoke=next(e for e in plan['jobs'] if e['job']['arm']=='CONTRIB_FREEZE' and e['job']['condition']=='RECENT30')
    smoke_result=execute('smoke',command(smoke,'smoke',['--smoke']))
    assert smoke_result['frozen_step']==1 and smoke_result['frozen_tail_verified'] and smoke_result['toggle_checks']>=1
    if args.smoke_only:return
    selection=[]
    for e in plan['jobs']:
        record=execute(e['key'],command(e,e['key']))
        assert record['job']==e['job'] and record['frozen_verified'] and record['replay_verified'] and not record['holdout_opened']
        assert sha(RUN/e['key']/'output/best.pt')==record['checkpoint_sha256']
        selection.append({**e,'best_step':record['best_step'],'checkpoint_sha256':record['checkpoint_sha256']})
    if (RUN/'selection.json').exists():assert json.loads((RUN/'selection.json').read_text())==selection
    else:save(RUN/'selection.json',selection)
    for ds in ('bdg2','jena'):
        key='f0/'+ds
        execute(key,[sys.executable,'-m','experiments.peft_contribution_freeze_v1.f0','--contract',str(RUN/'contract.json'),'--dataset',ds,'--output',str(RUN/key/'output')])
    for e in selection:
        key=e['key'].replace('fits/','eval/')
        result=execute(key,command(e,key,['--forecast-fit',str(RUN/e['key']/'output')]))
        assert result['job']==e['job']
    verify(plan)
    save(RUN/'completed.json',{'completed':True,'finished_utc':datetime.now(timezone.utc).isoformat(),
         'invocation_seconds':time.perf_counter()-started,'fits':48,'forecasts':50,'smokes':1,
         'selection_sha256':sha(RUN/'selection.json'),'plan_sha256':sha(RUN/'plan.json')})


if __name__=='__main__': main()
