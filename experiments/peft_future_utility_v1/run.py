"""Freeze diagnostic protocol and run serial guarded continuation jobs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from experiments.peft_optimization_control_v1.run import sha, save
from experiments.peft_contribution_freeze_v1.run import verify as verify31
from experiments.hospital_shared_strength_v1.resume_monitored import admission
from experiments.peft_adaptation_scope_v1.guard import run_guarded

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/peft_future_utility_v1'
CODE = Path(__file__).resolve().parent


def verify(plan):
    for p,h in {**plan['source_hashes'],**plan['input_hashes']}.items():
        assert sha(ROOT/p) == h, p


def prepare():
    assert not RUN.exists()
    old_path = ROOT/'runs/peft_contribution_freeze_v1/plan.json'
    old = json.loads(old_path.read_text()); verify31(old)
    jobs = [{'job':e['job'],'reference_fit':'runs/peft_contribution_freeze_v1/'+e['key'],
             'key':f"jobs/{e['job']['dataset']}/{e['job']['condition']}/s{e['job']['seed']}"}
            for e in old['jobs'] if e['job']['arm']=='FULL']
    assert len(jobs) == 12
    sources = dict(old['source_hashes'])
    sources.update({p.relative_to(ROOT).as_posix():sha(p) for p in CODE.iterdir() if p.suffix in ('.py','.md')})
    fit_data = {ds:{'path':spec['fit_data_path'],'sha256':spec['fit_data_sha256']} for ds,spec in old['reference']['datasets'].items()}
    inputs = {old_path.relative_to(ROOT).as_posix():sha(old_path)}
    for spec in fit_data.values(): inputs[spec['path']] = spec['sha256']
    for e in jobs:
        parent = ROOT/e['reference_fit']/'output'
        for p in [parent/'result.json',*parent.glob('point_*.npz')]: inputs[p.relative_to(ROOT).as_posix()] = sha(p)
    plan = {'created_utc':datetime.now(timezone.utc).isoformat(),'source_hashes':sources,'input_hashes':inputs,
            'checkpoint':old['reference']['checkpoint'],'fit_data':fit_data,'jobs':jobs,'schedules':old['schedules'],
            'fork_fractions':[1/3,2/3],'selection_rows':list(range(14)),'diagnostic_rows':list(range(16,30)),
            'negligible_band_pct_F0':.25,'expected_main_jobs':12,'expected_forks':24,'smokes':1,
            'exposure':'Previously observed validation only; no E outcomes used; hypothesis-generating diagnostic',
            'new_method_claim':False}
    verify(plan); RUN.mkdir(parents=True);save(RUN/'plan.json',plan)
    print(json.dumps({'prepared':True,'main_jobs':12,'forks':24,'smokes':1}))


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--smoke-only',action='store_true');a=p.parse_args()
    if a.prepare: prepare();return
    assert not (RUN/'completed.json').exists()
    plan=json.loads((RUN/'plan.json').read_text());verify(plan);start=time.perf_counter()
    jobs=[('smoke',0,True)]+[(e['key'],i,False) for i,e in enumerate(plan['jobs'])]
    for key,i,smoke in jobs:
        parent=RUN/key
        if not parent.exists():
            admission(RUN,key)
            cmd=[sys.executable,'-m','experiments.peft_future_utility_v1.fit','--plan',str(RUN/'plan.json'),
                 '--index',str(i),'--output',str(parent/'output')]+(['--smoke'] if smoke else [])
            print(json.dumps({'starting':key}),flush=True)
            status=run_guarded(cmd,parent/'guard',ROOT,900,require_gpu=True)
            assert status['completed'] and not status['reasons'],f'Failure preserved: {key}'
        status=json.loads((parent/'guard/status.json').read_text());result=json.loads((parent/'output/result.json').read_text())
        assert status['completed'] and status['returncode']==0 and not status['reasons']
        assert result['completed'] and result['plan_sha256']==sha(RUN/'plan.json') and not result['holdout_opened']
        assert result['job']==plan['jobs'][i]['job'] and len(result['forks'])==2
        if smoke and a.smoke_only:return
    verify(plan)
    save(RUN/'completed.json',{'completed':True,'main_jobs':12,'forks':24,'smokes':1,
         'plan_sha256':sha(RUN/'plan.json'),'seconds':time.perf_counter()-start,'finished_utc':datetime.now(timezone.utc).isoformat()})


if __name__=='__main__':main()
