"""Serial guarded count-control fits, short probes and sealed evaluation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from experiments.peft_optimization_control_v1.run import sha, save
from experiments.hospital_shared_strength_v1.resume_monitored import admission
from experiments.peft_adaptation_scope_v1.guard import run_guarded

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/peft_capacity_probe_v1'
CODE = ROOT / 'experiments/peft_capacity_probe_v1'
CONDITIONS = ('FULL90', 'SPREAD30', 'RECENT30')


def verify(plan):
    for p,digest in {**plan['source_hashes'], **plan['original_input_hashes'], **plan['preserved_inputs']}.items():
        assert sha(ROOT / p) == digest, p


def entry(period, job, short=False):
    condition, ds, seed = job['condition'], job['dataset'], job['seed']
    key = f'short/{period}/{ds}/{condition}/s{seed}' if short else f'fits/{period}/{ds}/{condition}/r{job["recipe"]}_s{seed}'
    return {'period': period, 'job': job, 'key': key, 'contract': f'{period}_{condition}' + ('_short' if short else '') + '_contract.json'}


def wide_job(old, recipe):
    lr = (1e-4, 1e-5)[recipe]
    return {**old, 'arm': 'WIDE', 'blocks': [], 'recipe': recipe, 'head_lr': lr, 'lora_lr': lr}


def prepare():
    old0 = ROOT / 'runs/peft_optimization_control_v1'
    old1 = ROOT / 'runs/peft_overlap_transfer_v1'
    c0 = json.loads((old0 / 'contract.json').read_text())
    c1 = json.loads((old1 / 'FULL90_contract.json').read_text())
    sources = {**c0['source_hashes'], **c1['source_hashes']}
    for p in [*CODE.glob('*.py'), CODE / 'PURPOSE.md']:
        sources[p.relative_to(ROOT).as_posix()] = sha(p)
    preserved, old_all, short_jobs, wide0 = {}, [], [], []
    for period, old in [('P0',old0), ('P1',old1)]:
        selection = json.loads((old / 'selection.json').read_text())
        selected = [e for e in selection if e['job']['arm'] == 'ALL' and e.get('regime','EXPOSURE') == 'EXPOSURE']
        assert len(selected) == 12
        for e in selected:
            job = e['job']
            assert job['head_lr'] == job['lora_lr'] == 1e-4
            fit_key = (old / e['key']).relative_to(ROOT).as_posix()
            eval_key = (old / e['key'].replace('fits/', 'eval/EXPOSURE/' if period == 'P0' else 'eval/')).relative_to(ROOT).as_posix()
            old_all.append({'period':period, 'job':job, 'key':fit_key, 'eval_key':eval_key})
            short_jobs.append({**entry(period, job, True), 'old_key':fit_key})
            for p in [ROOT / fit_key / 'output/result.json', ROOT / fit_key / 'output/EXPOSURE.pt', ROOT / fit_key / 'guard/status.json', ROOT / eval_key / 'output/result.json', ROOT / eval_key / 'output/predictions.npz']:
                preserved[p.relative_to(ROOT).as_posix()] = sha(p)
            if period == 'P0':
                for recipe in (0,1):
                    wide0.append(entry(period,wide_job(job,recipe)))
        preserved[(old / 'selection.json').relative_to(ROOT).as_posix()] = sha(old / 'selection.json')
    f0 = {}
    for period in ('P0','P1'):
        f0[period] = {}
        for ds in ('bike','household'):
            path = ROOT / (f'runs/peft_fullft_reference_v3/forecasts/{ds}/F0/Epredictions.npz' if period == 'P0' else f'runs/peft_overlap_transfer_v1/f0/{ds}/output/predictions.npz')
            f0[period][ds] = path.relative_to(ROOT).as_posix()
            preserved[path.relative_to(ROOT).as_posix()] = sha(path)
    plan = {'created_utc':datetime.now(timezone.utc).isoformat(), 'source_hashes':sources, 'original_input_hashes':c0['original_input_hashes'], 'preserved_inputs':preserved, 'references':{'P0':c0['reference'],'P1':c1['reference']}, 'old_all':old_all, 'wide_p0_jobs':wide0, 'short_jobs':short_jobs, 'f0_paths':f0, 'fresh_test':False, 'expected_guard_jobs':85, 'trainable_count':1768949, 'capacity_gate':{'period':'P1','dataset':'bike','condition':'FULL90','positive_both':True,'mean_min_pct_F0':1.}, 'probe_budgets':{'FULL90':[15,30],'SPREAD30':[5,10],'RECENT30':[5,10]}, 'probe_rule_threshold':0., 'cost_gate':{'macro_regret_max_pct_F0':.25,'source_regret_max_pct_F0':.5,'savings_min_percent':5.}}
    verify(plan)
    RUN.mkdir(parents=True, exist_ok=False)
    save(RUN / 'plan.json',plan)
    for period,reference in plan['references'].items():
        for condition in CONDITIONS:
            for short in (False,True):
                schedule = [0,*plan['probe_budgets'][condition]] if short else ([0,15,30,60,120,180] if condition == 'FULL90' else [0,5,10,20,40,60])
                path = RUN / f'{period}_{condition}{"_short" if short else ""}_contract.json'
                save(path,{'reference':reference,'source_hashes':sources,'steps':schedule[-1],'schedules':{condition:{'EXPOSURE':schedule}},'plan_sha256':sha(RUN/'plan.json')})
    print(json.dumps({'prepared':True,'wide_fits':36,'short_fits':24,'new_forecasts':24,'smokes':1}),flush=True)


def execute(e,key=None,extra=()):
    job = e['job']
    key = key or e['key']
    parent = RUN / key
    if not parent.exists():
        admission(RUN,key)
        cmd = [sys.executable,'-m','experiments.peft_capacity_probe_v1.fit','--contract',str(RUN/e['contract']),'--job',json.dumps(job),'--output',str(parent/'output'),*extra]
        print(json.dumps({'starting':key}),flush=True)
        status = run_guarded(cmd,parent/'guard',ROOT,900,require_gpu=True)
        assert status['completed'] and not status['reasons'],f'Preserved failure: {key}'
    status = json.loads((parent/'guard/status.json').read_text())
    result = json.loads((parent/'output/result.json').read_text())
    assert status['completed'] and result['completed'] and result['job']==job
    assert result['contract_sha256']==sha(RUN/e['contract'])
    for name,chosen in result.get('regimes',{}).items():
        assert chosen['replay_verified'] and sha(parent/'output'/f'{name}.pt')==chosen['checkpoint_sha256']
    return result


def seal(path,value):
    if path.exists():
        assert json.loads(path.read_text())==value
    else:
        save(path,value)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--smoke-only',action='store_true')
    args=parser.parse_args()
    if args.prepare:
        prepare(); return
    assert not (RUN/'completed.json').exists()
    started=time.perf_counter()
    plan=json.loads((RUN/'plan.json').read_text()); verify(plan)
    smoke=next(e for e in plan['wide_p0_jobs'] if e['job']['dataset']=='bike' and e['job']['condition']=='RECENT30' and e['job']['recipe']==0)
    execute(smoke,'smoke/WIDE',['--smoke'])
    if args.smoke_only:return
    for e in plan['wide_p0_jobs']:execute(e)
    selected0=[]
    recipes={}
    for ds in ('bike','household'):
        for condition in CONDITIONS:
            group=[e for e in plan['wide_p0_jobs'] if (e['job']['dataset'],e['job']['condition'])==(ds,condition)]
            means={recipe:sum(json.loads((RUN/e['key']/'output/result.json').read_text())['regimes']['EXPOSURE']['best_score'] for e in group if e['job']['recipe']==recipe)/2 for recipe in (0,1)}
            recipe=min(means,key=lambda k:(means[k],-k)); recipes[ds,condition]=recipe
            selected0.extend(e for e in group if e['job']['recipe']==recipe)
    seal(RUN/'wide_p0_selection.json',selected0)
    jobs1=[entry('P1',wide_job(e['job'],recipes[e['job']['dataset'],e['job']['condition']])) for e in plan['old_all'] if e['period']=='P1']
    seal(RUN/'p1_jobs.json',jobs1)
    for e in jobs1:execute(e)
    for e in plan['short_jobs']:
        record=execute(e)
        original=json.loads((ROOT/e['old_key']/'output/result.json').read_text())
        history={h['step']:h['score'] for h in original['history']}
        assert all(abs(h['score']-history[h['step']])<1e-10 for h in record['history'])
    selections=[]
    for e in selected0+jobs1:
        record=json.loads((RUN/e['key']/'output/result.json').read_text())
        selections.append({**e,'chosen':record['regimes']['EXPOSURE']})
    seal(RUN/'selection.json',selections)
    for e in selections:
        key=e['key'].replace('fits/','eval/')
        record=execute(e,key,['--regime','EXPOSURE','--forecast-fit',str(RUN/e['key']/'output')])
        assert record['regime']=='EXPOSURE' and Path(record['fit'])==RUN/e['key']/'output'
    verify(plan)
    save(RUN/'completed.json',{'completed':True,'finished_utc':datetime.now(timezone.utc).isoformat(),'invocation_seconds':time.perf_counter()-started,'wide_fits':36,'short_fits':24,'forecasts':24,'smokes':1,'selection_sha256':sha(RUN/'selection.json'),'fresh_test':False})
    print(json.dumps({'completed':True}),flush=True)


if __name__=='__main__':main()
