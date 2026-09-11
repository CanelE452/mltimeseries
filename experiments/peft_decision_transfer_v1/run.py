"""Serial development, sealed transfer, and forecast orchestration."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time
from experiments.peft_optimization_control_v1.run import sha,save
from experiments.hospital_shared_strength_v1.resume_monitored import admission
from experiments.peft_adaptation_scope_v1.guard import run_guarded

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/peft_decision_transfer_v1'
CODE=Path(__file__).resolve().parent


def verify(plan):
    for p,h in {**plan['source_hashes'],**plan['input_hashes']}.items():assert sha(ROOT/p)==h,p


def prepare():
    assert not (RUN/'plan.json').exists()
    data=json.loads((RUN/'prepared/summary.json').read_text())
    old=json.loads((ROOT/'runs/peft_future_utility_v1/plan.json').read_text())
    sources=dict(old['source_hashes'])
    for p in [*(p for p in CODE.glob('*.py') if not p.name.startswith('independent_')),*CODE.glob('*.md'),ROOT/'experiments/peft_capacity_probe_v1/model.py']:
        sources[p.relative_to(ROOT).as_posix()]=sha(p)
    inputs={'runs/peft_decision_transfer_v1/prepared/summary.json':sha(RUN/'prepared/summary.json')}
    for episode in data['data'].values():
        for spec in episode.values():
            for role in ('fit','holdout'):inputs[spec[f'{role}_path']]=spec[f'{role}_sha256']
    recipes=[{'head_lr':1e-4,'lora_lr':1e-4},{'head_lr':1e-4,'lora_lr':3e-5},{'head_lr':3e-5,'lora_lr':3e-5}]
    jobs=[]
    for episode,seeds in [('dev',[28000]),('test',[28002,28003,28004])]:
        for seed in seeds:
            for ds in sorted(data['data'][episode]):
                for condition in ('FULL90','SPREAD30'):
                    for family,ids in [('TREE',[0,1,2]),('WIDE',[0,2])]:
                        for recipe in ids:
                            rows=list(range(90)) if condition=='FULL90' else list(range(0,90,3))
                            job={'episode':episode,'dataset':ds,'condition':condition,'seed':seed,'family':family,
                                 'recipe':recipe,'arm':'WIDE' if family=='WIDE' else 'FULL','head':'mlp','rank':8,
                                 'blocks':[] if family=='WIDE' else list(range(12)),
                                 'training_rows':rows,**recipes[recipe]}
                            key=f'{episode}/{ds}/{condition}/{family}/r{recipe}/s{seed}'
                            jobs.append({'key':key,'job':job})
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'source_hashes':sources,'input_hashes':inputs,
          'run_dir':RUN.relative_to(ROOT).as_posix(),'data':data['data'],'checkpoint':old['checkpoint'],
          'jobs':jobs,'schedules':{'FULL90':[0,15,30,60,120,180],'SPREAD30':[0,5,10,20,40,60]},
          'margins_pct_F0':[0.,.25],'eval_origin_subsample_stride':4,'descriptive_band_pct_F0':.25,
          'gate':'Three prespecified pilot criteria in PURPOSE.md; no publication-acceptance claim.',
          'new_policy_sealed_before_test_E':True}
    verify(plan);save(RUN/'plan.json',plan)
    print(json.dumps({'prepared':True,'potential_jobs':len(jobs)}))


def execute(plan,index,stage,smoke=False):
    entry=plan['jobs'][index];key='smoke' if smoke else stage+'/'+entry['key']
    parent=RUN/key
    if not parent.exists():
        admission(RUN,key)
        cmd=[sys.executable,'-m','experiments.peft_decision_transfer_v1.fit','--plan',str(RUN/'plan.json'),
             '--index',str(index),'--output',str(parent/'output'),'--stage',stage]
        if smoke:cmd.append('--smoke')
        if stage=='forecast':cmd.extend(['--fit-dir',str(RUN/'fit'/entry['key']/'output')])
        print(json.dumps({'starting':key}),flush=True)
        status=run_guarded(cmd,parent/'guard',ROOT,900,require_gpu=True)
        assert status['completed'] and not status['reasons'],f'Failure preserved: {key}'
    status=json.loads((parent/'guard/status.json').read_text());result=json.loads((parent/'output/result.json').read_text())
    assert status['completed'] and status['returncode']==0 and not status['reasons']
    assert result['completed'] and result['plan_sha256']==sha(RUN/'plan.json') and result['job']==entry['job']


def replay_policies(plan):
    choice=json.loads((RUN/'development_choice.json').read_text())
    methods=['PROBE','FULL','FIXED','EARLY_STOP','HEAD','WIDE'];keys=[]
    cells=[(ds,c,s) for s in (28002,28003,28004) for ds in sorted(plan['data']['test']) for c in ('FULL90','SPREAD30')]
    for cell_index,(dataset,condition,seed) in enumerate(cells):
        order=methods[cell_index%len(methods):]+methods[:cell_index%len(methods)]
        for method in order:
            spec=choice['methods'][method]
            index=next(i for i,e in enumerate(plan['jobs']) if all(e['job'][k]==v for k,v in
                       {'episode':'test','dataset':dataset,'condition':condition,'seed':seed,'family':spec['family'],'recipe':spec['recipe']}.items()))
            key=f'replay/{dataset}/{condition}/s{seed}/{method}';keys.append(key);parent=RUN/key
            if not parent.exists():
                admission(RUN,key)
                cmd=[sys.executable,'-m','experiments.peft_decision_transfer_v1.replay','--plan',str(RUN/'plan.json'),
                     '--index',str(index),'--method',method,'--output',str(parent/'output')]
                print(json.dumps({'starting':key}),flush=True)
                status=run_guarded(cmd,parent/'guard',ROOT,900,require_gpu=True)
                assert status['completed'] and not status['reasons'],key
            status=json.loads((parent/'guard/status.json').read_text());r=json.loads((parent/'output/result.json').read_text())
            assert status['completed'] and status['returncode']==0 and not status['reasons']
            assert r['completed'] and r['reference_exact'] and r['method']==method
    save(RUN/'timing_completed.json',{'completed':True,'keys':keys,'replays':len(keys),
         'finished_utc':datetime.now(timezone.utc).isoformat(),'order':'deterministic cyclic rotation across cells, one timing per policy/cell'})


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--smoke-only',action='store_true');p.add_argument('--dev-only',action='store_true');a=p.parse_args()
    if a.prepare:prepare();return
    assert not (RUN/'completed.json').exists()
    plan=json.loads((RUN/'plan.json').read_text());verify(plan);start=time.perf_counter()
    execute(plan,0,'fit',True)
    if a.smoke_only:return
    for episode in ('dev','test'):
        indices=[i for i,e in enumerate(plan['jobs']) if e['job']['episode']==episode]
        if episode=='test':
            from experiments.peft_decision_transfer_v1.analyse import choose_development
            if not (RUN/'development_choice.json').exists():choose_development()
            choice=json.loads((RUN/'development_choice.json').read_text())
            tree_recipes={v['recipe'] for v in choice['methods'].values() if v['family']=='TREE'}
            wide_recipes={v['recipe'] for v in choice['methods'].values() if v['family']=='WIDE'}
            indices=[i for i in indices if (plan['jobs'][i]['job']['recipe'] in
                     (tree_recipes if plan['jobs'][i]['job']['family']=='TREE' else wide_recipes))
                     or (plan['jobs'][i]['job']['family']=='TREE' and plan['jobs'][i]['job']['seed']==28002)]
        for i in indices:execute(plan,i,'fit')
        seal=RUN/f'{episode}_fits_sealed.json'
        if not seal.exists():
            save(seal,{'sealed_utc':datetime.now(timezone.utc).isoformat(),'indices':indices,
                 'fit_result_hashes':{plan['jobs'][i]['key']:sha(RUN/'fit'/plan['jobs'][i]['key']/'output/result.json') for i in indices}})
        else:assert json.loads(seal.read_text())['indices']==indices
        for i in indices:execute(plan,i,'forecast')
        if episode=='dev':
            from experiments.peft_decision_transfer_v1.analyse import choose_development
            if not (RUN/'development_choice.json').exists():choose_development()
            if a.dev_only:return
    replay_policies(plan)
    verify(plan)
    save(RUN/'completed.json',{'completed':True,'finished_utc':datetime.now(timezone.utc).isoformat(),
         'seconds':time.perf_counter()-start,'plan_sha256':sha(RUN/'plan.json')})


if __name__=='__main__':main()
