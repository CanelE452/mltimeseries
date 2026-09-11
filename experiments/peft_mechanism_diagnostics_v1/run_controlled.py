"""Serial admission-guarded R1/R2 diagnostic, with bounded jobs and frozen selection."""
import hashlib
import json
import random
from pathlib import Path
import sys
import time
from experiments.peft_adaptation_scope_v1.guard import run_guarded
from experiments.hospital_shared_strength_v1.resume_monitored import admission

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/peft_mechanism_diagnostics_v1'
CODE=ROOT/'experiments/peft_mechanism_diagnostics_v1'


def save(path,data):Path(path).write_text(json.dumps(data,indent=2),encoding='utf-8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute(job,key,contract,forecast_fit=None,smoke=False):
    parent=RUN/key
    parent.mkdir(parents=True,exist_ok=True)
    done=parent/'output/result.json'
    if done.exists():
        receipt=json.loads(done.read_text());assert receipt['completed'] and receipt['job']==job
        assert receipt['contract_sha256']==sha(contract)
        return receipt
    if (parent/'output').exists():raise RuntimeError(f'Preserve failed attempt: {parent}')
    admission(RUN,key)
    command=[sys.executable,'-m','experiments.peft_mechanism_diagnostics_v1.controlled_fit','--contract',str(contract),'--job',json.dumps(job),'--output',str(parent/'output')]
    if forecast_fit:command+=['--forecast-fit',str(forecast_fit)]
    if smoke:command+=['--smoke']
    print(json.dumps({'starting':key}),flush=True)
    status=run_guarded(command,parent/'guard',ROOT,900,require_gpu=True)
    if not status['completed']:raise RuntimeError(f'Guard failed {key}: {status}')
    return json.loads(done.read_text())


def main():
    started=time.perf_counter();RUN.mkdir(parents=True,exist_ok=True)
    contract=RUN/'contract.json'
    if not contract.exists():
        reference=json.loads((ROOT/'runs/peft_fullft_reference_v3/study_contract.json').read_text())
        sources=[CODE/'controlled_fit.py',CODE/'run_controlled.py']
        sources += [ROOT/'experiments/peft_adaptation_scope_v1'/n for n in ('modeling.py','train.py','guard.py')]
        sources += [ROOT/'experiments/peft_external_gap_v1/train.py',ROOT/'experiments/peft_fullft_reference_v3/model.py']
        save(contract,{'reference':reference,'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in sources},
            'scope':'Exploratory matched-head and placement diagnostic on previously exposed Study20 data',
            'recipes':[[.0001,.0001],[.0003,.00003]],'seeds':[25000,25001],
            'r1_max_fits':24,'r2_max_fits':12,'r2_gate':'Bike matched MLP mean gain >1% F0 and both paired seeds positive',
            'r2_selection':'fixed R1 ALL recipe; choose location with mean V before E; rank6 four-block groups vs rank2 all12 budget match',
            'fresh_test':False,'head':'native frozen head plus zero-init residual 768-533-ReLU-336 MLP; linear768-336 reference'})
    saved=json.loads(contract.read_text())
    for p,h in saved['source_hashes'].items():assert sha(ROOT/p)==h
    recipes=saved['recipes'];seeds=saved['seeds']
    def job(dataset,arm,seed,r,blocks=None,rank=8):
        return {'dataset':dataset,'arm':arm,'seed':seed,'head':'linear' if arm=='LINEAR' else 'mlp',
                'blocks':list(range(12)) if arm=='ALL' else ([] if blocks is None else blocks),'rank':rank,'head_lr':recipes[r][0],'lora_lr':recipes[r][1],'recipe':r}
    for arm in ('MLP','ALL','LINEAR'):
        execute(job('bike',arm,seeds[0],0),'smoke/'+arm,contract,smoke=True)
    selected=[]
    for dataset in ('bike','household'):
        for arm in ('MLP','ALL','LINEAR'):
            candidates=[]
            for r in (0,1):
                entries=[]
                for seed in seeds:
                    j=job(dataset,arm,seed,r);key=f'fits/{dataset}/{arm}/r{r}_s{seed}'
                    receipt=execute(j,key,contract);entries.append((j,key,receipt))
                candidates.append((sum(e[2]['best_score'] for e in entries)/len(entries),r,entries))
            winner=min(candidates,key=lambda x:(x[0],x[1]))
            selected.extend({'job':j,'key':key,'val_score':v['best_score']} for j,key,v in winner[2])
    save(RUN/'r1_selection.json',selected)
    evaluations=[]
    for entry in selected:
        j=entry['job'];key=entry['key'].replace('fits/','eval/')
        result=execute(j,key,contract,RUN/entry['key']/'output');evaluations.append(result)
    save(RUN/'r1_evaluations.json',evaluations)
    old=json.loads((ROOT/'runs/peft_fullft_reference_v3/forecasts/bike/F0/result.json').read_text())['eval_score']
    pairs=[]
    for seed in seeds:
        a={r['job']['arm']:r['score'] for r in evaluations if r['job']['dataset']=='bike' and r['job']['seed']==seed}
        pairs.append((a['MLP']-a['ALL'])/old*100)
    gate=all(g>0 for g in pairs) and sum(pairs)/len(pairs)>1
    save(RUN/'r2_gate.json',{'passed':gate,'gain_pct_f0_by_seed':pairs,'rule':saved['r2_gate'],'exploratory_only':True})
    if gate:
        recipe=next(e['job']['recipe'] for e in selected if e['job']['dataset']=='bike' and e['job']['arm']=='ALL')
        groups={'EARLY':[0,1,2,3],'MIDDLE':[4,5,6,7],'LATE':[8,9,10,11],
                'RANDOM_A':sorted(random.Random(25099).sample(range(12),4)),
                'RANDOM_B':sorted(random.Random(25100).sample(range(12),4)),
                'ALL_LOW':list(range(12))}
        placement=[]
        for arm,blocks in groups.items():
            for seed in seeds:
                j=job('bike',arm,seed,recipe,blocks,2 if arm=='ALL_LOW' else 6)
                key=f'placement/{arm}/s{seed}';v=execute(j,key,contract)
                placement.append({'job':j,'key':key,'val_score':v['best_score']})
        means={arm:sum(e['val_score'] for e in placement if e['job']['arm']==arm)/len(seeds) for arm in groups}
        chosen=min(('EARLY','MIDDLE','LATE'),key=lambda x:(means[x],x))
        save(RUN/'r2_selection.json',{'selected_location':chosen,'means':means,'entries':placement})
        for entry in placement:
            execute(entry['job'],entry['key'].replace('placement/','placement_eval/'),contract,RUN/entry['key']/'output')
    save(RUN/'completed.json',{'completed':True,'r1_completed':True,'r2_gate_passed':gate,'r2_executed':gate,'seconds':time.perf_counter()-started,'new_test_evidence':False})
    print(json.dumps({'completed':True,'r2_gate':gate,'seconds':time.perf_counter()-started}),flush=True)


if __name__=='__main__':main()
