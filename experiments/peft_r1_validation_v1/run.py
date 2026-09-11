"""Bounded R1 continuation; original records and sources are immutable inputs."""
import hashlib
import json
from pathlib import Path
import sys
import time
from experiments.peft_adaptation_scope_v1.guard import run_guarded
from experiments.hospital_shared_strength_v1.resume_monitored import admission

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/peft_r1_validation_v1'
OLD=ROOT/'runs/peft_mechanism_diagnostics_v1'
CODE=ROOT/'experiments/peft_r1_validation_v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def save(path,data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')


def execute(key, command, gpu=True):
    parent=RUN/key
    if parent.exists():raise RuntimeError(f'Preserve prior attempt: {parent}')
    admission(RUN,key)
    print(json.dumps({'starting':key}),flush=True)
    status=run_guarded(command,parent/'guard',ROOT,900,require_gpu=gpu)
    if not status['completed']:raise RuntimeError(f'Guard failed: {key}')


def main():
    started=time.perf_counter()
    RUN.mkdir(parents=True,exist_ok=False)
    old=json.loads((OLD/'contract.json').read_text())
    hashes=json.loads((OLD/'original_input_audit.json').read_text())['original_code_model_data_hashes']
    for path,digest in hashes.items():assert sha(ROOT/path)==digest,path
    for path,digest in old['source_hashes'].items():assert sha(ROOT/path)==digest,path
    rows={'SPREAD30':[round(i*89/29) for i in range(30)],'RECENT30':list(range(60,90))}
    assert len(set(rows['SPREAD30']))==30 and rows['SPREAD30'][0]==0 and rows['SPREAD30'][-1]==89
    sources=dict(old['source_hashes'])
    for path in [*CODE.glob('*.py'),CODE/'PURPOSE.md',ROOT/'experiments/peft_fullft_reference_v3/baselines.py',ROOT/'experiments/hospital_shared_strength_v1/resume_monitored.py']:
        sources[str(path.relative_to(ROOT))]=sha(path)
    reused={}
    for ds in ('bike','household'):
        for arm in ('MLP','ALL'):
            for seed in (25000,25001):
                for folder,names in [('fits',('result.json','best.pt')),('eval',('result.json','predictions.npz'))]:
                    for name in names:
                        path=OLD/folder/ds/arm/f'r0_s{seed}'/'output'/name
                        reused[str(path.relative_to(ROOT))]=sha(path)
    contract={'reference':old['reference'],'source_hashes':sources,'original_input_hashes':hashes,'reused_full90_hashes':reused,
              'training_rows':rows,'seeds':[25000,25001],'steps':200,'head_lr':1e-4,'lora_lr':1e-4,
              'fresh_test':False,'scope':'Fixed-compute origin coverage/recency sensitivity plus past-only F0 residual corrections',
              'new_fits':16,'new_model_evaluations':16,'residual_selection':'ZERO/BIAS/seasonal/context ridge; train-only OOF, V selection before exposed E'}
    path=RUN/'contract.json';save(path,contract)
    save(RUN/'input_audit.json',{'verified':True,'original_hash_count':len(hashes),'reused_artifact_count':len(reused)})
    for ds in ('bike','household'):
        execute(f'f0/{ds}',[sys.executable,'-m','experiments.peft_r1_validation_v1.cache_f0','--contract',str(path),'--dataset',ds,'--output',str(RUN/'f0'/ds/'output')])
    execute('residual_select',[sys.executable,'-m','experiments.peft_r1_validation_v1.residual','select'],gpu=False)
    entries=[]
    for ds in ('bike','household'):
        for condition,indices in rows.items():
            for arm in ('MLP','ALL'):
                for seed in (25000,25001):
                    job={'dataset':ds,'condition':condition,'training_rows':indices,'arm':arm,'seed':seed,'head':'mlp',
                         'blocks':list(range(12)) if arm=='ALL' else [],'rank':8,'head_lr':1e-4,'lora_lr':1e-4,'recipe':0}
                    key=f'fits/{ds}/{condition}/{arm}/s{seed}'
                    command=[sys.executable,'-m','experiments.peft_r1_validation_v1.subset_fit','--contract',str(path),'--job',json.dumps(job),'--output',str(RUN/key/'output')]
                    execute(key,command)
                    r=json.loads((RUN/key/'output/result.json').read_text())
                    assert r['completed'] and not r['holdout_opened']
                    entries.append({'job':job,'key':key,'val_score':r['best_score'],'best_step':r['best_step']})
    save(RUN/'selection.json',entries)
    for e in entries:
        key=e['key'].replace('fits/','eval/')
        command=[sys.executable,'-m','experiments.peft_r1_validation_v1.subset_fit','--contract',str(path),'--job',json.dumps(e['job']),
                 '--output',str(RUN/key/'output'),'--forecast-fit',str(RUN/e['key']/'output')]
        execute(key,command)
    execute('residual_eval',[sys.executable,'-m','experiments.peft_r1_validation_v1.residual','evaluate'],gpu=False)
    for name,digest in {**hashes,**reused,**sources}.items():assert sha(ROOT/name)==digest,name
    save(RUN/'completed.json',{'completed':True,'seconds':time.perf_counter()-started,'new_fits':16,'new_model_evaluations':16,'guard_jobs':36,'fresh_test':False})
    print(json.dumps({'completed':True,'seconds':time.perf_counter()-started}),flush=True)


if __name__=='__main__':main()
