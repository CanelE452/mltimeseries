"""Post-freeze evidence packaging and descriptive review; never imported by training."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import numpy as np
import psutil
from experiments.hospital_shared_strength_v1.resume_monitored import memory

ROOT=Path(__file__).resolve().parents[2]; RUN=ROOT/'runs/peft_head_convergence_v1'; OUT=ROOT/'results/peft_head_convergence_v1'


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,obj):
    with path.open('x',encoding='utf-8') as f: json.dump(obj,f,indent=2,allow_nan=False)


def main():
    assert read(RUN/'completed.json')['completed'] and read(OUT/'completed.json')['completed']
    assert read(RUN/'independent_audit.json')['completed']
    destination=OUT/'completion_review.json'; assert not destination.exists()
    plan=read(RUN/'plan.json'); protected={}
    for study in ['peft_head_convergence_v1','peft_initial_headroom_v1','peft_decision_transfer_v1']:
        p=read(ROOT/'runs'/study/'plan.json'); count=0
        for category in ['source_hashes','input_hashes']:
            for path,digest in p[category].items(): assert sha(ROOT/path)==digest,path; count+=1
        protected[study]=count
    statuses=[]; samples=[]; finishes=0
    for path in sorted(RUN.glob('**/guard/status.json')):
        s=read(path); assert s['completed'] and s['returncode']==0 and not s['reasons'],path
        statuses.append({'key':path.parent.parent.relative_to(RUN).as_posix(),'status':s})
        for line in (path.parent/'resource_log.jsonl').read_text().splitlines():
            if not line.strip(): continue
            item=json.loads(line)
            if 'available_ram_gib' in item: samples.append(item)
            else: finishes+=1
    b=read(RUN/'completed.json')['stage_B_executed']; expected=138 if b else 75
    assert len(statuses)==expected,(len(statuses),expected)
    gpu=[g for s in samples for g in (s.get('gpus') or [])]
    active=[]
    for p in psutil.process_iter(['pid','name','cmdline']):
        cmd=' '.join(p.info.get('cmdline') or [])
        if p.info['pid']==__import__('os').getpid(): continue
        if (p.info.get('name') or '').lower() in ('python.exe','python') and 'peft_head_convergence_v1' in cmd and 'completion_review' not in cmd:
            active.append({'pid':p.pid,'command':cmd})
    lock=ROOT/'runs/peft_adaptation_scope_v1/.guard.lock'
    assert not active and not lock.exists(),(active,lock)
    refit_tensor_checks=0
    if b:
        import torch
        stage_a=read(RUN/'selection_A.json')
        for entry in plan['jobs']:
            job=entry['job']
            if job['family']!='REFIT': continue
            source=stage_a['cells'][f"{job['dataset']}/s{job['seed']}"]['L720']['JOINT']['key']
            before=torch.load(RUN/'fit'/source/'output/L720.pt',map_location='cpu',weights_only=True)
            after=torch.load(RUN/'fit'/entry['key']/'output/L720.pt',map_location='cpu',weights_only=True)
            names={n for n in before if 'lora_' in n}; assert len(names)==192
            assert {n for n in after if 'lora_' in n}==names
            assert all(torch.equal(before[n],after[n]) for n in names)
            refit_tensor_checks+=1
            del before,after
        assert refit_tensor_checks==16
    rows=list(csv.DictReader((OUT/'metrics.csv').open())); histories=read(OUT/'fit_histories.json')
    def metric(cell,budget,fam): return next(float(r['D_score']) for r in rows if r['cell']==cell and r['budget']==budget and r['family']==fam)
    cells=sorted({r['cell'] for r in rows}); budget_changes=[]; comparisons=[]; caps={}
    seals={phase:read(RUN/f'selection_{phase}.json') for phase in (['A','B'] if b else ['A'])}
    for cell in cells:
        for fam in ['HEAD','WIDE','JOINT']:
            short=metric(cell,'S180',fam); long=metric(cell,'L720',fam); f0=metric(cell,'L720','F0')
            budget_changes.append({'cell':cell,'family':fam,'S180_D':short,'L720_D':long,'improvement_pct_F0':100*(short-long)/f0})
        for phase,sealed in seals.items():
            for budget in (['S180','L720'] if phase=='A' else ['L720']):
                for fam,selected in sealed['cells'][cell][budget].items():
                    cap=180 if budget=='S180' else 720
                    key=f'{budget}/{fam}'; caps[key]=caps.get(key,0)+int(selected['step']==cap)
    def load(phase,cell,budget,fam):
        selected=seals[phase]['cells'][cell][budget][fam]
        with np.load(RUN/'forecast'/selected['key']/budget/'output/selected.npz') as z: return z['prediction'].copy()
    for cell in cells:
        comparisons.append({'cell':cell,'L720_JOINT_equals_HEAD':bool(np.array_equal(load('A',cell,'L720','JOINT'),load('A',cell,'L720','HEAD'))),
                            'L720_JOINT_equals_WIDE':bool(np.array_equal(load('A',cell,'L720','JOINT'),load('A',cell,'L720','WIDE')))})
    evidence=OUT/'evidence'; evidence.mkdir(exist_ok=True); copies=[]
    names=['plan.json','selection_A.json','completed.json','independent_audit.json','window_qc.json','download_receipt.json',
           'preflight_review.json','environment.json','prepared/summary.json','prepared/data_audit.json']
    if b: names+=['selection_B.json']
    for name in names:
        source=RUN/name; target=evidence/name.replace('/','_'); assert not target.exists()
        target.write_bytes(source.read_bytes()); assert sha(source)==sha(target)
        copies.append({'source':source.relative_to(ROOT).as_posix(),'copy':target.name,'sha256':sha(source)})
    write(evidence/'copies.json',copies); write(evidence/'guard_statuses.json',statuses)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,cell in zip(axes.flat,cells):
        for fam,col in [('HEAD','#cc6633'),('WIDE','#3269aa'),('JOINT','#087e77')]:
            key=seals['A']['cells'][cell]['L720'][fam]['key']; r=read(RUN/'fit'/key/'output/result.json')
            ax.plot([h['step'] for h in r['history']],[h['train_score']/r['history'][0]['train_score'] for h in r['history']],label=fam,color=col)
        ax.axvline(180,color='grey',ls=':'); ax.set_title(cell); ax.set_xlabel('Updates'); ax.set_ylabel('Train score / initial train score'); ax.legend()
    fig.savefig(OUT/'04_train_trajectories.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True)
    for fam,offset,col in [('HEAD',-.23,'#cc6633'),('WIDE',0,'#3269aa'),('JOINT',.23,'#087e77')]:
        vals=[next(r['improvement_pct_F0'] for r in budget_changes if r['cell']==cell and r['family']==fam) for cell in cells]
        ax.bar(np.arange(4)+offset,vals,width=.22,color=col,label=fam)
    ax.axhline(0,color='black',lw=.8); ax.set_xticks(range(4),cells,rotation=20,ha='right'); ax.set_ylabel('S180 minus L720 (%F0)'); ax.set_title('Effect of larger budget on V-selected D loss'); ax.legend()
    fig.savefig(OUT/'05_budget_effect.png',dpi=160); plt.close(fig)
    if b:
        fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True)
        for family,other,offset,col in [('REFIT','HEAD',-.17,'#087e77'),('WARM_JOINT','WARM_HEAD',.17,'#cc6633')]:
            values=[100*(min(metric(c,'L720','F0'),metric(c,'L720',other))-metric(c,'L720',family))/metric(c,'L720','F0') for c in cells]
            ax.bar(np.arange(4)+offset,values,width=.32,label=f'{family} vs min(F0, {other})',color=col)
        ax.axhline(0,color='black',lw=.8); ax.axhline(.25,color='grey',ls=':'); ax.set_xticks(range(4),cells,rotation=20,ha='right'); ax.set_ylabel('Gain (%F0)'); ax.set_title('Prespecified component interventions (development only)'); ax.legend()
        fig.savefig(OUT/'06_component_interventions.png',dpi=160); plt.close(fig)
    categories={}; phase_family_costs={}
    for item in statuses:
        category=item['key'].split('/')[0]; g=categories.setdefault(category,{'jobs':0,'seconds':0.})
        g['jobs']+=1; g['seconds']+=item['status']['elapsed_seconds']
        parts=item['key'].split('/'); family=parts[2] if category=='smoke' else parts[3]
        label=f'{category}/{parts[1]}/{family}'; detail=phase_family_costs.setdefault(label,{'jobs':0,'guard_seconds':0.})
        detail['jobs']+=1; detail['guard_seconds']+=item['status']['elapsed_seconds']
    prefix_search_costs={}
    for cell in cells:
        for source_family in ['HEAD','JOINT']:
            matches=[h for h in histories if h['job']['phase']=='A' and h['job']['family']==source_family and f"{h['job']['dataset']}/s{h['job']['seed']}"==cell]
            prefix_search_costs[f'{cell}/{source_family}']={'trials':len(matches),'full_fit_seconds':sum(h['seconds'] for h in matches),
                'trajectory_seconds':sum(h['trajectory_seconds'] for h in matches)}
    result={'completed':True,'created_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':sha(Path(__file__)),
        'scope':'Post-freeze packaging/descriptive plots only; no changes to frozen training/analysis',
        'protected_hashes':protected,'guards':len(statuses),'categories':categories,'phase_family_costs':phase_family_costs,
        'prefix_search_costs':prefix_search_costs,'cost_scope':'Guard totals include child startup but exclude admission/CPU preparation. Parent run time includes admission gaps. Prefix search is already part of A; shared WARM_HEAD/WARM_JOINT prefixes are not duplicated in actual total. S180 selection comes from full720 trajectories, not a separately timed180 run.',
        'resource_extrema_scope':'Sampled extrema, device-wide GPU memory; not continuous per-process peaks.','resource_samples':len(samples),
        'finish_records':finishes,'minimum_available_ram_gib':min(s['available_ram_gib'] for s in samples),
        'minimum_available_commit_gib':min(s['available_commit_gib'] for s in samples),
        'maximum_child_rss_gib':max(s['child_tree_rss_gib'] for s in samples),
        'maximum_gpu_memory_mib':max(g['memory_used_mib'] for g in gpu),
        'maximum_gpu_temperature_c':max(g['temperature_c'] for g in gpu),'current_memory':memory(),
        'active_study_processes':active,'guard_lock_exists':lock.exists(),'selected_at_cap':caps,
        'budget_changes':budget_changes,'prediction_equality':comparisons,'copied_evidence':len(copies)}
    result['independent_refit_saved_adapter_tensor_checks']=refit_tensor_checks
    result['tensor_check_scope']='CPU checkpoint comparison for REFIT saved adapters if B ran; initial WARM head ownership uses runtime restoration assertions and independent initial prediction equality, not a separately saved initial head tensor.'
    write(destination,result); print(json.dumps(result,indent=2))


if __name__=='__main__': main()
