"""Independent NumPy score and selection audit with diagnostic plots."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='2'
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/peft_mechanism_diagnostics_v1'
OUT=ROOT/'results/peft_mechanism_diagnostics_v1/controlled'


def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    complete=load(RUN/'completed.json');assert complete['completed']
    contract=load(RUN/'contract.json');chash=sha(RUN/'contract.json')
    for p,digest in contract['source_hashes'].items():assert sha(ROOT/p)==digest
    sources={'runs/peft_mechanism_diagnostics_v1/contract.json':chash}
    fits=[]
    for p in sorted((RUN/'fits').rglob('result.json')):
        r=load(p);assert r['steps']==200 and r['completed'] and not r['holdout_opened']
        assert r['identity_error']==0 and r['frozen_verified'] and r['replay_verified']
        assert r['contract_sha256']==chash
        assert sha(p.parent/'best.pt')==r['checkpoint_sha256']
        sources[str(p.relative_to(ROOT))]=sha(p);fits.append(r)
    assert len(fits)==24
    for seed in (25000,25001):
        assert len({r['initial_head_hash'] for r in fits if r['job']['seed']==seed and r['job']['head']=='mlp'})==1
    selection=load(RUN/'r1_selection.json');assert len(selection)==12
    for ds in ('bike','household'):
        for arm in ('MLP','ALL','LINEAR'):
            candidates={r:np.mean([f['best_score'] for f in fits if f['job']['dataset']==ds and f['job']['arm']==arm and f['job']['recipe']==r]) for r in (0,1)}
            winner=min(candidates,key=lambda r:(candidates[r],r))
            chosen=[s for s in selection if s['job']['dataset']==ds and s['job']['arm']==arm]
            assert len(chosen)==2 and all(s['job']['recipe']==winner for s in chosen)
    rows=[];max_error=0.;gains={}
    for folder in ('eval','placement_eval'):
        if not (RUN/folder).exists():continue
        for p in sorted((RUN/folder).rglob('result.json')):
            r=load(p);assert r['completed'] and r['contract_sha256']==chash
            with np.load(p.parent/'predictions.npz') as a:
                pred,y,q,scale=a['prediction'].astype(float),a['target'].astype(float),a['quantiles'],a['scale']
                valid=np.isfinite(y);e=y[:,:,None,:]-pred
                loss=np.where(valid[:,:,None,:],2*np.maximum(q[None,None,:,None]*e,(q[None,None,:,None]-1)*e),0)/scale[None,:,None,None]
                score=float(np.mean(loss.sum((0,2,3))/(valid.sum((0,2))*len(q))))
                med=pred[:,:,np.argmin(abs(q-.5)),:]
                mae=float(np.nanmean(abs(y-med)/scale[None,:,None]))
                per_origin=(loss.sum((2,3))/(valid.sum(2)*len(q))).mean(1)
            err=abs(score-r['score']);max_error=max(max_error,err);assert err<1e-10
            fit=load(Path(r['fit'])/'result.json')
            ds=r['job']['dataset'];f0=load(ROOT/f'runs/peft_fullft_reference_v3/forecasts/{ds}/F0/result.json')['eval_score']
            row={'stage':folder,'dataset':ds,'arm':r['job']['arm'],'seed':r['job']['seed'],'recipe':r['job']['recipe'],
                'score':score,'gain_pct_f0':100*(f0-score)/f0,'median_mae_scaled':mae,'trainable':r['trainable'],
                'fit_seconds':fit['seconds'],'forecast_seconds':r['seconds'],'best_step':fit['best_step'],
                'max_cuda_allocated_gib':fit['max_cuda_allocated_gib']}
            rows.append(row);gains[(folder,ds,r['job']['arm'],r['job']['seed'])]=per_origin
            sources[str(p.relative_to(ROOT))]=sha(p)
            sources[str((p.parent/'predictions.npz').relative_to(ROOT))]=sha(p.parent/'predictions.npz')
    assert len([r for r in rows if r['stage']=='eval'])==12
    effects={}
    for ds in ('bike','household'):
        f0=load(ROOT/f'runs/peft_fullft_reference_v3/forecasts/{ds}/F0/result.json')['eval_score']
        pairs=[];absolute_gains=[]
        for seed in (25000,25001):
            pair={r['arm']:r for r in rows if r['stage']=='eval' and r['dataset']==ds and r['seed']==seed}
            pairs.append((pair['MLP']['score']-pair['ALL']['score'])/f0*100)
            absolute_gains.append((f0-pair['ALL']['score'])/f0*100)
        effects[ds]={'matched_mlp_backbone_gain_pct_f0':pairs,'mean':float(np.mean(pairs)),
                     'all_absolute_gain_pct_f0':absolute_gains,'all_better_than_f0_both_seeds':all(g>0 for g in absolute_gains)}
    gate=load(RUN/'r2_gate.json');assert gate['passed']==(all(g>0 for g in effects['bike']['matched_mlp_backbone_gain_pct_f0']) and effects['bike']['mean']>1)
    placement=None
    if gate['passed']:
        placement=load(RUN/'r2_selection.json')
        assert len(placement['entries'])==12
        assert len([r for r in rows if r['stage']=='placement_eval'])==12
        means={}
        count=set()
        for entry in placement['entries']:
            r=load(RUN/entry['key']/'output/result.json')
            assert r['completed'] and not r['holdout_opened']
            assert r['steps']==200 and r['frozen_verified'] and r['replay_verified'] and r['identity_error']==0
            assert r['contract_sha256']==chash
            assert r['initial_head_hash']==next(f['initial_head_hash'] for f in fits if f['job']['head']=='mlp' and f['job']['seed']==r['job']['seed'])
            assert sha(RUN/entry['key']/'output/best.pt')==r['checkpoint_sha256']
            sources[str((RUN/entry['key']/'output/result.json').relative_to(ROOT))]=sha(RUN/entry['key']/'output/result.json')
            count.add(r['trainable']);means.setdefault(r['job']['arm'],[]).append(r['best_score'])
        assert len(count)==1
        winner=min(('EARLY','MIDDLE','LATE'),key=lambda a:(np.mean(means[a]),a))
        assert winner==placement['selected_location']
        placement={'selected_location':winner,'equal_total_trainable':count.pop(),'mean_val':{k:float(np.mean(v)) for k,v in means.items()}}
    statuses=[load(p) for p in RUN.rglob('guard/status.json')]
    assert len(statuses)==(63 if gate['passed'] else 39)
    assert all(s['completed'] and s['returncode']==0 for s in statuses)
    resource=[]
    for p in RUN.rglob('guard/resource_log.jsonl'):
        resource.extend(json.loads(x) for x in p.read_text().splitlines() if x.strip())
    commits=[r['available_commit_gib'] for r in resource if r.get('available_commit_gib') is not None]
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig,axs=plt.subplots(1,2,figsize=(12,4.8))
    labels=['LINEAR','MLP','ALL']
    for ax,ds in zip(axs,('bike','household')):
        vals=[[r['gain_pct_f0'] for r in rows if r['stage']=='eval' and r['dataset']==ds and r['arm']==arm] for arm in labels]
        ax.bar(range(3),[np.mean(v) for v in vals],color=['#94a3b8','#38bdf8','#0f766e'])
        for i,v in enumerate(vals):ax.scatter([i-.04,i+.04],v,color='black',s=22)
        ax.axhline(0,color='gray',lw=1);ax.set(xticks=range(3),xticklabels=['Frozen + Linear','Frozen + MLP','LoRA + same MLP'],ylabel='Loss reduction relative to F0 (%)',title=ds)
    fig.suptitle('Matched-head diagnostic: two seeds; exposed E, exploratory');fig.tight_layout();fig.savefig(OUT/'r1_matched_head.png',dpi=170);plt.close(fig)
    if placement:
        labels=['ALL','ALL_LOW','EARLY','MIDDLE','LATE','RANDOM_A','RANDOM_B']
        fig,axs=plt.subplots(1,2,figsize=(13,5))
        means=[];counts=[]
        for i,arm in enumerate(labels):
            v=[r for r in rows if r['dataset']=='bike' and r['arm']==arm]
            means.append(float(np.mean([r['gain_pct_f0'] for r in v])));counts.append(v[0]['trainable'])
            axs[0].scatter([i-.04,i+.04],[r['gain_pct_f0'] for r in v],c='black',s=20,zorder=3)
        colors=['#0f766e' if a==placement['selected_location'] else '#94a3b8' for a in labels]
        axs[0].bar(range(len(labels)),means,color=colors);axs[0].axhline(0,c='gray');axs[0].set(xticks=range(len(labels)),xticklabels=labels,ylabel='Loss reduction vs F0 (%)',title='Green: location chosen by V')
        axs[1].bar(range(len(labels)),np.array(counts)/1e6,color=colors);axs[1].set(xticks=range(len(labels)),xticklabels=labels,ylabel='Trainable parameters (millions)',title='Includes identical residual MLP')
        for ax in axs:ax.tick_params(axis='x',rotation=35)
        fig.suptitle('Placement intervention: fixed optimization recipe, two seeds, exposed E');fig.tight_layout();fig.savefig(OUT/'r2_placement.png',dpi=170);plt.close(fig)
    report={'completed':True,'independent_score_max_error':max_error,'r1_fits_verified':len(fits),'forecast_rows_verified':len(rows),
        'initial_mlp_pairing_verified':True,'selection_verified':True,'effects':effects,'r2':placement,
        'guard_jobs':len(statuses),'new_safety_stops':sum(not s['completed'] for s in statuses),'minimum_sampled_commit_gib':min(commits),
        'runner_seconds':complete['seconds'],'all_guard_seconds':sum(s['elapsed_seconds'] for s in statuses),
        'source_hashes':sources,'limits':['previously exposed E; no independent confirmation','two paired optimization recipes; no optimum guarantee',
        'two optimizer seeds; one Bike and one Household window','placement selection is validation scan, not a learned probe selector','no complexity causal intervention performed']}
    (OUT/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_hashes','limits')}))


if __name__=='__main__':main()
