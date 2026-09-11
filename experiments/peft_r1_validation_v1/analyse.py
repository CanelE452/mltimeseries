"""Independent numerical and artifact audit of the bounded R1 continuation."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='2'
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.peft_r1_validation_v1.run import sha

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/peft_r1_validation_v1'
OLD=ROOT/'runs/peft_mechanism_diagnostics_v1'
OUT=ROOT/'results/peft_r1_validation_v1'


def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def independent(pred,y,q,scale):
    pred,y,q,scale=(np.asarray(x,dtype=np.float64) for x in (pred,y,q,scale))
    valid=np.isfinite(y)
    d=np.where(valid[:,:,None,:],y[:,:,None,:]-pred,0)
    loss=2*np.maximum(q[None,None,:,None]*d,(q[None,None,:,None]-1)*d)/scale[None,:,None,None]
    score=float(np.mean(loss.sum((0,2,3))/(valid.sum((0,2))*len(q))))
    per_origin=(loss.sum((2,3))/(valid.sum(2)*len(q))).mean(1)
    mid=int(np.argmin(abs(q-.5)))
    mae=float(np.mean(np.where(valid,abs(y-pred[:,:,mid])/scale[None,:,None],0).sum((0,2))/valid.sum((0,2))))
    return score,per_origin,mae


def main():
    complete=load(RUN/'completed.json');assert complete['completed']
    c=load(RUN/'contract.json');ch=sha(RUN/'contract.json')
    for k in ('source_hashes','original_input_hashes','reused_full90_hashes'):
        for p,digest in c[k].items():assert sha(ROOT/p)==digest,p
    supplemental=load(RUN/'supplemental_input_audit.json')
    assert supplemental['verified'] and supplemental['residual_eval_not_started']
    for p,digest in supplemental['f0_original_prediction_hashes'].items():assert sha(ROOT/p)==digest,p
    selection=load(RUN/'selection.json');assert len(selection)==16
    rows=[];origin_losses={};histories={};maximum=0.;receipts={}
    for ds in ('bike','household'):
        spec=c['reference']['datasets'][ds]
        assert sha(ROOT/spec['holdout_data_path'])==spec['holdout_data_sha256']
        with np.load(ROOT/spec['holdout_data_path']) as a:
            q=a['quantiles'];scale=a['fit_std'][a['target_indices']]
            expected_origins=a['eval_origins']
            target=np.stack([a['target_values'][o:o+48,a['target_indices']].T for o in expected_origins])
        with np.load(ROOT/f'runs/peft_fullft_reference_v3/forecasts/{ds}/F0/Epredictions.npz') as a:
            np.testing.assert_array_equal(a['target'],target)
            f0_prediction=a['predictions'].copy()
            f0,_,_=independent(f0_prediction,target,q,scale)
        for condition in ('FULL90','SPREAD30','RECENT30'):
            for arm in ('MLP','ALL'):
                for seed in (25000,25001):
                    fit=OLD/'fits'/ds/arm/f'r0_s{seed}'/'output' if condition=='FULL90' else RUN/'fits'/ds/condition/arm/f's{seed}'/'output'
                    ev=OLD/'eval'/ds/arm/f'r0_s{seed}'/'output' if condition=='FULL90' else RUN/'eval'/ds/condition/arm/f's{seed}'/'output'
                    r=load(fit/'result.json');e=load(ev/'result.json')
                    assert r['completed'] and e['completed'] and r['steps']==200
                    assert r['identity_error']==0 and r['frozen_verified'] and r['replay_verified'] and not r['holdout_opened']
                    assert sha(fit/'best.pt')==r['checkpoint_sha256']
                    assert r['best_score']==min(x['score'] for x in r['history'])
                    old=load(OLD/'fits'/ds/arm/f'r0_s{seed}'/'output/result.json')
                    assert r['initial_head_hash']==old['initial_head_hash'] and r['trainable']==old['trainable']
                    if condition!='FULL90':
                        assert r['contract_sha256']==e['contract_sha256']==ch
                        assert r['job']['training_rows']==c['training_rows'][condition]
                    with np.load(ev/'predictions.npz') as a:
                        np.testing.assert_array_equal(a['origins'],expected_origins)
                        np.testing.assert_array_equal(a['target'],target)
                        np.testing.assert_array_equal(a['quantiles'],q)
                        np.testing.assert_array_equal(a['scale'],scale)
                        if r['best_step']==0:
                            np.testing.assert_allclose(a['prediction'],f0_prediction,rtol=0,atol=1e-5)
                        score,by_origin,mae=independent(a['prediction'],target,q,scale)
                    err=abs(score-e['score']);maximum=max(maximum,err);assert err<1e-10
                    row={'dataset':ds,'condition':condition,'arm':arm,'seed':seed,'f0_score':f0,'score':score,
                         'gain_pct_f0':100*(f0-score)/f0,'scaled_median_mae':mae,'best_step':r['best_step'],
                         'val_score':r['best_score'],'trainable':r['trainable'],'fit_seconds':r['seconds']}
                    rows.append(row);origin_losses[(ds,condition,arm,seed)]=by_origin
                    histories[(ds,condition,arm,seed)]=r['history']
                    for p in (fit/'result.json',ev/'result.json',ev/'predictions.npz'):receipts[str(p.relative_to(ROOT))]=sha(p)
    effects={}
    for ds in ('bike','household'):
        effects[ds]={}
        f0=next(r['f0_score'] for r in rows if r['dataset']==ds)
        for condition in ('FULL90','SPREAD30','RECENT30'):
            byseed=[];blocks=[]
            for seed in (25000,25001):
                pair={r['arm']:r for r in rows if r['dataset']==ds and r['condition']==condition and r['seed']==seed}
                byseed.append(100*(pair['MLP']['score']-pair['ALL']['score'])/f0)
            diff=np.mean([origin_losses[(ds,condition,'MLP',seed)]-origin_losses[(ds,condition,'ALL',seed)] for seed in (25000,25001)],axis=0)
            blocks=[float(100*b.mean()/f0) for b in np.array_split(diff,4)]
            effects[ds][condition]={'matched_gain_pct_f0':byseed,'mean':float(np.mean(byseed)),
                                    'chronological_blocks_20origins':blocks,
                                    'both_all_better_than_f0':all(r['gain_pct_f0']>0 for r in rows if r['dataset']==ds and r['condition']==condition and r['arm']=='ALL')}
        effects[ds]['spread_minus_full_by_seed']=(np.array(effects[ds]['SPREAD30']['matched_gain_pct_f0'])-effects[ds]['FULL90']['matched_gain_pct_f0']).tolist()
        effects[ds]['recent_minus_spread_by_seed']=(np.array(effects[ds]['RECENT30']['matched_gain_pct_f0'])-effects[ds]['SPREAD30']['matched_gain_pct_f0']).tolist()
    residual=load(OUT/'residual/evaluation.json')
    selections=load(OUT/'residual/selection.json')
    for item in residual['records']:
        ds=item['dataset'];s=selections['selected'][ds]
        assert min(s['candidates'],key=lambda x:(x['val_score'],x['index']))==s['winner']
        assert item['selection_sha256']==sha(OUT/'residual/selection.json')
        for candidate in s['candidates']:
            for fold in candidate['oof']:assert fold['fit_last_target_end']<=fold['first_origin']
        with np.load(OUT/'residual'/f'{ds}_eval.npz') as a:
            sc,_,_=independent(a['prediction'],a['target'],a['quantiles'],a['scale'])
        err=abs(sc-item['score']);maximum=max(maximum,err);assert err<1e-10
    statuses=[load(p) for p in RUN.rglob('guard/status.json')]
    assert len(statuses)==37 and sum(s['completed'] and s['returncode']==0 for s in statuses)==36
    failed=[s for s in statuses if not s['completed']]
    assert len(failed)==1 and failed[0]['state']=='child_failed' and failed[0]['returncode']==1 and not failed[0]['reasons']
    recovery=load(RUN/'dtype_recovery_contract.json')
    assert sha(ROOT/'experiments/peft_r1_validation_v1/residual_dtypefix.py')==recovery['fixed_source_sha256']
    for p,digest in recovery['protected_selection'].items():assert sha(ROOT/p)==digest
    assert load(RUN/'residual_select/guard/status.json')['finished_at']<load(RUN/'residual_eval_dtype_retry/guard/status.json')['started_at']
    resources=[json.loads(line) for p in RUN.rglob('guard/resource_log.jsonl') for line in p.read_text().splitlines() if line.strip()]
    values={'min_ram_gib':min(r['available_ram_gib'] for r in resources if 'available_ram_gib' in r),
            'min_commit_gib':min(r['available_commit_gib'] for r in resources if r.get('available_commit_gib') is not None),
            'max_gpu_mib':max(g['memory_used_mib'] for r in resources for g in (r.get('gpus') or [])),
            'max_gpu_temp_c':max(g['temperature_c'] for r in resources for g in (r.get('gpus') or []))}
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    conditions=('FULL90','SPREAD30','RECENT30')
    for col,ds in enumerate(('bike','household')):
        for j,arm in enumerate(('MLP','ALL')):
            vals=[[r['gain_pct_f0'] for r in rows if r['dataset']==ds and r['condition']==cnd and r['arm']==arm] for cnd in conditions]
            pos=np.arange(3)+(j-.5)*.32
            axs[0,col].bar(pos,[np.mean(v) for v in vals],width=.3,label=arm,color=['#38bdf8','#0f766e'][j])
            for x,v in zip(pos,vals):axs[0,col].scatter([x-.02,x+.02],v,c='black',s=16,zorder=3)
        vals=[effects[ds][cnd]['matched_gain_pct_f0'] for cnd in conditions]
        axs[1,col].bar(range(3),[np.mean(v) for v in vals],color='#6366f1')
        for x,v in enumerate(vals):axs[1,col].scatter([x-.025,x+.025],v,c='black',s=20,zorder=3)
        for ax in axs[:,col]:ax.axhline(0,c='gray',lw=1);ax.set_xticks(range(3),conditions)
        axs[0,col].set(title=ds,ylabel='Loss reduction vs F0 (%)');axs[0,col].legend()
        axs[1,col].set(ylabel='(MLP loss - ALL loss) / F0 (%)')
    fig.suptitle('Fixed 200 updates: origin coverage and recency diagnostic; two seeds, exposed E')
    fig.tight_layout();fig.savefig(OUT/'origin_coverage.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    for col,ds in enumerate(('bike','household')):
        for row,arm in enumerate(('MLP','ALL')):
            ax=axs[row,col]
            for condition,color in zip(conditions,('#0f766e','#f59e0b','#6366f1')):
                for seed in (25000,25001):
                    h=histories[(ds,condition,arm,seed)];start=h[0]['score']
                    ax.plot([v['step'] for v in h],[100*(start-v['score'])/start for v in h],color=color,
                            linestyle='-' if seed==25000 else '--',label=condition if seed==25000 else None)
            ax.axhline(0,c='gray',lw=1);ax.set(title=ds+' '+arm,xlabel='Optimizer updates',ylabel='V loss reduction vs initial F0 (%)');ax.legend(fontsize=8)
    fig.suptitle('Validation learning curves: solid/dashed = two seeds; best checkpoint includes step 0')
    fig.tight_layout();fig.savefig(OUT/'validation_learning_curves.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    for col,ds in enumerate(('bike','household')):
        s=selections['selected'][ds];v0=s['val_f0_score']
        vals=[100*(v0-r['val_score'])/v0 for r in s['candidates']]
        labels=['ZERO','BIAS','S .01','S 1','S 100','C .01','C 1','C 100']
        axs[0,col].bar(range(8),vals,color=['#0f766e' if r['index']==s['winner']['index'] else '#94a3b8' for r in s['candidates']])
        axs[0,col].set_xticks(range(8),labels,rotation=35);axs[0,col].set(title=ds+' validation candidates',ylabel='Loss reduction vs F0 (%)')
        ev=next(r for r in residual['records'] if r['dataset']==ds)
        vals=[f['gain_pct_f0'] for f in s['winner']['oof']]+[100*(v0-s['winner']['val_score'])/v0,ev['gain_pct_f0']]
        axs[1,col].bar(range(4),vals,color=['#94a3b8','#94a3b8','#38bdf8','#0f766e'])
        axs[1,col].set_xticks(range(4),['Past OOF 1','Past OOF 2','V selection','Exposed E'])
        axs[1,col].set(title='V-selected: '+s['winner']['method'],ylabel='Loss reduction vs segment F0 (%)')
        for ax in axs[:,col]:ax.axhline(0,c='gray',lw=1)
    fig.suptitle('Residual correction: OOF excludes not-yet-arrived labels; E only after selection')
    fig.tight_layout();fig.savefig(OUT/'residual_correction.png',dpi=160);plt.close(fig)
    coverage={}
    fig,axs=plt.subplots(1,2,figsize=(12,3.8))
    for ax,ds in zip(axs,('bike','household')):
        with np.load(ROOT/c['reference']['datasets'][ds]['fit_data_path']) as a:
            train=a['train_origins'];target=a['target_values'];ix=a['target_indices']
            weights=[];coverage[ds]={}
            for condition,indices in {'FULL90':list(range(90)),**c['training_rows']}.items():
                origins=train[indices];weight=np.zeros(int(train[-1]-train[0]+48))
                for o in origins:weight[o-train[0]:o-train[0]+48]+=1
                weights.append(weight.reshape(-1,24).mean(1))
                hours=np.flatnonzero(weight)+train[0]
                coverage[ds][condition]={'origins':len(origins),'unique_target_hours':len(hours),'valid_target_fraction':float(np.isfinite(target[hours][:,ix]).mean()),
                                          'first_origin':str(a['timestamps'][origins[0]]),'last_origin':str(a['timestamps'][origins[-1]]),'full90_scaling_held_fixed':True}
        ax.imshow(np.array(weights),aspect='auto',interpolation='nearest',vmin=0,vmax=2,cmap='Blues')
        ax.set(yticks=range(3),yticklabels=conditions,xlabel='Target day since training start',title=ds)
    fig.suptitle('Training target coverage: white = unused, darker = overlapping supervision')
    fig.tight_layout();fig.savefig(OUT/'training_coverage.png',dpi=160);plt.close(fig)
    save(OUT/'coverage_audit.json',coverage)
    summary={'completed':True,'independent_score_max_error':maximum,'new_fits_verified':16,'reused_full90_fits_verified':8,
             'model_evaluations_verified':24,'effects':effects,'residual':residual,'resources':values,
             'runner_seconds':complete['seconds'],'timing_basis':complete['timing_basis'],'guard_jobs':len(statuses),
             'successful_guard_jobs':36,'cpu_assertion_failures':1,'safety_stops':sum(s['state']=='safety_stop' for s in statuses),
             'guard_seconds':sum(s['elapsed_seconds'] for s in statuses),
             'source_hashes':receipts,'analysis_sha256':sha(Path(__file__)),'fresh_test':False,
             'limits':['Fixed LR and updates; not optimal sample complexity','SPREAD30 vs RECENT30 changes coverage and recency jointly',
                       'Ridge grid limited; failure cannot exclude all simple corrections','No synthetic frequency intervention or independent source confirmation']}
    save(OUT/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('source_hashes','limits')}))


if __name__=='__main__':main()
