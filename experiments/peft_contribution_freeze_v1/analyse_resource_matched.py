"""Frozen outcome definitions, raw forecast checks and actual measured costs."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/peft_contribution_freeze_v1'
OUT=ROOT/'results/peft_contribution_freeze_v1/resource_matched'
ARMS=('FULL','ES2','FIXED_FREEZE','CONTRIB_FREEZE')


def score(pred,target,scale,quantiles):
    pred=np.sort(np.asarray(pred,dtype=np.float64),axis=2)
    target=np.asarray(target,dtype=np.float64)
    error=target[:,:,None,:]-pred
    q=np.asarray(quantiles,dtype=np.float64)[None,None,:,None]
    valid=np.isfinite(target)
    loss=np.where(valid[:,:,None,:],2*np.maximum(q*error,(q-1)*error),0.)
    return float((loss.sum(axis=(0,3))/valid.sum(axis=(0,2))[:,None]/np.asarray(scale)[:,None]).mean())


def read(path): return json.loads(Path(path).read_text())


def archive_score(path):
    with np.load(path,allow_pickle=False) as z:
        return score(z['prediction'],z['target'],z['scale'],z['quantiles'])


def cost_seconds(key):
    import hashlib
    contract=read(RUN/'timing_contract.json')
    for p,d in contract['source_hashes'].items():
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==d,p
    replay=read(RUN/'timing_completed.json')
    assert replay['completed'] and replay['contract_sha256']==hashlib.sha256((RUN/'timing_contract.json').read_bytes()).hexdigest()
    found=[r for r in replay['rows'] if r['key']==key]
    if found:
        assert len(found)==1 and found[0]['checkpoint_exact_match']
        g=read(RUN/found[0]['replay_key']/'guard/status.json')
        assert g['completed'] and not g['reasons'] and g['elapsed_seconds']==found[0]['fit_seconds']
        return g['elapsed_seconds']
    return read(RUN/key/'guard/status.json')['elapsed_seconds']


def main():
    assert read(RUN/'completed.json')['completed']
    plan=read(RUN/'plan.json');OUT.mkdir(parents=True,exist_ok=True)
    rows=[];checks=[]
    f0={ds:archive_score(RUN/'f0'/ds/'output/predictions.npz') for ds in ('bdg2','jena')}
    for ds in f0:
        checks.append(abs(f0[ds]-read(RUN/'f0'/ds/'output/result.json')['score']))
    for e in plan['jobs']:
        job=e['job'];fit=RUN/e['key'];ev=RUN/e['key'].replace('fits/','eval/')
        result=read(fit/'output/result.json');prediction_score=archive_score(ev/'output/predictions.npz')
        checks.append(abs(prediction_score-read(ev/'output/result.json')['score']))
        checks.append(abs(archive_score(fit/'output/val_predictions.npz')-result['best_score']))
        rows.append({k:job[k] for k in ('dataset','condition','seed','arm')} | {
            'E_score':prediction_score,'F0_score':f0[job['dataset']],
            'fit_seconds':cost_seconds(e['key']),
            'original_fit_seconds':read(fit/'guard/status.json')['elapsed_seconds'],
            'eval_seconds':read(ev/'guard/status.json')['elapsed_seconds'],
            'steps':result['steps'],'adapter_updates':result['adapter_updates'],
            'frozen_step':result['frozen_step'],'best_step':result['best_step'],
            'best_V_score':result['best_score'],'toggle_checks':result['toggle_checks']})
    for row in rows:
        full=next(r for r in rows if r['arm']=='FULL' and all(r[k]==row[k] for k in ('dataset','condition','seed')))
        row['regret_pct_F0']=100*(row['E_score']-full['E_score'])/row['F0_score']
        row['fit_savings_percent']=100*(1-row['fit_seconds']/full['fit_seconds'])
    metrics=[]
    for seed in plan['seeds']:
        base=[r for r in rows if r['seed']==seed and r['arm']=='FULL']
        for arm in ARMS[1:]:
            group=[r for r in rows if r['seed']==seed and r['arm']==arm]
            sources={ds:float(np.mean([r['regret_pct_F0'] for r in group if r['dataset']==ds])) for ds in f0}
            regret=float(np.mean([r['regret_pct_F0'] for r in group]))
            seconds=sum(r['fit_seconds'] for r in group)
            savings=100*(1-seconds/sum(r['fit_seconds'] for r in base))
            g=plan['gates']
            quality=regret<=g['macro_regret_max_pct_F0'] and max(sources.values())<=g['source_regret_max_pct_F0']
            metrics.append({'seed':seed,'arm':arm,'macro_regret_pct_F0':regret,'source_regret_pct_F0':sources,
                'fit_seconds':seconds,'fit_savings_percent':savings,'quality_passed':bool(quality),
                'cost_passed':bool(savings>g['savings_min_percent']),'gate_passed':bool(quality and savings>g['savings_min_percent']),
                'freeze_count':sum(r['frozen_step'] is not None for r in group),'stopped_before_cap':sum(r['steps']<(180 if r['condition']=='FULL90' else 60) for r in group)})
    dominance=[]
    for seed in plan['seeds']:
        candidate=next(r for r in metrics if r['seed']==seed and r['arm']=='CONTRIB_FREEZE')
        for arm in ('ES2','FIXED_FREEZE'):
            b=next(r for r in metrics if r['seed']==seed and r['arm']==arm)
            no_worse=b['macro_regret_pct_F0']<=candidate['macro_regret_pct_F0'] and b['fit_seconds']<=candidate['fit_seconds']
            strict=b['macro_regret_pct_F0']<candidate['macro_regret_pct_F0'] or b['fit_seconds']<candidate['fit_seconds']
            dominance.append({'seed':seed,'baseline':arm,'dominates_candidate_macro_quality_cost':bool(no_worse and strict)})
    assert max(checks)<1e-10
    summary={'completed':True,'analysed_utc':datetime.now(timezone.utc).isoformat(),'F0':f0,'gates':plan['gates'],
             'metrics':metrics,'dominance':dominance,'raw_score_max_abs_error':max(checks),
             'candidate_gate_passed':all(r['gate_passed'] for r in metrics if r['arm']=='CONTRIB_FREEZE'),
             'independent_source_count':2,'seeds':plan['seeds'],'cells_not_independent':True,
             'new_method_claim':False,'closest_prior_AFLoRA_implemented':False, 'cost_basis':'13 pre-cleanup fits replayed under resumed environment; original35 resumed fit times; raw mixed-session analysis preserved separately'}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    with (OUT/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plot(rows,metrics)
    print(json.dumps(summary,indent=2),flush=True)


def plot(rows,metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'ES2':'#d38b25','FIXED_FREEZE':'#718399','CONTRIB_FREEZE':'#007e87'}
    conditions=('FULL90','SPREAD30','RECENT30')
    fig,axes=plt.subplots(2,2,figsize=(13,8.5))
    for di,ds in enumerate(('bdg2','jena')):
        for ai,arm in enumerate(ARMS[1:]):
            for col,metric in enumerate(('regret_pct_F0','fit_savings_percent')):
                ax=axes[di,col]
                values=[[r[metric] for r in rows if r['dataset']==ds and r['condition']==c and r['arm']==arm] for c in conditions]
                x=np.arange(3)+(ai-1)*.23
                ax.bar(x,[np.mean(v) for v in values],width=.21,color=colors[arm],label=arm,alpha=.85)
                for xi,vs in zip(x,values):ax.scatter([xi-.035,xi+.035],vs,color='black',s=15,zorder=3)
                ax.set_xticks(np.arange(3),conditions);ax.axhline(0,color='black',lw=.8)
                ax.set_title(f'{ds.upper()} | '+('E regret vs FULL (lower better)' if col==0 else 'Actual fit time saved (higher better)'))
                ax.set_ylabel('% of F0 loss' if col==0 else '% of FULL fit time');ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Study31: single-trajectory freezing | bars: 2-seed mean; points: seeds',fontsize=14)
    fig.text(.5,.012,'Existing source families, different E periods. Cell bars are descriptive; gates apply to per-seed aggregates.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.96]);fig.savefig(OUT/'01_quality_cost.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for i,metric in enumerate(('macro_regret_pct_F0','fit_savings_percent')):
        for a,arm in enumerate(ARMS[1:]):
            for j,seed in enumerate((27000,27001)):
                v=next(r[metric] for r in metrics if r['arm']==arm and r['seed']==seed)
                x=a+(j-.5)*.3;axes[i].bar(x,v,width=.27,color=colors[arm],alpha=1 if j==0 else .55)
                axes[i].annotate(f'{v:.2f}',(x,v),xytext=(0,4 if v>=0 else -12),textcoords='offset points',ha='center',fontsize=9)
        axes[i].set_xticks(range(3),ARMS[1:],fontsize=9);axes[i].axhline(.25 if i==0 else 5,color='#b54141',ls='--',label='gate: <=0.25' if i==0 else 'gate: >5')
        axes[i].axhline(0,color='black',lw=.7);axes[i].grid(axis='y',alpha=.2);axes[i].legend()
        axes[i].set_title('Macro E regret (%F0)' if i==0 else 'Actual aggregate fit savings (%)')
    fig.suptitle('Per-seed decision | solid: 27000; light: 27001 | all six cells equally weighted for regret')
    fig.tight_layout();fig.savefig(OUT/'02_gate_summary.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7.8),sharex='col')
    plan=read(RUN/'plan.json')
    for di,ds in enumerate(('bdg2','jena')):
        for ci,c in enumerate(conditions):
            ax=axes[di,ci]
            for seed,color in ((27000,'#007e87'),(27001,'#b54d2b')):
                e=next(e for e in plan['jobs'] if (e['job']['dataset'],e['job']['condition'],e['job']['seed'],e['job']['arm'])==(ds,c,seed,'CONTRIB_FREEZE'))
                r=read(RUN/e['key']/'output/result.json')
                points=[h for h in r['history'] if 'contribution_halves' in h]
                for half,style in ((0,'-'),(1,'--')):
                    ax.plot([h['step'] for h in points],[h['contribution_halves'][half] for h in points],style+'o',color=color,label=f'{seed} V{half+1}',ms=4)
                if r['frozen_step'] is not None:ax.axvline(r['frozen_step'],color=color,alpha=.4,lw=2)
            ax.set_title(f'{ds.upper()} {c}');ax.set_xlabel('Joint-training updates');ax.set_ylabel('Current adapter contribution (%F0_V)');ax.axhline(0,color='black',lw=.6);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    fig.suptitle('Observed on/off contribution | vertical lines: actual freeze | no observations after freeze/cap')
    fig.tight_layout();fig.savefig(OUT/'03_observed_contribution.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
