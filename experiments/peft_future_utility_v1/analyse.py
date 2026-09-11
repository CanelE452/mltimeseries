"""Fixed descriptive estimands and raw-array verification for the continuation study."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/peft_future_utility_v1'
OUT=ROOT/'results/peft_future_utility_v1'


def read(path): return json.loads(path.read_text())


def loss(pred,target,scale,q):
    pred=np.sort(np.asarray(pred,dtype=np.float64),axis=2)
    y=np.asarray(target,dtype=np.float64);error=y[:,:,None,:]-pred
    valid=np.isfinite(y);q=np.asarray(q)[None,None,:,None]
    pinball=np.where(valid[:,:,None,:],2*np.maximum(q*error,(q-1)*error),0.)
    return float((pinball.sum(axis=(0,3))/valid.sum(axis=(0,2))[:,None]/np.asarray(scale)[:,None]).mean())


def archive_scores(path):
    with np.load(path,allow_pickle=False) as z:
        return {label:loss(z['prediction'][sl],z['target'][sl],z['scale'],z['quantiles'])
                for label,sl in (('S',slice(0,14)),('D',slice(16,30)))}


def ranks(v):
    a=np.asarray(v);r=np.empty(len(a),dtype=float)
    for x in np.unique(a):
        ids=np.flatnonzero(a==x);r[ids]=(np.sum(a<x)+.5*(len(ids)-1))
    return r


def correlation(x,y):
    x=ranks(x);y=ranks(y)
    if np.std(x)==0 or np.std(y)==0:return None
    return float(np.corrcoef(x,y)[0,1])


def main():
    assert read(RUN/'completed.json')['completed']
    assert not (OUT/'summary.json').exists()
    plan=read(RUN/'plan.json');rows=[];errors=[];checks=[];all_histories=[]
    for e in plan['jobs']:
        out=RUN/e['key']/'output';r=read(out/'result.json');j=e['job']
        assert r['completed'] and r['prefix_exact_study31'] and not r['holdout_opened']
        all_histories.append(r)
        for f in r['forks']:
            fork=f['fork'];branches=f['branches'];joint=branches['JOINT']
            prefix=[h for h in joint['history'] if h['step']<=fork]
            current,previous=prefix[-1],prefix[-2];s0=r['initial']['S'];d0=r['initial']['D']
            contribution=100*(current['S_off']-current['S'])/s0
            previous_C=100*(previous['S_off']-previous['S'])/s0
            row={**{k:j[k] for k in ('dataset','condition','seed')},'fork':fork,'cap':r['cap'],
                 'C_S_pct_F0':contribution,'C_slope_per_update':(contribution-previous_C)/(fork-previous['step']),
                 'S_improvement_per_update':100*(previous['S']-current['S'])/(s0*(fork-previous['step'])),
                 'D_F0':d0,'D_STOP':current['D']}
            for mode,b in branches.items():
                assert b['selected']==min(b['history'],key=lambda h:h['S'])
                for h in b['history']:
                    folder=out/'joint' if mode=='JOINT' or h['step']<=fork else out/f'fork_{fork}'/mode
                    scores=archive_scores(folder/f"point_{h['step']}.npz")
                    for key in ('S','D'): errors.append(abs(scores[key]-h[key]))
                row[f'{mode}_D_final']=b['final']['D'];row[f'{mode}_D_selected']=b['selected']['D']
                row[f'{mode}_selected_step']=b['selected']['step']
                row[f'{mode}_gain_vs_STOP']=100*(current['D']-b['final']['D'])/d0
                row[f'{mode}_selection_benefit']=100*(b['final']['D']-b['selected']['D'])/d0
                if mode!='JOINT':
                    assert b['restore_verified'] and b['adapter_unchanged'] and b['head_changed']
                    row[f'{mode}_clipped_steps']=b['clipped_steps']
                    with np.load(out/f'fork_{fork}'/mode/f'point_{fork}.npz') as z,np.load(out/'joint'/f'point_{fork}.npz') as orig:
                        np.testing.assert_array_equal(z['prediction'],orig['prediction'])
            row['U_final_pct_F0']=100*(row['HEAD_ONLY_D_final']-row['JOINT_D_final'])/d0
            row['U_selected_pct_F0']=100*(row['HEAD_ONLY_D_selected']-row['JOINT_D_selected'])/d0
            row['U_masked_pct_F0']=100*(row['MASKED_UPDATE_D_final']-row['JOINT_D_final'])/d0
            row['clipping_control_difference_pct_F0']=100*(row['HEAD_ONLY_D_final']-row['MASKED_UPDATE_D_final'])/d0
            rows.append(row)
    assert len(rows)==24 and max(errors)<1e-10
    band=plan['negligible_band_pct_F0']
    groups={name:[r for r in rows if name=='all' or r['dataset']==name] for name in ('all','bdg2','jena')}
    metrics={}
    for name,group in groups.items():
        meaningful=[r for r in group if abs(r['U_final_pct_F0'])>band]
        rule_accuracy={}
        if meaningful:
            rules={'always_continue':lambda r:True,'always_freeze':lambda r:False,
                   'positive_C':lambda r:r['C_S_pct_F0']>0,
                   'increasing_C':lambda r:r['C_slope_per_update']>0,
                   'improving_S':lambda r:r['S_improvement_per_update']>0}
            rule_accuracy={k:sum(fn(r)==(r['U_final_pct_F0']>band) for r in meaningful)/len(meaningful) for k,fn in rules.items()}
        metrics[name]={'forks':len(group),'joint_better_than_head':sum(r['U_final_pct_F0']>band for r in group),
           'head_better_than_joint':sum(r['U_final_pct_F0']<-band for r in group),'negligible':len(group)-len(meaningful),
           'joint_better_than_head_and_STOP':sum(r['U_final_pct_F0']>band and r['JOINT_gain_vs_STOP']>band for r in group),
           'head_better_than_joint_and_STOP':sum(r['U_final_pct_F0']<-band and r['HEAD_ONLY_gain_vs_STOP']>band for r in group),
           'both_continuations_worse_than_STOP':sum(r['JOINT_gain_vs_STOP']<-band and r['HEAD_ONLY_gain_vs_STOP']<-band for r in group),
           'selection_reverses_meaningful_direction':sum(r['U_final_pct_F0']*r['U_selected_pct_F0']<0 and min(abs(r['U_final_pct_F0']),abs(r['U_selected_pct_F0']))>band for r in group),
           'mean_U_final_pct_F0':float(np.mean([r['U_final_pct_F0'] for r in group])),
           'mean_U_selected_pct_F0':float(np.mean([r['U_selected_pct_F0'] for r in group])),
           'max_abs_clipping_control_difference_pct_F0':max(abs(r['clipping_control_difference_pct_F0']) for r in group),
           'spearman_C_vs_U':correlation([r['C_S_pct_F0'] for r in group],[r['U_final_pct_F0'] for r in group]),
           'spearman_C_slope_vs_U':correlation([r['C_slope_per_update'] for r in group],[r['U_final_pct_F0'] for r in group]),
           'meaningful_forks_for_rule_accuracy':len(meaningful),'descriptive_rule_accuracy':rule_accuracy}
    sign_pairs=[]
    for ds in ('bdg2','jena'):
        for condition in ('FULL90','SPREAD30','RECENT30'):
            for fraction in (1/3,2/3):
                pair=[r for r in rows if r['dataset']==ds and r['condition']==condition and abs(r['fork']/r['cap']-fraction)<1e-9]
                assert len(pair)==2
                classes=[1 if r['U_final_pct_F0']>band else -1 if r['U_final_pct_F0']<-band else 0 for r in pair]
                sign_pairs.append({'dataset':ds,'condition':condition,'fork_fraction':fraction,'classes':classes,'agrees':classes[0]==classes[1]})
    statuses=[read(p) for p in RUN.rglob('guard/status.json')]
    assert len(statuses)==13 and all(s['completed'] and s['returncode']==0 and not s['reasons'] for s in statuses)
    summary={'completed':True,'analysed_utc':datetime.now(timezone.utc).isoformat(),'band_pct_F0':band,
       'metrics':metrics,'seed_sign_pairs':sign_pairs,'raw_score_max_abs_error':max(errors),
       'guard_jobs':len(statuses),'diagnostic_guard_seconds':sum(s['elapsed_seconds'] for s in statuses),
       'publication_ready':False,'policy_trained':False,'evaluation_E_used':False,
       'limitation':'24 dependent forks from12 cells/two previously observed source families; descriptive development evidence only'}
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'metrics.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    (OUT/'histories.json').write_text(json.dumps(all_histories,indent=2,allow_nan=False),encoding='utf-8')
    plot(rows,band)
    print(json.dumps(summary,indent=2))


def plot(rows,band):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'bdg2':'#007e87','jena':'#b54d2b'}
    fig,axes=plt.subplots(1,2,figsize=(13,5.5))
    for ds,color in colors.items():
        for seed,marker in ((27000,'o'),(27001,'^')):
            group=[r for r in rows if r['dataset']==ds and r['seed']==seed]
            for ax,key,title in zip(axes,('C_S_pct_F0','C_slope_per_update'),('Current adapter contribution on S','Current contribution slope on S')):
                ax.scatter([r[key] for r in group],[r['U_final_pct_F0'] for r in group],color=color,marker=marker,s=55,label=f'{ds} / {seed}')
                ax.set_xlabel(title);ax.set_ylabel('Future JOINT advantage vs HEAD (%F0_D)')
    for ax in axes:
        ax.axhspan(-band,band,color='gray',alpha=.1);ax.axhline(0,color='black',lw=.7);ax.axvline(0,color='gray',lw=.6);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Current contribution vs future update utility | positive y favors further LoRA updates')
    fig.text(.5,.015,'24 dependent development forks; no unseen-E evaluation or fitted selector. Shading: descriptive +/-0.25%F0 band.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.94]);fig.savefig(OUT/'01_current_vs_future.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,5.4))
    specs=[('U_final_pct_F0','U_selected_pct_F0','Fixed final vs S-selected','Fixed-final JOINT advantage','S-selected JOINT advantage'),
           ('U_masked_pct_F0','U_final_pct_F0','Clipping control','MASKED comparison advantage','HEAD_ONLY comparison advantage'),
           ('HEAD_ONLY_gain_vs_STOP','JOINT_gain_vs_STOP','Continuing vs STOP','HEAD gain vs STOP','JOINT gain vs STOP')]
    for ax,(xk,yk,title,xl,yl) in zip(axes,specs):
        for ds,color in colors.items():
            g=[r for r in rows if r['dataset']==ds]
            ax.scatter([r[xk] for r in g],[r[yk] for r in g],color=color,s=45,label=ds)
        vals=[r[k] for r in rows for k in (xk,yk)];lim=max(abs(v) for v in vals)*1.1+.1
        ax.plot([-lim,lim],[-lim,lim],'--',color='gray',lw=.8);ax.axhline(0,color='black',lw=.6);ax.axvline(0,color='black',lw=.6)
        ax.set_xlim(-lim,lim);ax.set_ylim(-lim,lim);ax.set_xlabel(xl+' (%F0_D)');ax.set_ylabel(yl+' (%F0_D)');ax.set_title(title);ax.grid(alpha=.2)
    axes[0].legend();fig.suptitle('Separate continuation, checkpoint selection and clipping effects | descriptive diagnostic')
    fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(OUT/'02_selection_clipping_stop.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
