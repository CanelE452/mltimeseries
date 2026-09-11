"""Development-only tuning followed by fixed out-of-period tests."""
import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
from experiments.peft_decision_transfer_v1.policy import paths,selected,ACTIONS

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/peft_decision_transfer_v1'
OUT=ROOT/'results/peft_decision_transfer_v1'


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def records(episode):
    plan=read(RUN/'plan.json');seal=read(RUN/f'{episode}_fits_sealed.json')
    result=[]
    for i in seal['indices']:
        e=plan['jobs'][i];fit=read(RUN/'fit'/e['key']/'output/result.json')
        forecast=read(RUN/'forecast'/e['key']/'output/result.json')
        assert fit['completed'] and forecast['completed'] and fit['job']==forecast['job']
        result.append((e,fit,forecast))
    return result


def option(fit,spec):
    if spec['family']=='WIDE':
        h=fit['histories']['JOINT'];return {'action':'WIDE','selected':selected(h)[0],'final':h[-1]}
    return paths(fit,spec.get('margin',0.),spec.get('constant','STOP'))[spec['path']]


def dev_value(items,spec,dataset=None):
    vals=[]
    for e,fit,pred in items:
        j=e['job']
        if j['family']!=spec['family'] or j['recipe']!=spec['recipe'] or (dataset and j['dataset']!=dataset):continue
        pick=option(fit,spec)['selected']['checkpoint']
        vals.append(pred['scores'][pick]/pred['scores']['JOINT_0'])
    assert len(vals)==(2 if dataset else 4),len(vals)
    return float(np.mean(vals))


def choose_development():
    destination=RUN/'development_choice.json';assert not destination.exists()
    items=records('dev');grids={};methods={}
    candidates={'FULL':['JOINT'],'HEAD':['HEAD0'],'FIXED':['HEAD1','HEAD2'],
                'EARLY_STOP':['ES1','ES2','ES3'],'PROBE':['PROBE']}
    for name,path_options in candidates.items():
        specs=[{'family':'TREE','recipe':recipe,'path':path,'margin':margin}
               for recipe in (0,1,2) for path in path_options for margin in ([0.,.25] if name=='PROBE' else [0.])]
        for spec in specs:spec['development_normalized_E']=dev_value(items,spec)
        grids[name]=specs
        methods[name]=min(specs,key=lambda s:(s['development_normalized_E'],s['recipe'],s['path'],s['margin']))
    specs=[{'family':'WIDE','recipe':recipe,'path':'WIDE'} for recipe in (0,2)]
    for spec in specs:spec['development_normalized_E']=dev_value(items,spec)
    grids['WIDE']=specs;methods['WIDE']=min(specs,key=lambda s:(s['development_normalized_E'],s['recipe']))
    chosen=methods['PROBE'];recipe=chosen['recipe'];margin=chosen['margin']
    for name in ('CURRENT_C','RANDOM','HEAD_PROBE','MASKED1'):
        methods[name]={'family':'TREE','recipe':recipe,'path':name,'margin':margin}
    specs=[{'family':'TREE','recipe':recipe,'path':'CONSTANT','constant':a,'margin':margin} for a in ACTIONS]
    for spec in specs:spec['development_normalized_E']=dev_value(items,spec)
    grids['CONSTANT']=specs;methods['CONSTANT']=min(specs,key=lambda s:(s['development_normalized_E'],ACTIONS.index(s['constant'])))
    by_source={}
    for dataset in sorted({e['job']['dataset'] for e,_,_ in items}):
        by_source[dataset]=min(ACTIONS,key=lambda a:(dev_value(items,{'family':'TREE','recipe':recipe,'path':'CONSTANT','constant':a},dataset),ACTIONS.index(a)))
    methods['SOURCE_CONSTANT']={'family':'TREE','recipe':recipe,'path':'CONSTANT','source_actions':by_source,'margin':margin}
    payload={'completed':True,'sealed_utc':datetime.now(timezone.utc).isoformat(),'methods':methods,
             'development_grids':grids,'development_cells':4,'test_E_used':False,
             'tie_rule':'exact normalized mean ties: recipe order, path lexical, margin order; constant STOP/HEAD/JOINT'}
    with destination.open('x',encoding='utf-8') as f:json.dump(payload,f,indent=2)
    print(json.dumps({'development_sealed':True,'methods':methods}),flush=True)


def raw_loss(path):
    with np.load(path,allow_pickle=False) as z:
        p=np.sort(z['prediction'].astype(float),axis=2);y=z['target'].astype(float)
        error=y[:,:,None,:]-p;q=z['quantiles'][None,None,:,None]
        valid=np.isfinite(y);loss=np.where(valid[:,:,None,:],2*np.maximum(q*error,(q-1)*error),0.)
        return float((loss.sum(axis=(0,3))/valid.sum(axis=(0,2))[:,None]/z['scale'][:,None]).mean())


def paired_intervals(rows):
    rng=np.random.default_rng(28333);draws={};cache={}
    for ds in sorted({r['dataset'] for r in rows}):
        starts=rng.integers(0,20,size=(2000,10))
        indices=np.stack([starts,(starts+1)%20],axis=-1).reshape(2000,20)
        draws[ds]=np.stack([np.bincount(i,minlength=20) for i in indices])
    def boot(row):
        key=(row['fit_key'],row['selected_checkpoint'])
        if key not in cache:
            path=RUN/'forecast'/key[0]/'output'/(key[1]+'.npz')
            with np.load(path,allow_pickle=False) as z:
                p=np.sort(z['prediction'].astype(float),axis=2);y=z['target'].astype(float)
                assert len(y)==20
                error=y[:,:,None,:]-p;q=z['quantiles'][None,None,:,None];valid=np.isfinite(y)
                num=np.where(valid[:,:,None,:],2*np.maximum(q*error,(q-1)*error),0.).sum(axis=3)
                den=valid.sum(axis=2);weights=draws[row['dataset']]
                counts=weights@den;assert np.all(counts>0)
                scores=(np.einsum('bn,ncq->bcq',weights,num)/counts[:,:,None]/z['scale'][None,:,None]).mean(axis=(1,2))
                cache[key]=scores
        return cache[key]
    probes=[r for r in rows if r['method']=='PROBE'];result={}
    for method in sorted({r['method'] for r in rows}):
        differences=[]
        for row in [r for r in rows if r['method']==method]:
            probe=next(p for p in probes if all(p[k]==row[k] for k in ('dataset','condition','seed')))
            differences.append(100*(boot(row)-boot(probe))/row['F0'])
        values=np.mean(differences,axis=0)
        result[method]=np.quantile(values,[.05,.95]).tolist()
    return {'interval':.9,'draws':2000,'block_origins':2,'circular':True,
            'PROBE_advantage_pct_F0':result,'scope':'Conditional on these two sources/windows; shared seed/condition/method draws; not population source inference.'}


def plot(rows,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods=list(summary['methods']);means=[summary['methods'][m]['mean_normalized_E'] for m in methods]
    fig,ax=plt.subplots(figsize=(11,5.5));ax.barh(methods,means,color=['#007e87' if m=='PROBE' else '#8295a5' for m in methods])
    ax.invert_yaxis();ax.set_xlabel('Selected-checkpoint E loss / initial E loss (lower is better)')
    ax.set_title('Prospective period transfer | 12 dependent cells, 2 sources, 3 seeds')
    fig.tight_layout();fig.savefig(OUT/'01_selected_quality.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ds,color in zip(sorted({r['dataset'] for r in rows}),('#007e87','#b54d2b')):
        s=[r for r in summary['signals'] if r['dataset']==ds]
        axes[0].scatter([r['trial_U_V_pct_F0'] for r in s],[r['final_U_E_pct_F0'] for r in s],label=ds,color=color)
    axes[0].axhline(0,color='gray');axes[0].axvline(0,color='gray');axes[0].legend()
    axes[0].set_xlabel('Short paired trial advantage on V (%F0_V)');axes[0].set_ylabel('Final JOINT advantage on E (%F0_E)')
    axes[0].set_title('Can the short response predict future updates?')
    for method,metrics in summary['timing']['methods'].items():
        axes[1].scatter(metrics['mean_adaptation_seconds'],summary['methods'][method]['mean_normalized_E'],s=60)
        axes[1].annotate(method,(metrics['mean_adaptation_seconds'],summary['methods'][method]['mean_normalized_E']),xytext=(4,4),textcoords='offset points',fontsize=8)
    axes[1].set_xlabel('Actual adaptation seconds, including probes');axes[1].set_ylabel('Selected E / F0 (lower is better)')
    axes[1].set_title('Actual quality and adaptation time')
    fig.tight_layout();fig.savefig(OUT/'02_signal_and_cost.png',dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--choose',action='store_true');a=p.parse_args()
    if a.choose:choose_development();return
    assert read(RUN/'completed.json')['completed'];assert not (OUT/'summary.json').exists()
    plan=read(RUN/'plan.json');choice=read(RUN/'development_choice.json');items=records('test')
    by_key={(e['job']['dataset'],e['job']['condition'],e['job']['seed'],e['job']['family'],e['job']['recipe']):(e,f,p) for e,f,p in items}
    rows=[];errors=[];signals=[];band=plan['descriptive_band_pct_F0']
    for dataset in sorted(plan['data']['test']):
        for condition in ('FULL90','SPREAD30'):
            for seed in (28002,28003,28004):
                for method,raw_spec in choice['methods'].items():
                    spec=dict(raw_spec)
                    if 'source_actions' in spec:spec['constant']=spec['source_actions'][dataset]
                    e,fit,forecast=by_key[dataset,condition,seed,spec['family'],spec['recipe']]
                    opt=option(fit,spec);scores=forecast['scores'];f0=scores['JOINT_0']
                    vals={k:scores[opt[k]['checkpoint']] for k in ('selected','final')}
                    for point in (opt['selected'],opt['final']):
                        actual=raw_loss(RUN/'forecast'/e['key']/'output'/(point['checkpoint']+'.npz'))
                        errors.append(abs(actual-scores[point['checkpoint']]))
                    rows.append({'dataset':dataset,'condition':condition,'seed':seed,'method':method,
                        'recipe':spec['recipe'],'action':opt['action'],'selected_step':opt['selected']['step'],
                        'E_selected':vals['selected'],'E_final':vals['final'],'F0':f0,
                        'normalized_E':vals['selected']/f0,'fit_key':e['key'],
                        'selected_checkpoint':opt['selected']['checkpoint']})
                    if method=='PROBE':
                        options=paths(fit,spec['margin']);branch={'STOP':'STOP','HEAD':'HEAD1','JOINT':'JOINT'}
                        losses={k:scores[options[v]['selected']['checkpoint']] for k,v in branch.items()}
                        best=min(losses.values());acceptable=[k for k in ACTIONS if 100*(losses[k]-best)/f0<=band]
                        fork=fit['fork'];qstep=fork+fit['probe_steps']
                        at=lambda mode,step:next(h for h in fit['histories'][mode] if h['step']==step)
                        trialU=100*(at('HEAD1',qstep)['V']-at('JOINT',qstep)['V'])/fit['histories']['JOINT'][0]['V']
                        finalU=100*(scores[options['HEAD1']['final']['checkpoint']]-scores[options['JOINT']['final']['checkpoint']])/f0
                        signals.append({'dataset':dataset,'condition':condition,'seed':seed,'action':opt['action'],
                             'acceptable_actions':acceptable,'correct_within_band':opt['action'] in acceptable,
                             'selected_oracle_regret_pct_F0':100*(losses[opt['action']]-best)/f0,
                             'trial_U_V_pct_F0':trialU,'final_U_E_pct_F0':finalU,
                             'C_V_pct_F0':fit['current_C_pct_F0'],'oracle_E':best,
                             'oracle_action_losses':losses})
    assert len(signals)==12 and max(errors)<1e-10
    grouped={}
    baseline=[r for r in rows if r['method']=='PROBE']
    for method in choice['methods']:
        current=[r for r in rows if r['method']==method];assert len(current)==12
        pairs=[(r,next(b for b in baseline if all(r[k]==b[k] for k in ('dataset','condition','seed')))) for r in current]
        grouped[method]={'mean_normalized_E':float(np.mean([r['normalized_E'] for r in current])),
           'PROBE_advantage_pct_F0':float(np.mean([100*(r['E_selected']-b['E_selected'])/r['F0'] for r,b in pairs])),
           'per_seed_PROBE_advantage':{str(s):float(np.mean([100*(r['E_selected']-b['E_selected'])/r['F0'] for r,b in pairs if r['seed']==s])) for s in (28002,28003,28004)},
           'per_source_PROBE_advantage':{d:float(np.mean([100*(r['E_selected']-b['E_selected'])/r['F0'] for r,b in pairs if r['dataset']==d])) for d in plan['data']['test']}}
    distinct=len({s['action'] for s in signals})
    predictive=distinct>=2 and all(grouped[m]['PROBE_advantage_pct_F0']>band for m in ('CONSTANT','SOURCE_CONSTANT','RANDOM'))
    ablation=all(grouped[m]['PROBE_advantage_pct_F0']>band for m in ('CURRENT_C','HEAD_PROBE'))
    quality_vs_strong={m:grouped[m]['PROBE_advantage_pct_F0'] for m in ('FULL','HEAD','WIDE','FIXED','EARLY_STOP')}
    timing_records=[]
    timing_seal=read(RUN/'timing_completed.json');assert timing_seal['replays']==72
    for key in timing_seal['keys']:
        r=read(RUN/key/'output/result.json');assert r['completed'] and r['reference_exact']
        timing_records.append(r)
    times={}
    for method in ('PROBE','FULL','FIXED','EARLY_STOP','HEAD','WIDE'):
        records_m=[r for r in timing_records if r['method']==method];assert len(records_m)==12
        times[method]={'mean_adaptation_seconds':float(np.mean([r['adaptation_seconds'] for r in records_m])),
             'per_seed_seconds':{str(s):sum(r['adaptation_seconds'] for r in records_m if r['job']['seed']==s) for s in (28002,28003,28004)}}
    efficiency={}
    for method in ('FULL','FIXED','EARLY_STOP','HEAD','WIDE'):
        savings={str(s):100*(1-times['PROBE']['per_seed_seconds'][str(s)]/times[method]['per_seed_seconds'][str(s)]) for s in (28002,28003,28004)}
        gains=grouped[method]['per_seed_PROBE_advantage']
        seed_ok={str(s):((gains[str(s)]>=-band and savings[str(s)]>5) or (gains[str(s)]>band and savings[str(s)]>=0)) for s in (28002,28003,28004)}
        source_ok=all(v>=-band for v in grouped[method]['per_source_PROBE_advantage'].values())
        efficiency[method]={'saving_pct_by_seed':savings,'seed_passes':seed_ok,'source_quality_guard':source_ok,'passes':all(seed_ok.values()) and source_ok}
    best_simple=min(('FULL','HEAD','WIDE'),key=lambda m:choice['methods'][m]['development_normalized_E'])
    practical=all(efficiency[m]['passes'] for m in {'FIXED','EARLY_STOP',best_simple})
    controls=[]
    for e,fit,pred in items:
        j=e['job']
        if j['family']!='TREE' or j['seed']!=28002:continue
        opts=paths(fit,choice['methods']['PROBE']['margin']);scores=pred['scores'];f0=scores['JOINT_0']
        controls.append({k:j[k] for k in ('dataset','condition','seed','recipe')}|{
            'selected_U_pct_F0':100*(scores[opts['HEAD1']['selected']['checkpoint']]-scores[opts['JOINT']['selected']['checkpoint']])/f0,
            'final_U_pct_F0':100*(scores[opts['HEAD1']['final']['checkpoint']]-scores[opts['JOINT']['final']['checkpoint']])/f0,
            'clipping_difference_pct_F0':100*(scores[opts['HEAD1']['final']['checkpoint']]-scores[opts['MASKED1']['final']['checkpoint']])/f0,
            'PROBE_action':opts['PROBE']['action'],'PROBE_selected_E':scores[opts['PROBE']['selected']['checkpoint']]})
    assert len(controls)==12
    summary={'completed':True,'analysed_utc':datetime.now(timezone.utc).isoformat(),'test_cells':12,
        'raw_score_max_abs_error':max(errors),'methods':grouped,
        'gates':{'predictive_conditional_screen':predictive,'signal_ablation_screen':ablation,
                 'quality_advantage_vs_strong':quality_vs_strong,'practical_quality_cost_screen':practical,
                 'best_development_simple_baseline':best_simple},
        'timing':{'replays':72,'methods':times,'comparisons':efficiency,'single_timing_per_cell':True},
        'conditional_quality_intervals':paired_intervals(rows),'LR_clipping_controls':controls,
        'signals':signals,'distinct_actions':distinct,'publication_ready':False,
        'limitations':['one development seed/four development cells','three test seeds share the same labels',
                       'two previously known source families, new checked periods; pretraining overlap unknown',
                       'nearest adaptive-freezing prior and second FM required if simple-baseline gate survives']}
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'metrics.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (OUT/'development_choice.json').write_bytes((RUN/'development_choice.json').read_bytes())
    plot(rows,summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='signals'},indent=2))


if __name__=='__main__':main()
