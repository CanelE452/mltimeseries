"""Post-outcome resource review and small reproducibility artifacts."""
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import psutil
import numpy as np
from experiments.hospital_shared_strength_v1.resume_monitored import memory

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/peft_decision_transfer_v1'
OUT=ROOT/'results/peft_decision_transfer_v1'


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def validation_plot(histories,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharey=False)
    colors={'JOINT':'#007e87','HEAD1':'#b54d2b'};styles={28002:'-',28003:'--',28004:':'}
    for i,dataset in enumerate(('bmra','jena')):
        for j,condition in enumerate(('FULL90','SPREAD30')):
            ax=axes[i,j]
            for row in rows:
                if row['method']!='PROBE' or row['dataset']!=dataset or row['condition']!=condition:continue
                fit=histories[row['fit_key']];initial=fit['histories']['JOINT'][0]['V'];seed=int(row['seed'])
                for path,color in colors.items():
                    h=fit['histories'][path]
                    ax.plot([p['step']/fit['cap'] for p in h],[p['V']/initial for p in h],
                            color=color,linestyle=styles[seed],alpha=.85,label=f'{path} / seed {seed}')
                selected_point=next(p for h in fit['histories'].values() for p in h if p['checkpoint']==row['selected_checkpoint'])
                ax.scatter(selected_point['step']/fit['cap'],selected_point['V']/initial,marker='*',s=100,color='#6b45a3',zorder=4)
            ax.axvline(1/3,color='gray',linewidth=1,alpha=.5)
            ax.set_title(f'{dataset.upper()} | {condition}');ax.set_xlabel('Fraction of maximum update budget')
            ax.set_ylabel('Validation loss / initial validation loss');ax.grid(alpha=.15)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8)
    fig.suptitle('Validation trajectories | stars: selected PROBE outputs; gray line: decision fork')
    fig.tight_layout(rect=(0,.08,1,.96));fig.savefig(OUT/'03_validation_paths.png',dpi=160);plt.close(fig)


def advantage_plot(summary):
    import matplotlib.pyplot as plt
    methods=[m for m in summary['methods'] if m!='PROBE']
    intervals=summary['conditional_quality_intervals']['PROBE_advantage_pct_F0']
    fig,ax=plt.subplots(figsize=(11,6));ys=np.arange(len(methods))
    for y,method in zip(ys,methods):
        stats=summary['methods'][method];low,high=intervals[method]
        ax.hlines(y,low,high,color='#526777',linewidth=2)
        ax.scatter(stats['PROBE_advantage_pct_F0'],y,color='#162e42',s=40,zorder=3)
        for offset,dataset,color in ((-.13,'bmra','#007e87'),(.13,'jena','#b54d2b')):
            ax.scatter(stats['per_source_PROBE_advantage'][dataset],y+offset,marker='x',color=color,s=35,
                       label=f'{dataset.upper()} mean' if y==0 else None)
    ax.axvline(0,color='black',linewidth=.8)
    ax.axvspan(-.25,.25,color='#8295a5',alpha=.12,label='Prespecified practical band')
    ax.set_yticks(ys,methods);ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
    ax.set_xlabel('PROBE quality advantage over comparator (% of initial E loss; positive favors PROBE)')
    ax.set_title('Selected-checkpoint differences | dots: overall mean; lines: conditional 90% block intervals')
    ax.legend(loc='best',fontsize=8)
    fig.text(.5,.015,'Intervals condition on two sources and their windows; three seeds share the same targets.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,1));fig.savefig(OUT/'04_quality_advantages.png',dpi=160);plt.close(fig)


def main():
    assert read(RUN/'completed.json')['completed'] and read(OUT/'summary.json')['completed']
    dest=OUT/'completion_review.json';assert not dest.exists()
    statuses=list(RUN.rglob('guard/status.json'));measurements=[];finishes=0;categories={}
    for path in statuses:
        s=read(path);assert s['completed'] and s['returncode']==0 and not s['reasons'],str(path)
        records=[json.loads(line) for line in path.with_name('resource_log.jsonl').read_text().splitlines()]
        terminal=[r for r in records if 'gpus' not in r]
        assert len(terminal)==1 and terminal[0]['event']=='finish' and terminal[0]['state']=='completed'
        finishes+=1;measurements.extend(r for r in records if 'gpus' in r)
        parts=path.relative_to(RUN).parts;key='/'.join(parts[:2]) if parts[0] in ('fit','forecast') else parts[0]
        group=categories.setdefault(key,{'jobs':0,'seconds':0.})
        group['jobs']+=1;group['seconds']+=s['elapsed_seconds']
    counts={episode:len(read(RUN/f'{episode}_fits_sealed.json')['indices']) for episode in ('dev','test')}
    assert len(statuses)==1+2*sum(counts.values())+72
    active=[];inaccessible=[]
    modules={'experiments.peft_decision_transfer_v1.'+name for name in ('fit','run','replay')}
    for process in psutil.process_iter(['name','pid']):
        if 'python' not in (process.info['name'] or '').lower():continue
        try:
            if modules.intersection(process.cmdline()):active.append(process.pid)
        except psutil.NoSuchProcess:pass
        except psutil.AccessDenied:inaccessible.append(process.pid)
    lock=ROOT/'runs/peft_adaptation_scope_v1/.guard.lock'
    assert not active and not inaccessible and not lock.exists()
    protected={}
    for study in ('peft_decision_transfer_v1','peft_future_utility_v1','peft_contribution_freeze_v1','peft_capacity_probe_v1'):
        p=read(ROOT/'runs'/study/'plan.json');hashes=dict(p['source_hashes'])
        if study in ('peft_decision_transfer_v1','peft_future_utility_v1'):hashes.update(p['input_hashes'])
        for path,h in hashes.items():assert sha(ROOT/path)==h,path
        protected[study]=len(hashes)
    rows=list(csv.DictReader((OUT/'metrics.csv').open(encoding='utf-8')));nums=[];dens=[];scales=[];decision_outputs=[]
    for row in rows:
        if row['method']=='PROBE':
            fit=read(RUN/'fit'/row['fit_key']/'output/result.json');histories=fit['histories']
            candidates={'JOINT':histories['JOINT'],'HEAD':histories['HEAD1'],
                        'STOP':[p for p in histories['JOINT'] if p['step']<=fit['fork']]}
            points={name:min(history,key=lambda p:p['V']) for name,history in candidates.items()}
            decision_outputs.append({'fit_key':row['fit_key'],'chosen_action':row['action'],
                 'cap':fit['cap'],'fork':fit['fork'],'selected_budget_fraction':int(row['selected_step'])/fit['cap'],
                 'selected_after_fork':int(row['selected_step'])>fit['fork'],
                 'all_three_selected_model_hashes_equal':len({p['model_hash'] for p in points.values()})==1,
                 'selected_points':points})
        path=RUN/'forecast'/row['fit_key']/'output'/(row['selected_checkpoint']+'.npz')
        with np.load(path,allow_pickle=False) as z:
            p=np.sort(z['prediction'].astype(float),axis=2);y=z['target'].astype(float)
            err=y[:,:,None,:]-p;q=z['quantiles'][None,None,:,None];valid=np.isfinite(y)
            num=np.where(valid[:,:,None,:],2*np.maximum(q*err,(q-1)*err),0.).sum(axis=3)
            den=valid.sum(axis=2);score=(num.sum(axis=0)/den.sum(axis=0)[:,None]/z['scale'][:,None]).mean()
            assert abs(score-float(row['E_selected']))<1e-10
            nums.append(num);dens.append(den);scales.append(z['scale'])
    loss_path=OUT/'per_example_losses.npz';assert not loss_path.exists()
    np.savez_compressed(loss_path,pinball_numerator=np.stack(nums),valid_target_count=np.stack(dens),scale=np.stack(scales),
       F0=np.asarray([float(r['F0']) for r in rows]),row_keys=np.asarray(['/'.join(r[k] for k in ('dataset','condition','seed','method')) for r in rows]),
       metadata_json=np.asarray(json.dumps({'meaning':'per-origin numerator and count to reproduce missingness-weighted mean loss and paired resampling; no raw targets or predictions','shape':'row,origin,target_channel,quantile','loss':'mean_cq(sum_n numerator / sum_n count / scale_c)'})))
    gpu=[g for r in measurements for g in r['gpus']]
    result={'completed':True,'created_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':sha(Path(__file__)),
       'guards':len(statuses),'categories':categories,'resource_samples':len(measurements),'finish_records':finishes,
       'minimum_available_ram_gib':min(r['available_ram_gib'] for r in measurements),
       'minimum_available_commit_gib':min(r['available_commit_gib'] for r in measurements),
       'maximum_child_rss_gib':max(r['child_tree_rss_gib'] for r in measurements),
       'maximum_gpu_memory_mib':max(g['memory_used_mib'] for g in gpu),'maximum_gpu_temperature_c':max(g['temperature_c'] for g in gpu),
       'current_memory':memory(),'active_study_processes':active,'inaccessible_python':inaccessible,'guard_lock_exists':lock.exists(),
       'protected_hash_checks':protected,'reproducible_loss_rows':len(rows),'loss_archive_sha256':sha(loss_path),
       'prepared_failure_preserved':(RUN/'prepared/attempt01_failed').exists(),
       'decision_output_comparisons':decision_outputs,
       'same_three_action_model_count':sum(r['all_three_selected_model_hashes_equal'] for r in decision_outputs),
       'scope':'post-outcome descriptive/resource audit only; no policy or thresholds changed'}
    with dest.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    evidence=OUT/'evidence';evidence.mkdir(exist_ok=True)
    for name in ('plan.json','development_choice.json','dev_fits_sealed.json','test_fits_sealed.json',
                 'completed.json','timing_completed.json','environment.json','cpu_checks.json',
                 'preflight_review.json','independent_audit.json','independent_policy_review.json','prepared/data_audit.json',
                 'prepared/summary.json','prepared/attempt01_failed/preparation_failures.json'):
        source=RUN/name;target=evidence/name.replace('/','_')
        assert source.exists() and not target.exists(),str(source)
        with target.open('xb') as f:f.write(source.read_bytes())
        assert sha(source)==sha(target)
    compact={}
    for episode in ('dev','test'):
        seal=read(RUN/f'{episode}_fits_sealed.json')
        compact[episode]={key:read(RUN/'fit'/key/'output/result.json') for key in seal['fit_result_hashes']}
    with (evidence/'validation_histories.json').open('x',encoding='utf-8') as f:json.dump(compact,f,indent=2)
    with (evidence/'replay_records.json').open('x',encoding='utf-8') as f:
        json.dump({key:read(RUN/key/'output/result.json') for key in read(RUN/'timing_completed.json')['keys']},f,indent=2)
    with (evidence/'guard_statuses.json').open('x',encoding='utf-8') as f:
        json.dump({p.relative_to(RUN).as_posix():read(p) for p in statuses},f,indent=2)
    validation_plot(compact['test'],rows)
    advantage_plot(read(OUT/'summary.json'))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
