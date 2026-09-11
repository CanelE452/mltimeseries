"""Publish verified Hospital summaries and figures from an already completed run."""
from pathlib import Path
from datetime import datetime
import csv
import hashlib
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/hospital_shared_strength_v1_run2'
OUT = ROOT/'results/hospital_shared_strength_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    done=read(RUN/'completed.json');audit=read(RUN/'independent_audit.json')
    assert done['completed'] and audit['passed'] and audit['score_rows_recomputed']==48
    stages={k:Path(v) for k,v in done['stages'].items()}
    summary=read(stages['analyse']/'summary.json')
    selected=read(stages['select_models']/'selected.json')
    policy=read(stages['select_gates']/'policies.json')
    assert summary['completed'] and len(summary['metrics'])==48
    assert np.isclose(summary['main_effect_pctF0'],audit['main_effect_pctF0'],atol=1e-11)
    receipts=0
    for receipt in RUN.glob('stages/*/attempt_*/receipt.json'):
        for path,digest in read(receipt)['files'].items():assert sha(receipt.parent/path)==digest
        receipts+=1
    assert receipts==10
    contract=read(RUN/'run_contract.json')
    for path,digest in contract['protected_files'].items():assert sha(path)==digest
    OUT.mkdir(exist_ok=True);figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    copies={stages['analyse']/'summary.json':'summary.json',stages['analyse']/'metrics.csv':'metrics.csv',
            stages['analyse']/'series_effects.csv':'series_effects.csv',
            RUN/'independent_audit.json':'independent_audit.json',
            stages['select_models']/'selected.json':'selected.json',
            stages['select_gates']/'policies.json':'policies.json'}
    for src,name in copies.items():
        dest=OUT/name
        if dest.exists():assert sha(dest)==sha(src)
        else:shutil.copyfile(src,dest)
    with (OUT/'series_effects.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==3068
    seeds=sorted({int(r['seed']) for r in rows})
    expected=summary['main_effect_pctF0']
    denominator=np.mean([float(r['F0_loss']) for r in rows])
    recomputed=100*np.mean([float(r['GLOBAL_loss'])-float(r['INDIVIDUAL_loss']) for r in rows])/denominator
    assert np.isclose(recomputed,expected,atol=1e-11)
    plt.rcParams.update({'font.family':'Malgun Gothic','font.size':10,'axes.unicode_minus':False,
                         'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    def save(name):
        for ext in ('png','svg'):plt.savefig(figdir/f'{name}.{ext}',dpi=190,bbox_inches='tight',facecolor='white')
        plt.close()
    colors=['#737373','#0072B2','#009E73','#D55E00','#CC79A7']
    methods=['F0','LORA','GLOBAL','INDIVIDUAL','SHUFFLED']
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    for i,name in enumerate(methods):
        vals=[r['score'] for r in summary['metrics'] if r['method']==name or (name=='SHUFFLED' and r['method'].startswith('SHUFFLED_'))]
        v=np.mean(vals);ax.bar(i,v,color=colors[i],alpha=.78,width=.62)
        ax.scatter(i+np.linspace(-.13,.13,len(vals)),vals,s=16,color='black',zorder=3)
        ax.annotate(f'{v:.6f}',(i,max(vals)),xytext=(0,8),textcoords='offset points',ha='center')
    ax.set(xticks=range(5),xticklabels=methods,ylabel='평균 scaled 2-pinball loss (낮을수록 좋음)',
           title='Hospital | 고정된 V1/V2 선택 이후 E1/E2 결과\n학습군 점=2 seed · SHUFFLED 점=20순열 × 2 seed · 시간 CI가 아님')
    ax.set_ylim(0,max(r['score'] for r in summary['metrics'] if r['method']!='SEASONAL_RESIDUAL_REFERENCE')*1.18)
    save('01_scores')

    effects=[]
    for seed in seeds:
        for origin in (60,72):
            subset=[r for r in rows if int(r['seed'])==seed and int(r['origin'])==origin]
            base=np.mean([float(r['F0_loss']) for r in subset])
            gain=100*np.mean([float(r['GLOBAL_loss'])-float(r['INDIVIDUAL_loss']) for r in subset])/base
            match=100*np.mean([float(r['mean_SHUFFLED_loss'])-float(r['INDIVIDUAL_loss']) for r in subset])/base
            effects.append({'seed':seed,'year':2000+origin//12,'individual_over_global_pctF0':gain,'individual_over_shuffled_pctF0':match})
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    y=np.arange(4)
    ax.scatter([r['individual_over_global_pctF0'] for r in effects],y-.09,label='GLOBAL 대비',color=colors[1],s=60)
    ax.scatter([r['individual_over_shuffled_pctF0'] for r in effects],y+.09,label='SHUFFLED 평균 대비',color=colors[3],marker='s',s=60)
    ax.axvline(0,color=colors[0],ls='--')
    ax.set(yticks=y,yticklabels=[f"seed {r['seed']} / {r['year']}" for r in effects],
           xlabel='INDIVIDUAL의 추가 이득 (% F0, 오른쪽이 개선)',
           title='Hospital | seed와 연도별 방향 확인\n767개 관련 계열 · 2개 평가 연도 · 기술적 점추정, 독립 도메인 확증 아님')
    ax.legend();ax.invert_yaxis();save('02_seed_year_effects')

    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax,p in zip(axes,policy['policies']):
        grid=np.array(list(p['alpha_histogram']),dtype=float)
        count=list(p['alpha_histogram'].values())
        ax.bar(grid,count,width=.07,color=colors[1],alpha=.8)
        ax.axvline(p['global_alpha'],color=colors[3],ls='--',label=f"GLOBAL={p['global_alpha']:.1f}")
        ax.set(xticks=grid,xlabel='V2에서 선택한 적용 강도 alpha',ylabel='계열 수',title=f"seed {p['seed']}")
        ax.legend()
    fig.suptitle('Hospital | 12개월 V2로 선택한 계열별 강도\n계수 분포의 다양성이 미래 예측 이득을 보장하지 않음')
    save('03_alpha_histogram')

    attempts=[]
    for run in [ROOT/'runs/hospital_shared_strength_v1_run',RUN]:
        for path in sorted(run.glob('stages/*/attempt_*/guard/status.json')):
            d=read(path)
            samples=[json.loads(line) for line in (path.parent/'resource_log.jsonl').read_text().splitlines() if line.strip()]
            samples=[x for x in samples if 'available_commit_gib' in x]
            attempts.append({'path':path.relative_to(ROOT).as_posix(),'state':d.get('state'),
                'started_at':d.get('started_at'),'elapsed_seconds':d.get('elapsed_seconds'),
                'min_sampled_commit_gib':min((x['available_commit_gib'] for x in samples),default=None),
                'max_sampled_gpu_mib':max((g['memory_used_mib'] for x in samples for g in (x.get('gpus') or [])),default=None)})
    monitor=sorted((ROOT/'runs').glob('hospital_shared_strength_v1_resume_monitor_*/completed.json'))[-1].parent
    trace=[json.loads(x) for x in (monitor/'private_memory.jsonl').read_text().splitlines() if x.strip()]
    owned=[p for x in trace for p in x.get('owned_tree',[]) if p['name']=='python.exe']
    costs={'attempts':attempts,'scope':'Original run S0 attempts and all run2 stages; excludes data export/preparation.',
           'all_listed_guard_seconds':sum(x['elapsed_seconds'] or 0 for x in attempts),
           'new_invocation_seconds':done['invocation_seconds'],'invocation_excludes_initial_admission_wait':True,
           'monitor':monitor.relative_to(ROOT).as_posix(),
           'max_observed_python_private_gib':max(p['private_gib'] for p in owned),
           'fits':{k:read(v/'fit/result.json') for k,v in stages.items() if k.startswith('fit_')},
           'forecasts':{k:read(stages[k]/'forecast_info.json') for k in ('forecast_gate','forecast_eval')}}
    # Keep only fit metadata needed to interpret results; large tensor-name audits stay in the run.
    keys=['seed','learning_rate','best_step','val_score','steps_completed','checkpoint_replay','seconds',
          'unique_sampled_series','unseen_train_series_this_fit','nominal_window_exposure_ratio','history']
    costs['fits']={k:{key:value[key] for key in keys} for k,value in costs['fits'].items()}
    (OUT/'execution_summary.json').write_text(json.dumps(costs,indent=2),encoding='utf-8')
    (OUT/'seed_year_effects.json').write_text(json.dumps(effects,indent=2),encoding='utf-8')
    verification={'completed':True,'verified_receipts':receipts,'verified_rows':len(rows),
        'main_effect_csv_recomputed':recomputed,'difference':abs(recomputed-expected),
        'sources_sha256':{src.relative_to(ROOT).as_posix():sha(src) for src in copies},
        'run_contract_sha256':sha(RUN/'run_contract.json'),'new_training_by_this_script':0,
        'figures':[p.name for p in sorted(figdir.glob('*.png'))]}
    (OUT/'publication_verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    print(json.dumps({'main_effect_pctF0':expected,'matching_effect_pctF0':summary['matching_effect_pctF0'],
        'flags':summary['flags'],'descriptive_interval':summary['descriptive_series_interval'],
        'per_seed':summary['per_seed'],'seed_year':effects,'selected':selected},ensure_ascii=False))


if __name__=='__main__':main()
