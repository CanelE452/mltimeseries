"""V-only selection followed by development-only budget and component contrasts."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.peft_initial_headroom_v1.analyse import choose_correction, apply_correction, score_arrays, score_npz, pinball_components

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/peft_head_convergence_v1'
OUT=ROOT/'results/peft_head_convergence_v1'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f: json.dump(payload,f,indent=2,allow_nan=False)


def choose(rows): return min(rows,key=lambda r:(r['V'],r['step'],r['recipe']))


def seal(phase):
    destination=RUN/f'selection_{phase}.json'
    if destination.exists(): return read(destination)
    plan=read(RUN/'plan.json'); cells={}; hashes={}; candidates={}; paired={}
    for index,e in enumerate(plan['jobs']):
        j=e['job']
        if j['phase']!=phase: continue
        folder=RUN/'fit'/e['key']/'output'; r=read(folder/'result.json')
        assert r['completed'] and r['job']==j and r['plan_sha256']==sha(RUN/'plan.json')
        assert r['frozen_verified'] and r['restore_exact'] and not r['D_opened']
        assert not r['smoke'] and r['steps']==720
        assert [h['step'] for h in r['history']]==plan['schedule']
        assert set(r['budgets'])==({'S180','L720'} if phase=='A' else {'L720'})
        hashes[e['key']]=sha(folder/'result.json'); cell=f"{j['dataset']}/s{j['seed']}"
        cells.setdefault(cell,{'dataset':j['dataset'],'seed':j['seed']})
        with np.load(folder/'initial_val.npz') as z: initial=z['prediction'].copy()
        paired.setdefault(cell,[]).append((j['family'],r['initial_head_hash'],r['sample_sha256'],initial,r['dependency']))
        for budget,b in r['budgets'].items():
            cap=180 if budget=='S180' else 720
            expected=min((p for p in r['history'] if p['step']<=cap),key=lambda p:(p['V'],p['step']))
            assert (b['V'],b['step'])==(expected['V'],expected['step'])
            assert sha(folder/f'{budget}.pt')==b['checkpoint_sha256']
            assert abs(score_npz(folder/f'{budget}_val.npz')-b['V'])<1e-12
            candidates.setdefault((cell,budget,j['family']),[]).append(b|{'index':index,'key':e['key'],
                 'recipe':j['recipe'],'trainable':r['trainable'],'trajectory_seconds':r['trajectory_seconds']})
    forecasts=[]
    for (cell,budget,family),options in candidates.items():
        assert len(options)==4
        selected=choose(options); cells[cell].setdefault(budget,{})[family]=selected
        forecasts.append([selected['index'],budget])
    for cell,items in paired.items():
        assert len(items)==12
        if phase=='A':
            assert len({i[1] for i in items if i[0] in ('HEAD','JOINT')})==1
            assert len({i[2] for i in items})==1
            for i in items[1:]: np.testing.assert_array_equal(i[3],items[0][3])
            entry=next(e for e in plan['jobs'] if e['job']['phase']=='A' and e['job']['family']=='HEAD' and e['job']['recipe']==0 and f"{e['job']['dataset']}/s{e['job']['seed']}"==cell)
            folder=RUN/'fit'/entry['key']/'output'
            cells[cell]['correction']=choose_correction(folder/'F0_train.npz',folder/'initial_val.npz')
        else:
            warm=[i for i in items if i[0].startswith('WARM_')]
            assert len({i[2] for i in warm})==1
            for i in warm: assert i[4]['initial_prediction_exact']; np.testing.assert_array_equal(i[3],warm[0][3])
            a=read(RUN/'selection_A.json')['cells'][cell]['L720']['HEAD']
            reference=read(RUN/'fit'/a['key']/'output/result.json')
            for i in items:
                if i[0]=='REFIT':
                    assert i[1]==reference['initial_head_hash'] and i[2]==reference['sample_sha256']
    payload={'phase':phase,'plan_sha256':sha(RUN/'plan.json'),'cells':cells,'fit_hashes':hashes,
        'forecasts':sorted(forecasts),'sealed_utc':datetime.now(timezone.utc).isoformat(),'selection_uses_V_only':True}
    write(destination,payload); return payload


def prediction(phase,cell,budget,family):
    if family in ('F0','CORRECTION'): selected=cell[budget]['HEAD']; name='F0'
    else: selected=cell[budget][family]; name='selected'
    folder=RUN/'forecast'/selected['key']/budget/'output'
    with np.load(folder/f'{name}.npz') as z:
        arrays={k:z[k].copy() for k in ('prediction','target','quantiles','scale')}
    if family=='CORRECTION':
        arrays['prediction']=apply_correction(arrays['prediction'],cell['correction']['parameters'],cell['correction']['selected'])
    return arrays,selected


def rows_for(phase):
    sealed=read(RUN/f'selection_{phase}.json'); rows=[]; components=[]
    a_rows=None
    if phase=='B': a_rows=read(OUT/'summary_A.json')['rows']
    for cell_key,cell in sorted(sealed['cells'].items()):
        for budget in (('S180','L720') if phase=='A' else ('L720',)):
            families=('F0','HEAD','WIDE','JOINT','CORRECTION') if phase=='A' else ('REFIT','WARM_HEAD','WARM_JOINT')
            f0=None if phase=='A' else next(r['D_score'] for r in a_rows if r['cell']==cell_key and r['budget']=='L720' and r['family']=='F0')
            reference=None
            for family in families:
                arrays,selected=prediction(phase,cell,budget,family)
                p,y,q,s=(arrays[k] for k in ('prediction','target','quantiles','scale'))
                if reference is None: reference=(y,q,s)
                else:
                    for x,v in zip((y,q,s),reference): np.testing.assert_array_equal(x,v)
                score=score_arrays(p,y,q,s)
                if family=='F0': f0=score
                if family not in ('F0','CORRECTION'):
                    record=read(RUN/'forecast'/selected['key']/budget/'output/result.json')
                    assert abs(record['D_score']-score)<1e-12
                num,den=pinball_components(p,y,q)
                row={'phase':phase,'cell':cell_key,'dataset':cell['dataset'],'seed':cell['seed'],'budget':budget,
                     'family':family,'D_score':score,'F0_score':f0,'D_over_F0':score/f0,
                     'V':selected['V'] if family not in ('F0','CORRECTION') else None,
                     'step':selected['step'] if family not in ('F0','CORRECTION') else 0,
                     'recipe':selected['recipe'] if family not in ('F0','CORRECTION') else None}
                rows.append(row); components.append((num,den,s))
    return rows,components


def gate(rows,family,comparators):
    values=[]
    for r in rows:
        if r['family']!=family or r['budget']!='L720': continue
        other=[o['D_score'] for o in rows if o['cell']==r['cell'] and o['budget']=='L720' and o['family'] in comparators]
        assert len(other)==len(comparators)
        values.append({'cell':r['cell'],'dataset':r['dataset'],'seed':r['seed'],
                       'gain_pct_F0':100*(min(other)-r['D_score'])/r['F0_score']})
    by_source={d:all(r['gain_pct_F0']>.25 for r in values if r['dataset']==d) for d in sorted({r['dataset'] for r in values})}
    assert all(sum(r['dataset']==d for r in values)==2 for d in by_source)
    return {'passes':any(by_source.values()),'source_passes':by_source,'values':values}


def analyse(phase):
    destination=OUT/f'summary_{phase}.json'
    if destination.exists(): return read(destination)
    rows,parts=rows_for(phase)
    if phase=='A': gates={'G1':gate(rows,'JOINT',('F0','HEAD','WIDE','CORRECTION'))}
    else:
        a=read(OUT/'summary_A.json'); combined=a['rows']+rows
        gates={'G2_refit':gate(combined,'REFIT',('F0','HEAD')),'G3_warm':gate(combined,'WARM_JOINT',('F0','WARM_HEAD'))}
        gates['same_source_all_three']=[d for d in a['gates']['G1']['source_passes'] if a['gates']['G1']['source_passes'][d] and gates['G2_refit']['source_passes'][d] and gates['G3_warm']['source_passes'][d]]
    payload={'completed':True,'phase':phase,'plan_sha256':sha(RUN/'plan.json'),
             'analysed_utc':datetime.now(timezone.utc).isoformat(),'gates':gates,'rows':rows,
             'D_is_development':True,'not_final_test':True}
    write(destination,payload); return payload


def finish():
    OUT.mkdir(exist_ok=True)
    phases=['A']+(['B'] if (OUT/'summary_B.json').exists() else [])
    rows=[]; parts=[]
    for phase in phases:
        r,p=rows_for(phase); rows+=r; parts+=p
    with (OUT/'metrics.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    nums=np.stack([p[0] for p in parts]); counts=np.stack([p[1] for p in parts]); scales=np.stack([p[2] for p in parts])
    np.savez_compressed(OUT/'per_example_losses.npz',pinball_numerator=nums,valid_target_count=counts,scale=scales,
                        F0=np.array([r['F0_score'] for r in rows]),row_keys=np.array([json.dumps(r,sort_keys=True) for r in rows]))
    rng=np.random.default_rng(30355); weights={}
    for d in sorted({r['dataset'] for r in rows}):
        starts=rng.integers(0,20,size=(2000,10)); draws=np.stack([starts,(starts+1)%20],axis=-1).reshape(2000,20)
        weights[d]=np.stack([np.bincount(v,minlength=20) for v in draws])
    boots=[]
    for i,r in enumerate(rows):
        w=weights[r['dataset']]; boots.append((np.einsum('bo,ocq->bcq',w,nums[i])/(w@counts[i])[:,:,None]/scales[i][None,:,None]).mean((1,2)))
    def index(cell,budget,family): return next(i for i,r in enumerate(rows) if r['cell']==cell and r['budget']==budget and r['family']==family)
    contrasts=[('JOINT','HEAD'),('JOINT','WIDE'),('JOINT','CORRECTION')]
    if 'B' in phases: contrasts += [('REFIT','HEAD'),('WARM_JOINT','WARM_HEAD')]
    intervals={}
    for family,other in contrasts:
        diffs=[]
        for cell in sorted({r['cell'] for r in rows}):
            i=index(cell,'L720',family); j=index(cell,'L720',other); f=index(cell,'L720','F0')
            diffs.append(100*(boots[j]-boots[i])/boots[f])
        intervals[f'{family}_vs_{other}']=np.quantile(np.mean(diffs,axis=0),[.05,.95]).tolist()
    write(OUT/'intervals.json',{'conditional_90pct':intervals,'draws':2000,'block_origins':2,'fixed_source_periods':True})
    histories=[]
    for p in sorted((RUN/'fit').glob('**/output/result.json')):
        r=read(p); histories.append(r)
    write(OUT/'fit_histories.json',histories)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cells=sorted({r['cell'] for r in rows})
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for ax,fam in zip(axes,('HEAD','WIDE')):
        for budget,offset,color in [('S180',-.17,'#7a8da5'),('L720',.17,'#187d79')]:
            vals=[100*(rows[index(c,budget,fam)]['D_score']-rows[index(c,budget,'JOINT')]['D_score'])/rows[index(c,budget,'F0')]['D_score'] for c in cells]
            ax.bar(np.arange(4)+offset,vals,width=.32,label=budget,color=color)
        ax.axhline(0,color='black',lw=.8); ax.set_xticks(range(4),cells,rotation=25,ha='right'); ax.set_title(f'JOINT gain over {fam}'); ax.set_ylabel('%F0'); ax.legend()
    fig.savefig(OUT/'01_budget_gap.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    seal_a=read(RUN/'selection_A.json')
    for ax,cell in zip(axes.flat,cells):
        for family,color in [('HEAD','#cc6633'),('WIDE','#3269aa'),('JOINT','#087e77')]:
            key=seal_a['cells'][cell]['L720'][family]['key']; r=read(RUN/'fit'/key/'output/result.json')
            ax.plot([h['step'] for h in r['history']],[h['V']/r['initial_V'] for h in r['history']],label=family,color=color)
            selected=r['budgets']['L720']; ax.scatter(selected['step'],selected['V']/r['initial_V'],marker='*',s=90,color=color)
        ax.axvline(180,color='grey',ls=':'); ax.set_title(cell); ax.set_xlabel('Updates'); ax.set_ylabel('V / initial V'); ax.legend()
    fig.savefig(OUT/'02_validation_trajectories.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True)
    fams=['F0','HEAD','WIDE','JOINT','CORRECTION']+(['REFIT','WARM_HEAD','WARM_JOINT'] if 'B' in phases else [])
    matrix=np.array([[rows[index(c,'L720',f)]['D_over_F0'] for c in cells] for f in fams])
    im=ax.imshow(matrix,cmap='viridis_r',aspect='auto'); norm=im.norm
    for i in range(len(fams)):
        for j in range(4):
            rgba=im.cmap(norm(matrix[i,j])); lum=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
            ax.text(j,i,f'{matrix[i,j]:.4f}',ha='center',va='center',color='black' if lum>.5 else 'white')
    ax.set_yticks(range(len(fams)),fams); ax.set_xticks(range(4),cells,rotation=20,ha='right'); ax.set_title('L720 development loss / F0 (lower is better)'); fig.colorbar(im,ax=ax)
    fig.savefig(OUT/'03_family_scores.png',dpi=160); plt.close(fig)
    write(OUT/'completed.json',{'completed':True,'phases':phases,'rows':len(rows),'finished_utc':datetime.now(timezone.utc).isoformat()})
