"""Past-only residual correction selection and separately dispatched exposed-E scoring."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.peft_fullft_reference_v3 import baselines as simple

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/peft_r1_validation_v1'
OUT = ROOT/'results/peft_r1_validation_v1/residual'
CANDIDATES = [('ZERO',None),('BIAS',None)] + [(method,lam) for method in ('SEASONAL_RIDGE','CONTEXT_RIDGE') for lam in (.01,1.,100.)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')


def features(panel, origins, median, kind):
    values = panel['context_values']; ix = panel['target_indices']
    context = np.stack([values[o-336:o].T for o in origins]).astype(float)
    if kind == 'CONTEXT_RIDGE':
        return np.concatenate([context.reshape(len(origins),-1),median.reshape(len(origins),-1)],axis=1)
    weekly = np.stack([values[o-168:o-120,ix].T for o in origins])
    daily = np.stack([np.tile(values[o-24:o,ix].T,(1,2)) for o in origins])
    return np.concatenate([weekly.reshape(len(origins),-1),daily.reshape(len(origins),-1),
                           context[:,ix].mean(2),context[:,ix].std(2),median.reshape(len(origins),-1)],axis=1)


def fit(method, lam, x, residual):
    if method == 'ZERO': return {'offset':np.zeros(residual.shape[1:])}
    if method == 'BIAS': return {'offset':np.nanmedian(residual,axis=0)}
    return simple.fit_ridge(x,residual.reshape(len(residual),-1),lam)


def correction(model, x):
    if 'offset' in model: return np.broadcast_to(model['offset'],(len(x),2,48)).copy()
    return simple.predict_ridge(model,x).reshape(len(x),2,48)


def score(pred, target, scale, q):
    return simple.score_predictions(pred,target,scale,q)['score']


def select():
    OUT.mkdir(parents=True,exist_ok=True)
    c = json.loads((RUN/'contract.json').read_text())
    selected = {}
    for ds in ('bike','household'):
        panel = simple.load_panel(ROOT/c['reference']['datasets'][ds]['fit_data_path'],'fit')
        archives = {}
        for split in ('train','val'):
            with np.load(RUN/'f0'/ds/'output'/(split+'.npz')) as a:
                archives[split] = {k:a[k].copy() for k in a.files}
                assert np.array_equal(a['origins'],panel['origins'][split])
        tr,va = archives['train'],archives['val']
        q,scale = tr['quantiles'],tr['scale']; mid = int(np.argmin(abs(q-.5)))
        residual = tr['target']-tr['prediction'][:,:,mid]
        records = []; models = []
        for index,(method,lam) in enumerate(CANDIDATES):
            kind = 'CONTEXT_RIDGE' if method=='CONTEXT_RIDGE' else 'SEASONAL_RIDGE'
            tx = features(panel,tr['origins'],tr['prediction'][:,:,mid],kind)
            vx = features(panel,va['origins'],va['prediction'][:,:,mid],kind)
            folds = []
            for past,future in simple.oof_folds(tr['origins'],48):
                assert tr['origins'][past].max()+48 <= tr['origins'][future].min()
                model = fit(method,lam,tx[past],residual[past])
                pred = tr['prediction'][future]+correction(model,tx[future])[:,:,None,:]
                f0 = score(tr['prediction'][future],tr['target'][future],scale,q)
                value = score(pred,tr['target'][future],scale,q)
                folds.append({'first_origin':int(tr['origins'][future[0]]),'fit_last_target_end':int(tr['origins'][past[-1]]+48),
                              'fit_rows':past.tolist(),'forecast_rows':future.tolist(),'f0_score':f0,'score':value,'gain_pct_f0':100*(f0-value)/f0})
            model = fit(method,lam,tx,residual)
            vp = va['prediction']+correction(model,vx)[:,:,None,:]
            value = score(vp,va['target'],scale,q)
            model_path = OUT/f'{ds}_candidate{index}.npz'
            np.savez_compressed(model_path,**model)
            models.append(model_path)
            records.append({'index':index,'method':method,'lambda':lam,'val_score':value,'oof':folds,'model_sha256':sha(model_path)})
        winner = min(records,key=lambda r:(r['val_score'],r['index']))
        selected[ds] = {'winner':winner,'candidates':records,'val_f0_score':records[0]['val_score'],
                        'holdout_opened':False,'model_path':str(models[winner['index']].relative_to(ROOT)),
                        'fit_forecast_hashes':{s:sha(RUN/'f0'/ds/'output'/(s+'.npz')) for s in ('train','val')}}
    save(OUT/'selection.json',{'contract_sha256':sha(RUN/'contract.json'),'selected':selected,'selection_only':True})
    print(json.dumps({ds:r['winner'] for ds,r in selected.items()}),flush=True)


def evaluate():
    c = json.loads((RUN/'contract.json').read_text())
    chosen = json.loads((OUT/'selection.json').read_text())
    assert chosen['contract_sha256']==sha(RUN/'contract.json')
    records = []
    for ds,selection in chosen['selected'].items():
        spec = c['reference']['datasets'][ds]
        panel = simple.load_panel(ROOT/spec['holdout_data_path'],'forecast')
        with np.load(ROOT/f'runs/peft_fullft_reference_v3/forecasts/{ds}/F0/Epredictions.npz') as a:
            pred,target,origins = a['predictions'].copy(),a['target'].copy(),a['origins'].copy()
        assert np.array_equal(origins,panel['origins']['eval'])
        expected = np.stack([panel['target_values'][o:o+48,panel['target_indices']].T for o in origins])
        np.testing.assert_array_equal(target,expected)
        q=panel['quantiles']; scale=panel['fit_std'][panel['target_indices']]
        mid=int(np.argmin(abs(q-.5))); win=selection['winner']
        path=ROOT/selection['model_path'];assert sha(path)==win['model_sha256']
        with np.load(path) as a: model={k:a[k].copy() for k in a.files}
        kind='CONTEXT_RIDGE' if win['method']=='CONTEXT_RIDGE' else 'SEASONAL_RIDGE'
        x=features(panel,origins,pred[:,:,mid],kind)
        corrected=pred+correction(model,x)[:,:,None,:]
        assert np.all(np.diff(corrected,axis=2)>=0)
        np.testing.assert_allclose(np.diff(corrected,axis=2),np.diff(pred,axis=2),atol=1e-10,rtol=1e-10)
        f0=score(pred,target,scale,q);value=score(corrected,target,scale,q)
        np.savez_compressed(OUT/f'{ds}_eval.npz',prediction=corrected,target=target,origins=origins,quantiles=q,scale=scale)
        records.append({'dataset':ds,'method':win['method'],'lambda':win['lambda'],'f0_score':f0,'score':value,
                        'gain_pct_f0':100*(f0-value)/f0,'selection_sha256':sha(OUT/'selection.json'),'previously_exposed_E':True})
    save(OUT/'evaluation.json',{'records':records,'completed':True})
    print(json.dumps(records),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['select','evaluate']);args=p.parse_args()
    select() if args.stage=='select' else evaluate()
