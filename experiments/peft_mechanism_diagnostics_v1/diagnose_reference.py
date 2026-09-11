"""Descriptive decomposition of the exposed Study20 evaluation, without model updates."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/peft_fullft_reference_v3'
OUT = ROOT/'results/peft_mechanism_diagnostics_v1/reference'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def corr(a, b):
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3 or np.std(a[keep]) == 0 or np.std(b[keep]) == 0:
        return None
    return float(np.corrcoef(a[keep], b[keep])[0, 1])


def features(x):
    x = np.asarray(x, float)
    t = np.arange(len(x), dtype=float)
    detrended = x - np.polyval(np.polyfit(t, x, 1), t)
    power = abs(np.fft.rfft(detrended * np.hanning(len(x))))[1:]**2
    p = power/power.sum() if power.sum() else np.zeros_like(power)
    entropy = float(-np.sum(p[p > 0]*np.log(p[p > 0]))/np.log(len(p))) if power.sum() else None
    return {'spectral_entropy': entropy, 'acf24': corr(x[:-24], x[24:]),
            'acf168': corr(x[:-168], x[168:]),
            'level_shift': float(abs(np.mean(x[:168])-np.mean(x[168:]))/(np.std(x)+1e-12))}


def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'summary.json').exists():
        raise FileExistsError('Preserve completed diagnostic')
    contract = json.loads((RUN/'study_contract.json').read_text())
    selection = json.loads((RUN/'selection.json').read_text())
    source_hashes = {}
    summaries, records, feature_rows = {}, [], []
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for row, dataset in enumerate(('bike', 'household')):
        path = ROOT/contract['datasets'][dataset]['holdout_data_path']
        assert digest(path) == contract['datasets'][dataset]['holdout_data_sha256']
        source_hashes[str(path.relative_to(ROOT))] = digest(path)
        with np.load(path) as archive:
            values, indices, std, qs, stamps = [archive[k] for k in ('context_values', 'target_indices', 'fit_std', 'quantiles', 'timestamps')]
        scale = std[indices].astype(float)
        metrics, losses, meds, preds = {}, {}, {}, {}
        for method in ('F0', 'HEAD_ONLY', 'LORA', 'FULL_FT'):
            keys = ['F0'] if method == 'F0' else [f'{method}_seed_{s}' for s in (20000,20001,20002)]
            method_losses, method_meds, method_preds = [], [], []
            for key in keys:
                pth = RUN/'forecasts'/dataset/key/'Epredictions.npz'
                source_hashes[str(pth.relative_to(ROOT))] = digest(pth)
                with np.load(pth) as a:
                    pred, truth, origins = a['predictions'].astype(float), a['target'].astype(float), a['origins']
                    residual = truth[:,:,None,:]-pred
                    valid = np.isfinite(truth)
                    loss = np.where(valid[:,:,None,:], 2*np.maximum(qs[None,None,:,None]*residual,(qs[None,None,:,None]-1)*residual),0)/scale[None,:,None,None]
                    np.testing.assert_allclose(loss.sum((2,3)), a['loss_sums'], rtol=1e-10,atol=1e-10)
                    score = float(np.mean(loss.sum((0,2,3))/(valid.sum((0,2))*len(qs))))
                med = pred[:,:,np.argmin(abs(qs-.5)),:]
                metrics[key] = {'score':score,'median_mae_scaled':float(np.nanmean(abs(truth-med)/scale[None,:,None])),
                    'coverage80':float(((truth>=pred[:,:,2,:])&(truth<=pred[:,:,-3,:]))[valid].mean())}
                method_losses.append(loss.mean(2));method_meds.append(med);method_preds.append(pred)
            losses[method], meds[method], preds[method] = np.stack(method_losses), np.stack(method_meds), np.stack(method_preds)
        f0 = metrics['F0']['score']
        gain = (losses['HEAD_ONLY']-losses['LORA']).mean((0,2,3))/f0*100
        feature_values = []
        for i, origin in enumerate(origins):
            per_channel = [features(values[origin-336:origin,int(c)]) for c in indices]
            feat = {k:float(np.mean([f[k] for f in per_channel])) for k in per_channel[0]}
            feature_values.append(feat)
            feature_rows.append({'dataset':dataset,'origin':int(origin),'timestamp':str(stamps[origin]),**feat,'gain_pct_f0':float(gain[i])})
        gain_by_h = (losses['HEAD_ONLY']-losses['LORA']).mean((0,1,2))/f0*100
        by_seed = [(metrics[f'HEAD_ONLY_seed_{s}']['score']-metrics[f'LORA_seed_{s}']['score'])/f0*100 for s in (20000,20001,20002)]
        correlation = {k:corr(np.array([f[k] for f in feature_values]),gain) for k in feature_values[0]}
        blocks = [float(np.mean(gain[i:i+20])) for i in range(0,80,20)]
        daily_residual = (truth-meds['F0'][0])[:,:,:24].transpose(0,2,1).reshape(-1,2)
        residual_acf = {str(lag):[corr(daily_residual[:-lag,c],daily_residual[lag:,c]) for c in range(2)] for lag in (24,168)}
        histories=[]
        for item in selection['selected']:
            if item['dataset'] != dataset: continue
            pth=ROOT/item['fit_dir']/'result.json'
            source_hashes[str(pth.relative_to(ROOT))]=digest(pth)
            result=json.loads(pth.read_text())
            histories.append({'arm':item['arm'],'seed':item['seed'],'best_step':item['best_step'],'lr':item['lr'],'history':result.get('history')})
        summaries[dataset]={'metrics':metrics,'gain_head_lora_pct_f0_by_seed':by_seed,
            'gain_by_20_origin_block':blocks,'positive_origin_fraction':float(np.mean(gain>0)),
            'feature_mean':{k:float(np.mean([f[k] for f in feature_values])) for k in feature_values[0]},
            'descriptive_within_dataset_correlations':correlation,'f0_residual_acf_first24h':residual_acf,'selected_fit_histories':histories}
        for i,g in enumerate(gain_by_h):records.append({'dataset':dataset,'horizon':i+1,'gain_pct_f0':float(g)})
        axes[row,0].plot(gain);axes[row,0].axhline(0,color='gray',ls='--');axes[row,0].set(xlabel='Exposed E origin (daily)',ylabel='Head - LoRA (% F0)',title=dataset+' / per-origin gain')
        axes[row,1].plot(np.arange(1,49),gain_by_h);axes[row,1].axhline(0,color='gray',ls='--');axes[row,1].set(xlabel='Forecast hour',title='Horizon profile')
        axes[row,2].scatter([f['spectral_entropy'] for f in feature_values],gain,s=15,alpha=.7);axes[row,2].set(xlabel='Past-context spectral entropy',title='Association only; overlapping windows')
    fig.suptitle('Study20 diagnostic: mean of paired seed losses, not an ensemble',fontsize=14)
    fig.tight_layout();fig.savefig(OUT/'reference_diagnostic.png',dpi=160);plt.close(fig)
    for name,rows in [('origin_features.csv',feature_rows),('horizon_effects.csv',records)]:
        with (OUT/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report={'completed':True,'exploratory_exposed_E':True,'new_training':0,'datasets':summaries,
        'limits':['two datasets; overlapping origins not independent','past-only features, but correlations assessed on previously exposed E','residual ACF is posthoc; not an OOF predictor for new selection','coverage uses observed target mask; no causal inference'],
        'source_hashes':source_hashes,'seconds':time.perf_counter()-start}
    (OUT/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:{'gain':v['gain_head_lora_pct_f0_by_seed'],'blocks':v['gain_by_20_origin_block'],'features':v['feature_mean'],'corr':v['descriptive_within_dataset_correlations']} for k,v in summaries.items()}))


if __name__=='__main__':main()
