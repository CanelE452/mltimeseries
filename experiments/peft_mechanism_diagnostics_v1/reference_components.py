"""Target and quantile decomposition, plus influence diagnostics of the exposed E."""
import json
from pathlib import Path
import numpy as np
from experiments.peft_mechanism_diagnostics_v1.diagnose_reference import ROOT,RUN,OUT,digest


def main():
    contract=json.loads((RUN/'study_contract.json').read_text())
    report={}
    for ds in ('bike','household'):
        with np.load(ROOT/contract['datasets'][ds]['holdout_data_path']) as a:
            ids=a['target_indices'];names=a['channels'][ids];std=a['fit_std'][ids].astype(float);qs=a['quantiles']
        sums={};provenance={}
        for arm in ('F0','HEAD_ONLY','LORA'):
            per=[]
            for seed in ([None] if arm=='F0' else [20000,20001,20002]):
                path=RUN/'forecasts'/ds/(arm if seed is None else f'{arm}_seed_{seed}')/'Epredictions.npz'
                provenance[str(path.relative_to(ROOT))]=digest(path)
                with np.load(path) as a:
                    p=a['predictions'].astype(float);y=a['target'].astype(float);valid=np.isfinite(y)
                    e=y[:,:,None,:]-p
                    loss=np.where(valid[:,:,None,:],2*np.maximum(qs[None,None,:,None]*e,(qs[None,None,:,None]-1)*e),np.nan)/std[None,:,None,None]
                per.append(loss)
            sums[arm]=np.stack(per)
        f0=float(np.nanmean(sums['F0']))
        difference=np.nanmean(sums['HEAD_ONLY']-sums['LORA'],axis=0)/f0*100
        origins=np.nanmean(difference,axis=(1,2,3))
        high=int(np.argmax(origins));low=int(np.argmin(origins))
        assert abs(float(origins.mean())-float(difference.mean()))<1e-10
        report[ds]={'target_names':names.tolist(),'additional_gain_by_target_pct_f0':dict(zip(names.tolist(),np.nanmean(difference,axis=(0,2,3)).tolist())),
            'additional_gain_by_quantile_pct_f0':dict(zip(map(str,qs),np.nanmean(difference,axis=(0,1,3)).tolist())),
            'first24_gain_pct_f0':float(np.nanmean(difference[:,:,:,:24])),
            'second24_gain_pct_f0':float(np.nanmean(difference[:,:,:,24:])),
            'mean_gain_pct_f0':float(origins.mean()),'largest_benefit_origin':high,'largest_harm_origin':low,
            'gain_without_largest_benefit_origin_pct_f0':float(np.delete(origins,high).mean()),
            'gain_without_largest_harm_origin_pct_f0':float(np.delete(origins,low).mean()),
            'loss_mask_all_valid':bool(valid.all()),'sources_sha256':provenance,
            'limits':'Component profiles and deletion diagnostics are descriptive; omitted origins are not a new primary evaluation.'}
    path=OUT/'components.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:{n:v for n,v in a.items() if n not in ('sources_sha256','additional_gain_by_quantile_pct_f0')} for k,a in report.items()}))


if __name__=='__main__':main()
