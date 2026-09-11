"""Supplemental provenance check without changing the in-flight frozen protocol."""
from datetime import datetime,timezone
import json
from pathlib import Path
from experiments.peft_r1_validation_v1.run import sha

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/peft_r1_validation_v1'


def main():
    c=json.loads((RUN/'contract.json').read_text())
    paths={};initial=[]
    for ds in ('bike','household'):
        folder=ROOT/f'runs/peft_fullft_reference_v3/forecasts/{ds}/F0'
        r=json.loads((folder/'result.json').read_text())
        assert sha(folder/'Epredictions.npz')==r['Epredictions_sha256']
        assert sha(ROOT/c['reference']['datasets'][ds]['holdout_data_path'])==r['holdout_data_sha256']
        assert sha(ROOT/'runs/peft_fullft_reference_v3/study_contract.json')==r['contract_sha256']
        for p in (folder/'Epredictions.npz',folder/'result.json'):
            paths[str(p.relative_to(ROOT))]=sha(p)
        cached=json.loads((RUN/'f0'/ds/'output/result.json').read_text())
        old=json.loads((ROOT/f'runs/peft_mechanism_diagnostics_v1/fits/{ds}/MLP/r0_s25000/output/result.json').read_text())
        assert cached['scores']['val']==old['history'][0]['score']
        initial.append({'dataset':ds,'cached_val_score':cached['scores']['val'],'original_step0_score':old['history'][0]['score'],'difference':0.0})
    path=RUN/'supplemental_input_audit.json'
    report={'verified':True,'utc':datetime.now(timezone.utc).isoformat(),'f0_original_prediction_hashes':paths,
            'cached_f0_initial_score_checks':initial,'residual_eval_not_started':not (RUN/'residual_eval').exists()}
    if path.exists():
        old=json.loads(path.read_text())
        assert old['f0_original_prediction_hashes']==paths
        print('PASS supplemental provenance recheck')
    else:
        path.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report))


if __name__=='__main__':main()
