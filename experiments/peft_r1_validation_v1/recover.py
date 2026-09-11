"""Resume only failed CPU dtype assertion; preserve frozen source, fits and selection."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
from experiments.peft_r1_validation_v1.run import ROOT,RUN,CODE,sha,save,execute


def main():
    failed=json.loads((RUN/'residual_eval/guard/status.json').read_text())
    assert failed['state']=='child_failed' and failed['returncode']==1 and not failed['reasons']
    assert not (RUN/'completed.json').exists()
    original=(CODE/'residual.py').read_text()
    fixed=(CODE/'residual_dtypefix.py').read_text()
    needle="pred,target,origins = a['predictions'].copy(),a['target'].copy(),a['origins'].copy()"
    assert original.count(needle)==1
    assert fixed==original.replace(needle,"pred,target,origins = a['predictions'].astype(np.float64),a['target'].copy(),a['origins'].copy()")
    out=ROOT/'results/peft_r1_validation_v1/residual'
    assert not (out/'evaluation.json').exists() and not list(out.glob('*_eval.npz'))
    protected={str(p.relative_to(ROOT)):sha(p) for p in [out/'selection.json',*out.glob('*_candidate*.npz')]}
    save(RUN/'dtype_recovery_contract.json',{'change':'Cast stored F0 prediction to float64 before adding the unchanged selected offset; remove float32 addition rounding, assertion tolerance unchanged',
                                          'fixed_source_sha256':sha(CODE/'residual_dtypefix.py'),'protected_selection':protected})
    execute('residual_eval_dtype_retry',[sys.executable,'-m','experiments.peft_r1_validation_v1.residual_dtypefix','evaluate'],gpu=False)
    c=json.loads((RUN/'contract.json').read_text())
    for p,d in {**c['source_hashes'],**c['original_input_hashes'],**c['reused_full90_hashes'],**protected}.items():assert sha(ROOT/p)==d,p
    first=json.loads((RUN/'admission.jsonl').read_text().splitlines()[0])['utc']
    elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(first)).total_seconds()
    save(RUN/'completed.json',{'completed':True,'seconds':elapsed,'timing_basis':'first admission sample to recovered completion, including CPU diagnosis delay',
                              'new_fits':16,'new_model_evaluations':16,'guard_jobs':37,'successful_guard_jobs':36,
                              'original_controller_failed':True,'cpu_dtype_recovery':True,'fresh_test':False})
    print(json.dumps({'completed':True,'seconds':elapsed,'retrained':False}))


if __name__=='__main__':main()
