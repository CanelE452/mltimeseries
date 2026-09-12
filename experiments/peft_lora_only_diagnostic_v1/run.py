"""Prepare, zero-update smoke, sixteen serial fits, seal V, eight D forecasts."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
import traceback
from experiments.peft_lora_only_diagnostic_v1.audit import ROOT, CODE, RUN, OUT, read, write, sha, verify_old, verify_plan, seal
from experiments.hospital_shared_strength_v1.resume_monitored import admission
from experiments.peft_adaptation_scope_v1.guard import run_guarded


def prepare():
    assert not (CODE/'plan.json').exists()
    old_audit = verify_old()
    old = read(ROOT/'runs/peft_head_convergence_v1/plan.json')
    jobs = [{'key': f'{d}_s{s}_r{i}', 'dataset': d, 'seed': s, 'recipe': i, 'lora_lr': lr}
            for d in sorted(old['data']) for s in (30000, 30001)
            for i, lr in enumerate((1e-5, 3e-5, 1e-4, 3e-4))]
    sources = dict(old['source_hashes'])
    for path in CODE.glob('*.py'):
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
    sources[(CODE/'PURPOSE.md').relative_to(ROOT).as_posix()] = sha(CODE/'PURPOSE.md')
    inputs = dict(old['input_hashes'])
    evidence = read(ROOT/'results/peft_paper_closure_v1/evidence_reconstruction.json')
    inputs.update({p: i['sha256'] for p, i in evidence['provenance'].items()})
    for path in ('runs/peft_head_convergence_v1/plan.json', 'runs/peft_head_convergence_v1/selection_A.json',
                 'results/peft_paper_closure_v1/evidence_reconstruction.json'):
        inputs[path] = sha(ROOT/path)
    plan = {k: old[k] for k in ('checkpoint', 'data', 'schedule', 'eval_stride')}
    plan.update(study='peft_lora_only_diagnostic_v1', run_dir=RUN.relative_to(ROOT).as_posix(),
                created_utc=datetime.now(timezone.utc).isoformat(), jobs=jobs, source_hashes=sources,
                input_hashes=inputs, rank=8, alpha=16, dropout=0, trainable=1179648,
                native_head_frozen=True, residual_head=False, max_fits=16, steps=720,
                optimizer={'name': 'AdamW', 'weight_decay': 0, 'effective_batch': 8, 'microbatch': 4, 'clip': 1},
                selection='V only; then earlier step; then recipe index; step0 eligible',
                step0_tolerance=1e-6, case_near_zero_pct_F0=0.25, development_only=True,
                lr_grid_note='Numeric candidates reused from Study35 HEAD/WIDE. JOINT LoRA rates were 1e-5,3e-5,3e-5,1e-4; 3e-4 was not a JOINT LoRA rate. Requested four distinct defaults retained; not the identical joint recipe grid.',
                old_audit=old_audit)
    write(CODE/'plan.json', plan)
    write(OUT/'integrity_audit.json', {'status': 'PREFLIGHT_PASS', 'old': old_audit})


def execute(stage, index, budget='L720'):
    plan = verify_plan()
    job = plan['jobs'][index]
    folder = RUN/stage/job['key']
    if stage == 'forecast':
        folder /= budget
    assert not folder.exists(), f'No automatic rerun of existing artifacts: {folder}'
    RUN.mkdir(parents=True, exist_ok=True)
    admission(RUN, f'{stage}/{job["key"]}/{budget}')
    command = [sys.executable, '-m', 'experiments.peft_lora_only_diagnostic_v1.fit',
               '--index', str(index), '--stage', stage, '--budget', budget, '--output', str(folder/'output')]
    guard = run_guarded(command, folder/'guard', ROOT, 900, require_gpu=True)
    assert guard['completed'], guard
    assert read(folder/'output/result.json')['completed']
    print(f'Completed {stage} {job["key"]} {budget}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare', 'gate', 'run'])
    args = p.parse_args()
    started = time.perf_counter()
    try:
        if args.action == 'prepare':
            prepare()
        elif args.action == 'gate':
            for index in (0, 4, 8, 12):
                execute('gate', index)
        else:
            assert not (OUT/'STOP.json').exists()
            for index in (0, 4, 8, 12):
                assert read(RUN/'gate'/verify_plan()['jobs'][index]['key']/'output/gate.json')['passed']
            for index in range(16):
                execute('fit', index)
            seal()
            for index, budget in read(RUN/'selection.json')['forecasts']:
                execute('forecast', index, budget)
            write(RUN/'completed.json', {'completed': True, 'seconds': time.perf_counter()-started})
    except Exception:
        write(OUT/'STOP.json', {'action': args.action, 'traceback': traceback.format_exc(),
              'seconds': time.perf_counter()-started, 'artifacts_preserved': True})
        raise


if __name__ == '__main__':
    main()
