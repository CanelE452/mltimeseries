"""Guarded serial future-period validation of fixed past-only decisions."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from experiments.peft_optimization_control_v1.run import sha, save
from experiments.peft_adaptation_scope_v1.guard import run_guarded
from experiments.hospital_shared_strength_v1.resume_monitored import admission

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/peft_overlap_transfer_v1'
CODE = ROOT / 'experiments/peft_overlap_transfer_v1'
OLD = ROOT / 'runs/peft_optimization_control_v1'


def verify(plan):
    for path, digest in {**plan['source_hashes'], **plan['original_input_hashes'], **plan['development_hashes']}.items():
        assert sha(ROOT / path) == digest, path


def execute(key, command, gpu=True):
    parent = RUN / key
    if parent.exists():
        status = json.loads((parent / 'guard/status.json').read_text())
        assert status['completed'] and not status['reasons'], key
        return
    admission(RUN, key)
    print(json.dumps({'starting': key}), flush=True)
    status = run_guarded(command, parent / 'guard', ROOT, 900, require_gpu=gpu)
    assert status['completed'] and not status['reasons'], f'Preserved failure: {key}'


def prepare():
    gate = json.loads((ROOT / 'results/peft_optimization_control_v1/summary.json').read_text())
    assert gate['completed'] and gate['gate_passed']
    old = json.loads((OLD / 'contract.json').read_text())
    selected = json.loads((OLD / 'selection.json').read_text())
    sources = dict(old['source_hashes'])
    for p in [*CODE.glob('*.py'), CODE / 'PURPOSE.md', ROOT / 'experiments/peft_fullft_reference_v3/data.py', ROOT / 'experiments/peft_external_gap_v1/data.py', ROOT / 'experiments/peft_optimization_control_v1/analyse.py']:
        sources[p.relative_to(ROOT).as_posix()] = sha(p)
    rows = {'FULL90': list(range(90)), 'SPREAD30': [round(i*89/29) for i in range(30)], 'RECENT30': list(range(60, 90))}
    features = {}
    for condition, indices in rows.items():
        unique = len({24*i+h for i in indices for h in range(48)})
        ratio = len(indices)*48/unique
        features[condition] = {'origin_count': len(indices), 'unique_target_hours': unique, 'overlap_ratio': ratio, 'OVERLAP': 'ALL' if ratio >= 1.5 else 'MLP', 'COUNT': 'ALL' if len(indices) >= 60 else 'MLP'}
    jobs = []
    for ds in ('bike', 'household'):
        for condition in rows:
            for arm in ('MLP', 'ALL'):
                group = [e for e in selected if e['regime'] == 'EXPOSURE' and (e['job']['dataset'], e['job']['condition'], e['job']['arm']) == (ds, condition, arm)]
                assert len(group) == 2 and group[0]['job']['recipe'] == group[1]['job']['recipe']
                for seed in (26000, 26001):
                    job = {**group[0]['job'], 'seed': seed}
                    jobs.append({'job': job, 'key': f'fits/{ds}/{condition}/{arm}/s{seed}', 'contract': f'{condition}_contract.json'})
    plan = {'source_hashes': sources, 'original_input_hashes': old['original_input_hashes'], 'development_hashes': {p.relative_to(ROOT).as_posix(): sha(p) for p in [OLD / 'contract.json', OLD / 'selection.json', ROOT / 'results/peft_optimization_control_v1/summary.json']}, 'features': features, 'jobs': jobs, 'created_utc': datetime.now(timezone.utc).isoformat(), 'gates': {'macro_regret_max_pct_F0': .25, 'source_regret_max_pct_F0': .5, 'fit_time_savings_min_percent': 5., 'each_seed_required': True}, 'new_fits': 24, 'new_evaluations': 26, 'new_E_only': True, 'new_sources': False}
    verify(plan)
    RUN.mkdir(parents=True, exist_ok=False)
    save(RUN / 'plan.json', plan)
    execute('data', [sys.executable, '-m', 'experiments.peft_overlap_transfer_v1.prepare', '--output', str(RUN / 'prepared')], gpu=False)
    prepared = json.loads((RUN / 'prepared/manifest.json').read_text())
    assert prepared['all_qc_passed']
    reference = {**old['reference'], 'datasets': prepared['datasets']}
    for condition in rows:
        schedule = [0, 15, 30, 60, 120, 180] if condition == 'FULL90' else [0, 5, 10, 20, 40, 60]
        save(RUN / (condition + '_contract.json'), {'reference': reference, 'source_hashes': sources, 'steps': schedule[-1], 'schedules': {condition: {'EXPOSURE': schedule}}, 'plan_sha256': sha(RUN / 'plan.json'), 'prepared_manifest_sha256': sha(RUN / 'prepared/manifest.json')})
    print(json.dumps({'prepared': True, 'fits': 24, 'evaluations': 26}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return
    assert not (RUN / 'completed.json').exists(), 'Already complete; do not overwrite'
    started = time.perf_counter()
    plan = json.loads((RUN / 'plan.json').read_text())
    verify(plan)
    selections = []
    for entry in plan['jobs']:
        job, key = entry['job'], entry['key']
        command = [sys.executable, '-m', 'experiments.peft_optimization_control_v1.fit', '--contract', str(RUN / entry['contract']), '--job', json.dumps(job), '--output', str(RUN / key / 'output')]
        execute(key, command)
        result = json.loads((RUN / key / 'output/result.json').read_text())
        contract = json.loads((RUN / entry['contract']).read_text())
        assert result['completed'] and result['job'] == job and result['contract_sha256'] == sha(RUN / entry['contract'])
        assert result['steps'] == contract['steps'] and result['frozen_verified'] and not result['holdout_opened']
        winner = result['regimes']['EXPOSURE']
        assert winner['replay_verified'] and sha(RUN / key / 'output/EXPOSURE.pt') == winner['checkpoint_sha256']
        selections.append({**entry, 'chosen': winner})
    selected_path = RUN / 'selection.json'
    if selected_path.exists():
        assert json.loads(selected_path.read_text()) == selections
    else:
        save(selected_path, selections)
    for ds in ('bike', 'household'):
        key = 'f0/' + ds
        execute(key, [sys.executable, '-m', 'experiments.peft_overlap_transfer_v1.f0', '--contract', str(RUN / 'FULL90_contract.json'), '--dataset', ds, '--output', str(RUN / key / 'output')])
    for entry in selections:
        key = entry['key'].replace('fits/', 'eval/')
        execute(key, [sys.executable, '-m', 'experiments.peft_optimization_control_v1.fit', '--contract', str(RUN / entry['contract']), '--job', json.dumps(entry['job']), '--output', str(RUN / key / 'output'), '--regime', 'EXPOSURE', '--forecast-fit', str(RUN / entry['key'] / 'output')])
        record = json.loads((RUN / key / 'output/result.json').read_text())
        assert record['completed'] and record['job'] == entry['job'] and record['regime'] == 'EXPOSURE'
        assert record['contract_sha256'] == sha(RUN / entry['contract']) and Path(record['fit']) == RUN / entry['key'] / 'output'
    verify(plan)
    save(RUN / 'completed.json', {'completed': True, 'finished_utc': datetime.now(timezone.utc).isoformat(), 'invocation_seconds': time.perf_counter()-started, 'fits': 24, 'evaluations': 26, 'selection_sha256': sha(selected_path), 'plan_sha256': sha(RUN / 'plan.json')})
    print(json.dumps({'completed': True}), flush=True)


if __name__ == '__main__':
    main()
