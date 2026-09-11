"""Serial guarded optimization controls; selection is sealed before E opens."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from experiments.peft_adaptation_scope_v1.guard import run_guarded
from experiments.hospital_shared_strength_v1.resume_monitored import admission

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/peft_optimization_control_v1'
CODE = ROOT / 'experiments/peft_optimization_control_v1'
OLD = ROOT / 'runs/peft_mechanism_diagnostics_v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def prepare():
    RUN.mkdir(parents=True, exist_ok=False)
    old = json.loads((OLD / 'contract.json').read_text())
    original = json.loads((OLD / 'original_input_audit.json').read_text())['original_code_model_data_hashes']
    sources = dict(old['source_hashes'])
    for p in (CODE / 'fit.py', CODE / 'run.py', CODE / 'PURPOSE.md', ROOT / 'experiments/hospital_shared_strength_v1/resume_monitored.py'):
        sources[str(p.relative_to(ROOT))] = sha(p)
    for p, digest in {**original, **sources}.items():
        assert sha(ROOT / p) == digest, p
    rows = {'FULL90': list(range(90)), 'SPREAD30': [round(i*89/29) for i in range(30)], 'RECENT30': list(range(60, 90))}
    schedules = {condition: {'UPDATE': [0, 5, 10, 20, 40, 60, 100, 140, 180], 'EXPOSURE': [0, 15, 30, 60, 120, 180] if condition == 'FULL90' else [0, 5, 10, 20, 40, 60]} for condition in rows}
    for condition in rows:
        assert [8*s/len(rows[condition]) for s in schedules[condition]['EXPOSURE']] == [8*s/90 for s in schedules['FULL90']['EXPOSURE']]
    jobs = []
    for ds in ('bike', 'household'):
        for condition, indices in rows.items():
            for arm in ('MLP', 'ALL'):
                for recipe, lr in enumerate((1e-4, 1e-5)):
                    for seed in (25000, 25001):
                        job = {'dataset': ds, 'condition': condition, 'training_rows': indices, 'arm': arm, 'seed': seed, 'head': 'mlp', 'blocks': list(range(12)) if arm == 'ALL' else [], 'rank': 8, 'head_lr': lr, 'lora_lr': lr, 'recipe': recipe}
                        jobs.append({'job': job, 'key': f'fits/{ds}/{condition}/{arm}/r{recipe}_s{seed}'})
    save(RUN / 'contract.json', {'reference': old['reference'], 'source_hashes': sources, 'original_input_hashes': original, 'steps': 180, 'schedules': schedules, 'jobs': jobs, 'fresh_test': False, 'created_utc': datetime.now(timezone.utc).isoformat(), 'gate': {'primary_dataset': 'bike', 'full_mean_gain_min_percent_F0': 1., 'full_minus_spread_mean_min_percent_F0': 1., 'both_seeds_positive': True, 'both_regimes_required': True}, 'new_fits': 48, 'new_evaluations': 48, 'smokes': 2})
    print(json.dumps({'prepared': str(RUN), 'verified_original_hashes': len(original), 'jobs': len(jobs)}), flush=True)


def execute(key, job, extra=()):
    parent = RUN / key
    if parent.exists():
        status = json.loads((parent / 'guard/status.json').read_text())
        result = json.loads((parent / 'output/result.json').read_text())
        assert status['completed'] and result['completed'] and result['job'] == job
        assert result['contract_sha256'] == sha(RUN / 'contract.json')
        for name, record in result.get('regimes', {}).items():
            assert sha(parent / 'output' / (name + '.pt')) == record['checkpoint_sha256']
        return
    admission(RUN, key)
    command = [sys.executable, '-m', 'experiments.peft_optimization_control_v1.fit', '--contract', str(RUN / 'contract.json'), '--job', json.dumps(job), '--output', str(parent / 'output'), *extra]
    print(json.dumps({'starting': key}), flush=True)
    status = run_guarded(command, parent / 'guard', ROOT, 900, require_gpu=True)
    if not status['completed']:
        raise RuntimeError(f'Preserved failed attempt: {key}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return
    started = time.perf_counter()
    contract = json.loads((RUN / 'contract.json').read_text())
    for p, digest in {**contract['original_input_hashes'], **contract['source_hashes']}.items():
        assert sha(ROOT / p) == digest, p
    for arm in ('MLP', 'ALL'):
        job = next(e['job'] for e in contract['jobs'] if e['job']['dataset'] == 'bike' and e['job']['condition'] == 'RECENT30' and e['job']['arm'] == arm)
        execute('smoke/' + arm, job, ['--smoke'])
    if args.smoke_only:
        return
    for entry in contract['jobs']:
        execute(entry['key'], entry['job'])
    selections = []
    for regime in ('UPDATE', 'EXPOSURE'):
        for ds in ('bike', 'household'):
            for condition in ('FULL90', 'SPREAD30', 'RECENT30'):
                for arm in ('MLP', 'ALL'):
                    candidates = []
                    for recipe in (0, 1):
                        group = [e for e in contract['jobs'] if (e['job']['dataset'], e['job']['condition'], e['job']['arm'], e['job']['recipe']) == (ds, condition, arm, recipe)]
                        assert len(group) == 2
                        scores = [json.loads((RUN / e['key'] / 'output/result.json').read_text())['regimes'][regime]['best_score'] for e in group]
                        candidates.append({'recipe': recipe, 'mean_val': sum(scores)/2, 'entries': group})
                    chosen = min(candidates, key=lambda c: (c['mean_val'], -c['recipe']))
                    for entry in chosen['entries']:
                        selections.append({'regime': regime, **entry, 'recipe_mean_val': chosen['mean_val'], 'candidate_means': {str(c['recipe']): c['mean_val'] for c in candidates}})
    path = RUN / 'selection.json'
    if path.exists():
        assert json.loads(path.read_text()) == selections
    else:
        save(path, selections)
    for entry in selections:
        key = entry['key'].replace('fits/', 'eval/' + entry['regime'] + '/')
        execute(key, entry['job'], ['--regime', entry['regime'], '--forecast-fit', str(RUN / entry['key'] / 'output')])
    for p, digest in {**contract['original_input_hashes'], **contract['source_hashes']}.items():
        assert sha(ROOT / p) == digest, p
    save(RUN / 'completed.json', {'completed': True, 'invocation_seconds': time.perf_counter()-started, 'finished_utc': datetime.now(timezone.utc).isoformat(), 'fits': 48, 'evaluations': 48, 'smokes': 2, 'fresh_test': False, 'selection_sha256': sha(path)})
    print(json.dumps({'completed': True}), flush=True)


if __name__ == '__main__':
    main()
