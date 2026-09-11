"""Post-outcome resource and descriptive review; no policy fitting or new gate."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import psutil
from experiments.hospital_shared_strength_v1.resume_monitored import memory

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/peft_future_utility_v1'
OUT = ROOT / 'results/peft_future_utility_v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert read(RUN / 'completed.json')['completed']
    assert read(RUN / 'independent_audit.json')['completed']
    destination = OUT / 'completion_review.json'
    assert not destination.exists()
    statuses = list(RUN.rglob('guard/status.json'))
    assert len(statuses) == 13
    samples, finishes = [], 0
    for path in statuses:
        status = read(path)
        assert status['completed'] and status['returncode'] == 0 and not status['reasons']
        records = [json.loads(line) for line in path.with_name('resource_log.jsonl').read_text().splitlines()]
        terminal = [r for r in records if 'gpus' not in r]
        assert len(terminal) == 1 and terminal[0]['event'] == 'finish' and terminal[0]['state'] == 'completed'
        finishes += len(terminal)
        samples.extend(r for r in records if 'gpus' in r)
    gpu = [g for r in samples for g in r['gpus']]
    protected = {}
    for study in ('peft_future_utility_v1', 'peft_contribution_freeze_v1', 'peft_capacity_probe_v1'):
        plan = read(ROOT / 'runs' / study / 'plan.json')
        hashes = dict(plan['source_hashes'])
        if study == 'peft_future_utility_v1':
            hashes.update(plan['input_hashes'])
        for path, digest in hashes.items():
            assert sha(ROOT / path) == digest, path
        protected[study] = {'checked_hashes': len(hashes), 'scope': 'source and registered inputs' if study == 'peft_future_utility_v1' else 'frozen sources only'}
    active, inaccessible = [], []
    for process in psutil.process_iter(['pid', 'name']):
        if 'python' not in (process.info['name'] or '').lower():
            continue
        try:
            cmd = process.cmdline()
            if any(x in cmd for x in ('experiments.peft_future_utility_v1.fit', 'experiments.peft_future_utility_v1.run')):
                active.append(process.pid)
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            inaccessible.append(process.pid)
    lock = ROOT / 'runs/peft_adaptation_scope_v1/.guard.lock'
    assert not active and not inaccessible and not lock.exists()
    rows = list(csv.DictReader((OUT / 'metrics.csv').open(encoding='utf-8')))
    val = lambda r, k: float(r[k])
    band = read(OUT / 'summary.json')['band_pct_F0']
    cls = lambda v: 1 if v > band else -1 if v < -band else 0
    selected = [r for r in rows if val(r, 'U_selected_pct_F0') == 0.]
    extra = {
        'role': 'Post-outcome descriptive counts using existing estimands and band; not new confirmatory tests.',
        'positive_current_contribution': sum(val(r, 'C_S_pct_F0') > 0 for r in rows),
        'positive_contribution_but_HEAD_better': sum(val(r, 'C_S_pct_F0') > 0 and val(r, 'U_final_pct_F0') < -band for r in rows),
        'exact_zero_selected_score_difference': len(selected),
        'zero_selected_both_same_prefix_step': sum(r['JOINT_selected_step'] == r['HEAD_ONLY_selected_step'] and val(r, 'JOINT_selected_step') <= val(r, 'fork') for r in selected),
        'clipping_control_same_band_class': sum(cls(val(r, 'U_final_pct_F0')) == cls(val(r, 'U_masked_pct_F0')) for r in rows),
        'clipping_control_same_raw_sign': sum((val(r, 'U_final_pct_F0') > 0) == (val(r, 'U_masked_pct_F0') > 0) for r in rows),
    }
    result = {
        'completed': True, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'script_sha256': sha(Path(__file__)), 'protected_hashes': protected,
        'active_study_fit_or_controller': active, 'inaccessible_python_processes': inaccessible,
        'guard_lock_exists': lock.exists(), 'current_memory': memory(),
        'resource': {'guards': len(statuses), 'samples': len(samples), 'finish_records': finishes,
            'minimum_available_ram_gib': min(r['available_ram_gib'] for r in samples),
            'minimum_available_commit_gib': min(r['available_commit_gib'] for r in samples),
            'maximum_child_rss_gib': max(r['child_tree_rss_gib'] for r in samples),
            'maximum_gpu_memory_mib': max(g['memory_used_mib'] for g in gpu),
            'maximum_gpu_temperature_c': max(g['temperature_c'] for g in gpu)},
        'descriptive_counts': extra,
        'E_outcomes_used': False,
    }
    with destination.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
