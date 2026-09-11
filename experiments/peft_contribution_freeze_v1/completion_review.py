"""Descriptive mechanism and resource audit; does not choose methods or gates."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/peft_contribution_freeze_v1'
OUT = ROOT / 'results/peft_contribution_freeze_v1'


def read(path):
    return json.loads(path.read_text())


def same_prediction(a, b):
    with np.load(a, allow_pickle=False) as x, np.load(b, allow_pickle=False) as y:
        return bool(np.array_equal(x['prediction'], y['prediction'], equal_nan=True))


def main():
    assert read(RUN / 'completed.json')['completed']
    assert read(RUN / 'timing_completed.json')['completed']
    assert read(RUN / 'independent_timing_analysis_audit.json')['completed']
    output = OUT / 'completion_review.json'
    assert not output.exists()
    plan = read(RUN / 'plan.json')
    rows = []
    for e in plan['jobs']:
        j = e['job']
        if j['arm'] != 'CONTRIB_FREEZE':
            continue
        p = RUN / e['key']
        c = read(p / 'output/result.json')
        full = RUN / e['key'].replace('/CONTRIB_FREEZE/', '/FULL/')
        fixed = RUN / e['key'].replace('/CONTRIB_FREEZE/', '/FIXED_FREEZE/')
        rows.append({**{k: j[k] for k in ('dataset', 'condition', 'seed')},
            'frozen_step': c['frozen_step'], 'selected_step': c['best_step'],
            'cap': c['cap'], 'adapter_updates': c['adapter_updates'],
            'toggle_checks': c['toggle_checks'],
            'selected_before_freeze': c['frozen_step'] is not None and c['best_step'] < c['frozen_step'],
            'selected_after_freeze': c['frozen_step'] is not None and c['best_step'] > c['frozen_step'],
            'E_prediction_exact_FULL': same_prediction(
                RUN / e['key'].replace('fits/', 'eval/') / 'output/predictions.npz',
                RUN / e['key'].replace('fits/', 'eval/').replace('/CONTRIB_FREEZE/', '/FULL/') / 'output/predictions.npz'),
            'E_prediction_exact_FIXED': same_prediction(
                RUN / e['key'].replace('fits/', 'eval/') / 'output/predictions.npz',
                RUN / e['key'].replace('fits/', 'eval/').replace('/CONTRIB_FREEZE/', '/FIXED_FREEZE/') / 'output/predictions.npz'),
            'V_prediction_exact_FULL': same_prediction(p / 'output/val_predictions.npz', full / 'output/val_predictions.npz'),
            'V_prediction_exact_FIXED': same_prediction(p / 'output/val_predictions.npz', fixed / 'output/val_predictions.npz')})
    statuses = list(RUN.rglob('guard/status.json'))
    assert len(statuses) == 112, len(statuses)
    first_E = min(datetime.fromisoformat(read(p)['started_at']) for p in statuses
                  if p.relative_to(RUN).parts[0] in ('eval', 'f0'))
    last_fit = max(datetime.fromisoformat(read(p)['finished_at']) for p in statuses
                   if p.relative_to(RUN).parts[0] == 'fits')
    timing_sealed = datetime.fromisoformat(read(RUN / 'timing_contract.json')['created_utc'])
    assert last_fit < first_E and timing_sealed < first_E
    categories = {}
    all_samples = []
    finish_records = 0
    for path in statuses:
        status = read(path)
        assert status['completed'] and status['returncode'] == 0 and not status['reasons'], str(path)
        category = path.relative_to(RUN).parts[0]
        records = [json.loads(line) for line in path.with_name('resource_log.jsonl').read_text().splitlines()]
        terminal = [s for s in records if 'gpus' not in s]
        assert len(terminal) == 1 and terminal[0].get('event') == 'finish' and terminal[0].get('state') == 'completed'
        finish_records += len(terminal)
        samples = [s for s in records if 'gpus' in s]
        all_samples.extend(samples)
        item = categories.setdefault(category, {'guards': 0, 'seconds': 0., 'samples': 0})
        item['guards'] += 1
        item['seconds'] += status['elapsed_seconds']
        item['samples'] += len(samples)
    gpu = [g for s in all_samples for g in s['gpus']]
    data = {'completed': True, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'role': 'Post-outcome descriptive explanation; no additional selection or statistical gate.',
        'chronology': {'last_original_fit_finished_utc': last_fit.isoformat(),
            'timing_contract_sealed_utc': timing_sealed.isoformat(),
            'first_E_started_utc': first_E.isoformat(), 'ordering_verified': True},
        'candidate_cells': rows,
        'candidate_counts': {k: sum(r[k] for r in rows) for k in (
            'selected_before_freeze', 'selected_after_freeze', 'E_prediction_exact_FULL',
            'E_prediction_exact_FIXED', 'V_prediction_exact_FULL', 'V_prediction_exact_FIXED')},
        'freeze_count': sum(r['frozen_step'] is not None for r in rows),
        'resource': {'guards': len(statuses), 'samples': len(all_samples), 'finish_event_records': finish_records, 'categories': categories,
            'minimum_available_ram_gib': min(s['available_ram_gib'] for s in all_samples),
            'minimum_available_commit_gib': min(s['available_commit_gib'] for s in all_samples),
            'maximum_child_rss_gib': max(s['child_tree_rss_gib'] for s in all_samples),
            'maximum_gpu_memory_mib': max(g['memory_used_mib'] for g in gpu),
            'maximum_gpu_temperature_c': max(g['temperature_c'] for g in gpu)}}
    output.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    with (OUT / 'mechanism_diagnostics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({k: v for k, v in data.items() if k != 'candidate_cells'}, indent=2))


if __name__ == '__main__':
    main()
