"""Read-only independent CSV/score audit; does not import the study implementation."""

import calendar
import csv
from datetime import date, timedelta, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results/peft_revision_entry_v1'
RAW = ROOT / 'data_external/alfred_revision_entry_v1/complete'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shift_month(event, n):
    year, zero_month = divmod(event.year * 12 + event.month - 1 + n, 12)
    return date(year, zero_month + 1, 1)


def mature_cutoff(event):
    shifted = shift_month(event, 6)
    return date(shifted.year, shifted.month, calendar.monthrange(shifted.year, shifted.month)[1])


def parse_intervals(path, name):
    intervals = {}
    with path.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            event = date.fromisoformat(row['period_start_date'])
            begin = date.fromisoformat(row['realtime_start_date'])
            finish = date.fromisoformat(row['realtime_end_date']) if row['realtime_end_date'] else date.max
            value = float(row[name])
            if begin > finish or not math.isfinite(value) or value <= 0:
                raise ValueError('Invalid independent raw interval')
            intervals.setdefault(event, []).append((begin, finish, value))
    return intervals


def level(intervals, event, cutoff):
    matches = [value for begin, finish, value in intervals[event] if begin <= cutoff <= finish]
    if len(matches) != 1:
        raise ValueError(f'Expected one independently matched interval: {event}, {cutoff}, {len(matches)}')
    return matches[0]


def growth(intervals, event, cutoff):
    return 100 * math.log(level(intervals, event, cutoff) / level(intervals, shift_month(event, -1), cutoff))


def first_growth(intervals, event):
    previous = shift_month(event, -1)
    intersections = [max(a, c) for a, b, _ in intervals[event] for c, d, _ in intervals[previous] if max(a, c) <= min(b, d)]
    first_date = min(intersections)
    return first_date, growth(intervals, event, first_date)


def execute():
    started = time.perf_counter()
    input_paths = [RESULTS / name for name in (
        'cpu_results.json', 'evaluation_predictions.json', 'ridge_identity.json', 'origin_audit.json', 'event_labels.csv')]
    selection_path = ROOT / 'runs/peft_revision_entry_v1/validation_selection.json'
    contract_path = ROOT / 'runs/peft_revision_entry_v1/cpu_contract.json'
    matched_path = RESULTS / 'posthoc_matched_config.json'
    input_paths += [selection_path, contract_path, matched_path]
    input_paths += [RAW / name / 'obs._by_real-time_period.csv' for name in ('PAYEMS', 'INDPRO')]
    before = {str(path.relative_to(ROOT)): sha(path) for path in input_paths}
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    results = json.loads((RESULTS / 'cpu_results.json').read_text(encoding='utf-8'))
    predictions = json.loads((RESULTS / 'evaluation_predictions.json').read_text(encoding='utf-8'))
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    matched = json.loads(matched_path.read_text(encoding='utf-8'))
    origins = json.loads((RESULTS / 'origin_audit.json').read_text(encoding='utf-8'))
    identities = json.loads((RESULTS / 'ridge_identity.json').read_text(encoding='utf-8'))
    with (RESULTS / 'event_labels.csv').open(encoding='utf-8', newline='') as handle:
        label_rows = list(csv.DictReader(handle))
    failures, maxima, counts = [], {}, {}

    def check(condition, label):
        counts['logical'] = counts.get('logical', 0) + 1
        if not condition:
            failures.append(label)

    def close(observed, expected, label, tolerance=1e-8):
        difference = float(np.max(np.abs(np.asarray(observed) - np.asarray(expected))))
        maxima[label] = max(maxima.get(label, 0.0), difference)
        counts[label] = counts.get(label, 0) + 1
        if not np.isfinite(difference) or difference > tolerance:
            failures.append(f'{label}: {difference}')

    for section in ('source_hashes', 'input_hashes', 'protected_hashes'):
        for name, expected in contract[section].items():
            check(sha(ROOT / name) == expected, f'Frozen hash mismatch: {name}')
    e_dates = [date(year, month, 1) for year in range(2020, 2025) for month in range(1, 13)]
    v_dates = [date(year, month, 1) for year in range(2016, 2019) for month in range(1, 13)]
    scales, normalized, posthoc, series_summary, matched_summary = {}, {}, {}, {}, {}
    for name in ('PAYEMS', 'INDPRO'):
        intervals = parse_intervals(RAW / name / 'obs._by_real-time_period.csv', name)
        historic = [growth(intervals, date(year, month, 1), mature_cutoff(date(year, month, 1)))
                    for year in range(1991, 2015) for month in range(1, 13)]
        centered = np.asarray(historic) - sum(historic) / len(historic)
        scales[name] = float(centered @ centered / len(historic))
        close(results['revision_qc'][name]['historical_variance'], scales[name], 'historical_variance', 1e-12)
        expected_truth = np.asarray([growth(intervals, event, mature_cutoff(event)) for event in e_dates])
        close(predictions[name]['truth_six_month_mature'], expected_truth, 'truth_from_raw_csv', 1e-10)
        own_labels = [row for row in label_rows if row['series'] == name]
        check(len(own_labels) == 408, f'Event label count: {name}')
        for row in own_labels:
            event = date.fromisoformat(row['target_event'])
            first_date, first_value = first_growth(intervals, event)
            check(row['first_release'] == first_date.isoformat(), f'First date mismatch: {name}/{event}')
            check(row['maturity_date'] == mature_cutoff(event).isoformat(), f'Maturity mismatch: {name}/{event}')
            close(float(row['first_growth']), first_value, 'first_growth_from_raw_csv', 1e-10)
            close(float(row['mature_growth']), growth(intervals, event, mature_cutoff(event)), 'all_mature_labels', 1e-10)
        normalized[name], posthoc[name] = {}, {}
        for arm, score in results['metrics'][name].items():
            point = np.asarray(predictions[name][arm])
            check(point.shape == (60,) and score['n'] == 60, f'E count: {name}/{arm}')
            error = point - expected_truth
            squared = error * error
            close(score['mse'], np.mean(squared), 'mse')
            close(score['mae'], np.mean(np.abs(error)), 'mae')
            normalized[name][arm] = squared / scales[name]
            close(score['normalized_mse'], np.mean(normalized[name][arm]), 'normalized_mse')
            posthoc[name][arm] = {
                'yearly_mse': {str(year): float(np.mean(squared[12*(year-2020):12*(year-2019)])) for year in range(2020, 2025)},
                'fraction_of_total_squared_error_in_2020': float(squared[:12].sum() / squared.sum()),
                'mse_2021_2024': float(np.mean(squared[12:])),
            }
        v_truth = np.asarray([growth(intervals, event, mature_cutoff(event)) for event in v_dates])
        matched_summary[name] = []
        for comparison in matched['series'][name]:
            computed_mse = {}
            for arm, saved_metrics in comparison['metrics'].items():
                error = np.asarray(comparison['predictions'][arm]) - expected_truth
                computed_mse[arm] = float(np.mean(error * error))
                close(saved_metrics['mse'], computed_mse[arm], 'matched_mse')
                close(saved_metrics['mae'], np.mean(np.abs(error)), 'matched_mae')
            improvement = 100 * (computed_mse['FIRST_FIXED_X'] - computed_mse['REVISED_FIXED_X']) / computed_mse['FIRST_FIXED_X']
            close(comparison['same_config_revision_improvement_percent'], improvement, 'matched_improvement_percent')
            prediction_delta = np.max(np.abs(np.asarray(comparison['predictions']['FIRST_FIXED_X']) - np.asarray(comparison['predictions']['REVISED_FIXED_X'])))
            close(comparison['revision_max_prediction_difference'], prediction_delta, 'matched_prediction_difference')
            matched_summary[name].append({'lam': comparison['lam'], 'window': comparison['window'],
                                          'mse': computed_mse, 'revision_improvement_percent': improvement})
        for arm, chosen in selection['selected'][name].items():
            candidates = selection['all_scores'][name][arm]
            expected = sorted(candidates, key=lambda c: (c['mse'], c['lam'], 0 if c['window'] is None else 1))[0]
            check(chosen == expected, f'Selection tie/minimum: {name}/{arm}')
            v_errors = np.asarray(selection['predictions'][name][arm]) - v_truth
            close(chosen['mse'], np.dot(v_errors, v_errors) / 36, 'selected_validation_mse')
            check(results['selection'][name][arm] == chosen, f'Published selection: {name}/{arm}')
        series_summary[name] = {'evaluation_origins': 60, 'historical_scale_rows': 288,
                                'historical_variance': scales[name],
                                'best_reported_mse_arm': min(results['metrics'][name], key=lambda arm: results['metrics'][name][arm]['mse'])}

    base = np.stack([normalized[name]['FIRST_FIXED_X'] for name in normalized])
    bootstrap_starts = np.random.default_rng(1908).integers(0, 60, size=(2000, 5))
    indices = ((bootstrap_starts[:, :, None] + np.arange(12)) % 60).reshape(2000, 60)
    for arm, pooled in results['pooled'].items():
        loss = np.stack([normalized[name][arm] for name in normalized])
        close(pooled['normalized_mse'], loss.mean(), 'pooled_normalized_mse')
        close(pooled['improvement_percent_vs_first'], 100 * (base - loss).mean() / base.mean(), 'pooled_improvement_percent')
        paired_difference = (base - loss).mean(axis=0)
        boot_means = paired_difference[indices].mean(axis=1)
        interval = np.percentile(boot_means, [2.5, 97.5])
        close(pooled['normalized_mse_difference_95pct_block_ci'], interval, 'paired_circular_block_ci')
    check(selection['evaluation_used_for_selection'] is False, 'Selection E-use flag')
    check(selection['selection_cutoff'] == '2019-07-01', 'Selection cutoff')
    check({(row['series'], row['target_event']) for row in origins} == {(name, e.isoformat()) for name in normalized for e in e_dates}, 'Origin coverage')
    for row in origins:
        event = date.fromisoformat(row['target_event'])
        check(row['origin_date'] == event.isoformat(), 'Origin timestamp')
        check(row['vintage_cutoff'] == (event - timedelta(days=1)).isoformat(), 'Origin cutoff')
        check(row['latest_available_event'] == shift_month(event, -2).isoformat() and row['event_step_gap'] == 2, 'Publication step gap')
    check(len(identities) == 192 and all(row['passed'] and row['n_rows'] == row['expected_rows'] for row in identities), 'Reported external identity results')
    for path in input_paths:
        check(sha(path) == before[str(path.relative_to(ROOT))], f'Input mutated: {path}')
    return {'status': 'PASS' if not failures else 'FAIL', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'source_sha256': sha(Path(__file__)), 'input_hashes': before,
            'implementation_imports': 'No imports from run.py, data.py or model.py',
            'scope': 'Raw CSV interval lookup; first/mature labels; historical scales; saved prediction metrics; paired CI; saved V argmin and scores. Does not refit every candidate or independently replay ridge state.',
            'counts': counts, 'maximum_absolute_errors': maxima, 'failures': failures,
            'series': series_summary, 'posthoc_descriptive_only': posthoc,
            'posthoc_matched_config_metric_audit': matched_summary,
            'posthoc_limit': 'Year splits and 2020 concentration are descriptive after seeing E; do not change original gate or replace full E.',
            'wall_seconds': time.perf_counter() - started}


if __name__ == '__main__':
    report = execute()
    path = RESULTS / 'independent_audit.json'
    with path.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({key: report[key] for key in ('status', 'counts', 'maximum_absolute_errors', 'series', 'wall_seconds')}))
    raise SystemExit(0 if report['status'] == 'PASS' else 1)
