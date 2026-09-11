"""Descriptive join of V2 sensitivity and already exposed E regret; no selector fit."""
import csv
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2] / 'results/peft_mechanism_diagnostics_v1/hospital_strength'
    with (root / 'hospital_alpha_series.csv').open() as f:
        source = {(r['seed'], r['series_id']): r for r in csv.DictReader(f)}
    with (root / 'month_sensitivity.csv').open() as f:
        sensitivity = list(csv.DictReader(f))
    assert len(source) == len(sensitivity) == 1534
    assert set(source) == {(r['seed'], r['series_id']) for r in sensitivity}
    rows = []
    for seed in ('24000', '24001'):
        for stable in (True, False):
            group = [r for r in sensitivity if r['seed'] == seed and (int(r['changed_months']) == 0) == stable]
            values = [float(source[(r['seed'], r['series_id'])]['eval_regret_individual_minus_global']) for r in group]
            rows.append(dict(seed=int(seed), stable_all_month_deletions=stable, n=len(group),
                             mean_eval_regret_individual_minus_global=sum(values) / len(values),
                             sum_eval_regret=sum(values)))
    report = {'scope': 'Posthoc descriptive join; not a tested selection rule or causal separation of noise and drift', 'rows': rows}
    (root / 'stability_transfer_descriptive.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
