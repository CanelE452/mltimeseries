"""Independent future-score and counterfactual rule/cost audit."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
from datetime import datetime
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.peft_optimization_control_v1.analyse import score
from experiments.peft_overlap_transfer_v1.run import ROOT, RUN, sha, save, verify

OUT = ROOT / 'results/peft_overlap_transfer_v1'
CONDITIONS = ('FULL90', 'SPREAD30', 'RECENT30')
SEEDS = (26000, 26001)
RULES = ('ALWAYS_ALL', 'ALWAYS_MLP', 'COUNT', 'OVERLAP')


def chosen_arm(rule, condition, plan):
    if rule == 'ALWAYS_ALL':
        return 'ALL'
    if rule == 'ALWAYS_MLP':
        return 'MLP'
    return plan['features'][condition][rule]


def main():
    plan = json.loads((RUN / 'plan.json').read_text())
    done = json.loads((RUN / 'completed.json').read_text())
    assert done['completed'] and done['selection_sha256'] == sha(RUN / 'selection.json')
    assert done['plan_sha256'] == sha(RUN / 'plan.json')
    verify(plan)
    clock = json.loads((RUN / 'analysis_clock_contract.json').read_text())
    assert sha(__file__) == clock['analysis_sha256']
    assert sha(ROOT / 'experiments/peft_overlap_transfer_v1/COST_CLOCK.md') == clock['note_sha256']
    selections = json.loads((RUN / 'selection.json').read_text())
    assert len(selections) == 24
    f0, targets, errors, starts = {}, {}, [], []
    contracts = {c: json.loads((RUN / (c + '_contract.json')).read_text()) for c in CONDITIONS}
    for c, contract in contracts.items():
        assert contract['plan_sha256'] == sha(RUN / 'plan.json')
        assert contract['prepared_manifest_sha256'] == sha(RUN / 'prepared/manifest.json')
        for spec in contract['reference']['datasets'].values():
            for split in ('fit', 'holdout'):
                assert sha(ROOT / spec[split + '_data_path']) == spec[split + '_data_sha256']
    for ds in ('bike', 'household'):
        parent = RUN / 'f0' / ds
        record = json.loads((parent / 'output/result.json').read_text())
        assert record['completed'] and record['identity_error'] < 1e-5
        assert record['contract_sha256'] == sha(RUN / 'FULL90_contract.json')
        with np.load(parent / 'output/predictions.npz', allow_pickle=False) as z:
            f0[ds] = score(z['prediction'], z['target'], z['scale'], z['quantiles'])
            targets[ds] = {k: z[k].copy() for k in ('target', 'origins', 'scale', 'quantiles')}
        errors.append(abs(f0[ds] - record['score']))
        starts.append(datetime.fromisoformat(json.loads((parent / 'guard/status.json').read_text())['started_at']).timestamp())
    rows, fits = [], {}
    for entry in selections:
        assert {k: entry[k] for k in ('job', 'key', 'contract')} in plan['jobs']
        job, key = entry['job'], entry['key']
        fit = json.loads((RUN / key / 'output/result.json').read_text())
        contract = contracts[job['condition']]
        assert fit['completed'] and fit['job'] == job and fit['frozen_verified'] and not fit['holdout_opened']
        assert fit['contract_sha256'] == sha(RUN / entry['contract']) and fit['steps'] == contract['steps']
        assert fit['identity_error'] < 1e-5 and set(fit['regimes']) == {'EXPOSURE'}
        winner = fit['regimes']['EXPOSURE']
        assert winner == entry['chosen'] and winner['replay_verified']
        assert sha(RUN / key / 'output/EXPOSURE.pt') == winner['checkpoint_sha256']
        schedule = contract['schedules'][job['condition']]['EXPOSURE']
        history = {h['step']: h['score'] for h in fit['history']}
        assert set(history) == set(schedule)
        best = min(schedule, key=lambda s: (history[s], s))
        assert best == winner['best_step'] and history[best] == winner['best_score']
        fits[key] = fit
        parent = RUN / key.replace('fits/', 'eval/')
        record = json.loads((parent / 'output/result.json').read_text())
        assert record['completed'] and record['job'] == job and record['regime'] == 'EXPOSURE'
        assert record['contract_sha256'] == sha(RUN / entry['contract']) and Path(record['fit']) == RUN / key / 'output'
        with np.load(parent / 'output/predictions.npz', allow_pickle=False) as z:
            for field, expected in targets[job['dataset']].items():
                np.testing.assert_equal(z[field], expected)
            actual = score(z['prediction'], z['target'], z['scale'], z['quantiles'])
        errors.append(abs(actual-record['score']))
        starts.append(datetime.fromisoformat(json.loads((parent / 'guard/status.json').read_text())['started_at']).timestamp())
        rows.append({'dataset': job['dataset'], 'condition': job['condition'], 'arm': job['arm'], 'seed': job['seed'], 'lr': job['head_lr'], 'score': actual, 'f0_score': f0[job['dataset']], 'gain_pct_F0': 100*(f0[job['dataset']]-actual)/f0[job['dataset']], 'best_step': best, 'fit_seconds': json.loads((RUN / key / 'guard/status.json').read_text())['elapsed_seconds'], 'internal_fit_seconds': fit['seconds'], 'trainable': fit['trainable']})
    assert (RUN / 'selection.json').stat().st_mtime <= min(starts)
    assert (RUN / 'analysis_clock_contract.json').stat().st_mtime <= min(starts)
    assert max(errors) < 1e-10
    for ds in ('bike', 'household'):
        for condition in CONDITIONS:
            for seed in SEEDS:
                pair = [f for f in fits.values() if (f['job']['dataset'], f['job']['condition'], f['job']['seed']) == (ds, condition, seed)]
                assert len(pair) == 2
                for field in ('initial_head_hash', 'sample_index_sha256'):
                    assert pair[0][field] == pair[1][field], field
                assert pair[0]['history'][0]['score'] == pair[1]['history'][0]['score']
    for arm in ('MLP', 'ALL'):
        assert len({f['frozen_hash'] for f in fits.values() if f['job']['arm'] == arm}) == 1
    pairs = {(r['dataset'], r['condition'], r['seed'], r['arm']): r for r in rows}
    rule_rows, rule_summary, effects = [], {}, {}
    for ds in ('bike', 'household'):
        effects[ds] = {}
        for c in CONDITIONS:
            g = [100*(pairs[ds,c,s,'MLP']['score']-pairs[ds,c,s,'ALL']['score'])/f0[ds] for s in SEEDS]
            effects[ds][c] = {'by_seed': g, 'mean': float(np.mean(g))}
    for rule in RULES:
        by_seed = []
        for seed in SEEDS:
            cell_rows = []
            for ds in ('bike', 'household'):
                for c in CONDITIONS:
                    arm = chosen_arm(rule, c, plan)
                    chosen, baseline = pairs[ds,c,seed,arm], pairs[ds,c,seed,'ALL']
                    row = {'rule': rule, 'dataset': ds, 'condition': c, 'seed': seed, 'arm': arm, 'regret_pct_F0': 100*(chosen['score']-baseline['score'])/f0[ds], 'fit_seconds': chosen['fit_seconds'], 'all_fit_seconds': baseline['fit_seconds']}
                    cell_rows.append(row)
                    rule_rows.append(row)
            macro = float(np.mean([r['regret_pct_F0'] for r in cell_rows]))
            source = {ds: float(np.mean([r['regret_pct_F0'] for r in cell_rows if r['dataset'] == ds])) for ds in ('bike', 'household')}
            seconds = sum(r['fit_seconds'] for r in cell_rows)
            savings = 100*(1-seconds/sum(r['all_fit_seconds'] for r in cell_rows))
            gates = plan['gates']
            quality = macro <= gates['macro_regret_max_pct_F0'] and max(source.values()) <= gates['source_regret_max_pct_F0']
            by_seed.append({'seed': seed, 'macro_regret_pct_F0': macro, 'source_regret_pct_F0': source, 'fit_seconds': seconds, 'savings_percent': savings, 'quality_passed': quality, 'cost_passed': savings > gates['fit_time_savings_min_percent'], 'passed': quality and savings > gates['fit_time_savings_min_percent']})
        rule_summary[rule] = {'by_seed': by_seed, 'quality_both': all(r['quality_passed'] for r in by_seed), 'passed_both': all(r['passed'] for r in by_seed)}
    dominating = [rule for rule in ('ALWAYS_MLP', 'COUNT') if rule_summary[rule]['quality_both'] and all(rule_summary[rule]['by_seed'][i]['fit_seconds'] < rule_summary['OVERLAP']['by_seed'][i]['fit_seconds'] for i in range(2))]
    statuses = [json.loads(p.read_text()) for p in RUN.glob('**/guard/status.json')]
    assert len(statuses) == 51 and all(s['completed'] and not s['reasons'] for s in statuses)
    resources = [json.loads(line) for p in RUN.glob('**/guard/resource_log.jsonl') for line in p.read_text().splitlines()]
    resources = [r for r in resources if 'available_commit_gib' in r]
    gpu = [g for r in resources for g in r.get('gpus', []) or []]
    resource_summary = {'guard_jobs': len(statuses), 'minimum_ram_gib': min(r['available_ram_gib'] for r in resources), 'minimum_commit_gib': min(r['available_commit_gib'] for r in resources), 'maximum_gpu_mib': max(g['memory_used_mib'] for g in gpu), 'maximum_temperature_c': max(g['temperature_c'] for g in gpu)}
    summary = {'completed': True, 'effects': effects, 'rules': rule_summary, 'overlap_gate_passed': rule_summary['OVERLAP']['passed_both'], 'cheaper_quality_passing_controls': dominating, 'overlap_unique_practical_support': rule_summary['OVERLAP']['passed_both'] and not dominating, 'f0_scores': f0, 'independent_score_max_error': max(errors), 'fits_verified': 24, 'forecasts_verified': 26, 'resources': resource_summary, 'campaign_fit_seconds': sum(r['fit_seconds'] for r in rows), 'campaign_invocation_seconds': done['invocation_seconds'], 'new_E_only': True, 'new_sources': False, 'limitations': ['Two familiar sources, one new E period each; two seeds are not independent datasets.', 'Overlap rule is a development heuristic and may encode the sampling condition; no causal identification.', 'Same head but unequal total capacity; fixed development LR may not be future optimal.', 'Time savings are retrospective counterfactuals, not realized research-campaign savings.', 'Practical thresholds are not statistical noninferiority tests; pretraining overlap unknown.']}
    OUT.mkdir(parents=True, exist_ok=False)
    save(OUT / 'summary.json', summary)
    for name, records in [('metrics.csv', rows), ('rule_metrics.csv', rule_rows)]:
        with (OUT / name).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    old = json.loads((ROOT / 'results/peft_optimization_control_v1/summary.json').read_text())['effects']['EXPOSURE']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout='constrained')
    for ax, ds in zip(axes, ('bike', 'household')):
        for offset, label, data, color in [(-.18, 'Development', old, '#999999'), (.18, 'New E period', effects, '#0072B2')]:
            ax.bar(np.arange(3)+offset, [data[ds][c]['mean'] for c in CONDITIONS], width=.34, label=label, color=color)
            for i,c in enumerate(CONDITIONS):
                ax.scatter([i+offset]*2, data[ds][c]['by_seed'], s=20, color='black', zorder=3)
        ax.axhline(0, color='black', linewidth=.7)
        ax.set(title=ds.title(), xticks=np.arange(3), xticklabels=CONDITIONS, ylabel='Additional LoRA gain (% of F0 score)')
        ax.legend(fontsize=8)
    fig.suptitle('Frozen adaptation recipe: transfer to a new evaluation period\nDots are two training seeds, not confidence intervals')
    fig.savefig(OUT / 'future_gain.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout='constrained')
    colors = ['#999999', '#56B4E9', '#009E73', '#D55E00']
    for i,rule in enumerate(RULES):
        for ax,key in zip(axes, ('macro_regret_pct_F0', 'savings_percent')):
            values = [s[key] for s in rule_summary[rule]['by_seed']]
            ax.bar(i, np.mean(values), color=colors[i])
            ax.scatter([i]*2, values, color='black', s=24, zorder=3)
    axes[0].axhline(.25, color='red', linestyle='--', label='Maximum regret: 0.25')
    axes[1].axhline(5, color='red', linestyle='--', label='Minimum savings: 5%')
    for ax in axes:
        ax.axhline(0, color='black', linewidth=.7)
        ax.set(xticks=range(4), xticklabels=['All LoRA', 'MLP only', 'Count', 'Overlap'])
        ax.legend(fontsize=8)
    axes[0].set(ylabel='Regret vs all LoRA (% of F0 score)', title='Accuracy cost (lower is better)')
    axes[1].set(ylabel='Counterfactual fit-time savings (%)', title='Adaptation cost (higher is better)')
    fig.suptitle('Past-only decision rules on six future source/condition cells\nPractical thresholds, not statistical noninferiority tests')
    fig.savefig(OUT / 'rule_tradeoff.png', dpi=180)
    plt.close(fig)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
