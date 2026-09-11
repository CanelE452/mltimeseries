"""CPU-only independent score, selection, pairing, and gate audit."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
from collections import Counter
from datetime import datetime
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.peft_optimization_control_v1.run import ROOT, RUN, sha, save

OUT = ROOT / 'results/peft_optimization_control_v1'


def score(prediction, target, scale, quantiles):
    pred = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    assert pred.shape == (len(target), 2, 21, 48) and target.shape == (len(target), 2, 48)
    assert np.isfinite(pred).all() and np.all(np.asarray(scale) > 0)
    assert np.all(np.diff(pred, axis=2) >= 0)
    mask = np.isfinite(target)
    assert np.all(mask.sum(axis=(0, 2)) > 0)
    error = np.where(mask[:, :, None, :], target[:, :, None, :] - pred, 0.)
    q = np.asarray(quantiles, dtype=np.float64)[None, None, :, None]
    loss = np.maximum(q * error, (q - 1) * error)
    return float(2 * np.mean(loss.sum(axis=(0, 3)) / mask.sum(axis=(0, 2))[:, None] / np.asarray(scale, dtype=np.float64)[:, None]))


def main():
    contract = json.loads((RUN / 'contract.json').read_text())
    completed = json.loads((RUN / 'completed.json').read_text())
    assert completed['completed'] and sha(RUN / 'selection.json') == completed['selection_sha256']
    for path, digest in {**contract['source_hashes'], **contract['original_input_hashes']}.items():
        assert sha(ROOT / path) == digest, path
    OUT.mkdir(parents=True, exist_ok=True)
    selection = json.loads((RUN / 'selection.json').read_text())
    fits = {}
    for entry in contract['jobs']:
        result = json.loads((RUN / entry['key'] / 'output/result.json').read_text())
        assert result['completed'] and result['frozen_verified'] and not result['holdout_opened']
        assert result['job'] == entry['job'] and result['steps'] == 180
        assert result['identity_error'] < 1e-5 and result['contract_sha256'] == sha(RUN / 'contract.json')
        history = {r['step']: r['score'] for r in result['history']}
        for regime, winner in result['regimes'].items():
            schedule = contract['schedules'][entry['job']['condition']][regime]
            best = min(schedule, key=lambda s: (history[s], s))
            assert best == winner['best_step'] and history[best] == winner['best_score']
            assert winner['replay_verified']
            assert sha(RUN / entry['key'] / 'output' / (regime + '.pt')) == winner['checkpoint_sha256']
        fits[entry['key']] = result
    for ds in ('bike', 'household'):
        for condition in ('FULL90', 'SPREAD30', 'RECENT30'):
            for seed in (25000, 25001):
                matched = [r for r in fits.values() if (r['job']['dataset'], r['job']['condition'], r['job']['seed']) == (ds, condition, seed)]
                assert len({r['initial_head_hash'] for r in matched}) == 1
                assert len({r['sample_index_sha256'] for r in matched}) == 1
                assert len({r['history'][0]['score'] for r in matched}) == 1
    f0, targets = {}, {}
    for ds in ('bike', 'household'):
        spec = contract['reference']['datasets'][ds]
        assert sha(ROOT / spec['holdout_data_path']) == spec['holdout_data_sha256']
        with np.load(ROOT / spec['holdout_data_path'], allow_pickle=False) as z:
            scales = z['fit_std'][z['target_indices']]
            quantiles = z['quantiles']
        with np.load(ROOT / f'runs/peft_fullft_reference_v3/forecasts/{ds}/F0/Epredictions.npz', allow_pickle=False) as z:
            f0[ds] = score(z['predictions'], z['target'], scales, quantiles)
            targets[ds] = (z['target'].copy(), z['origins'].copy())
    errors, rows = [], []
    first_eval = None
    for entry in selection:
        job, regime = entry['job'], entry['regime']
        group = [r for r in fits.values() if all(r['job'][k] == job[k] for k in ('dataset', 'condition', 'arm'))]
        candidates = {recipe: np.mean([r['regimes'][regime]['best_score'] for r in group if r['job']['recipe'] == recipe]) for recipe in (0, 1)}
        assert job['recipe'] == min(candidates, key=lambda r: (candidates[r], -r))
        assert candidates[job['recipe']] == entry['recipe_mean_val']
        key = entry['key'].replace('fits/', 'eval/' + regime + '/')
        result = json.loads((RUN / key / 'output/result.json').read_text())
        status = json.loads((RUN / key / 'guard/status.json').read_text())
        started = datetime.fromisoformat(status['started_at']).timestamp()
        first_eval = min(first_eval, started) if first_eval is not None else started
        assert result['completed'] and result['job'] == job and result['regime'] == regime
        assert Path(result['fit']) == RUN / entry['key'] / 'output'
        with np.load(RUN / key / 'output/predictions.npz', allow_pickle=False) as z:
            np.testing.assert_equal(z['target'], targets[job['dataset']][0])
            np.testing.assert_equal(z['origins'], targets[job['dataset']][1])
            actual = score(z['prediction'], z['target'], z['scale'], z['quantiles'])
        errors.append(abs(actual - result['score']))
        chosen = fits[entry['key']]['regimes'][regime]
        rows.append({'dataset': job['dataset'], 'condition': job['condition'], 'regime': regime, 'arm': job['arm'], 'seed': job['seed'], 'recipe': job['recipe'], 'lr': job['head_lr'], 'best_step': chosen['best_step'], 'val_score': chosen['best_score'], 'score': actual, 'f0_score': f0[job['dataset']], 'gain_pct_f0': 100*(f0[job['dataset']]-actual)/f0[job['dataset']], 'fit_key': entry['key']})
    assert max(errors) < 1e-10
    assert (RUN / 'selection.json').stat().st_mtime <= first_eval
    effects, gate_details = {}, {}
    for regime in ('UPDATE', 'EXPOSURE'):
        effects[regime] = {}
        for ds in ('bike', 'household'):
            values = {}
            for condition in ('FULL90', 'SPREAD30', 'RECENT30'):
                gains = []
                for seed in (25000, 25001):
                    pair = {r['arm']: r['score'] for r in rows if (r['dataset'], r['condition'], r['regime'], r['seed']) == (ds, condition, regime, seed)}
                    gains.append(100*(pair['MLP']-pair['ALL'])/f0[ds])
                values[condition] = {'by_seed': gains, 'mean': float(np.mean(gains))}
            values['full_minus_spread'] = np.subtract(values['FULL90']['by_seed'], values['SPREAD30']['by_seed']).tolist()
            values['recent_minus_spread'] = np.subtract(values['RECENT30']['by_seed'], values['SPREAD30']['by_seed']).tolist()
            effects[regime][ds] = values
        bike = effects[regime]['bike']
        gate_details[regime] = {'full_positive_both': min(bike['FULL90']['by_seed']) > 0, 'full_mean_ge_1': bike['FULL90']['mean'] >= 1, 'difference_positive_both': min(bike['full_minus_spread']) > 0, 'difference_mean_ge_1': float(np.mean(bike['full_minus_spread'])) >= 1}
    passed = all(all(g.values()) for g in gate_details.values())
    statuses = [json.loads(p.read_text()) for p in RUN.glob('**/guard/status.json')]
    assert len(statuses) == 98 and all(s['completed'] and not s['reasons'] for s in statuses)
    resources = [json.loads(line) for p in RUN.glob('**/guard/resource_log.jsonl') for line in p.read_text().splitlines()]
    resources = [r for r in resources if 'available_commit_gib' in r]
    gpu = [g for r in resources for g in r.get('gpus', []) or []]
    resource_summary = {'guard_jobs': len(statuses), 'minimum_ram_gib': min(r['available_ram_gib'] for r in resources), 'minimum_commit_gib': min(r['available_commit_gib'] for r in resources), 'maximum_gpu_mib': max(g['memory_used_mib'] for g in gpu), 'maximum_temperature_c': max(g['temperature_c'] for g in gpu)}
    summary = {'completed': True, 'effects': effects, 'gate_details': gate_details, 'gate_passed': passed, 'independent_score_max_error': max(errors), 'fits_verified': len(fits), 'evaluations_verified': len(rows), 'resource_audit': resource_summary, 'selected_steps': dict(Counter(str(r['best_step']) for r in rows)), 'selected_lr_counts': dict(Counter(str(r['lr']) for r in rows)), 'fresh_test': False, 'limits': ['Two sources and two seeds; discovery E previously exposed.', 'Same MLP head but unequal total parameter counts.', 'Expected exposure matching does not match unique labels, optimization paths, or time coverage.', 'Two learning rates do not establish global optimization.', 'Gate is a prespecified practical follow-up rule, not a significance test.']}
    save(OUT / 'summary.json', summary)
    with (OUT / 'metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    old = json.loads((ROOT / 'results/peft_r1_validation_v1/summary.json').read_text())['effects']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for ax, ds in zip(axes, ('bike', 'household')):
        for label, color, offset in [('OLD', '#999999', -.23), ('UPDATE', '#0072B2', 0), ('EXPOSURE', '#D55E00', .23)]:
            means = [old[ds][c]['mean'] if label == 'OLD' else effects[label][ds][c]['mean'] for c in ('FULL90', 'SPREAD30', 'RECENT30')]
            ax.bar(np.arange(3)+offset, means, width=.22, color=color, label=label)
            if label != 'OLD':
                for i, c in enumerate(('FULL90', 'SPREAD30', 'RECENT30')):
                    ax.scatter([i+offset]*2, effects[label][ds][c]['by_seed'], color='black', s=17, zorder=3)
        ax.axhline(0, color='black', linewidth=.7)
        ax.set(xticks=np.arange(3), xticklabels=['FULL90', 'SPREAD30', 'RECENT30'], title=ds.title(), ylabel='Additional LoRA gain (% of F0 score)')
        ax.legend(fontsize=8)
    fig.suptitle('Same head: original vs optimization controls\nDots are two training seeds, not confidence intervals')
    fig.savefig(OUT / 'matched_head_gain.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout='constrained')
    for i, ds in enumerate(('bike', 'household')):
        for j, condition in enumerate(('FULL90', 'SPREAD30', 'RECENT30')):
            ax = axes[i, j]
            for arm, color in [('MLP', '#0072B2'), ('ALL', '#D55E00')]:
                for recipe, style in [(0, '-'), (1, '--')]:
                    group = [r for r in fits.values() if (r['job']['dataset'], r['job']['condition'], r['job']['arm'], r['job']['recipe']) == (ds, condition, arm, recipe)]
                    x = [h['step'] for h in group[0]['history']]
                    y = np.mean([[100*(r['history'][0]['score']-h['score'])/r['history'][0]['score'] for h in r['history']] for r in group], axis=0)
                    ax.plot(x, y, style, color=color, label=f'{arm} {1e-4 if recipe == 0 else 1e-5:g}')
            ax.axhline(0, color='black', linewidth=.6)
            ax.set(title=f'{ds.title()} / {condition}', xlabel='Updates', ylabel='V gain (% of F0 V score)')
            ax.legend(fontsize=7)
    fig.suptitle('Mean validation trajectories; checkpoint candidates differ by regime')
    fig.savefig(OUT / 'validation_trajectories.png', dpi=180)
    plt.close(fig)
    units = json.loads((RUN / 'exposure_unit_audit.json').read_text())
    unit_rows = [r for r in units['rows'] if r['regime'] == 'EXPOSURE']
    fig, ax = plt.subplots(figsize=(8, 4.8), layout='constrained')
    x = np.arange(3)
    for offset, key, label, color in [(-.18, 'expected_origin_exposures', 'Per origin', '#0072B2'), (.18, 'average_target_hour_exposures_before_missing_mask', 'Per available target hour', '#D55E00')]:
        bars = ax.bar(x+offset, [r[key] for r in unit_rows], width=.35, label=label, color=color)
        ax.bar_label(bars, fmt='%.2f', padding=3)
    ax.set(xticks=x, xticklabels=[r['condition'] for r in unit_rows], ylabel='Mean repetitions at EXPOSURE budget cap', ylim=(0, 38), title='Matching origin exposure leaves target-overlap differences')
    ax.legend(loc='upper right', fontsize=9)
    fig.supxlabel('Deterministic schedule audit, before missing-label masks; not a performance result', fontsize=9)
    fig.savefig(OUT / 'exposure_units.png', dpi=180)
    plt.close(fig)
    save(OUT / 'exposure_unit_audit.json', units)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
