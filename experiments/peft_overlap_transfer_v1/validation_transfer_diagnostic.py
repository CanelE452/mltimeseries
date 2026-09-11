"""Descriptive V-to-E utility transfer; no fitted rule or decision changes."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.peft_overlap_transfer_v1.run import ROOT, RUN, sha, save

OUT = ROOT / 'results/peft_overlap_transfer_v1'


def main():
    seal = json.loads((RUN / 'validation_diagnostic_contract.json').read_text())
    assert sha(__file__) == seal['code_sha256']
    assert (RUN / 'validation_diagnostic_contract.json').stat().st_mtime <= (RUN / 'selection.json').stat().st_mtime
    main_result = json.loads((OUT / 'summary.json').read_text())
    assert main_result['completed'] and main_result['forecasts_verified'] == 26
    metrics = list(csv.DictReader((OUT / 'metrics.csv').open()))
    selection = json.loads((RUN / 'selection.json').read_text())
    records = []
    for ds in ('bike', 'household'):
        for c in ('FULL90', 'SPREAD30', 'RECENT30'):
            for seed in (26000, 26001):
                entries = [e for e in selection if (e['job']['dataset'], e['job']['condition'], e['job']['seed']) == (ds, c, seed)]
                fits = {e['job']['arm']: json.loads((RUN / e['key'] / 'output/result.json').read_text()) for e in entries}
                f0 = fits['MLP']['history'][0]['score']
                assert f0 == fits['ALL']['history'][0]['score'] and f0 > 0
                v_gain = 100*(fits['MLP']['regimes']['EXPOSURE']['best_score']-fits['ALL']['regimes']['EXPOSURE']['best_score'])/f0
                pair = {m['arm']: m for m in metrics if (m['dataset'], m['condition'], int(m['seed'])) == (ds, c, seed)}
                e_gain = 100*(float(pair['MLP']['score'])-float(pair['ALL']['score']))/float(pair['ALL']['f0_score'])
                records.append({'dataset': ds, 'condition': c, 'seed': seed, 'V_gain_pct_F0V': v_gain, 'E_gain_pct_F0E': e_gain, 'V_positive_E_nonpositive': v_gain > 0 and e_gain <= 0})
    result = {'records': records, 'positive_V_to_nonpositive_E_pairs': sum(r['V_positive_E_nonpositive'] for r in records), 'pairs': len(records), 'decision_or_threshold_changed': False, 'limits': ['V selected checkpoints and is optimistic.', 'V and E use their own F0 denominators.', 'Selection bias, optimization and distribution shift are not separated.', 'Two familiar sources and two seeds; pairs are not independent datasets.']}
    save(OUT / 'validation_transfer.json', result)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout='constrained')
    colors = {'FULL90': '#0072B2', 'SPREAD30': '#D55E00', 'RECENT30': '#009E73'}
    for ax, ds in zip(axes, ('bike', 'household')):
        for c, color in colors.items():
            group = [r for r in records if (r['dataset'],r['condition']) == (ds,c)]
            ax.scatter([r['V_gain_pct_F0V'] for r in group], [r['E_gain_pct_F0E'] for r in group], s=48, color=color, label=c)
            for r in group:
                ax.annotate(str(r['seed']-26000), (r['V_gain_pct_F0V'],r['E_gain_pct_F0E']), xytext=(5,3), textcoords='offset points', fontsize=8)
        ax.axhline(0, color='black', linewidth=.7)
        ax.axvline(0, color='black', linewidth=.7)
        ax.set(title=ds.title(), xlabel='V additional LoRA gain (% of F0 V)', ylabel='E additional LoRA gain (% of F0 E)')
        ax.legend(fontsize=8)
    fig.suptitle('Selected validation benefit versus new-period benefit\nLabels 0/1 identify seeds; V is used for selection and is optimistic')
    fig.savefig(OUT / 'validation_transfer.png', dpi=180)
    plt.close(fig)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
