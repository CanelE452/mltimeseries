"""Recompute all selected D losses and prespecified descriptive contrasts."""
import csv
import numpy as np
from experiments.peft_lora_only_diagnostic_v1.audit import ROOT, CODE, RUN, OUT, read, write, sha, verify_plan, verify_old
from experiments.peft_head_convergence_v1.analyse import prediction, score_arrays, score_npz


def csv_write(path, rows, fields=None):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def resources():
    guards = [read(p) for p in RUN.glob('**/guard/status.json')]
    samples = []
    for p in RUN.glob('**/guard/resource_log.jsonl'):
        samples.extend(read_line for line in p.read_text().splitlines()
                       if 'available_ram_gib' in (read_line := __import__('json').loads(line)))
    fits = [read(p) for p in RUN.glob('fit/*/output/result.json')]
    forecasts = [read(p) for p in RUN.glob('forecast/*/*/output/result.json')]
    records = fits+forecasts
    payload = {'guard_count': len(guards), 'all_guards_completed': all(g['completed'] for g in guards),
               'fit_count': len(fits), 'forecast_count': len(forecasts),
               'zero_update_gate_count': len(list(RUN.glob('gate/*/output/result.json'))),
               'guard_wall_seconds_sum': sum(g['elapsed_seconds'] for g in guards),
               'fit_wall_seconds_sum': sum(r['seconds'] for r in fits),
               'fit_trajectory_seconds_sum': sum(r['trajectory_seconds'] for r in fits),
               'peak_gpu_allocated_bytes': max((r.get('peak_allocated', 0) for r in records), default=0),
               'peak_gpu_reserved_bytes': max((r.get('peak_reserved', 0) for r in records), default=0),
               'sampled_device_memory_mib': max((g['memory_used_mib'] for s in samples for g in s['gpus']), default=None),
               'min_available_ram_gib': min((s['available_ram_gib'] for s in samples), default=None),
               'min_available_commit_gib': min((s['available_commit_gib'] for s in samples), default=None),
               'completed_run_wall_seconds': read(RUN/'completed.json')['seconds'] if (RUN/'completed.json').exists() else None}
    write(OUT/'resource_summary.json', payload)
    return payload


def main():
    plan = verify_plan()
    old_audit = verify_old()
    assert not (OUT/'STOP.json').exists()
    selected = read(RUN/'selection.json')
    assert selected['plan_sha256'] == sha(CODE/'plan.json') and selected['V_only']
    old = read(ROOT/'runs/peft_head_convergence_v1/selection_A.json')
    rows, winners = [], []
    for key, winner in sorted(selected['selected'].items()):
        dataset, seed, budget = key.split('/')
        cell = old['cells'][dataset+'/'+seed]
        folder = RUN/'forecast'/winner['key']/budget/'output'
        r = read(folder/'result.json')
        assert r['completed'] and r['frozen_verified']
        with np.load(folder/'selected.npz') as z:
            arrays = {k: z[k].copy() for k in z.files}
        values = {}
        for family in ('F0', 'HEAD', 'WIDE', 'JOINT'):
            a, old_winner = prediction('A', cell, budget, family)
            for k in ('target', 'scale', 'quantiles'):
                np.testing.assert_array_equal(a[k], arrays[k])
            values[family] = score_arrays(*(a[k] for k in ('prediction', 'target', 'quantiles', 'scale')))
            old_folder = ROOT/'runs/peft_head_convergence_v1/forecast'/old_winner['key']/budget/'output'
            with np.load(old_folder/('F0.npz' if family == 'F0' else 'selected.npz')) as z:
                np.testing.assert_array_equal(z['origins'], arrays['origins'])
            if family == 'F0':
                with np.load(folder/'F0.npz') as z:
                    np.testing.assert_array_equal(z['prediction'], a['prediction'])
        values['LORA_ONLY'] = score_npz(folder/'selected.npz')
        assert abs(values['LORA_ONLY']-r['D_score']) < 1e-12
        f, l, j, w = [values[k] for k in ('F0', 'LORA_ONLY', 'JOINT', 'WIDE')]
        case = 'A' if l < f and l <= j else 'B' if j < f and j < l else 'C' if l > f and j > f else 'D' if all(abs(100*(v-f)/f) <= plan['case_near_zero_pct_F0'] for v in values.values()) else 'UNRESOLVED'
        rows.append({'dataset': dataset, 'seed': int(seed[1:]), 'budget': budget, **values,
                     'G_LORA_ONLY': 100*(f-l)/f, 'G_JOINT': 100*(f-j)/f,
                     'I_LORA_ONLY': 100*(w-l)/f, 'I_JOINT': 100*(w-j)/f,
                     'H_EFFECT': 100*(l-j)/f, 'case': case})
        winners.append({'cell': dataset+'/'+seed, 'budget': budget, **winner})
    assert len(rows) == 8
    gates = [read(p) for p in RUN.glob('**/output/gate.json')]
    assert all(g['passed'] for g in gates)
    audit = {'status': 'PASS', 'old': old_audit, 'gate_count': len(gates),
             'step0_max_abs_error': max(g['step0_max_abs_error'] for g in gates),
             'trainable_counts': sorted({g['trainable_count'] for g in gates}),
             'gate_example': gates[0], 'metric_audit': 'same Study35 implementation + score_npz recomputation',
             'native_F0_D_exact': True, 'target_scale_quantile_origin_exact': True,
             'plan_sha256': sha(CODE/'plan.json'), 'selection_sha256': sha(RUN/'selection.json'),
             'artifact_hashes': {p.relative_to(ROOT).as_posix(): sha(p) for p in RUN.glob('**/output/*') if p.is_file()}}
    write(OUT/'integrity_audit.json', audit)
    csv_write(OUT/'metrics.csv', rows)
    csv_write(OUT/'selected_models.csv', winners)
    write(OUT/'summary.json', {'status': 'COMPLETED_DEVELOPMENT_DIAGNOSTIC', 'rows': rows,
          'overall_case': 'E' if len({r['case'] for r in rows}) > 1 else rows[0]['case'],
          'pretraining_overlap': 'UNKNOWN', 'independent_test': False, 'fresh_fits': 0})
    resources()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = OUT/'figures'
    figures.mkdir(exist_ok=True)
    for name, columns, ylabel in (
        ('01_loss_ratios', ['F0', 'HEAD', 'WIDE', 'LORA_ONLY', 'JOINT'], 'D loss / F0 (lower better)'),
        ('02_contrasts', ['G_LORA_ONLY', 'G_JOINT', 'I_LORA_ONLY', 'I_JOINT'], '%F0 (positive better)')):
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), layout='constrained')
        for ax, budget in zip(axes, ('S180', 'L720')):
            rr = [r for r in rows if r['budget'] == budget]
            x = np.arange(len(rr))
            width = .8/len(columns)
            for i, column in enumerate(columns):
                values = [r[column]/r['F0'] if name == '01_loss_ratios' else r[column] for r in rr]
                ax.bar(x+(i-(len(columns)-1)/2)*width, values, width, label=column)
            ax.set_xticks(x, [f"{r['dataset']} / {r['seed']}" for r in rr])
            ax.axhline(1 if name == '01_loss_ratios' else 0, color='black', linewidth=.7)
            ax.set(title=budget+' — exposed development data', ylabel=ylabel)
            ax.legend(ncol=len(columns), fontsize=8)
        fig.savefig(figures/f'{name}.png', dpi=160)
        plt.close(fig)


if __name__ == '__main__':
    main()
