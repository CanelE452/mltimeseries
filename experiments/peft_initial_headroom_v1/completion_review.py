"""Post-run evidence packaging and descriptive resource/trajectory review."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import psutil
from experiments.hospital_shared_strength_v1.resume_monitored import memory

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/peft_initial_headroom_v1'
OUT = ROOT/'results/peft_initial_headroom_v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plots(plan, fits, seal, metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'HEAD': '#c66637', 'WIDE': '#3468b2', 'JOINT': '#008177'}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, ds in enumerate(sorted(plan['data'])):
        for j, condition in enumerate(('FULL90', 'SPREAD30')):
            ax = axes[i, j]
            for seed, style in ((29000, '-'), (29001, '--')):
                cell = seal['cells'][f'{ds}/{condition}/s{seed}']
                for family, selected in cell['families'].items():
                    fit = fits[selected['key']]
                    initial = fit['initial_V']
                    points = fit['history']
                    ax.plot([p['step'] for p in points], [p['V']/initial for p in points],
                            color=colors[family], linestyle=style, label=f'{family} / {seed}')
                    ax.scatter(fit['best_step'], fit['best_V']/initial, color=colors[family], marker='*', s=90, zorder=3)
            ax.set_title(f'{ds.upper()} | {condition}')
            ax.set_xlabel('Updates'); ax.set_ylabel('V / initial V'); ax.grid(alpha=.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8)
    fig.suptitle('V-selected recipe trajectories | stars: actual returned checkpoints')
    fig.tight_layout(rect=(0, .08, 1, .96)); fig.savefig(OUT/'04_selected_trajectories.png', dpi=160); plt.close(fig)

    methods = ['F0', 'HEAD', 'WIDE', 'CORRECTION']
    fig, ax = plt.subplots(figsize=(10, 5.5))
    summary = read(OUT/'summary.json')
    for y, method in enumerate(methods):
        differences = []
        for ds, offset, color in (('bmra', -.13, '#008177'), ('jena', .13, '#c66637')):
            vals = []
            for condition in ('FULL90', 'SPREAD30'):
                for seed in (29000, 29001):
                    rows = [r for r in metrics if r['dataset'] == ds and r['condition'] == condition and int(r['seed']) == seed]
                    by = {r['family']: float(r['D_over_F0']) for r in rows}
                    vals.append(100*(by[method]-by['JOINT']))
            differences.extend(vals)
            ax.scatter(np.mean(vals), y+offset, marker='x', s=65, color=color, label=ds.upper() if y == 0 else None)
        ax.scatter(np.mean(differences), y, s=45, color='#182d44', zorder=3)
        if method != 'F0':
            low, high = summary['conditional_intervals']['intervals'][f'JOINT_minus_{method}_pct_F0']
            ax.hlines(y, low, high, color='#182d44', linewidth=2)
    ax.axvline(0, color='black', linewidth=.8)
    ax.axvspan(-.25, .25, color='gray', alpha=.12)
    ax.set_yticks(range(len(methods)), methods); ax.invert_yaxis()
    ax.set_xlabel('JOINT gain over comparator (%F0); positive favors JOINT')
    ax.set_title('Initial LoRA differences | dots: mean; lines: conditional 90% intervals')
    ax.legend(); ax.grid(axis='x', alpha=.15)
    fig.text(.5, .02, 'Two fixed source periods; seeds and conditions share target labels. Development diagnostic only.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig(OUT/'05_quality_differences.png', dpi=160); plt.close(fig)


def main():
    assert read(RUN/'completed.json')['completed'] and read(OUT/'summary.json')['completed']
    assert (RUN/'independent_audit.json').exists()
    destination = OUT/'completion_review.json'
    assert not destination.exists()
    protected = {}
    for study in ('peft_initial_headroom_v1', 'peft_decision_transfer_v1'):
        plan = read(ROOT/'runs'/study/'plan.json')
        hashes = {**plan['source_hashes'], **plan['input_hashes']}
        for path, expected in hashes.items():
            assert sha(ROOT/path) == expected, path
        protected[study] = len(hashes)
    plan = read(RUN/'plan.json'); seal = read(RUN/'selection_sealed.json')
    fits = {e['key']: read(RUN/'fit'/e['key']/'output/result.json') for e in plan['jobs']}
    statuses = {p.relative_to(RUN).as_posix(): read(p) for p in RUN.rglob('guard/status.json')}
    assert len(statuses) == 123
    measurements, categories, finish_count = [], {}, 0
    for name, status in statuses.items():
        assert status['completed'] and status['returncode'] == 0 and not status['reasons']
        category = name.split('/')[0]
        group = categories.setdefault(category, {'jobs': 0, 'guard_seconds': 0.})
        group['jobs'] += 1; group['guard_seconds'] += status['elapsed_seconds']
        records = [json.loads(line) for line in (RUN/name).with_name('resource_log.jsonl').read_text().splitlines()]
        terminal = [r for r in records if 'gpus' not in r]
        assert len(terminal) == 1 and terminal[0]['event'] == 'finish' and terminal[0]['state'] == 'completed'
        finish_count += 1; measurements.extend(r for r in records if 'gpus' in r)
    gpu = [g for r in measurements for g in r['gpus']]
    active = []
    for process in psutil.process_iter(['name', 'pid']):
        if 'python' not in (process.info['name'] or '').lower():
            continue
        try:
            if {'experiments.peft_initial_headroom_v1.fit', 'experiments.peft_initial_headroom_v1.run'}.intersection(process.cmdline()):
                active.append(process.pid)
        except psutil.NoSuchProcess:
            pass
    lock = ROOT/'runs/peft_adaptation_scope_v1/.guard.lock'
    assert not active and not lock.exists()
    metrics = list(csv.DictReader((OUT/'metrics.csv').open(encoding='utf-8')))
    with np.load(OUT/'per_example_losses.npz') as loss:
        per_target = (loss['pinball_numerator'].sum(axis=1)/loss['valid_target_count'].sum(axis=1)[:, :, None]/loss['scale'][:, :, None]).mean(axis=2)
    assert len(per_target) == len(metrics) == 40
    target_contributions = []
    for i, row in enumerate(metrics):
        if row['family'] != 'JOINT':
            continue
        for comparator in ('HEAD', 'WIDE', 'CORRECTION'):
            j = next(j for j, other in enumerate(metrics) if all(other[k] == row[k] for k in ('dataset', 'condition', 'seed')) and other['family'] == comparator)
            contributions = 50*(per_target[j]-per_target[i])/float(row['F0_score'])
            assert abs(contributions.sum()-100*(float(metrics[j]['D_over_F0'])-float(row['D_over_F0']))) < 1e-10
            target_contributions.append({'dataset': row['dataset'], 'condition': row['condition'], 'seed': int(row['seed']),
                'comparator': comparator, 'target_names': plan['data'][row['dataset']]['channels'][:2],
                'contributions_to_total_gain_pct_F0': contributions.tolist()})
    comparisons = []
    for cell_name, cell in seal['cells'].items():
        predictions = {}
        for family, selected in cell['families'].items():
            with np.load(RUN/'forecast'/selected['key']/'output/selected.npz') as z:
                predictions[family] = z['prediction'].copy()
        with np.load(RUN/'forecast'/cell['f0_key']/'output/F0.npz') as z:
            f0 = z['prediction'].copy()
        comparisons.append({'cell': cell_name,
            'selected_steps': {f: s['best_step'] for f, s in cell['families'].items()},
            'same_as_F0': {f: bool(np.array_equal(v, f0)) for f, v in predictions.items()},
            'JOINT_equals_HEAD': bool(np.array_equal(predictions['JOINT'], predictions['HEAD'])),
            'JOINT_equals_WIDE': bool(np.array_equal(predictions['JOINT'], predictions['WIDE']))})
    evidence = OUT/'evidence'; evidence.mkdir(exist_ok=True)
    copies = []
    for source_name in ('plan.json', 'selection_sealed.json', 'completed.json', 'preflight_review.json',
                        'environment.json', 'independent_audit.json', 'prepared/summary.json', 'prepared/data_audit.json'):
        source = RUN/source_name; target = evidence/source_name.replace('/', '_')
        assert not target.exists()
        target.write_bytes(source.read_bytes())
        assert sha(source) == sha(target)
        copies.append({'source': source.relative_to(ROOT).as_posix(), 'copy': target.name, 'sha256': sha(source)})
    (evidence/'validation_histories.json').write_text(json.dumps(fits, indent=2), encoding='utf-8')
    (evidence/'guard_statuses.json').write_text(json.dumps(statuses, indent=2), encoding='utf-8')
    (evidence/'copies.json').write_text(json.dumps(copies, indent=2), encoding='utf-8')
    plots(plan, fits, seal, metrics)
    result = {'completed': True, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'script_sha256': sha(Path(__file__)), 'scope': 'Post-freeze audit/packaging only; not imported by frozen training or analysis',
        'guards': len(statuses), 'categories': categories, 'resource_samples': len(measurements),
        'finish_records': finish_count, 'protected_hash_checks': protected,
        'minimum_available_ram_gib': min(r['available_ram_gib'] for r in measurements),
        'minimum_available_commit_gib': min(r['available_commit_gib'] for r in measurements),
        'maximum_child_rss_gib': max(r['child_tree_rss_gib'] for r in measurements),
        'maximum_gpu_memory_mib': max(g['memory_used_mib'] for g in gpu),
        'maximum_gpu_temperature_c': max(g['temperature_c'] for g in gpu),
        'current_memory': memory(), 'active_study_processes': active, 'guard_lock_exists': lock.exists(),
        'selected_output_comparisons': comparisons, 'copied_evidence': len(copies)}
    result['target_gain_contributions'] = target_contributions
    result['selected_at_budget_cap'] = {family: sum(fits[s['key']]['best_step'] == fits[s['key']]['steps'] for cell in seal['cells'].values() for f, s in cell['families'].items() if f == family) for family in ('HEAD', 'WIDE', 'JOINT')}
    destination.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
