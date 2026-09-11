"""Check persisted Study36 evidence and independently recompute state-mean risk."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'

import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from experiments.peft_dependence_screen_v1 import data_gate as data


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / 'results/peft_dependence_screen_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    summary = json.loads((RESULT / 'data_gate/summary.json').read_text())
    assert summary['source_sha256'] == sha(Path(data.__file__))
    assert summary['plan_sha256_at_execution'] == sha(data.PLAN)
    population = json.loads((RESULT / 'independent_gate_audit/audit.json').read_text())
    assert population['source_sha256'] == sha(Path(__file__).with_name('independent_gate_audit.py'))
    assert population['plan_sha256_before_execution_update'] == summary['plan_sha256_at_execution']
    with (RESULT / 'data_gate/predictor_metrics.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 108, len(rows)
    recomputed = []
    path_hashes = set()
    initial_states = {}
    for condition_index, condition in enumerate(data.CONDITIONS):
        for seed in data.SEEDS:
            splits = data.generate_condition(condition, condition_index, seed)
            initial_states[condition.name, seed] = splits['train'][:, 0]
            for split, values in splits.items():
                assert values.shape == (data.SPLITS[split], 512)
                for path in values:
                    digest = hashlib.sha256(path.tobytes()).hexdigest()
                    assert digest not in path_hashes, (condition.name, seed, split)
                    path_hashes.add(digest)
            if condition.name not in ('markov_forward', 'markov_reverse'):
                continue
            train = splits['train']
            positions = np.linspace(128, 511, 16, dtype=int)
            x = train[:, positions - 1].flatten()
            y = train[:, positions].flatten()
            estimated_means = np.array([y[x == value].mean() for value in (-1, 0, 1)])
            oracle_means = condition.transition @ np.array([-1., 0., 1.])
            population_risk = .46 + np.mean((estimated_means - oracle_means) ** 2)
            subset = [r for r in rows if r['condition'] == condition.name and int(r['seed']) == seed
                      and r['predictor'] == 'lag1_quadratic' and r['split'] == 'devtest']
            assert len(subset) == 1
            row = subset[0]
            assert int(row['train_examples_origins']) == 1024
            assert abs(population_risk - float(row['population_mse'])) < 1e-12
            evaluation = splits['devtest']
            latest = evaluation[:, positions - 1].astype(int) + 1
            empirical_mse = np.mean((estimated_means[latest] - evaluation[:, positions]) ** 2)
            assert abs(empirical_mse - float(row['mse'])) < 1e-12
            recomputed.append({'condition': condition.name, 'seed': seed,
                'population_mse': float(population_risk), 'finite_sample_D_mse': float(empirical_mse),
                'max_possible_population_improvement_pct': float(100 * (1 - .46 / population_risk))})
    for seed in data.SEEDS:
        assert not np.array_equal(initial_states['markov_forward', seed], initial_states['markov_reverse', seed])
        assert not np.array_equal(initial_states['ar_rho_neg06', seed], initial_states['ar_rho_pos06', seed])
    old = json.loads((ROOT / 'runs/peft_head_convergence_v1/plan.json').read_text())
    protected = {**old['source_hashes'], **old['input_hashes']}
    assert all(sha(ROOT / path) == expected for path, expected in protected.items())
    note = ROOT / '_docs/notes/tsfm_topics/07_research_direction/36_dependence_phase0_results_20260911.md'
    links = re.findall(r'\]\(([^)]+)\)', note.read_text(encoding='utf-8'))
    assert all((note.parent / link).exists() for link in links if not link.startswith('https://'))
    receipt = {'completed': True, 'decision': 'STOP_CURRENT_DESIGN_AT_PHASE0', 'gpu_fits': 0,
        'lora_performance_measured': False, 'recomputed': recomputed,
        'unique_paths_checked': len(path_hashes), 'predictor_csv_rows_checked': len(rows),
        'old_protected_files_unchanged': len(protected), 'linked_artifacts_checked': len(links),
        'data_summary_sha256': sha(RESULT / 'data_gate/summary.json'),
        'independent_audit_sha256': sha(RESULT / 'independent_gate_audit/audit.json')}
    (RESULT / 'verification.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
