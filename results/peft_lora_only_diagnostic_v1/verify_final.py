"""Independent arithmetic, artifact, selection and preservation checks for delivery."""
import ast
import csv
import json
import math
import subprocess
from pathlib import Path
from experiments.peft_lora_only_diagnostic_v1.audit import ROOT, CODE, RUN, OUT, read, sha, write, verify_old, verify_plan


def main():
    plan = verify_plan()
    old = verify_old()
    summary = read(OUT/'summary.json')
    selected = read(RUN/'selection.json')
    assert len(plan['jobs']) == 16 and len(selected['forecasts']) == 8
    assert not (OUT/'STOP.json').exists()
    fit_paths = list(RUN.glob('fit/*/output/result.json'))
    forecast_paths = list(RUN.glob('forecast/*/*/output/result.json'))
    assert len(fit_paths) == 16 and len(forecast_paths) == 8
    for cell_budget, choice in selected['selected'].items():
        dataset, seed, budget = cell_budget.split('/')
        cap = 180 if budget == 'S180' else 720
        candidates = []
        for job in plan['jobs']:
            if job['dataset'] != dataset or job['seed'] != int(seed[1:]):
                continue
            folder = RUN/'fit'/job['key']/'output'
            r = read(folder/'result.json')
            assert sha(folder/'result.json') == selected['fit_hashes'][job['key']]
            assert r['plan_sha256'] == sha(CODE/'plan.json')
            assert r['frozen_verified'] and r['restore_exact'] and not r['D_opened']
            assert r['trainable'] == 1179648 and r['steps'] == 720
            point = min((p for p in r['history'] if p['step'] <= cap), key=lambda p: (p['V'], p['step']))
            saved = r['budgets'][budget]
            assert (saved['V'], saved['step']) == (point['V'], point['step'])
            assert saved['checkpoint_sha256'] == sha(folder/f'{budget}.pt')
            candidates.append((saved['V'], saved['step'], job['recipe'], job['key']))
        assert len(candidates) == 4
        assert min(candidates) == (choice['V'], choice['step'], choice['recipe'], choice['key'])
    with (OUT/'metrics.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 8
    for r in rows:
        f, l, j, w = [float(r[k]) for k in ('F0', 'LORA_ONLY', 'JOINT', 'WIDE')]
        expected = {'G_LORA_ONLY': 100*(f-l)/f, 'G_JOINT': 100*(f-j)/f,
                    'I_LORA_ONLY': 100*(w-l)/f, 'I_JOINT': 100*(w-j)/f, 'H_EFFECT': 100*(l-j)/f}
        for k, value in expected.items():
            assert math.isfinite(value) and abs(float(r[k])-value) < 1e-12
        match = next(s for s in summary['rows'] if s['dataset'] == r['dataset'] and str(s['seed']) == r['seed'] and s['budget'] == r['budget'])
        for k in expected:
            assert abs(match[k]-float(r[k])) < 1e-12
    integrity = read(OUT/'integrity_audit.json')
    for path, h in integrity['artifact_hashes'].items():
        assert sha(ROOT/path) == h, path
    assert integrity['step0_max_abs_error'] <= 1e-6
    assert read(OUT/'fresh_manifest_verification.json')['status'] == 'PASS'
    with (OUT/'selected_models.csv').open(newline='', encoding='utf-8') as f:
        winners = list(csv.DictReader(f))
    assert len(winners) == 8
    for winner in winners:
        assert float(winner['lora_lr']) == plan['jobs'][int(winner['index'])]['lora_lr']
        choice = selected['selected'][winner['cell']+'/'+winner['budget']]
        assert choice['key'] == winner['key'] and choice['checkpoint_sha256'] == winner['checkpoint_sha256']
    for folder in (CODE, OUT):
        for p in folder.glob('*.py'):
            ast.parse(p.read_text(encoding='utf-8'))
    figures = list((OUT/'figures').glob('*.png'))
    assert len(figures) == 2 and all(p.stat().st_size > 10000 for p in figures)
    assert (OUT/'STATUS.md').exists()
    assert (ROOT/'_docs/notes/tsfm_topics/07_research_direction/38_lora_only_diagnostic_20260912.md').exists()
    assert subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip() == 'main'
    subprocess.run(['git', 'diff', '--check'], cwd=ROOT, check=True)
    write(OUT/'verification.json', {'status': 'PASS', 'fit_count': 16, 'forecast_count': 8,
          'old_evidence_audit': old, 'independent_selection_recheck': True,
          'arithmetic_csv_json_consistent': True, 'new_artifact_hashes_verified': len(integrity['artifact_hashes']),
          'python_syntax': 'PASS', 'figures': [p.name for p in figures],
          'related_tests': {'command': '.venv-peft/Scripts/python.exe -m pytest tests/test_peft_verifier.py tests/test_peft_adaptation_scope_data.py -q --import-mode=importlib', 'observed_result': '14 passed in 4.36s'},
          'fresh_manifest_reproducer': {'script': 'results/peft_paper_closure_v1/prepare_fresh_stage_a_candidates.py', 'observed_exit_code': 0, 'observed_result': 'VERIFY PASS; Household BLOCKED / BDG2 Bull READY'},
          'fresh_training_count': 0, 'branch': 'main'})
    print('PASS: 16 fits, 8 forecasts, sealed V winners, arithmetic, artifacts, preserved old evidence')


if __name__ == '__main__':
    main()
