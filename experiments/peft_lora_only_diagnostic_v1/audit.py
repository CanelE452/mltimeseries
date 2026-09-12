"""Fail-closed Study35 provenance, metric and V-selection verification."""
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.peft_head_convergence_v1.analyse import prediction, score_arrays, score_npz

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
RUN = ROOT/'runs/peft_lora_only_diagnostic_v1'
OUT = ROOT/'results/peft_lora_only_diagnostic_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def verify_old():
    old = ROOT/'runs/peft_head_convergence_v1'
    plan = read(old/'plan.json')
    evidence = read(ROOT/'results/peft_paper_closure_v1/evidence_reconstruction.json')
    hashes = plan['source_hashes'] | plan['input_hashes']
    for p, h in hashes.items():
        assert sha(ROOT/p) == h, p
    for p, item in evidence['provenance'].items():
        assert sha(ROOT/p) == item['sha256'], p
    selection = read(old/'selection_A.json')
    assert selection['plan_sha256'] == sha(old/'plan.json')
    assert selection['selection_uses_V_only']
    old_rows = read(ROOT/'results/peft_head_convergence_v1/summary_A.json')['rows']
    recomputed = []
    for cell_key, cell in sorted(selection['cells'].items()):
        assert cell['seed'] in (30000, 30001)
        for budget, cap in (('S180', 180), ('L720', 720)):
            reference = None
            for family in ('F0', 'HEAD', 'WIDE', 'JOINT'):
                arrays, selected = prediction('A', cell, budget, family)
                if reference is None:
                    reference = arrays
                else:
                    for k in ('target', 'scale', 'quantiles'):
                        np.testing.assert_array_equal(arrays[k], reference[k])
                score = score_arrays(*(arrays[k] for k in ('prediction', 'target', 'quantiles', 'scale')))
                recorded = next(r for r in old_rows if r['cell'] == cell_key and r['budget'] == budget and r['family'] == family)
                assert abs(score-recorded['D_score']) < 1e-12
                recomputed.append(recorded)
                if family == 'F0':
                    continue
                options = []
                for entry in plan['jobs']:
                    j = entry['job']
                    if j['phase'] != 'A' or j['dataset'] != cell['dataset'] or j['seed'] != cell['seed'] or j['family'] != family:
                        continue
                    folder = old/'fit'/entry['key']/'output'
                    r = read(folder/'result.json')
                    assert sha(folder/'result.json') == selection['fit_hashes'][entry['key']]
                    assert r['job'] == j and r['plan_sha256'] == sha(old/'plan.json')
                    b = r['budgets'][budget]
                    point = min((h for h in r['history'] if h['step'] <= cap), key=lambda h: (h['V'], h['step']))
                    assert (b['V'], b['step']) == (point['V'], point['step'])
                    assert sha(folder/f'{budget}.pt') == b['checkpoint_sha256']
                    assert abs(score_npz(folder/f'{budget}_val.npz')-b['V']) < 1e-12
                    options.append((b['V'], b['step'], j['recipe'], entry['key']))
                assert len(options) == 4
                assert min(options) == (selected['V'], selected['step'], selected['recipe'], selected['key'])
    for ref in ('HEAD', 'WIDE', 'F0'):
        gains = []
        for cell in selection['cells']:
            rows = {r['family']: r['D_score'] for r in recomputed if r['cell'] == cell and r['budget'] == 'L720'}
            gains.append(100*(rows[ref]-rows['JOINT'])/rows['F0'])
        assert abs(np.mean(gains)-evidence['details']['35'][ref]['mean_gain_pct_F0']) < 1e-10
    return {'passed': True, 'study35_hash_count': len(hashes), 'closure_hash_count': len(evidence['provenance']),
            'old_rows_recomputed': len(recomputed), 'old_V_selection_verified': True,
            'study35_plan_sha256': sha(old/'plan.json'), 'selection_sha256': sha(old/'selection_A.json'),
            'closure_sha256': sha(ROOT/'results/peft_paper_closure_v1/evidence_reconstruction.json')}


def verify_plan():
    plan = read(CODE/'plan.json')
    for p, h in (plan['source_hashes'] | plan['input_hashes']).items():
        assert sha(ROOT/p) == h, p
    return plan


def seal():
    plan = verify_plan()
    assert not (RUN/'selection.json').exists()
    options, hashes = {}, {}
    for index, job in enumerate(plan['jobs']):
        folder = RUN/'fit'/job['key']/'output'
        r = read(folder/'result.json')
        assert r['completed'] and r['job'] == job and r['plan_sha256'] == sha(CODE/'plan.json')
        assert r['frozen_verified'] and r['restore_exact'] and not r['D_opened'] and r['steps'] == 720
        assert [h['step'] for h in r['history']] == plan['schedule']
        assert read(folder/'gate.json')['passed']
        hashes[job['key']] = sha(folder/'result.json')
        for budget, cap in (('S180', 180), ('L720', 720)):
            b = r['budgets'][budget]
            p = min((h for h in r['history'] if h['step'] <= cap), key=lambda h: (h['V'], h['step']))
            assert (b['V'], b['step']) == (p['V'], p['step'])
            assert sha(folder/f'{budget}.pt') == b['checkpoint_sha256']
            assert abs(score_npz(folder/f'{budget}_val.npz')-b['V']) < 1e-12
            cell = f"{job['dataset']}/s{job['seed']}"
            options.setdefault(cell+'/'+budget, []).append(b | {'index': index, 'key': job['key'], 'recipe': job['recipe']})
    assert len(hashes) == 16 and len(options) == 8
    chosen = {}
    for cell, candidates in options.items():
        assert len(candidates) == 4
        chosen[cell] = min(candidates, key=lambda c: (c['V'], c['step'], c['recipe']))
    write(RUN/'selection.json', {'plan_sha256': sha(CODE/'plan.json'), 'V_only': True,
          'selected': chosen, 'fit_hashes': hashes,
          'forecasts': sorted([[v['index'], k.split('/')[-1]] for k, v in chosen.items()])})
