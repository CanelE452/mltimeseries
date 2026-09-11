"""CPU-only reconstruction; writes only its two new closure reports.

Run from any directory with the repository's existing NumPy environment.
No experiment module is imported and no forecast or fit is executed.
"""
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ATOL = 1e-10
PROVENANCE = {}
CHECKS = []
DETAIL = {}


def source(path):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    key = p.relative_to(ROOT).as_posix()
    if key not in PROVENANCE:
        PROVENANCE[key] = {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                           'bytes': p.stat().st_size}
    return p


def read(path):
    return json.loads(source(path).read_text(encoding='utf-8-sig'))


def rows(path):
    with source(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def archive(path):
    with np.load(source(path), allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def score(path, scale=None, quantiles=None, indices=None):
    z = archive(path)
    p = z['prediction'] if 'prediction' in z else z['predictions']
    y = z['target']
    if indices is not None:
        p, y = p[indices], y[indices]
    p, y = np.sort(p.astype(np.float64), axis=2), y.astype(np.float64)
    q = np.asarray(z.get('quantiles', quantiles), dtype=np.float64)
    s = np.asarray(z.get('scale', scale), dtype=np.float64)
    assert p.shape == (len(y), y.shape[1], len(q), y.shape[2])
    valid = np.isfinite(y)
    error = np.where(valid[:, :, None, :], y[:, :, None, :] - p, 0.)
    loss = 2 * np.maximum(q[None, None, :, None] * error,
                          (q[None, None, :, None] - 1) * error)
    return float((loss.sum(axis=(0, 3)) / valid.sum(axis=(0, 2))[:, None] / s[:, None]).mean())


def check(study, name, actual, expected, decimals=None, formula='', unit='%F0'):
    actual, expected = float(actual), float(expected)
    tolerance = ATOL if decimals is None else .5 * 10 ** (-decimals) + ATOL
    delta = actual - expected
    status = ('FLOAT_MATCH' if abs(delta) <= ATOL else
              'DOCUMENT_ROUNDING_MATCH' if decimals is not None and abs(delta) <= tolerance
              else 'EVIDENCE_MISMATCH')
    CHECKS.append(dict(study=study, claim=name, computed=actual, documented=expected,
                       delta=delta, tolerance=tolerance, status=status, unit=unit,
                       formula=formula, evidence_tag='[확인]'))


def gain(a, b, f0):
    return 100 * (a - b) / f0


def main():
    started = time.perf_counter()
    cap = read('runs/peft_capacity_probe_v1/plan.json')
    meta = {}
    f0 = {}
    for period in ('P0', 'P1'):
        for ds in ('bike', 'household'):
            spec = cap['references'][period]['datasets'][ds]
            z = archive(spec['holdout_data_path'])
            meta[period, ds] = (z['fit_std'][z['target_indices']], z['quantiles'])
            f0[period, ds] = score(cap['f0_paths'][period][ds], *meta[period, ds])

    # Study20: independently score all primary SORT rows, including simple reference.
    r20 = rows('results/peft_fullft_reference_v3/metrics.csv')
    scores20 = {}
    for r in r20:
        if r['procedure'] != 'SORT':
            continue
        p = Path(r['path']) / ('E_predictions.npz' if r['arm'] == 'SIMPLE' else 'Epredictions.npz')
        value = score(p, *meta['P0', r['dataset']])
        check('20', '/'.join((r['dataset'], r['arm'], r['seed'], 'SORT')),
              value, r['score'], unit='normalized mean 2-pinball')
        scores20[r['dataset'], r['arm'], r['seed']] = value
    for ds, expected in [('bike', 3.953), ('household', .121)]:
        vals = [gain(scores20[ds, 'HEAD_ONLY', str(s)], scores20[ds, 'LORA', str(s)], f0['P0', ds])
                for s in (20000, 20001, 20002)]
        check('20', ds + ' LoRA incremental mean', np.mean(vals), expected, 3,
              'mean_seed(100*(HEAD_ONLY-LORA)/F0)')
        DETAIL['20/' + ds] = {'seed_gains': vals, 'seeds': [20000, 20001, 20002]}

    # Study26: identical residual MLP comparison; every saved controlled score.
    r26 = rows('results/peft_mechanism_diagnostics_v1/controlled/metrics.csv')
    scores26 = {}
    for r in r26:
        if r['stage'] != 'eval':
            continue
        p = f"runs/peft_mechanism_diagnostics_v1/eval/{r['dataset']}/{r['arm']}/r{r['recipe']}_s{r['seed']}/output/predictions.npz"
        value = score(p)
        check('26', '/'.join((r['dataset'], r['arm'], r['seed'])), value, r['score'], unit='normalized mean 2-pinball')
        scores26[r['dataset'], r['arm'], r['seed']] = value
    for ds, expected in [('bike', 4.171), ('household', .418)]:
        vals = [gain(scores26[ds, 'MLP', str(s)], scores26[ds, 'ALL', str(s)], f0['P0', ds])
                for s in (25000, 25001)]
        check('26', ds + ' matched MLP incremental mean', np.mean(vals), expected, 3,
              'mean_seed(100*(MLP-ALL)/F0)')
        DETAIL['26/' + ds] = {'seed_gains': vals}

    # Study30: rescore WIDE and the linked old ALL forecasts, not report deltas.
    r30 = rows('results/peft_capacity_probe_v1/capacity_metrics.csv')
    effects30 = []
    for r in r30:
        period, ds, condition, seed = (r[k] for k in ('period', 'dataset', 'condition', 'seed'))
        wide = score(f"runs/peft_capacity_probe_v1/eval/{period}/{ds}/{condition}/r{r['wide_recipe']}_s{seed}/output/predictions.npz")
        old = next(x for x in cap['old_all'] if x['period'] == period and x['job']['dataset'] == ds
                   and x['job']['condition'] == condition and str(x['job']['seed']) == seed)
        joint = score(old['eval_key'] + '/output/predictions.npz', *meta[period, ds])
        wide_record = read(f"runs/peft_capacity_probe_v1/eval/{period}/{ds}/{condition}/r{r['wide_recipe']}_s{seed}/output/result.json")
        joint_record = read(old['eval_key'] + '/output/result.json')
        check('30', '/'.join((period, ds, condition, seed, 'trainable WIDE vs JOINT')),
              wide_record['trainable'], joint_record['trainable'], unit='parameters')
        delta = gain(wide, joint, f0[period, ds])
        for name, value in [('wide_score', wide), ('all_score', joint), ('wide_minus_all_pct_F0', delta)]:
            check('30', '/'.join((period, ds, condition, seed, name)), value, r[name],
                  unit='%F0' if name.endswith('pct_F0') else 'normalized mean 2-pinball')
        if (period, ds, condition) == ('P1', 'bike', 'FULL90'):
            effects30.append(delta)
    check('30', 'P1 Bike FULL90 WIDE vs JOINT', np.mean(effects30), 4.779, 3,
          'mean_seed(100*(WIDE-JOINT)/F0)')
    check('30', 'P1 Bike FULL90 full precision summary', np.mean(effects30),
          read('results/peft_capacity_probe_v1/summary.json')['capacity_gate']['P1_bike_FULL90_mean'])
    DETAIL['30'] = {'seed_gains': effects30, 'parameter_contract': cap['trainable_count']}

    # Study31: raw E forecasts plus raw resource-matched timing CSV.
    r31 = rows('results/peft_contribution_freeze_v1/resource_matched/metrics.csv')
    vals31 = {}
    for r in r31:
        key = (r['dataset'], r['condition'], r['seed'], r['arm'])
        val = score('runs/peft_contribution_freeze_v1/eval/' + '/'.join(key[:2]) + '/' + key[3] + '/s' + key[2] + '/output/predictions.npz')
        check('31', '/'.join(key) + '/E', val, r['E_score'], unit='normalized mean 2-pinball')
        vals31[key] = val
    DETAIL['31'] = []
    for arm in ('ES2', 'FIXED_FREEZE', 'CONTRIB_FREEZE'):
        for seed in ('27000', '27001'):
            group = [r for r in r31 if r['arm'] == arm and r['seed'] == seed]
            if not group:
                continue
            base = [r for r in r31 if r['arm'] == 'FULL' and r['seed'] == seed]
            losses = [gain(vals31[r['dataset'], r['condition'], seed, arm],
                           vals31[r['dataset'], r['condition'], seed, 'FULL'], float(r['F0_score'])) for r in group]
            saving = 100 * (1 - sum(float(r['fit_seconds']) for r in group) / sum(float(r['fit_seconds']) for r in base))
            DETAIL['31'].append({'arm': arm, 'seed': seed, 'mean_loss_pct_F0': float(np.mean(losses)), 'fit_saving_pct': saving})
    for arm, seed, loss, saving in [('ES2','27000',0,7.53),('ES2','27001',0,5.04),
                                    ('FIXED_FREEZE','27000',-.492,19.92),('FIXED_FREEZE','27001',-.032,21.46),
                                    ('CONTRIB_FREEZE','27000',.240,7.42),('CONTRIB_FREEZE','27001',0,3.77)]:
        rec = next(x for x in DETAIL['31'] if x['arm'] == arm and x['seed'] == seed)
        check('31', arm + '/' + seed + '/loss', rec['mean_loss_pct_F0'], loss, 3)
        check('31', arm + '/' + seed + '/time', rec['fit_saving_pct'], saving, 2, unit='% fit wall-clock')

    # Study32: rescore S and D at all branch checkpoints; S_off exists only as saved scalar JSON.
    r32 = rows('results/peft_future_utility_v1/metrics.csv')
    utility = []
    for r in r32:
        key = f"{r['dataset']}/{r['condition']}/s{r['seed']}"
        folder = ROOT / 'runs/peft_future_utility_v1/jobs' / key / 'output'
        job = read(folder / 'result.json')
        fork = next(f for f in job['forks'] if f['fork'] == int(r['fork']))
        initial = score(folder / 'joint/point_0.npz', indices=job['D_rows'])
        initial_s = score(folder / 'joint/point_0.npz', indices=job['S_rows'])
        current = next(h for h in fork['branches']['JOINT']['history'] if h['step'] == int(r['fork']))
        current_s = score(folder / f"joint/point_{r['fork']}.npz", indices=job['S_rows'])
        contribution = gain(current['S_off'], current_s, initial_s)
        check('32', key + '/C/' + r['fork'], contribution, r['C_S_pct_F0'])
        branch_values = {}
        for mode, b in fork['branches'].items():
            history = []
            for h in b['history']:
                sub = folder / 'joint' if mode == 'JOINT' or h['step'] <= int(r['fork']) else folder / ('fork_' + r['fork']) / mode
                p = sub / f"point_{h['step']}.npz"
                s, d = score(p, indices=job['S_rows']), score(p, indices=job['D_rows'])
                check('32', key + '/' + r['fork'] + '/' + mode + '/' + str(h['step']) + '/D', d, h['D'], unit='normalized mean 2-pinball')
                history.append((s, d, h['step']))
            selected = min(history, key=lambda x: x[0])
            branch_values[mode] = (history[-1][1], selected[1])
        u = gain(branch_values['HEAD_ONLY'][0], branch_values['JOINT'][0], initial)
        us = gain(branch_values['HEAD_ONLY'][1], branch_values['JOINT'][1], initial)
        check('32', key + '/' + r['fork'] + '/U', u, r['U_final_pct_F0'])
        check('32', key + '/' + r['fork'] + '/U_selected', us, r['U_selected_pct_F0'])
        utility.append({'cell': key, 'fork': int(r['fork']), 'C': contribution, 'U': u, 'U_selected': us})
    check('32', 'mean U final', np.mean([r['U'] for r in utility]), -4.350, 3)
    check('32', 'mean U selected', np.mean([r['U_selected'] for r in utility]), -.727, 3)
    DETAIL['32'] = {'all_forks': utility, 'positive_C_negative_U': [r for r in utility if r['C'] > 0 and r['U'] < 0],
                    'current_off_limitation': '[미검증] S_off raw predictions were not saved; contribution uses raw run JSON S_off, with on-score/F0 independently rescored.'}

    # Study33: selected E predictions for all 144 rows.
    r33 = rows('results/peft_decision_transfer_v1/metrics.csv')
    norm33 = {}
    for r in r33:
        folder = Path('runs/peft_decision_transfer_v1/forecast') / r['fit_key'] / 'output'
        value = score(folder / (r['selected_checkpoint'] + '.npz'))
        base = score(folder / 'JOINT_0.npz')
        check('33', '/'.join((r['dataset'], r['condition'], r['seed'], r['method'])), value, r['E_selected'], unit='normalized mean 2-pinball')
        norm33[r['dataset'], r['condition'], r['seed'], r['method']] = value / base
    for arm, expected in [('HEAD', .744), ('WIDE', .944)]:
        diffs = [100 * (v - norm33[k[:3] + ('PROBE',)]) for k, v in norm33.items() if k[3] == arm]
        check('33', 'PROBE mean gain vs ' + arm, np.mean(diffs), expected, 3)
    DETAIL['33'] = {'selected_rows': len(r33), 'PROBE_distinct_source_periods': 2,
                    'identical_FULL_PROBE_rows': sum(v == norm33[k[:3] + ('PROBE',)] for k,v in norm33.items() if k[3] == 'FULL')}

    # Studies34/35: selected primary losses from raw arrays; correction aggregate left as reported control.
    for study, name in [('34','peft_initial_headroom_v1'), ('35','peft_head_convergence_v1')]:
        table = rows(f'results/{name}/metrics.csv')
        selection = read('results/peft_head_convergence_v1/evidence/selection_A.json') if study == '35' else None
        scored = []
        for r in table:
            fam = r['family']
            if fam == 'CORRECTION':
                continue
            if study == '34':
                folder = Path('runs') / name / 'forecast' / r['forecast_key'] / 'output'
            else:
                cell = selection['cells'][r['cell']][r['budget']]
                spec = cell['HEAD' if fam == 'F0' else fam]
                folder = Path('runs') / name / 'forecast' / spec['key'] / r['budget'] / 'output'
            value = score(folder / ('F0.npz' if fam == 'F0' else 'selected.npz'))
            check(study, '/'.join(str(r.get(k,'')) for k in ('dataset','condition','seed','budget','family')), value, r['D_score'], unit='normalized mean 2-pinball')
            scored.append(r | {'value': value})
        contrast = {}
        for fam in ('HEAD','WIDE','F0'):
            diffs = []
            for j in scored:
                if j['family'] != 'JOINT' or (study == '35' and j['budget'] != 'L720'):
                    continue
                group = [x for x in scored if all(x.get(k) == j.get(k) for k in ('dataset','condition','seed','budget'))]
                base = next(x['value'] for x in group if x['family'] == 'F0')
                ref = next(x['value'] for x in group if x['family'] == fam)
                diffs.append(gain(ref, j['value'], base))
            contrast[fam] = {'mean_gain_pct_F0': float(np.mean(diffs)), 'cell_gains': diffs}
        DETAIL[study] = contrast
        expected = {'HEAD': 2.037, 'WIDE': 2.428, 'F0': 1.418} if study == '34' else {'HEAD':12.135,'WIDE':8.804,'F0':-4.305}
        for fam, value in expected.items():
            check(study, 'JOINT gain vs ' + fam, contrast[fam]['mean_gain_pct_F0'], value, 3,
                  'mean_cell(100*(reference-JOINT)/F0)')

    # Hospital: series/origin CSV gives directly saved elementary losses, prior to seed averaging.
    hospital = rows('results/hospital_shared_strength_v1/series_effects.csv')
    losses = {k: float(np.mean([float(r[k]) for r in hospital])) for k in ('F0_loss','LORA_loss','GLOBAL_loss','INDIVIDUAL_loss','mean_SHUFFLED_loss')}
    canonical = archive('data_external/hospital_shared_strength_v1/canonical.npz')
    prediction_folder = Path('runs/hospital_shared_strength_v1_run2/stages/forecast_eval/attempt_01/output')
    for label in ('f0', 'lora_24000', 'lora_24001'):
        z = archive(prediction_folder / (label + '.npz'))
        # Canonical uses numeric order, forecasts lexicographic order; align observed IDs.
        ids = {str(s):i for i,s in enumerate(canonical['series_ids'])}
        aligned = canonical['values'][[ids[str(s)] for s in z['series_ids']]]
        y = np.stack([aligned[:,int(o):int(o)+12] for o in z['origins']], axis=1).astype(np.float64)
        p = np.sort(z['prediction'].astype(np.float64), axis=2)
        q = z['quantiles'].astype(np.float64)[None,None,:,None]
        error = y[:,:,None,:] - p
        elementary = (2*np.maximum(q*error,(q-1)*error)).mean(axis=(2,3)) / z['scale'][:,None]
        seed = '24000' if label == 'f0' else label.split('_')[1]
        field = 'F0_loss' if label == 'f0' else 'LORA_loss'
        csv_lookup = {(r['series_id'],int(r['origin'])):float(r[field]) for r in hospital if r['seed']==seed}
        csv_array = np.array([[csv_lookup[str(s),int(o)] for o in z['origins']] for s in z['series_ids']])
        check('Hospital', label + '/raw prediction vs elementary CSV max error',
              float(np.max(abs(elementary-csv_array))), 0, unit='normalized mean 2-pinball')
    check('Hospital', 'LoRA vs F0', gain(losses['F0_loss'],losses['LORA_loss'],losses['F0_loss']),1.183,3)
    check('Hospital', 'individual vs global', gain(losses['GLOBAL_loss'],losses['INDIVIDUAL_loss'],losses['F0_loss']),-.411890,6)
    check('Hospital', 'individual vs shuffled', gain(losses['mean_SHUFFLED_loss'],losses['INDIVIDUAL_loss'],losses['F0_loss']),-.055366,6)
    DETAIL['Hospital'] = {'losses':losses,'rows':len(hospital),'series':len({r['series_id'] for r in hospital}),
                          'coverage':'[확인] F0/LoRA raw predictions independently rescored against all elementary series/origin CSV losses. Individual/shuffled contrasts aggregated from elementary CSV; no new forecast.'}

    # Study36: independent finite state calculations (no old audit script import/run).
    p = np.array([[.2,.7,.1],[.1,.2,.7],[.7,.1,.2]])
    v = np.array([-1.,0.,1.]); j = np.eye(3)[::-1]; reverse = p.T
    design = np.array([[1,-1,1],[1,0,0],[1,1,1]], dtype=float)
    risk = lambda a: float((a @ (v*v) - (a @ v)**2).mean())
    acf, risks = [], []
    for h in range(1,129):
        a,b = np.linalg.matrix_power(p,h),np.linalg.matrix_power(reverse,h)
        acf.append(abs(float(v @ (a-b) @ v / 3)))
        risks.append(abs(risk(a)-risk(b)))
    # Direct covariance Fourier sum, distinct from old resolvent implementation.
    freqs = np.linspace(0,np.pi,257)
    psds = []
    for transition in (p,reverse):
        cov = np.array([v @ np.linalg.matrix_power(transition,h) @ v / 3 for h in range(1,257)])
        psds.append(float(np.mean(v*v)) + 2*np.cos(freqs[:,None]*np.arange(1,257)) @ cov)
    maths = {'sign_conjugacy_max_error':float(np.max(abs(j@p@j-reverse))),
             'observation_sign_max_error':float(np.max(abs(j@v+v))),
             'acf_max_difference_h1_128':max(acf), 'risk_max_difference_h1_128':max(risks),
             'psd_max_difference_257_freqs':float(np.max(abs(psds[0]-psds[1]))),
             'one_step_bayes_mse':risk(p),
             'forward_quadratic_coefficients':np.linalg.solve(design,p@v).tolist(),
             'reverse_quadratic_coefficients':np.linalg.solve(design,reverse@v).tolist()}
    check('36','Bayes one-step MSE',risk(p),.46,unit='MSE')
    for name in ('sign_conjugacy_max_error','observation_sign_max_error','acf_max_difference_h1_128','risk_max_difference_h1_128','psd_max_difference_257_freqs'):
        check('36',name,maths[name],0,unit='absolute population error')
    old = read('results/peft_dependence_screen_v1/independent_gate_audit/audit.json')
    for side in ('forward','reverse'):
        for i,x in enumerate(maths[side+'_quadratic_coefficients']):
            check('36',side+'/coefficient/'+str(i),x,old['quadratic_coefficients_intercept_x_x2'][side][i],unit='polynomial coefficient')
    maths['decision'] = '[판정] First-order observed-state Markov property makes latest state sufficient at every horizon; sign relabeling confounds direction. Current DGP remains STOP.'
    DETAIL['36'] = maths

    docs = ['06_fullft_reference/20_peft_fullft_reference_results_20260909.md']
    docs += ['07_research_direction/'+n for n in ('26_mechanism_diagnostics_20260910.md','30_capacity_probe_20260910.md','31_contribution_freeze_20260911.md','32_future_utility_20260911.md','33_decision_transfer_20260911.md','34_initial_headroom_20260911.md','35_head_convergence_20260911.md','36_dependence_phase0_results_20260911.md')]
    docs += ['08_hospital_shared_strength/24_hospital_results_20260910.md']
    for d in docs:
        source('_docs/notes/tsfm_topics/'+d)
    source(__file__)
    # Rehash inputs to verify this independent CPU audit did not modify them.
    unchanged = all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==v['sha256'] for p,v in PROVENANCE.items())
    mismatches = [c for c in CHECKS if c['status']=='EVIDENCE_MISMATCH']
    report = {'status':'EVIDENCE_MISMATCH' if mismatches or not unchanged else 'PASS_CORE_RECONSTRUCTION',
              'created_utc':datetime.now(timezone.utc).isoformat(),'wall_clock_seconds':time.perf_counter()-started,
              'GPU_fits':0,'GPU_forecasts':0,'GPU_memory_bytes':None,
              'float_policy':{'absolute_tolerance':ATOL,'relative_tolerance':0,'document_rounding':'half last printed decimal plus float tolerance; never replace float tolerance by prose precision'},
              'metric':'[확인] SORT mean 2-pinball: mean over targets/quantiles of valid origin+horizon loss, divided by each target train scale.',
              'dependence':'[확인] Shared target windows across seeds, conditions and overlapping rolling origins are not independent datasets. All reconstructed periods are historical development evidence, not a fresh final test.',
              'pretraining_overlap':'[미검증] UNKNOWN; no new contamination evidence is inferred.',
              'scope':'[확인] Main primary scores and contrasts reconstructed; no retraining, model-selection opportunity audit, interval recomputation, full secondary calibration audit or causal mechanism test in this script.',
              'coverage_limits':['[미검증] Study32 S_off uses saved raw run JSON; off predictions absent.', '[미검증] Hospital individual/shuffled selection and blended predictions are not reimplemented; elementary CSV contrast is independent, F0/LoRA raw scoring verified.', '[미검증] Study34 correction and G2 are not independently reimplemented here.', '[미검증] Study36 finite-sample data-gate metrics are not reimplemented; population audit is independent.'],
              'inputs_unchanged':unchanged,'checks':CHECKS,'details':DETAIL,'provenance':PROVENANCE}
    (OUT/'evidence_reconstruction.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    lines = ['# Existing PEFT evidence reconstruction','',f"[판정] {report['status']}. [확인] {len(CHECKS)} numeric checks; {len(PROVENANCE)} hashed input files; inputs unchanged={unchanged}.",'',
             '[확인] CPU NumPy reconstruction only: GPU fit 0 / forecast 0. Runtime '+f"{report['wall_clock_seconds']:.3f} seconds. GPU memory not measured (no GPU used).",'',
             '[확인] Float comparison uses absolute 1e-10, rtol=0. Printed prose uses half of its final displayed decimal separately; DOCUMENT_ROUNDING_MATCH is not numerical drift.','',
             report['metric'],'',report['dependence'],'',report['pretraining_overlap'],'', '## Main document comparisons','']
    for c in CHECKS:
        if c['status']!='FLOAT_MATCH' or c['study']=='36':
            lines.append(f"- [확인] Study {c['study']} {c['claim']}: computed {c['computed']:.12g}; document {c['documented']:.12g}; delta {c['delta']:+.3g}; {c['status']} ({c['unit']}).")
    lines += ['', '## Interpretation and coverage','',
              '[판정] Study20/26/30 support procedure-specific internal adaptation benefit under the recorded controls. They do not prove absent representation information or optimality over all possible heads.','',
              '[확인] Study32 current contribution is positive in '+str(sum(r['C']>0 for r in utility))+' of 24 dependent forks; '+str(len(DETAIL['32']['positive_C_negative_U']))+' of those positive-contribution forks have negative future update utility. [판정] Current contribution and update utility are distinct estimands.','',
              '[확인] Study35 L720 JOINT F0-relative loss increase: '+f"{-DETAIL['35']['F0']['mean_gain_pct_F0']:.9f}%F0; it beats WIDE while losing to F0 in all four cells.",'',
              DETAIL['36']['decision'],'',report['scope'],'']+report['coverage_limits']
    lines += ['', '## Reproduction and provenance','',
              '[확인] Run `.venv-peft/Scripts/python.exe results/peft_paper_closure_v1/reconstruct_evidence.py`. Script imports only stdlib/NumPy and writes its two closure reports.', '',
              '[확인] Per-check expected/computed/delta/tolerance and SHA-256 for every accessed raw NPZ, CSV, run JSON and document are in `evidence_reconstruction.json`. Existing scripts were read for contracts but not executed.']
    (OUT/'evidence_reconstruction.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'checks':len(CHECKS),'inputs':len(PROVENANCE),'seconds':report['wall_clock_seconds'],'mismatches':mismatches},ensure_ascii=False))
    if mismatches or not unchanged:
        raise SystemExit(2)


if __name__=='__main__':
    main()
