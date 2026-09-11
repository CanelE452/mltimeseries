"""Independent population audit of the proposed Study36 direction contrast."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'

import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/peft_dependence_screen_v1/independent_gate_audit'
PLAN = ROOT / '_docs/notes/tsfm_topics/07_research_direction/36_dependence_peft_paper_plan_20260911.md'


def spectrum(p, values, frequencies):
    q = p - np.ones_like(p) / len(p)
    identity = np.eye(len(p))
    gamma0 = np.mean(values ** 2)
    result = []
    for frequency in frequencies:
        z = np.exp(-1j * frequency)
        tail = np.linalg.solve(identity - z * q, z * q @ values)
        result.append(float(gamma0 + 2 * np.real(np.mean(values * tail))))
    return np.array(result)


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=False)
    p = np.array([[.2, .7, .1], [.1, .2, .7], [.7, .1, .2]])
    reverse = p.T
    reflection = np.eye(3)[::-1]
    values = np.array([-1., 0., 1.])
    features = np.column_stack([np.ones(3), values, values ** 2])
    forward_mean = p @ values
    reverse_mean = reverse @ values
    forward_coef = np.linalg.solve(features, forward_mean)
    reverse_coef = np.linalg.solve(features, reverse_mean)
    risk = lambda transition: float(np.mean(transition @ (values ** 2) - (transition @ values) ** 2))
    acf_forward = []
    acf_reverse = []
    risk_forward = []
    risk_reverse = []
    for horizon in range(1, 129):
        a, b = np.linalg.matrix_power(p, horizon), np.linalg.matrix_power(reverse, horizon)
        acf_forward.append(float(np.mean(values * (a @ values))))
        acf_reverse.append(float(np.mean(values * (b @ values))))
        risk_forward.append(risk(a))
        risk_reverse.append(risk(b))
    frequencies = np.linspace(0, np.pi, 257)
    psd_forward = spectrum(p, values, frequencies)
    psd_reverse = spectrum(reverse, values, frequencies)
    checks = {
        'stationary_uniform_error': float(np.max(np.abs(np.ones(3) / 3 @ p - np.ones(3) / 3))),
        'reflection_conjugacy_error': float(np.max(np.abs(reflection @ p @ reflection - reverse))),
        'observation_sign_flip_error': float(np.max(np.abs(reflection @ values + values))),
        'conditional_mean_sign_conjugacy_error': float(np.max(np.abs(reverse_mean + reflection @ forward_mean))),
        'acf_difference_h1_to_128': float(np.max(np.abs(np.array(acf_forward) - acf_reverse))),
        'risk_difference_h1_to_128': float(np.max(np.abs(np.array(risk_forward) - risk_reverse))),
        'psd_difference_257_frequencies': float(np.max(np.abs(psd_forward - psd_reverse))),
        'quadratic_forward_error': float(np.max(np.abs(features @ forward_coef - forward_mean))),
        'quadratic_reverse_error': float(np.max(np.abs(features @ reverse_coef - reverse_mean))),
    }
    assert all(error < 1e-12 for error in checks.values()), checks
    assert np.min(psd_forward) > 0
    np.testing.assert_allclose(risk(p), .46, atol=1e-14)
    np.testing.assert_allclose(forward_coef, [.6, -.2, -.9], atol=1e-14)
    np.testing.assert_allclose(reverse_coef, [-.6, -.2, .9], atol=1e-14)
    record = {
        'finished_utc': datetime.now(timezone.utc).isoformat(),
        'plan_sha256_before_execution_update': hashlib.sha256(PLAN.read_bytes()).hexdigest(),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'checks': checks,
        'technical_checks_pass': True,
        'construct_validity': 'FAIL_FOR_ISOLATING_TEMPORAL_DIRECTION_FROM_AMPLITUDE_SIGN',
        'risk_h1': {'forward': risk(p), 'reverse': risk(reverse)},
        'quadratic_coefficients_intercept_x_x2': {'forward': forward_coef.tolist(), 'reverse': reverse_coef.tolist()},
        'conditional_means': {'forward': forward_mean.tolist(), 'reverse': reverse_mean.tolist()},
        'population_implications': [
            'Reverse process is exactly the sign-relabelled forward process in law; independent finite samples need not be exact sign flips.',
            'Any forward/reverse model score asymmetry could reflect amplitude-sign equivariance, not a uniquely temporal mechanism.',
            'The observed first-order Markov state is sufficient at every horizon; older observations provide no additional conditional information.',
            'A quadratic of the latest observation represents the one-step Bayes predictor exactly with three coefficients.',
            'These are representability and identification results, not measurements of trained LoRA or learned-head performance.',
        ],
        'gpu_started': False,
        'seconds': time.perf_counter() - started,
    }
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), layout='constrained')
    axes[0].plot(frequencies / np.pi, psd_forward, label='Forward', lw=3)
    axes[0].plot(frequencies / np.pi, psd_reverse, '--', label='Reverse', lw=2)
    axes[0].set(xlabel='Frequency / pi', ylabel='Population PSD', title='Control works: same spectrum')
    axes[0].legend()
    axes[1].plot(values, forward_mean, 'o-', label='Forward')
    axes[1].plot(values, reverse_mean, 's--', label='Reverse')
    axes[1].set(xlabel='Latest observed value', ylabel='E[next value | latest value]', title='3 coefficients recover the oracle')
    axes[1].set_xticks(values)
    axes[1].text(.04, .95, 'Forward: 0.6 - 0.2x - 0.9x²\nReverse: -0.6 - 0.2x + 0.9x²',
                 transform=axes[1].transAxes, va='top', fontsize=9)
    axes[1].legend(loc='lower left')
    axes[2].axis('off')
    axes[2].text(0, .95, 'Phase 0: design does not isolate\nthe intended mechanism', fontsize=13, weight='bold', va='top')
    axes[2].text(0, .72, 'Reverse = sign relabelling of Forward\n\nOne-step Bayes MSE = 0.46 in both\n\nLatest state contains all predictive information\n\nNo PEFT training result is implied', va='top', fontsize=11, linespacing=1.5)
    fig.suptitle('Study36 independent audit — population results, not trained-model scores', fontsize=14)
    fig.savefig(OUT / '01_population_audit.png', dpi=160)
    plt.close(fig)
    (OUT / 'audit.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
