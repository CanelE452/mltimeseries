"""Synthetic-only end-to-end check; all outputs are confined to pytest's temporary path."""

import importlib
import json
import sys
import types

import numpy as np
import pytest

from experiments.peft_revision_entry_v1 import run
from experiments.peft_revision_entry_v1.tests.test_run import FakeSeries


class ScaledFakeSeries(FakeSeries):
    def __init__(self, scale, offset):
        super().__init__()
        self.scale, self.offset = scale, offset

    def growth(self, event, asof):
        return self.scale * super().growth(event, asof) + self.offset


def test_synthetic_pipeline_selection_identity_and_independent_scores(tmp_path, monkeypatch):
    providers = {'PAYEMS': ScaledFakeSeries(1.0, 0.0), 'INDPRO': ScaledFakeSeries(1.7, -0.2)}
    module_name = 'experiments.peft_revision_entry_v1.data'
    try:
        data_module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise
        data_module = types.ModuleType(module_name)
        monkeypatch.setitem(sys.modules, module_name, data_module)
    monkeypatch.setattr(data_module, 'load_series', lambda raw, name: providers[name], raising=False)
    monkeypatch.setattr(run, 'RAW', tmp_path / 'synthetic_raw')
    monkeypatch.setattr(run, 'OUT', tmp_path / 'results')
    monkeypatch.setattr(run, 'RUN', tmp_path / 'runs')
    monkeypatch.setattr(run, 'verify_contract', lambda: {'synthetic_test_only': True})
    original_write = run.write_json
    write_order = []

    def record_write(path, value):
        write_order.append(path.name)
        original_write(path, value)

    monkeypatch.setattr(run, 'write_json', record_write)
    run.execute()

    def read(path):
        return json.loads(path.read_text(encoding='utf-8'))

    selection = read(run.RUN / 'validation_selection.json')
    saved = read(run.OUT / 'evaluation_predictions.json')
    results = read(run.OUT / 'cpu_results.json')
    identity = read(run.OUT / 'ridge_identity.json')
    assert selection['evaluation_used_for_selection'] is False
    assert selection['selection_cutoff'] == '2019-07-01'
    assert write_order.index('validation_selection.json') < write_order.index('evaluation_predictions.json')
    assert all(row['passed'] and row['n_rows'] == row['expected_rows'] for row in identity)
    expected_origins = set(range(run.month(2016, 1), run.month(2018, 12) + 1)) | set(
        range(run.month(2020, 1), run.month(2024, 12) + 1))
    assert {(row['series'], row['event']) for row in identity} == {
        (name, run.start(event).isoformat()) for name in providers for event in expected_origins}
    assert results['gpu_runs'] == 0
    assert results['new_method_claim'] is False

    normalized_losses = {}
    for name, provider in providers.items():
        # Fixture's six-month values follow its declared law, independently of CausalPanel.
        e_events = np.arange(run.month(2020, 1), run.month(2024, 12) + 1)
        v_events = np.arange(run.month(2016, 1), run.month(2018, 12) + 1)
        scale_events = np.arange(run.month(1991, 1), run.month(2014, 12) + 1)
        truth = provider.scale * (0.1 * (e_events % 11) + 0.06) + provider.offset
        v_truth = provider.scale * (0.1 * (v_events % 11) + 0.06) + provider.offset
        historical = provider.scale * (0.1 * (scale_events % 11) + 0.06) + provider.offset
        variance = np.var(historical)
        np.testing.assert_allclose(saved[name]['truth_six_month_mature'], truth, atol=1e-14)
        normalized_losses[name] = {}
        for arm, metrics in results['metrics'][name].items():
            errors = np.asarray(saved[name][arm]) - truth
            assert metrics['n'] == len(e_events)
            assert metrics['mse'] == pytest.approx(np.mean(errors ** 2), abs=1e-14)
            assert metrics['mae'] == pytest.approx(np.mean(np.abs(errors)), abs=1e-14)
            assert metrics['normalized_mse'] == pytest.approx(np.mean(errors ** 2) / variance, abs=1e-12)
            normalized_losses[name][arm] = errors ** 2 / variance
        for arm in run.ARMS:
            chosen = selection['selected'][name][arm]
            expected = min(selection['all_scores'][name][arm], key=lambda c: (c['mse'], c['lam'], c['window'] is not None))
            assert chosen == expected
            v_errors = np.asarray(selection['predictions'][name][arm]) - v_truth
            assert chosen['mse'] == pytest.approx(np.mean(v_errors ** 2), abs=1e-14)

    first = np.stack([normalized_losses[name]['FIRST_FIXED_X'] for name in providers])
    for arm, pooled in results['pooled'].items():
        loss = np.stack([normalized_losses[name][arm] for name in providers])
        assert pooled['normalized_mse'] == pytest.approx(loss.mean(), abs=1e-12)
        assert pooled['improvement_percent_vs_first'] == pytest.approx(100 * (first - loss).mean() / first.mean(), abs=1e-10)
