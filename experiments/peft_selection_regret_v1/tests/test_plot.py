from types import SimpleNamespace

import numpy as np
import pytest

from experiments.peft_selection_regret_v1 import analyse, plot


def plot_fixture():
    candidates, selected, effects = [], [], {}
    for ci, cell in enumerate(plot.CELLS):
        rows = []
        for method, rates in (("F0", [0.]), ("OFF_LORA", [1e-5, 3e-5, 1e-4]),
                               ("H_FULL", [3e-5, 1e-4, 3e-4]), ("H_MLP", [1e-4, 3e-4, 1e-3])):
            for rank, lr in enumerate(rates):
                candidate_id = f"{cell}/{method}/{lr}"
                for procedure in ("SORT", "QCAL"):
                    score = 1. if method == "F0" else .95+rank*.035+(method.startswith("H"))*.02
                    row = {"cell": cell, "candidate_id": candidate_id, "method": method, "lr": lr,
                           "procedure": procedure, "score": score, "val_score": 1. if method == "F0" else .99-rank*.02}
                    candidates.append(row)
                    rows.append(row)
        for selector in analyse.SELECTORS:
            method, rank = ("F0", 0) if selector == "F0" else ("H_FULL", 2) if selector == "HEAD_V" else ("OFF_LORA", 0 if selector == "FIXED_LOW" else 2)
            for procedure in ("SORT", "QCAL"):
                chosen = [r for r in rows if r["method"] == method and r["procedure"] == procedure][rank]
                selected.append({**chosen, "selector": selector, "all_oracle_regret_over_f0": chosen["score"]-.95})
        f0 = np.ones((83, 2))
        comparison = analyse.comparison_effects(f0*1.02, f0*.95, f0, f0)
        effects[cell] = {"SORT": {"LORA_V_vs_FIXED_LOW": comparison, "LORA_RECENT7_vs_FIXED_LOW": comparison}}
    return candidates, selected, {"completed": True, "cells": effects, "primary_confidence_each": .9875,
                                  "primary_family_size": 4, "bootstrap_seed": 2026090814}


def test_three_fixture_figures_render_without_opening_prediction_archives(tmp_path):
    candidates, selected, effects = plot_fixture()
    plot.configure()
    plot.plot_candidates(candidates, selected, tmp_path)
    plot.plot_primary(effects, tmp_path)
    plot.plot_oracle_regret(selected, tmp_path)
    files = list(tmp_path.iterdir())
    assert len(files) == 6
    assert {p.suffix for p in files} == {".png", ".pdf"}
    assert all(p.stat().st_size > 10000 for p in files)
    assert not plot.plt.get_fignums()


def test_partial_plot_verification_cannot_be_presented_as_complete(tmp_path):
    folder = tmp_path / "results" / analyse.STUDY
    folder.mkdir(parents=True)
    analyse.write_json(folder / "verification.json", {"completed": True, "passed": True,
                       "candidate_count": 39, "candidate_procedure_rows": 78, "selected_procedure_rows": 56})
    with pytest.raises(AssertionError, match="All fixed candidates"):
        plot.load_verified(tmp_path)
