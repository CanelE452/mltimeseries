import numpy as np
import pytest

from experiments.peft_objective_alignment_v1 import analyse as a, plot


def fixture_results(root):
    folder = root/"results"/a.STUDY
    folder.mkdir(parents=True)
    rows, effects, diagnostics = [], {}, {"sources": {}}
    for source in a.DATASETS:
        scores = {"F0": .3, "NATIVE": .29, "NORM_ALIGNED": .295, "RAW_ALIGNED": .285}
        effects[source] = {}
        for procedure in a.PROCEDURES:
            for arm, score in scores.items():
                rows.append({"source": source, "arm": arm, "procedure": procedure, "score": score})
            counts = np.ones((83, 2))
            effects[source][procedure] = {"NORM_ALIGNED_vs_RAW_ALIGNED": a.comparison_effects(
                counts*scores["NORM_ALIGNED"], counts*scores["RAW_ALIGNED"], counts*.3, counts)}
        calibration = {"cosines": {"NATIVE__NORM_ALIGNED": .96, "NATIVE__RAW_ALIGNED": .73, "NORM_ALIGNED__RAW_ALIGNED": .82},
                       "multipliers": {"NATIVE": 1., "NORM_ALIGNED": 7., "RAW_ALIGNED": 2.}}
        diagnostics["sources"][source] = {}
        for arm in a.ARMS:
            diagnostics["sources"][source][arm] = {"validation_history": [{"step": step, "val_score": .28-step*.0001} for step in range(0, 201, 40)],
                "gradient_calibration": calibration, "gradient_norms": np.linspace(.2, 2., 200).tolist(), "clipping_steps": list(range(90, 201))}
    a.write_csv(folder/"selected_results.csv", rows)
    a.write_json(folder/"effects.json", {"completed": True, "sources": effects, "bootstrap_seed": 2026090815,
                "primary_confidence_each": .975, "primary_family_size": 2})
    a.write_json(folder/"diagnostics.json", diagnostics)
    a.write_json(folder/"verification.json", {"completed": True, "passed": True, "selected_procedure_rows": 16,
                "new_fits": 6, "new_forecasts": 6, "reused_forecasts": 2, "s0_count": 6, "native_replay_verified": True,
                "output_hashes": {name: a.sha(folder/name) for name in ("selected_results.csv", "effects.json", "diagnostics.json")}})
    return folder


def test_three_figures_render_and_preserve_previous_output(tmp_path):
    folder = fixture_results(tmp_path)
    result = plot.run(tmp_path)
    assert result["completed"] and len(result["figure_hashes"]) == 6
    assert all((folder/"figures"/name).stat().st_size > 1000 for name in result["figure_hashes"])
    with pytest.raises(FileExistsError):
        plot.run(tmp_path)


def test_partial_results_are_rejected(tmp_path):
    folder = fixture_results(tmp_path)
    value = a.read_json(folder/"verification.json")
    value["selected_procedure_rows"] = 14
    a.write_json(folder/"verification.json", value)
    with pytest.raises(AssertionError):
        plot.load_verified(tmp_path)
