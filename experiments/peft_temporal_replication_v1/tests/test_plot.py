import importlib

import pytest

from experiments.peft_temporal_replication_v1 import analyse, plot
from experiments.peft_temporal_replication_v1.tests.test_analyse import result_fixture, temporal_fixture


def comparison_fixture():
    meta, current, old, previous = temporal_fixture()
    dates = {"previous_boundaries": old["contract"]["boundaries"], "replication_boundaries": meta["contract"]["boundaries"]}
    return {"completed": True, "blocks": {str(n): analyse.summarize_block(*result_fixture(n), n) for n in (12, 13)},
            "calendar_audits": {"bike": dates, "household": dates}, "input_hashes": {},
            "pooled_estimate_computed": False, "pooled_confidence_interval_computed": False,
            "cross_block_difference_test_computed": False}


def test_plot_uses_private_namespace_and_preserves_original_footer():
    original = importlib.import_module("experiments.peft_external_gap_v1.plot")
    assert original is not plot._core
    assert original.STUDY == "peft_external_gap_v1"
    assert plot._core.STUDY == "peft_temporal_replication_v1"
    assert "Blocks 12/13 are not pooled" in plot._core.FOOTER
    assert original.FOOTER != plot._core.FOOTER


def test_unverified_or_pooled_comparison_is_rejected(tmp_path):
    path = tmp_path / "results" / analyse.STUDY / "temporal_comparison.json"
    path.parent.mkdir(parents=True)
    comparison = comparison_fixture()
    analyse.write_json(path, comparison)
    with pytest.raises(AssertionError, match="not covered"):
        plot.load_comparison(tmp_path, {})
    comparison["pooled_estimate_computed"] = True
    analyse.write_json(path, comparison)
    with pytest.raises(AssertionError, match="separate temporal"):
        plot.load_comparison(tmp_path, {str(path): analyse.sha256_file(path)})


def test_comparison_fixture_renders_four_distinct_estimates_in_png_and_pdf(tmp_path):
    plot._core.configure()
    plot.plot_temporal_comparison(comparison_fixture(), tmp_path)
    files = {p.suffix: p for p in tmp_path.iterdir()}
    assert set(files) == {".png", ".pdf"}
    assert files[".png"].stat().st_size > 10000
    assert files[".pdf"].stat().st_size > 10000
    assert not plot._core.plt.get_fignums()
