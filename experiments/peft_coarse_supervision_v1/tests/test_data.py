from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _ids(site: str, role: str, n: int = 4) -> list[str]:
    return [f"{site}_{role}_{i:02d}" for i in range(n)]


def _write_metadata(path: Path, ids: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["building_id", "site_id", "primaryspaceusage", "timezone", "electricity"],
        )
        writer.writeheader()
        for building_id in ids:
            site = building_id.split("_", 1)[0]
            writer.writerow(
                {
                    "building_id": building_id,
                    "site_id": site,
                    "primaryspaceusage": "Office",
                    "timezone": "US/Eastern" if site == "Eagle" else "Europe/London",
                    "electricity": "Yes",
                }
            )


def _write_qc(path: Path, donors: dict[str, list[str]], targets: dict[str, list[str]]) -> None:
    candidate_stats = {}
    selection_sites = {}
    for site in ["Eagle", "Lamb"]:
        eligible = donors[site] + targets[site]
        selection_sites[site] = {
            "site_id": site,
            "eligible_count": len(eligible),
            "eligible_ids": eligible,
            "n": len(donors[site]),
            "decision": "PASS",
            "donor_ids": donors[site],
            "target_ids": targets[site],
        }
        for building_id in eligible:
            candidate_stats[building_id] = {
                "site_id": site,
                "finite_count": 8784,
                "missing_count": 0,
                "negative_count": 0,
                "zero_count": 0,
                "annual_sum": 8784.0,
                "eligible": True,
            }
    record = {
        "qc": {
            "completed": True,
            "decision": "PASS",
            "candidate_train_2016_qc": candidate_stats,
            "selection": {
                "sites": selection_sites,
                "total_donor_count": sum(len(v) for v in donors.values()),
                "total_target_count": sum(len(v) for v in targets.values()),
            },
            "timestamp_qc": {
                "matches_expected_grid": True,
                "total_rows": 17544,
                "rows_2016": 8784,
                "rows_2017": 8760,
            },
        }
    }
    path.write_text(json.dumps(record), encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_raw(
    path: Path,
    ids: list[str],
    *,
    permute_target_2016: bool = False,
    perturb_april_2017_target: str | None = None,
    blank_evaluation_2017_target: str | None = None,
) -> None:
    timestamps = pd.date_range("2016-01-01 00:00:00", "2017-12-31 23:00:00", freq="h")
    frame = pd.DataFrame({"timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S")})
    hour = timestamps.hour.to_numpy()
    month = timestamps.month.to_numpy()
    for col_index, building_id in enumerate(ids):
        base = float(col_index + 2)
        values = base + 0.1 * hour + 0.01 * month
        if "_target_" in building_id:
            values = base + 0.02 * month + 0.001 * np.arange(len(timestamps))
        frame[building_id] = values
    if permute_target_2016:
        mask_2016 = timestamps.year == 2016
        for building_id in [c for c in ids if "_target_" in c]:
            for month_number in range(1, 13):
                mask = mask_2016 & (timestamps.month == month_number)
                frame.loc[mask, building_id] = frame.loc[mask, building_id].to_numpy()[::-1]
    if perturb_april_2017_target is not None:
        april = (timestamps >= pd.Timestamp("2017-04-01")) & (
            timestamps <= pd.Timestamp("2017-04-30 23:00:00")
        )
        frame.loc[april, perturb_april_2017_target] += 100.0
    if blank_evaluation_2017_target is not None:
        evaluation = (timestamps >= pd.Timestamp("2017-04-01")) & (
            timestamps <= pd.Timestamp("2017-06-30 23:00:00")
        )
        frame.loc[evaluation, blank_evaluation_2017_target] = np.nan
    frame.to_csv(path, index=False)


def _fixture(
    tmp_path: Path,
    *,
    permute_target_2016: bool = False,
    perturb_april: bool = False,
    blank_april: bool = False,
) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    donors = {"Eagle": _ids("Eagle", "donor"), "Lamb": _ids("Lamb", "donor")}
    targets = {"Eagle": _ids("Eagle", "target"), "Lamb": _ids("Lamb", "target")}
    all_ids = donors["Eagle"] + targets["Eagle"] + donors["Lamb"] + targets["Lamb"]
    metadata = tmp_path / "metadata.csv"
    raw = tmp_path / "electricity.csv"
    qc = tmp_path / "qc.json"
    audit = tmp_path / "independent_audit.json"
    plan = tmp_path / "plan.md"
    output = tmp_path / "out"
    _write_metadata(metadata, all_ids)
    _write_raw(
        raw,
        all_ids,
        permute_target_2016=permute_target_2016,
        perturb_april_2017_target=targets["Eagle"][0] if perturb_april else None,
        blank_evaluation_2017_target=targets["Eagle"][0] if blank_april else None,
    )
    _write_qc(qc, donors, targets)
    _write_json(audit, {"completed": True, "verdict": "PASS"})
    plan.write_text("fixture plan", encoding="utf-8")
    return {"metadata": metadata, "raw": raw, "qc": qc, "audit": audit, "plan": plan, "output": output}


def _load_split(output: Path, name: str) -> dict[str, np.ndarray]:
    with np.load(output / name, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def test_build_outputs_declared_schema_and_no_current_month_total_in_evaluation_inputs(tmp_path: Path) -> None:
    from experiments.peft_coarse_supervision_v1.data import build_from_sources

    paths = _fixture(tmp_path)
    result = build_from_sources(
        metadata_path=paths["metadata"],
        raw_path=paths["raw"],
        qc_path=paths["qc"],
        audit_path=paths["audit"],
        output_dir=paths["output"],
        plan_path=paths["plan"],
    )

    assert result["completed"] is True
    train = _load_split(paths["output"], "train.npz")
    validation = _load_split(paths["output"], "validation.npz")
    evaluation_inputs = _load_split(paths["output"], "evaluation_inputs.npz")
    evaluation_truth = _load_split(paths["output"], "evaluation_truth.npz")
    assert train["context"].shape == (88, 512)
    assert validation["context"].shape == (24, 512)
    assert evaluation_inputs["context"].shape == (24, 512)
    assert evaluation_truth["target"].shape == (24, 744)
    assert "total" in train and "label_valid" in train
    assert "target" not in train and "target" not in validation and "target" not in evaluation_inputs
    assert "total" not in evaluation_inputs and "label_valid" not in evaluation_inputs
    assert train["context"].dtype == np.float32
    assert train["scale"].dtype == np.float64
    assert train["month"].dtype.kind == "U"
    assert evaluation_inputs["target_id"].dtype.kind == "U"


def test_target_2016_fine_permutation_with_same_monthly_totals_leaves_inputs_unchanged(tmp_path: Path) -> None:
    from experiments.peft_coarse_supervision_v1.data import build_from_sources

    base = _fixture(tmp_path / "base")
    permuted = _fixture(tmp_path / "permuted", permute_target_2016=True)
    build_from_sources(
        metadata_path=base["metadata"],
        raw_path=base["raw"],
        qc_path=base["qc"],
        audit_path=base["audit"],
        output_dir=base["output"],
        plan_path=base["plan"],
    )
    build_from_sources(
        metadata_path=permuted["metadata"],
        raw_path=permuted["raw"],
        qc_path=permuted["qc"],
        audit_path=permuted["audit"],
        output_dir=permuted["output"],
        plan_path=permuted["plan"],
    )

    for split_name in ["train.npz", "validation.npz", "evaluation_inputs.npz"]:
        a = _load_split(base["output"], split_name)
        b = _load_split(permuted["output"], split_name)
        for key in ["context", "profile", "scale", "horizon", "site", "target_id", "month"]:
            np.testing.assert_array_equal(a[key], b[key])
        if "total" in a:
            np.testing.assert_allclose(a["total"], b["total"])


def test_current_evaluation_month_fine_values_do_not_change_same_month_inputs(tmp_path: Path) -> None:
    from experiments.peft_coarse_supervision_v1.data import build_from_sources

    base = _fixture(tmp_path / "base")
    perturbed = _fixture(tmp_path / "perturbed", perturb_april=True)
    build_from_sources(
        metadata_path=base["metadata"],
        raw_path=base["raw"],
        qc_path=base["qc"],
        audit_path=base["audit"],
        output_dir=base["output"],
        plan_path=base["plan"],
    )
    build_from_sources(
        metadata_path=perturbed["metadata"],
        raw_path=perturbed["raw"],
        qc_path=perturbed["qc"],
        audit_path=perturbed["audit"],
        output_dir=perturbed["output"],
        plan_path=perturbed["plan"],
    )

    a_inputs = _load_split(base["output"], "evaluation_inputs.npz")
    b_inputs = _load_split(perturbed["output"], "evaluation_inputs.npz")
    april = a_inputs["month"] == "2017-04"
    np.testing.assert_array_equal(a_inputs["context"][april], b_inputs["context"][april])
    np.testing.assert_array_equal(a_inputs["profile"][april], b_inputs["profile"][april])
    truth_a = _load_split(base["output"], "evaluation_truth.npz")
    truth_b = _load_split(perturbed["output"], "evaluation_truth.npz")
    assert np.nanmax(np.abs(truth_a["target"][april] - truth_b["target"][april])) > 0


def test_build_refuses_to_overwrite_outputs(tmp_path: Path) -> None:
    from experiments.peft_coarse_supervision_v1.data import build_from_sources

    paths = _fixture(tmp_path)
    build_from_sources(
        metadata_path=paths["metadata"],
        raw_path=paths["raw"],
        qc_path=paths["qc"],
        audit_path=paths["audit"],
        output_dir=paths["output"],
        plan_path=paths["plan"],
    )

    with pytest.raises(FileExistsError, match="output already exists"):
        build_from_sources(
            metadata_path=paths["metadata"],
            raw_path=paths["raw"],
            qc_path=paths["qc"],
            audit_path=paths["audit"],
            output_dir=paths["output"],
            plan_path=paths["plan"],
        )


def test_insufficient_evaluation_coverage_is_recorded_with_artifacts(tmp_path: Path) -> None:
    from experiments.peft_coarse_supervision_v1.data import build_from_sources

    paths = _fixture(tmp_path, blank_april=True)
    result = build_from_sources(
        metadata_path=paths["metadata"],
        raw_path=paths["raw"],
        qc_path=paths["qc"],
        audit_path=paths["audit"],
        output_dir=paths["output"],
        plan_path=paths["plan"],
    )

    assert result["completed"] is False
    metadata = json.loads((paths["output"] / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["decision"] == "INSUFFICIENT_STRUCTURAL_COVERAGE"
    assert "forecast score" in metadata["validation_gate"]["scope"]
    assert (paths["output"] / "evaluation_truth.npz").exists()
