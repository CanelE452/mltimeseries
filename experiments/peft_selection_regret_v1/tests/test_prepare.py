import json
import hashlib
from pathlib import Path

import numpy as np
import pytest

from experiments.peft_selection_regret_v1 import prepare


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")


def _write_bytes(path: Path, payload: bytes = b"artifact") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_hashed(path: Path, payload: bytes) -> str:
    _write_bytes(path, payload)
    return _sha(payload)


def _report_path(root: Path, study: str) -> Path:
    suffix = "12_peft_external_gap_results_20260908.md"
    if study == "peft_temporal_replication_v1":
        suffix = "13_peft_temporal_replication_results_20260908.md"
    return root / "_docs" / "notes" / "tsfm_topics" / suffix


def _plot_manifest_path(root: Path, study: str) -> Path:
    if study == "peft_temporal_replication_v1":
        return root / "results" / study / "figures" / "plot_manifest.json"
    return root / "results" / study / "figures_verified" / "plot_manifest.json"


def _write_minimal_archive(path: Path, *, role: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    origins = {
        "train_origins": np.arange(3, dtype=np.int64) if role == "fit" else np.asarray([], dtype=np.int64),
        "val_origins": np.arange(3, 5, dtype=np.int64) if role == "fit" else np.asarray([], dtype=np.int64),
        "cal_origins": np.arange(5, 7, dtype=np.int64) if role == "holdout" else np.asarray([], dtype=np.int64),
        "eval_origins": np.arange(7, 10, dtype=np.int64) if role == "holdout" else np.asarray([], dtype=np.int64),
    }
    np.savez_compressed(
        path,
        context_values=np.zeros((16, 2), dtype=np.float32),
        target_values=np.zeros((16, 2), dtype=np.float32),
        observed_mask=np.ones((16, 2), dtype=bool),
        target_loss_mask=np.ones((16, 2), dtype=bool),
        timestamps=np.asarray([f"2020-01-01T{i:02d}:00:00" for i in range(16)]),
        channels=np.asarray(["a", "b"]),
        target_indices=np.asarray([0, 1], dtype=np.int64),
        quantiles=np.linspace(0.01, 0.99, 21, dtype=np.float64),
        fit_mean=np.zeros(2, dtype=np.float32),
        fit_std=np.ones(2, dtype=np.float32),
        manifest_json=np.asarray(json.dumps({"archive_role": role})),
        **origins,
    )


def _allow_fixture_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prepare, "REQUIRED_SOURCE_FILES", ("prepare.py",))


def _loss_arrays(full_score: float, recent_score: float) -> tuple[np.ndarray, np.ndarray]:
    loss = np.zeros((13, 2), dtype=np.float64)
    counts = np.full((13, 2), 42, dtype=np.int64)
    loss[:6, :] = (13 * full_score - 7 * recent_score) / 6 * counts[:6, :]
    loss[6:, :] = recent_score * counts[6:, :]
    return loss, counts


def _trial(root: Path, study: str, dataset: str, method: str, lr_tag: str, lr: float, full: float, recent: float) -> Path:
    base = root / "runs" / study / "trials" / dataset / method / f"lr_{lr_tag}_seed_12000"
    loss, counts = _loss_arrays(full, recent)
    _write_bytes(base / "best_trainable.pt", b"checkpoint")
    checkpoint_hash = hashlib.sha256(b"checkpoint").hexdigest()
    _write_json(
        base / "result.json",
        {
            "completed": True,
            "stage": "fit",
            "dataset": dataset,
            "method": method,
            "seed": 12000,
            "lr": lr,
            "best_step": 40 if method != "F0" else 0,
            "val_score": full,
            "checkpoint_sha256": checkpoint_hash,
            "predictions_sha256": "1" * 64,
            "audits": {"restored_adaptation_sha256": "2" * 64},
            "native_checkpoint_hashes": {"model.safetensors": "3" * 64},
            "source_hashes": {"train.py": "4" * 64},
            "protected_hashes": {},
            "cache_contract_sha256": "5" * 64,
        },
    )
    np.savez_compressed(
        base / "predictions.npz",
        val_loss_sums=loss,
        val_valid_counts=counts,
        val_origins=np.arange(13, dtype=np.int64),
        val_timestamps=np.asarray([f"2020-01-{i + 1:02d}T00:00:00" for i in range(13)]),
        quantiles=np.linspace(0.01, 0.99, 21, dtype=np.float64),
        target_indices=np.asarray([0, 1], dtype=np.int64),
        target_channels=np.asarray(["a", "b"]),
    )
    _write_json(base / "trial_contract.json", {"method": method, "lr": lr})
    return base


def _build_fake_study(root: Path, study: str, selected_h: str, selected_h_lr: str, selected_h_value: float) -> None:
    run_root = root / "runs" / study
    result_root = root / "results" / study
    _write_json(run_root / "study_contract.json", {"contract": study})
    _write_json(run_root / "fit_completed.json", {"completed": True})
    _write_json(run_root / "selection.json", {"completed": True, "choices": {}, "selected": []})
    extra = result_root / "extra_verified.txt"
    extra_sha = _write_hashed(extra, b"extra")
    output_csv = result_root / "all_results.csv"
    output_sha = _write_hashed(output_csv, b"method,score\nF0,1.0\n")
    analysis_source = root / "experiments" / study / "analyse.py"
    analysis_sha = _write_hashed(analysis_source, f"# {study} analysis\n".encode("utf-8"))
    parent_artifact = root / "runs" / study / "parent_artifact.json"
    parent_sha = _write_hashed(parent_artifact, f'{{"study":"{study}"}}'.encode("utf-8"))
    _write_json(
        result_root / "verification.json",
        {
            "passed": True,
            "artifact_hashes": {str(extra.relative_to(root)).replace("\\", "/"): extra_sha},
            "output_hashes": {"all_results.csv": output_sha},
            "analysis_sources": {str(analysis_source.relative_to(root)).replace("\\", "/"): analysis_sha},
            "parent_artifacts": {str(parent_artifact.relative_to(root)).replace("\\", "/"): parent_sha},
        },
    )
    figure_path = _plot_manifest_path(root, study).parent / "figure.png"
    figure_sha = _write_hashed(figure_path, b"figure")
    plot_manifest = _plot_manifest_path(root, study)
    _write_json(
        plot_manifest,
        {
            "completed": True,
            "input_hashes": {str(output_csv): output_sha},
            "figure_hashes": {"figure.png": figure_sha},
        },
    )
    report_sha = _write_hashed(_report_path(root, study), f"# {study} report\n".encode("utf-8"))
    windows_sha = _write_hashed(result_root / "windows_event_audit.json", b"windows")
    final_audit = {
        "passed": True,
        "report_sha256": report_sha,
        "verification_sha256": prepare.sha256_file(result_root / "verification.json"),
        "plot_manifest_sha256": prepare.sha256_file(plot_manifest),
        "windows_event_audit_sha256": windows_sha,
    }
    if study == "peft_temporal_replication_v1":
        final_audit["recovery_audit_sha256"] = _write_hashed(result_root / "recovery_audit.json", b"recovery")
    _write_json(result_root / "final_audit.json", final_audit)
    for dataset in ("bike", "household"):
        _write_minimal_archive(run_root / "prepared" / f"{dataset}_fit.npz", role="fit")
        _write_minimal_archive(run_root / "prepared" / f"{dataset}_holdout.npz", role="holdout")
        _trial(root, study, dataset, "F0", "0e+00", 0.0, 5.0, 5.0)
        _trial(root, study, dataset, "H_MLP", "1e-04", 1e-4, 4.4, 4.4)
        _trial(root, study, dataset, "H_MLP", "3e-04", 3e-4, 4.3, 4.3)
        _trial(root, study, dataset, "H_MLP", "1e-03", 1e-3, 4.2, 4.2)
        _trial(root, study, dataset, "H_FULL", "3e-05", 3e-5, selected_h_value, 3.0)
        _trial(root, study, dataset, "H_FULL", "1e-04", 1e-4, 3.8, 2.8)
        _trial(root, study, dataset, "H_FULL", "3e-04", 3e-4, 3.9, 2.7)
        _trial(root, study, dataset, "OFF_LORA", "1e-05", 1e-5, 2.5, 3.5)
        _trial(root, study, dataset, "OFF_LORA", "3e-05", 3e-5, 2.0, 2.0)
        _trial(root, study, dataset, "OFF_LORA", "1e-04", 1e-4, 2.2, 1.5)
        selection = json.loads((run_root / "selection.json").read_text(encoding="utf-8"))
        selection["choices"][dataset] = {
            "H": {
                "dataset": dataset,
                "method": selected_h,
                "lr": float(selected_h_lr),
                "seed": 12000,
                "path": str(run_root / "trials" / dataset / selected_h / f"lr_{prepare.lr_tag(float(selected_h_lr))}_seed_12000"),
            },
            "OFF_LORA": {
                "dataset": dataset,
                "method": "OFF_LORA",
                "lr": 3e-5,
                "seed": 12000,
                "path": str(run_root / "trials" / dataset / "OFF_LORA" / "lr_3e-05_seed_12000"),
            },
        }
        _write_json(run_root / "selection.json", selection)
        for role, rel in {
            "F0": f"F0_seed_12000",
            "H": f"H_seed_12000",
            "OFF_LORA": f"OFF_LORA_seed_12000",
        }.items():
            forecast = run_root / "forecasts" / dataset / rel
            _write_bytes(forecast / "result.json", b"not json and should not be parsed")
            _write_bytes(forecast / "predictions.npz", b"holdout bytes only hashed")


def test_score_from_loss_arrays_uses_target_macro_and_recent7():
    loss = np.array([[100.0, 1.0], [100.0, 1.0], [1.0, 100.0]], dtype=np.float64)
    counts = np.array([[100, 1], [100, 1], [1, 100]], dtype=np.int64)

    full = prepare.score_from_loss_arrays(loss, counts)
    recent = prepare.score_from_loss_arrays(loss, counts, rows=[2])

    assert full == pytest.approx(((201 / 201) + (102 / 102)) / 2)
    assert recent == pytest.approx(((1 / 1) + (100 / 100)) / 2)


def test_prepare_freezes_validation_only_selection_and_forecast_jobs(tmp_path, monkeypatch):
    _allow_fixture_sources(monkeypatch)
    _build_fake_study(tmp_path, "peft_external_gap_v1", "H_FULL", "3e-05", 1.9)
    _build_fake_study(tmp_path, "peft_temporal_replication_v1", "H_FULL", "3e-05", 1.9)
    _write_bytes(tmp_path / "_docs" / "notes" / "tsfm_topics" / "14_peft_selection_regret_plan_20260908.md", b"plan")
    out = tmp_path / "runs" / "peft_selection_regret_v1"

    contract = prepare.prepare(tmp_path, out)

    assert contract["completed"] is True
    assert contract["selection_rules_frozen"] is True
    assert len(contract["cells"]) == 4
    assert sum(len(cell["candidates"]) for cell in contract["cells"].values()) == 40
    assert len(contract["unselected_forecast_jobs"]) == 28
    assert len(contract["reuse_forecast_jobs"]) == 12
    for cell_id, cell in contract["cells"].items():
        assert isinstance(cell["candidates"], dict)
        assert set(cell["selectors"]) == set(prepare.SELECTORS)
        assert cell["smoke_candidate"] == cell["selectors"]["LORA_V"]
        chosen = {name: cell["candidates"][candidate_id] for name, candidate_id in cell["selectors"].items()}
        assert cell["selectors"]["F0"].startswith(f"{cell_id}__F0")
        assert chosen["FIXED_LOW"]["method"] == "OFF_LORA"
        assert chosen["FIXED_LOW"]["lr"] == 1e-5
        assert chosen["LORA_V"]["lr"] == 3e-5
        assert chosen["LORA_RECENT7"]["lr"] == 1e-4
        assert chosen["ALL_V"]["method"] == "H_FULL"
        assert chosen["ALL_RECENT7"]["method"] == "OFF_LORA"
        assert chosen["HEAD_V"]["method"] == "H_FULL"
        assert chosen["LORA_V"]["reuse_forecast_path"] is not None
        assert chosen["ALL_RECENT7"]["reuse_forecast_path"] is None
    assert contract["leakage_audit"]["selection_inputs"] == ["trial_predictions_val_loss_sums", "trial_predictions_val_valid_counts"]
    assert contract["leakage_audit"]["holdout_prediction_values_read"] is False
    assert any(path.endswith("extra_verified.txt") for path in contract["protected_hashes"])
    assert "results/peft_external_gap_v1/all_results.csv" in contract["protected_hashes"]
    assert (out / "selection_contract.json").exists()


def test_verify_protected_hashes_detects_tampering(tmp_path, monkeypatch):
    _allow_fixture_sources(monkeypatch)
    _build_fake_study(tmp_path, "peft_external_gap_v1", "H_FULL", "3e-05", 1.9)
    _build_fake_study(tmp_path, "peft_temporal_replication_v1", "H_FULL", "3e-05", 1.9)
    _write_bytes(tmp_path / "_docs" / "notes" / "tsfm_topics" / "14_peft_selection_regret_plan_20260908.md", b"plan")
    contract = prepare.prepare(tmp_path, tmp_path / "runs" / "peft_selection_regret_v1")
    prepare.verify_protected_hashes(contract, tmp_path)

    first_path = next(iter(contract["protected_hashes"]))
    (tmp_path / first_path).write_bytes(b"changed")

    with pytest.raises(AssertionError, match="Protected artifact changed"):
        prepare.verify_protected_hashes(contract, tmp_path)


def test_prepare_rejects_different_existing_contract_and_does_not_rewrite_equal(tmp_path, monkeypatch):
    _allow_fixture_sources(monkeypatch)
    _build_fake_study(tmp_path, "peft_external_gap_v1", "H_FULL", "3e-05", 1.9)
    _build_fake_study(tmp_path, "peft_temporal_replication_v1", "H_FULL", "3e-05", 1.9)
    _write_bytes(tmp_path / "_docs" / "notes" / "tsfm_topics" / "14_peft_selection_regret_plan_20260908.md", b"plan")
    out = tmp_path / "runs" / "peft_selection_regret_v1"
    contract = prepare.prepare(tmp_path, out)

    def fail_write(path, data):
        raise AssertionError("equal existing contract must be returned without rewriting")

    monkeypatch.setattr(prepare, "_write_json", fail_write)
    assert prepare.prepare(tmp_path, out) == contract
    with pytest.raises(TypeError):
        prepare.prepare(tmp_path, out, allow_existing=True)

    (out / "selection_contract.json").write_text(json.dumps({**contract, "tampered": True}), encoding="utf-8")
    with pytest.raises(FileExistsError, match="Existing selection contract differs"):
        prepare.prepare(tmp_path, out)


def test_build_contract_rejects_missing_required_source(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "REQUIRED_SOURCE_FILES", ("prepare.py", "definitely_missing_source.py"))
    _build_fake_study(tmp_path, "peft_external_gap_v1", "H_FULL", "3e-05", 1.9)
    _build_fake_study(tmp_path, "peft_temporal_replication_v1", "H_FULL", "3e-05", 1.9)
    _write_bytes(tmp_path / "_docs" / "notes" / "tsfm_topics" / "14_peft_selection_regret_plan_20260908.md", b"plan")

    with pytest.raises(FileNotFoundError, match="Required study14 source missing"):
        prepare.build_contract(tmp_path, tmp_path / "runs" / "peft_selection_regret_v1")


def test_validate_contract_rebuilds_canonical_contract_and_rejects_tamper(tmp_path, monkeypatch):
    _allow_fixture_sources(monkeypatch)
    _build_fake_study(tmp_path, "peft_external_gap_v1", "H_FULL", "3e-05", 1.9)
    _build_fake_study(tmp_path, "peft_temporal_replication_v1", "H_FULL", "3e-05", 1.9)
    _write_bytes(tmp_path / "_docs" / "notes" / "tsfm_topics" / "14_peft_selection_regret_plan_20260908.md", b"plan")
    out = tmp_path / "runs" / "peft_selection_regret_v1"
    contract = prepare.prepare(tmp_path, out)
    prepare.validate_contract(tmp_path, out / "selection_contract.json")

    tampered = json.loads((out / "selection_contract.json").read_text(encoding="utf-8"))
    tampered["cells"]["s12_bike"]["selectors"]["LORA_V"] = tampered["cells"]["s12_bike"]["selectors"]["FIXED_LOW"]
    (out / "selection_contract.json").write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")

    with pytest.raises(AssertionError, match="does not match canonical"):
        prepare.validate_contract(tmp_path, out / "selection_contract.json")


def test_original_lora_choice_must_match_recalculated_validation_minimum(tmp_path, monkeypatch):
    _allow_fixture_sources(monkeypatch)
    _build_fake_study(tmp_path, "peft_external_gap_v1", "H_FULL", "3e-05", 1.9)
    _build_fake_study(tmp_path, "peft_temporal_replication_v1", "H_FULL", "3e-05", 1.9)
    _write_bytes(tmp_path / "_docs" / "notes" / "tsfm_topics" / "14_peft_selection_regret_plan_20260908.md", b"plan")
    selection_path = tmp_path / "runs" / "peft_external_gap_v1" / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["choices"]["bike"]["OFF_LORA"]["lr"] = 1e-4
    selection["choices"]["bike"]["OFF_LORA"]["path"] = str(
        tmp_path / "runs" / "peft_external_gap_v1" / "trials" / "bike" / "OFF_LORA" / "lr_1e-04_seed_12000"
    )
    _write_json(selection_path, selection)

    with pytest.raises(AssertionError, match="Original OFF_LORA choice"):
        prepare.build_contract(tmp_path, tmp_path / "runs" / "peft_selection_regret_v1")


def test_verification_output_hashes_are_result_relative_and_missing_is_rejected(tmp_path, monkeypatch):
    _allow_fixture_sources(monkeypatch)
    _build_fake_study(tmp_path, "peft_external_gap_v1", "H_FULL", "3e-05", 1.9)
    _build_fake_study(tmp_path, "peft_temporal_replication_v1", "H_FULL", "3e-05", 1.9)
    _write_bytes(tmp_path / "_docs" / "notes" / "tsfm_topics" / "14_peft_selection_regret_plan_20260908.md", b"plan")

    contract = prepare.build_contract(tmp_path, tmp_path / "runs" / "peft_selection_regret_v1")
    assert "results/peft_external_gap_v1/all_results.csv" in contract["protected_hashes"]
    assert "all_results.csv" not in contract["protected_hashes"]

    (tmp_path / "results" / "peft_external_gap_v1" / "all_results.csv").unlink()
    with pytest.raises(FileNotFoundError, match="all_results.csv"):
        prepare.build_contract(tmp_path, tmp_path / "runs" / "peft_selection_regret_v1")


def test_final_audit_references_and_plot_manifest_hashes_are_protected(tmp_path, monkeypatch):
    _allow_fixture_sources(monkeypatch)
    _build_fake_study(tmp_path, "peft_external_gap_v1", "H_FULL", "3e-05", 1.9)
    _build_fake_study(tmp_path, "peft_temporal_replication_v1", "H_FULL", "3e-05", 1.9)
    _write_bytes(tmp_path / "_docs" / "notes" / "tsfm_topics" / "14_peft_selection_regret_plan_20260908.md", b"plan")

    contract = prepare.build_contract(tmp_path, tmp_path / "runs" / "peft_selection_regret_v1")
    protected = contract["protected_hashes"]

    assert "_docs/notes/tsfm_topics/12_peft_external_gap_results_20260908.md" in protected
    assert "_docs/notes/tsfm_topics/13_peft_temporal_replication_results_20260908.md" in protected
    assert "results/peft_external_gap_v1/windows_event_audit.json" in protected
    assert "results/peft_temporal_replication_v1/windows_event_audit.json" in protected
    assert "results/peft_temporal_replication_v1/recovery_audit.json" in protected
    assert "results/peft_external_gap_v1/figures_verified/figure.png" in protected
    assert "results/peft_temporal_replication_v1/figures/figure.png" in protected

    (tmp_path / "results" / "peft_temporal_replication_v1" / "figures" / "figure.png").write_bytes(b"changed")
    with pytest.raises(AssertionError, match="Declared verification hash mismatch"):
        prepare.build_contract(tmp_path, tmp_path / "runs" / "peft_selection_regret_v1")
