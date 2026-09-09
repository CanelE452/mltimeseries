"""Freeze validation-only PEFT selection-regret candidates before future reads."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


STUDY = "peft_selection_regret_v1"
PLAN_PATH = Path("_docs/notes/tsfm_topics/14_peft_selection_regret_plan_20260908.md")
DEFAULT_OUTPUT = Path("runs/peft_selection_regret_v1")
SEED0 = 12000
STUDIES = {12: "peft_external_gap_v1", 13: "peft_temporal_replication_v1"}
DATASETS = ("bike", "household")
SELECTORS = ("F0", "FIXED_LOW", "LORA_V", "LORA_RECENT7", "ALL_V", "ALL_RECENT7", "HEAD_V")
REQUIRED_SOURCE_FILES = ("prepare.py", "forecast.py", "run_study.py", "analyse.py", "plot.py")
LORA_LRS = (1e-5, 3e-5, 1e-4)
H_FULL_LRS = (3e-5, 1e-4, 3e-4)
H_MLP_LRS = (1e-4, 3e-4, 1e-3)
LR_TAGS = {
    0.0: "0e+00",
    1e-5: "1e-05",
    3e-5: "3e-05",
    1e-4: "1e-04",
    3e-4: "3e-04",
    1e-3: "1e-03",
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
REPORT_PATHS = {
    "peft_external_gap_v1": Path("_docs/notes/tsfm_topics/12_peft_external_gap_results_20260908.md"),
    "peft_temporal_replication_v1": Path(
        "_docs/notes/tsfm_topics/13_peft_temporal_replication_results_20260908.md"
    ),
}
PLOT_MANIFEST_PATHS = {
    "peft_external_gap_v1": Path("results/peft_external_gap_v1/figures_verified/plot_manifest.json"),
    "peft_temporal_replication_v1": Path("results/peft_temporal_replication_v1/figures/plot_manifest.json"),
}


def sha256_file(path: Path | str) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    hasher = hashlib.sha256()
    hasher.update(str(arr.dtype).encode("utf-8"))
    hasher.update(json.dumps(arr.shape).encode("utf-8"))
    hasher.update(arr.view(np.uint8))
    return hasher.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def _rel(root: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(resolved)


def _under_root(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _hash_existing(root: Path, path: Path, protected: dict[str, str] | None = None) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    digest = sha256_file(path)
    if protected is not None:
        protected[_rel(root, path)] = digest
    return digest


def lr_tag(lr: float) -> str:
    for value, tag in LR_TAGS.items():
        if abs(float(lr) - value) < 1e-12:
            return tag
    raise ValueError(f"Unsupported learning rate for the fixed study14 grid: {lr!r}")


def candidate_id(cell_id: str, method: str, lr: float, seed: int = SEED0) -> str:
    return f"{cell_id}__{method}__lr_{lr_tag(lr)}__seed_{seed}"


def score_from_loss_arrays(loss_sums: np.ndarray, valid_counts: np.ndarray, rows: list[int] | np.ndarray | None = None) -> float:
    losses = np.asarray(loss_sums, dtype=np.float64)
    counts = np.asarray(valid_counts, dtype=np.float64)
    if losses.ndim != 2 or counts.shape != losses.shape or losses.shape[1] != 2:
        raise ValueError("Expected [origin, 2 target] validation loss sums and counts")
    if rows is not None:
        losses = losses[rows]
        counts = counts[rows]
    total_counts = counts.sum(axis=0)
    if np.any(total_counts <= 0):
        raise ValueError("Each target needs at least one observed validation label")
    return float((losses.sum(axis=0) / total_counts).mean())


def _candidate_specs() -> list[tuple[str, float]]:
    return [("F0", 0.0)] + [("H_MLP", lr) for lr in H_MLP_LRS] + [
        ("H_FULL", lr) for lr in H_FULL_LRS
    ] + [("OFF_LORA", lr) for lr in LORA_LRS]


def _all_priority(candidate: dict[str, Any]) -> tuple[int, float]:
    method = candidate["method"]
    lr = float(candidate["lr"])
    if method == "F0":
        return (0, 0.0)
    if method == "OFF_LORA":
        return (1, lr)
    if method == "H_FULL":
        return (2, lr)
    if method == "H_MLP":
        return (3, lr)
    return (9, lr)


def _forecast_reuse(root: Path, study: str, dataset: str, role: str) -> dict[str, Any]:
    folder = root / "runs" / study / "forecasts" / dataset / f"{role}_seed_{SEED0}"
    result_path = folder / "result.json"
    predictions_path = folder / "predictions.npz"
    return {
        "role": role,
        "path": _rel(root, folder),
        "result_path": _rel(root, result_path),
        "result_sha256": _hash_existing(root, result_path),
        "predictions_path": _rel(root, predictions_path),
        "predictions_sha256": _hash_existing(root, predictions_path),
    }


def _declared_path(root: Path, base: Path, rel_or_abs: str) -> Path:
    path = Path(rel_or_abs)
    if path.is_absolute():
        return path
    return base / path


def _protect_declared_hash(
    root: Path,
    rel_or_abs: str,
    expected: str,
    protected: dict[str, str],
    *,
    base: Path | None = None,
) -> None:
    if not isinstance(expected, str) or not HEX64.match(expected):
        return
    path = _declared_path(root, base or root, rel_or_abs)
    if not path.exists():
        raise FileNotFoundError(f"Declared verification hash target missing: {rel_or_abs}")
    actual = sha256_file(path)
    if actual != expected:
        raise AssertionError(f"Declared verification hash mismatch: {rel_or_abs}")
    protected[_rel(root, path)] = actual


def _protect_declared_hashes(root: Path, payload: Any, protected: dict[str, str], *, base: Path | None = None) -> None:
    if isinstance(payload, dict):
        if isinstance(payload.get("path"), str) and isinstance(payload.get("sha256"), str):
            _protect_declared_hash(root, payload["path"], payload["sha256"], protected, base=base)
        for key, value in payload.items():
            if key in {"path", "sha256"} and isinstance(payload.get("path"), str):
                continue
            if isinstance(key, str) and isinstance(value, str):
                _protect_declared_hash(root, key, value, protected, base=base)
            else:
                _protect_declared_hashes(root, value, protected, base=base)
    elif isinstance(payload, list):
        for value in payload:
            _protect_declared_hashes(root, value, protected, base=base)


def _protect_plot_manifest_references(root: Path, plot_manifest_path: Path, protected: dict[str, str]) -> None:
    manifest = _read_json(plot_manifest_path)
    _protect_declared_hashes(root, manifest.get("input_hashes", {}), protected)
    _protect_declared_hashes(root, manifest.get("figure_hashes", {}), protected, base=plot_manifest_path.parent)


def _require_audit_hash(audit: dict[str, Any], key: str, final_audit_path: Path) -> str:
    digest = audit.get(key)
    if not isinstance(digest, str) or not HEX64.match(digest):
        raise AssertionError(f"Final audit missing {key}: {final_audit_path}")
    return digest


def _protect_final_audit_references(
    root: Path,
    study: str,
    final_audit_path: Path,
    verification_path: Path,
    protected: dict[str, str],
) -> None:
    audit = _read_json(final_audit_path)
    if audit.get("passed") is not True:
        raise AssertionError(f"Final audit is not passed: {final_audit_path}")
    _protect_declared_hash(
        root,
        str(verification_path),
        _require_audit_hash(audit, "verification_sha256", final_audit_path),
        protected,
    )
    _protect_declared_hash(
        root,
        str(REPORT_PATHS[study]),
        _require_audit_hash(audit, "report_sha256", final_audit_path),
        protected,
    )
    _protect_declared_hash(
        root,
        str(root / "results" / study / "windows_event_audit.json"),
        _require_audit_hash(audit, "windows_event_audit_sha256", final_audit_path),
        protected,
    )
    plot_manifest = root / PLOT_MANIFEST_PATHS[study]
    _protect_declared_hash(
        root,
        str(plot_manifest),
        _require_audit_hash(audit, "plot_manifest_sha256", final_audit_path),
        protected,
    )
    _protect_plot_manifest_references(root, plot_manifest, protected)
    if study == "peft_temporal_replication_v1":
        recovery_hash = _require_audit_hash(audit, "recovery_audit_sha256", final_audit_path)
        _protect_declared_hash(root, str(root / "results" / study / "recovery_audit.json"), recovery_hash, protected)


def _protect_verification_references(root: Path, verification_path: Path, protected: dict[str, str]) -> None:
    verification = _read_json(verification_path)
    for key in ("artifact_hashes", "analysis_sources", "parent_artifacts"):
        _protect_declared_hashes(root, verification.get(key, {}), protected)
    _protect_declared_hashes(root, verification.get("output_hashes", {}), protected, base=verification_path.parent)


def _reuse_role(candidate: dict[str, Any], choices: dict[str, Any]) -> str | None:
    if candidate["method"] == "F0" and abs(float(candidate["lr"])) < 1e-12:
        return "F0"
    for role in ("H", "OFF_LORA"):
        chosen = choices.get(role)
        if not chosen:
            continue
        if (
            candidate["method"] == chosen.get("method")
            and int(candidate["seed"]) == int(chosen.get("seed", SEED0))
            and abs(float(candidate["lr"]) - float(chosen.get("lr"))) < 1e-12
        ):
            return role
    return None


def _validation_payload(predictions_path: Path) -> dict[str, Any]:
    with np.load(predictions_path, allow_pickle=False) as archive:
        loss_sums = archive["val_loss_sums"].astype(np.float64)
        valid_counts = archive["val_valid_counts"].astype(np.int64)
        recent = np.arange(max(0, loss_sums.shape[0] - 7), loss_sums.shape[0])
        payload = {
            "val_full_score": score_from_loss_arrays(loss_sums, valid_counts),
            "val_recent7_score": score_from_loss_arrays(loss_sums, valid_counts, rows=recent),
            "val_origin_count": int(loss_sums.shape[0]),
            "val_recent_origin_count": int(len(recent)),
            "val_loss_sums_sha256": sha256_array(loss_sums),
            "val_valid_counts_sha256": sha256_array(valid_counts),
        }
        for key in ("quantiles", "target_indices", "target_channels"):
            if key in archive.files:
                value = archive[key]
                if value.dtype.kind in "US":
                    payload[key] = [str(x) for x in value.tolist()]
                else:
                    payload[key] = np.asarray(value).tolist()
                payload[f"{key}_sha256"] = sha256_array(np.asarray(value))
    return payload


def _archive_payload(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {"path_exists": path.exists()}
    if not path.exists():
        return payload
    with np.load(path, allow_pickle=False) as archive:
        payload.update(
            {
                "context_values_shape": list(archive["context_values"].shape) if "context_values" in archive.files else None,
                "target_values_shape": list(archive["target_values"].shape) if "target_values" in archive.files else None,
                "quantiles_sha256": sha256_array(archive["quantiles"]) if "quantiles" in archive.files else None,
                "target_indices": archive["target_indices"].astype(int).tolist() if "target_indices" in archive.files else None,
                "origin_counts": {
                    split: int(archive[split].shape[0])
                    for split in ("train_origins", "val_origins", "cal_origins", "eval_origins")
                    if split in archive.files
                },
            }
        )
        missing = {"context_values", "target_values", "quantiles", "target_indices"} - set(archive.files)
        if missing:
            raise KeyError(f"Prepared archive missing required keys: {sorted(missing)}")
        if archive["quantiles"].dtype != np.float64:
            raise ValueError(f"Prepared archive quantiles must be float64: {path}")
        if archive["target_indices"].astype(int).tolist() != [0, 1]:
            raise ValueError(f"Prepared archive target indices changed: {path}")
        if not any(
            split in archive.files and archive[split].shape[0] > 0
            for split in ("train_origins", "val_origins", "cal_origins", "eval_origins")
        ):
            raise ValueError(f"Prepared archive has no origins: {path}")
    return payload


def _build_candidate(
    root: Path,
    output_dir: Path,
    cell_id_value: str,
    study: str,
    dataset: str,
    method: str,
    lr: float,
    choices: dict[str, Any],
    protected: dict[str, str],
) -> dict[str, Any]:
    trial_path = root / "runs" / study / "trials" / dataset / method / f"lr_{lr_tag(lr)}_seed_{SEED0}"
    result_path = trial_path / "result.json"
    predictions_path = trial_path / "predictions.npz"
    checkpoint_path = trial_path / "best_trainable.pt"
    result = _read_json(result_path)
    if result.get("completed") is not True or result.get("stage") != "fit":
        raise AssertionError(f"Incomplete fit result: {result_path}")
    if result.get("method") != method or result.get("dataset") != dataset or int(result.get("seed")) != SEED0:
        raise AssertionError(f"Fit result identity mismatch: {result_path}")
    if abs(float(result.get("lr", 0.0)) - lr) >= 1e-12:
        raise AssertionError(f"Fit result LR mismatch: {result_path}")
    validation = _validation_payload(predictions_path)
    if abs(float(result["val_score"]) - validation["val_full_score"]) > 1e-8:
        raise AssertionError(f"Validation arrays do not reproduce result val_score: {result_path}")
    result_sha = _hash_existing(root, result_path, protected)
    pred_sha = _hash_existing(root, predictions_path, protected)
    cp_sha = _hash_existing(root, checkpoint_path, protected)
    trial_contract = trial_path / "trial_contract.json"
    trial_contract_sha = _hash_existing(root, trial_contract, protected) if trial_contract.exists() else None
    if result.get("checkpoint_sha256") and result["checkpoint_sha256"] != cp_sha:
        raise AssertionError(f"Checkpoint hash mismatch: {checkpoint_path}")
    role = _reuse_role({"method": method, "lr": lr, "seed": SEED0}, choices)
    reuse = _forecast_reuse(root, study, dataset, role) if role else None
    candidate = {
        "candidate_id": candidate_id(cell_id_value, method, lr),
        "cell_id": cell_id_value,
        "study": study,
        "dataset": dataset,
        "method": method,
        "seed": SEED0,
        "lr": lr,
        "lr_tag": lr_tag(lr),
        "fit_path": _rel(root, trial_path),
        "fit_result_path": _rel(root, result_path),
        "fit_result_sha256": result_sha,
        "fit_predictions_path": _rel(root, predictions_path),
        "fit_predictions_sha256": pred_sha,
        "checkpoint_path": _rel(root, checkpoint_path),
        "checkpoint_sha256": cp_sha,
        "reported_checkpoint_sha256": result.get("checkpoint_sha256"),
        "trial_contract_path": _rel(root, trial_contract) if trial_contract.exists() else None,
        "trial_contract_sha256": trial_contract_sha,
        "restored_adaptation_sha256": result.get("audits", {}).get("restored_adaptation_sha256"),
        "native_checkpoint_hashes": result.get("native_checkpoint_hashes", {}),
        "fit_source_hashes": result.get("source_hashes", {}),
        "fit_protected_hashes": result.get("protected_hashes", {}),
        "best_step": result.get("best_step"),
        "reported_val_score": result.get("val_score"),
        "reuse_forecast": reuse,
        "reuse_forecast_path": reuse["path"] if reuse else None,
        "needs_forecast": reuse is None,
        "new_forecast_output": _rel(root, output_dir / "forecasts" / cell_id_value / candidate_id(cell_id_value, method, lr)),
        **validation,
    }
    return candidate


def _select(candidates: list[dict[str, Any]], choices: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    selected: dict[str, dict[str, Any]] = {}
    selected["F0"] = min((c for c in candidates if c["method"] == "F0"), key=lambda c: c["candidate_id"])
    selected["FIXED_LOW"] = next(c for c in candidates if c["method"] == "OFF_LORA" and abs(float(c["lr"]) - 1e-5) < 1e-12)
    selected["LORA_V"] = min(
        (c for c in candidates if c["method"] == "OFF_LORA"),
        key=lambda c: (c["val_full_score"], float(c["lr"])),
    )
    selected["LORA_RECENT7"] = min(
        (c for c in candidates if c["method"] == "OFF_LORA"),
        key=lambda c: (c["val_recent7_score"], float(c["lr"])),
    )
    selected["ALL_V"] = min(candidates, key=lambda c: (c["val_full_score"], _all_priority(c)))
    selected["ALL_RECENT7"] = min(candidates, key=lambda c: (c["val_recent7_score"], _all_priority(c)))
    head = choices.get("H")
    if head:
        head_id = candidate_id("", head["method"], float(head["lr"])).split("__", 1)[1]
        selected["HEAD_V"] = next(c for c in candidates if c["candidate_id"].endswith(head_id))
    else:
        selected["HEAD_V"] = min(
            (c for c in candidates if c["method"].startswith("H_")),
            key=lambda c: (c["val_full_score"], c["method"], float(c["lr"])),
        )
    if set(selected) != set(SELECTORS) or any(c["candidate_id"] not in by_id for c in selected.values()):
        raise AssertionError("Incomplete selector result")
    return {name: _selection_view(candidate) for name, candidate in selected.items()}


def _assert_original_lora_choice_matches(selected: dict[str, dict[str, Any]], choices: dict[str, Any], cell: str) -> None:
    original = choices.get("OFF_LORA")
    chosen = selected["LORA_V"]
    if not original:
        raise AssertionError(f"Original OFF_LORA choice missing for {cell}")
    if (
        original.get("method") != chosen["method"]
        or int(original.get("seed", SEED0)) != int(chosen["seed"])
        or abs(float(original.get("lr")) - float(chosen["lr"])) >= 1e-12
    ):
        raise AssertionError(f"Original OFF_LORA choice disagrees with recalculated LORA_V in {cell}")


def _selection_view(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate["candidate_id"],
        "method": candidate["method"],
        "lr": candidate["lr"],
        "seed": candidate["seed"],
        "val_full_score": candidate["val_full_score"],
        "val_recent7_score": candidate["val_recent7_score"],
        "needs_forecast": candidate["needs_forecast"],
        "reuse_role": candidate["reuse_forecast"]["role"] if candidate["reuse_forecast"] else None,
    }


def _study_choice(root: Path, study: str, dataset: str) -> dict[str, Any]:
    selection_path = root / "runs" / study / "selection.json"
    selection = _read_json(selection_path)
    if selection.get("completed") is not True:
        raise AssertionError(f"Study selection is not complete: {selection_path}")
    return selection.get("choices", {}).get(dataset, {})


def _source_hashes(root: Path, protected: dict[str, str]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    sources: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    module_dir = Path(__file__).resolve().parent
    for name in REQUIRED_SOURCE_FILES:
        path = module_dir / name
        if path.exists():
            digest = sha256_file(path)
            sources[name] = {"path": _rel(root, path), "sha256": digest}
            if _under_root(root, path):
                protected[_rel(root, path)] = digest
        else:
            missing.append(name)
    if missing:
        raise FileNotFoundError(f"Required study14 source missing: {', '.join(missing)}")
    plan = root / PLAN_PATH
    digest = _hash_existing(root, plan, protected)
    sources["plan14"] = {"path": _rel(root, plan), "sha256": digest}
    return sources, missing


def _build_cell(root: Path, output_dir: Path, study_number: int, dataset: str, protected: dict[str, str]) -> dict[str, Any]:
    study = STUDIES[study_number]
    cell = f"s{study_number}_{dataset}"
    run_root = root / "runs" / study
    result_root = root / "results" / study
    selection_path = run_root / "selection.json"
    study_contract_path = run_root / "study_contract.json"
    verification_path = result_root / "verification.json"
    final_audit_path = result_root / "final_audit.json"
    fit_path = run_root / "prepared" / f"{dataset}_fit.npz"
    holdout_path = run_root / "prepared" / f"{dataset}_holdout.npz"
    choices = _study_choice(root, study, dataset)
    candidate_list = [
        _build_candidate(root, output_dir, cell, study, dataset, method, lr, choices, protected)
        for method, lr in _candidate_specs()
    ]
    if len({c["candidate_id"] for c in candidate_list}) != 10:
        raise AssertionError(f"Expected 10 unique candidates for {cell}")
    selected = _select(candidate_list, choices)
    _assert_original_lora_choice_matches(selected, choices, cell)
    selectors = {name: view["candidate_id"] for name, view in selected.items()}
    candidates = {candidate["candidate_id"]: candidate for candidate in candidate_list}
    _protect_verification_references(root, verification_path, protected)
    if final_audit_path.exists():
        _protect_final_audit_references(root, study, final_audit_path, verification_path, protected)
    cell_payload = {
        "cell_id": cell,
        "study_number": study_number,
        "study": study,
        "dataset": dataset,
        "fit_data_path": _rel(root, fit_path),
        "fit_data_sha256": _hash_existing(root, fit_path, protected),
        "holdout_data_path": _rel(root, holdout_path),
        "holdout_data_sha256": _hash_existing(root, holdout_path, protected),
        "fit_archive": _archive_payload(fit_path),
        "holdout_archive": _archive_payload(holdout_path),
        "selection_json_path": _rel(root, selection_path),
        "selection_json_sha256": _hash_existing(root, selection_path, protected),
        "study_contract_path": _rel(root, study_contract_path),
        "study_contract_sha256": _hash_existing(root, study_contract_path, protected),
        "verification_path": _rel(root, verification_path),
        "verification_sha256": _hash_existing(root, verification_path, protected),
        "final_audit_path": _rel(root, final_audit_path) if final_audit_path.exists() else None,
        "final_audit_sha256": _hash_existing(root, final_audit_path, protected) if final_audit_path.exists() else None,
        "candidate_ids": [candidate["candidate_id"] for candidate in candidate_list],
        "candidates": candidates,
        "selectors": selectors,
        "smoke_candidate": selectors["LORA_V"],
        "selected": selected,
    }
    return cell_payload


def build_contract(root: Path | str = Path("."), output_dir: Path | str = DEFAULT_OUTPUT) -> dict[str, Any]:
    root = Path(root).resolve()
    output_dir = (root / output_dir).resolve() if not Path(output_dir).is_absolute() else Path(output_dir).resolve()
    protected: dict[str, str] = {}
    source_hashes, missing_sources = _source_hashes(root, protected)
    cells = {
        f"s{study_number}_{dataset}": _build_cell(root, output_dir, study_number, dataset, protected)
        for study_number in (12, 13)
        for dataset in DATASETS
    }
    all_candidates = [candidate for cell in cells.values() for candidate in cell["candidates"].values()]
    unselected_jobs = [
        {
            "cell_id": candidate["cell_id"],
            "candidate_id": candidate["candidate_id"],
            "study": candidate["study"],
            "dataset": candidate["dataset"],
            "method": candidate["method"],
            "lr": candidate["lr"],
            "seed": candidate["seed"],
            "fit_path": candidate["fit_path"],
            "checkpoint_path": candidate["checkpoint_path"],
            "output": candidate["new_forecast_output"],
        }
        for candidate in all_candidates
        if candidate["needs_forecast"]
    ]
    reuse_jobs = [
        {
            "cell_id": candidate["cell_id"],
            "candidate_id": candidate["candidate_id"],
            "study": candidate["study"],
            "dataset": candidate["dataset"],
            "method": candidate["method"],
            "lr": candidate["lr"],
            "seed": candidate["seed"],
            "reuse_forecast": candidate["reuse_forecast"],
        }
        for candidate in all_candidates
        if not candidate["needs_forecast"]
    ]
    if len(all_candidates) != 40 or len(unselected_jobs) != 28 or len(reuse_jobs) != 12:
        raise AssertionError("Study14 expects 40 candidates, 28 new forecasts, and 12 reused forecasts")
    return {
        "completed": True,
        "selection_rules_frozen": True,
        "study": STUDY,
        "plan_path": str(PLAN_PATH).replace("\\", "/"),
        "plan_sha256": source_hashes["plan14"]["sha256"],
        "source_hashes": source_hashes,
        "missing_optional_sources": missing_sources,
        "protected_hashes": dict(sorted(protected.items())),
        "cells": cells,
        "selectors": {
            "F0": "seed0 frozen base model",
            "FIXED_LOW": "OFF_LORA lr=1e-5 with its existing validation-selected checkpoint",
            "LORA_V": "minimum full validation SORT score among three seed0 LoRA candidates",
            "LORA_RECENT7": "minimum last-seven validation origins SORT score among three seed0 LoRA candidates",
            "ALL_V": "minimum full validation SORT score among F0 and nine adaptation candidates with fixed tie priority",
            "ALL_RECENT7": "minimum last-seven validation origins SORT score among all ten candidates",
            "HEAD_V": "original H-family validation choice from the source study selection.json",
        },
        "selected": {cell_id: cell["selected"] for cell_id, cell in cells.items()},
        "unselected_forecast_jobs": unselected_jobs,
        "reuse_forecast_jobs": reuse_jobs,
        "leakage_audit": {
            "selection_inputs": ["trial_predictions_val_loss_sums", "trial_predictions_val_valid_counts"],
            "holdout_prediction_values_read": False,
            "holdout_forecast_files_used_for_selection": False,
            "holdout_forecast_hash_only_before_selection_freeze": True,
            "c_cal_or_eval_metrics_read": False,
            "nested_validation": False,
            "note": "Existing fit checkpoints were already validation-selected inside studies 12/13; this contract freezes a post-hoc diagnostic selector before additional future forecasts.",
        },
    }


def prepare(
    root: Path | str = Path("."),
    output_dir: Path | str = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    root = Path(root).resolve()
    output_dir = (root / output_dir).resolve() if not Path(output_dir).is_absolute() else Path(output_dir).resolve()
    contract = build_contract(root, output_dir)
    path = output_dir / "selection_contract.json"
    if path.exists():
        existing = _read_json(path)
        if existing != contract:
            raise FileExistsError(f"Existing selection contract differs: {path}")
        return existing
    _write_json(path, contract)
    return contract


def run(root: Path | str = Path("."), output_dir: Path | str = DEFAULT_OUTPUT) -> dict[str, Any]:
    return prepare(root=root, output_dir=output_dir)


def verify_protected_hashes(contract: dict[str, Any], root: Path | str = Path(".")) -> None:
    root = Path(root).resolve()
    for rel, expected in contract.get("protected_hashes", {}).items():
        path = Path(rel)
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            raise AssertionError(f"Protected artifact missing: {rel}")
        actual = sha256_file(path)
        if actual != expected:
            raise AssertionError(f"Protected artifact changed: {rel}")


def validate_contract(root: Path | str, path: Path | str) -> dict[str, Any]:
    root = Path(root).resolve()
    path = Path(path)
    contract = _read_json(path)
    output_dir = path.parent
    canonical = build_contract(root, output_dir)
    if contract != canonical:
        raise AssertionError("Selection contract does not match canonical build")
    if contract.get("completed") is not True or contract.get("selection_rules_frozen") is not True:
        raise AssertionError("Selection contract is not frozen")
    if contract.get("study") != STUDY or set(contract.get("selectors", {})) != set(SELECTORS):
        raise AssertionError("Selection contract identity or rules changed")
    cells = contract.get("cells", {})
    if set(cells) != {f"s{study}_{dataset}" for study in (12, 13) for dataset in DATASETS}:
        raise AssertionError("Unexpected selection-regret cells")
    all_ids = set()
    for cell_id_value, cell in cells.items():
        candidates = cell.get("candidates", {})
        if not isinstance(candidates, dict):
            raise AssertionError(f"Candidates must be a mapping in {cell_id_value}")
        candidate_ids = list(candidates)
        if len(candidate_ids) != 10 or len(set(candidate_ids)) != 10:
            raise AssertionError(f"Expected 10 candidates in {cell_id_value}")
        all_ids.update(candidate_ids)
        selectors = cell.get("selectors", {})
        if set(selectors) != set(SELECTORS):
            raise AssertionError(f"Expected all selectors in {cell_id_value}")
        if any(candidate_id not in candidate_ids for candidate_id in selectors.values()):
            raise AssertionError(f"Selector points outside the candidate universe in {cell_id_value}")
        if cell.get("smoke_candidate") != selectors["LORA_V"]:
            raise AssertionError(f"Smoke candidate must be the LORA_V choice in {cell_id_value}")
    if len(contract.get("unselected_forecast_jobs", [])) != 28 or len(contract.get("reuse_forecast_jobs", [])) != 12:
        raise AssertionError("Unexpected forecast job split")
    if any(job["candidate_id"] not in all_ids for job in contract["unselected_forecast_jobs"]):
        raise AssertionError("Forecast job points outside candidate universe")
    if any(job["candidate_id"] not in all_ids for job in contract["reuse_forecast_jobs"]):
        raise AssertionError("Reuse job points outside candidate universe")
    verify_protected_hashes(contract, root)
    return contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    contract = prepare(args.root, args.output)
    path = (args.root / args.output / "selection_contract.json").resolve() if not args.output.is_absolute() else args.output / "selection_contract.json"
    print(json.dumps({"completed": True, "path": str(path), "cells": len(contract["cells"]),
                      "candidates": sum(len(cell["candidates"]) for cell in contract["cells"].values()),
                      "new_forecast_jobs": len(contract["unselected_forecast_jobs"]),
                      "reuse_forecast_jobs": len(contract["reuse_forecast_jobs"])}, indent=2))


if __name__ == "__main__":
    main()
