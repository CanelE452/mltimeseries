"""CPU-only SORT/QCAL closure on the already observed train-lag development study."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "2"

import numpy as np


STUDY = "peft_calibration_closure_v1"
PARENT = "peft_trainlag_v1"
METHODS = ("F0", "ATTN", "ALIGN_F0", "ALIGN_ATTN")
PROCEDURES = ("SORT", "QCAL")
PLAN = "_docs/notes/tsfm_topics/11_peft_calibration_closure_plan_20260908.md"
PARENT_PLAN = "_docs/notes/tsfm_topics/10_peft_trainlag_plan_20260908.md"
BOOTSTRAP_SEED = 2026090811
BOOTSTRAP_REPLICATES = 4000
FIELDS = ("method", "procedure", "corpus", "lr", "best_step", "source_path", "score", "coverage80",
          "width80", "median_mse", "crossing", "native_score", "native_coverage80", "native_width80",
          "score_change_from_native_over_F0", "F0_denominator", "calibration_labels", "offset_parameters")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def validate_prediction(prediction, target, quantiles):
    prediction, target, quantiles = (np.asarray(value, dtype=np.float64) for value in (prediction, target, quantiles))
    if prediction.ndim != 3 or target.ndim != 2 or prediction.shape != (len(target), len(quantiles), target.shape[1]):
        raise ValueError("Expected prediction[N,Q,H], target[N,H], quantiles[Q]")
    if quantiles.ndim != 1 or np.any(np.diff(quantiles) <= 0) or np.any((quantiles <= 0) | (quantiles >= 1)):
        raise ValueError("Quantile levels must be strictly increasing inside (0,1)")
    if not all(np.isfinite(value).all() for value in (prediction, target, quantiles)):
        raise ValueError("This complete-data diagnostic requires finite predictions and labels")
    return prediction, target, quantiles


def sort_quantiles(prediction):
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.ndim != 3 or not np.isfinite(prediction).all():
        raise ValueError("Expected finite prediction[N,Q,H]")
    return np.sort(prediction, axis=1)


def fit_offsets(val_prediction, val_target, quantiles):
    """Fit only 21 marginal offsets; this API cannot receive evaluation or oracle arrays."""
    prediction, target, quantiles = validate_prediction(val_prediction, val_target, quantiles)
    sorted_prediction = sort_quantiles(prediction)
    residuals = target[:, None, :] - sorted_prediction
    return np.asarray([np.quantile(residuals[:, index, :].reshape(-1), float(q), method="linear")
                       for index, q in enumerate(quantiles)], dtype=np.float64)


def apply_offsets(prediction, offsets):
    sorted_prediction = sort_quantiles(prediction)
    offsets = np.asarray(offsets, dtype=np.float64)
    if offsets.shape != (sorted_prediction.shape[1],) or not np.isfinite(offsets).all():
        raise ValueError("One finite offset per quantile is required")
    return sort_quantiles(sorted_prediction + offsets[None, :, None])


def evaluate(prediction, target, quantiles):
    prediction, target, quantiles = validate_prediction(prediction, target, quantiles)
    error = target[:, None, :] - prediction
    losses = (2 * np.maximum(error * quantiles[None, :, None],
                             error * (quantiles[None, :, None] - 1))).mean(axis=(1, 2))
    indices = []
    for level in (.1, .5, .9):
        matches = np.flatnonzero(np.isclose(quantiles, level, rtol=0, atol=1e-7))
        if len(matches) != 1:
            raise ValueError("The quantile grid must contain unique .1/.5/.9 levels")
        indices.append(int(matches[0]))
    lower, median, upper = (prediction[:, index] for index in indices)
    coverage = ((target >= lower) & (target <= upper)).mean(axis=1)
    return {"score": float(losses.mean()), "coverage80": float(coverage.mean()),
            "width80": float((upper - lower).mean()), "median_mse": float(((median - target)**2).mean()),
            "crossing": float((np.diff(prediction, axis=1) < 0).mean())}, losses, coverage


def bootstrap_weights(n=512, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    return np.random.default_rng(seed).multinomial(n, np.full(n, 1 / n), size=replicates) / n


def paired_effects(sort_losses, qcal_losses, f0_losses, sort_coverage, qcal_coverage, weights):
    values = [np.asarray(value, dtype=np.float64) for value in
              (sort_losses, qcal_losses, f0_losses, sort_coverage, qcal_coverage)]
    if any(value.shape != values[0].shape for value in values) or values[0].ndim != 2 or values[0].shape[0] != 3:
        raise ValueError("Use matching [three fitted corpora, shared episodes] arrays")
    if weights.ndim != 2 or weights.shape[1] != values[0].shape[1] or not np.allclose(weights.sum(axis=1), 1):
        raise ValueError("Bootstrap must resample the common episode axis with unit total weights")
    if not all(np.isfinite(value).all() for value in values) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Bootstrap inputs must be finite and weights nonnegative")
    sorted_loss, calibrated_loss, f0, sorted_coverage, calibrated_coverage = values
    numerator = calibrated_loss - sorted_loss
    denominator = weights @ f0.mean(axis=0)
    if np.any(denominator <= 0) or np.any(f0.mean(axis=1) <= 0):
        raise ValueError("Native F0 score denominators must be positive")
    score_draws = (weights @ numerator.mean(axis=0)) / denominator
    coverage_delta = calibrated_coverage - sorted_coverage
    coverage_draws = weights @ coverage_delta.mean(axis=0)
    return {"score_change_over_F0": {"value": float(numerator.mean() / f0.mean()),
                "corpus_values": (numerator.mean(axis=1) / f0.mean(axis=1)).tolist(),
                "ci95": np.quantile(score_draws, [.025, .975]).tolist(),
                "definition": "(QCAL score - SORT score) / original native F0 score; positive is worse"},
            "coverage_change": {"value": float(coverage_delta.mean()),
                "corpus_values": coverage_delta.mean(axis=1).tolist(),
                "ci95": np.quantile(coverage_draws, [.025, .975]).tolist(),
                "unit": "fraction; multiply by 100 for percentage points"}}


def closure_decision(sort_scores, qcal_scores, f0_scores, qcal_coverages):
    values = [np.asarray(value, dtype=np.float64) for value in (sort_scores, qcal_scores, f0_scores, qcal_coverages)]
    if any(value.shape != (3,) or not np.isfinite(value).all() for value in values) or np.any(values[2] <= 0):
        raise ValueError("Closure requires three finite corpus results and positive native F0 scores")
    score_change = (values[1] - values[0]) / values[2]
    coverage = float(values[3].mean())
    score_ok, coverage_ok = bool(np.all(score_change <= .01)), bool(.78 <= coverage <= .82)
    return {"supports_simple_calibration_closure": score_ok and coverage_ok,
            "all_corpora_score_worsening_at_most_1pct_F0": score_ok,
            "mean_coverage_between_78_and_82pct": coverage_ok,
            "corpus_score_change_over_F0": score_change.tolist(),
            "corpus_qcal_coverage80": values[3].tolist(), "mean_qcal_coverage80": coverage,
            "interpretation": "Supports simple correction of this observed development phenomenon"
                              if score_ok and coverage_ok else "Criteria unmet; this does not establish a need for new PEFT"}


def verify_episode_pairing(reference_ids, reference_target, ids, target):
    if len(np.unique(ids)) != len(ids) or not np.array_equal(ids, reference_ids) or not np.array_equal(target, reference_target):
        raise AssertionError("Predictions are not paired on the same ordered episodes and targets")


def verify_hashes(hashes):
    for name, expected in hashes.items():
        if file_hash(name) != expected:
            raise AssertionError(f"Protected source/input/artifact changed: {name}")


def selected_entries(selected, trials, choices):
    lookup = {(int(entry["corpus"]), entry["method"], float(entry["lr"])): entry for entry in trials}
    expected_trials = {(c, method, lr) for c in range(3) for method in METHODS
                       for lr in ((3e-5, 1e-4) if method in ("ATTN", "ALIGN_ATTN") else (.001,))}
    if len(trials) != 18 or set(lookup) != expected_trials or set(choices) != {"ATTN", "ALIGN_ATTN"}:
        raise AssertionError("Source study selection grid is incomplete")
    for (corpus, method, lr), entry in lookup.items():
        expected_path = f"runs/{PARENT}/trials/Q00_c{corpus}/{method}/lr_{lr:.0e}"
        expected_mode = "aligned" if method.startswith("ALIGN_") else "raw"
        if entry["path"] != expected_path or entry["input_mode"] != expected_mode or entry["condition"] != "Q00":
            raise AssertionError("Source trial identity or canonical path changed")
    for method, choice in choices.items():
        candidates = [lookup[(0, method, lr)] for lr in (3e-5, 1e-4)]
        if choice["candidates"] != candidates or choice["lr"] != min(candidates, key=lambda e: (e["val_score"], e["lr"]))["lr"]:
            raise AssertionError("Source LR was not fixed by corpus0 validation")
    wanted = {(c, method, choices[method]["lr"] if method in choices else .001) for c in range(3) for method in METHODS}
    keys = [(int(entry["corpus"]), entry["method"], float(entry["lr"])) for entry in selected]
    if len(selected) != 12 or set(keys) != wanted or any(entry != lookup[key] for entry, key in zip(selected, keys)):
        raise AssertionError("Only the twelve previously selected FM procedures may be calibrated")
    return sorted(selected, key=lambda e: (e["corpus"], METHODS.index(e["method"])))


def source_contract(root):
    source_results, source_runs = root / "results" / PARENT, root / "runs" / PARENT
    verification_path = source_results / "verification.json"
    verification = read_json(verification_path)
    contract = read_json(source_runs / "study_contract.json")
    if not verification.get("passed") or verification.get("gpu_trials") != 18:
        raise AssertionError("The source study must have a completed verified result")
    if hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest() != verification["contract_sha256"]:
        raise AssertionError("Source verification and execution contract disagree")
    done = read_json(source_runs / "completed.json")
    if any(done.get(k) != v for k, v in {"completed": True, "fit_count": 12, "f0_count": 6, "trial_count": 18}.items()):
        raise AssertionError("The source study is incomplete")
    if read_json(source_runs / "smoke_contract.json") != contract or not read_json(source_runs / "smoke_completed.json")["completed"]:
        raise AssertionError("Source S0 and production contracts disagree")
    expected_hashes = {str(root / name): expected for name, expected in {**contract["sources"], **contract["data_files"]}.items()}
    expected_hashes[str(root / PARENT_PLAN)] = contract["plan_sha256"]
    expected_hashes[str(root / "experiments" / PARENT / "analyse.py")] = verification["analysis_sha256"]
    for name, files in verification["artifacts"].items():
        expected_hashes.update({str(root / name / file): expected for file, expected in files.items()})
    expected_hashes.update({str(source_results / name): expected for name, expected in verification["outputs"].items()})
    verify_hashes(expected_hashes)
    entries = selected_entries(read_json(source_runs / "selected.json"), read_json(source_runs / "trials.json"),
                               read_json(source_runs / "selection.json"))
    references = [verification_path, *sorted(source_results.glob("*.json")), *sorted(source_results.glob("*.csv")),
                  *(source_runs / name for name in ("selected.json", "trials.json", "selection.json", "completed.json",
                    "study_contract.json", "smoke_contract.json", "smoke_completed.json")), root / PLAN]
    sources = list((root / "experiments" / STUDY).rglob("*.py")) + [root / "experiments" / STUDY / "README.md"]
    references.extend(sources)
    for entry in entries:
        path = root / entry["path"]
        meta = read_json(path / "result.json")
        guard = read_json(path / "guard/status.json")
        if not meta["completed"] or not meta["wrapped_completed"] or not guard["completed"] or guard["returncode"] != 0 or guard["reasons"]:
            raise AssertionError("Selected source trial or guard is incomplete")
        if (meta["method"], meta["metadata"]["corpus"], meta["lr"], meta["val_score"]) != (
                entry["method"], entry["corpus"], entry["lr"], entry["val_score"]):
            raise AssertionError("Source selected entry does not match its trial")
        expected_hashes.update(meta["wrapper_contract"]["protected_hashes"])
        expected_hashes.update(meta["wrapper_audits"]["cache_hashes"])
        references.extend((path / "wrapper_contract.json", path / "guard/status.json"))
    expected_hashes.update({str(path.resolve()): file_hash(path) for path in references})
    verify_hashes(expected_hashes)
    return entries, {"source_study": PARENT, "source_verification_sha256": file_hash(verification_path),
                     "source_contract_sha256": verification["contract_sha256"], "protected_hashes": expected_hashes,
                     "plan_sha256": file_hash(root / PLAN), "analysis_sha256": file_hash(Path(__file__)),
                     "selected_entries": entries, "bootstrap_seed": BOOTSTRAP_SEED,
                     "bootstrap_replicates": BOOTSTRAP_REPLICATES, "new_gpu_fits": 0,
                     "denominator": "Original native F0, before SORT or QCAL, recomputed per bootstrap draw",
                     "calibration": "Per-q linear empirical quantile of SORT validation residuals, followed by quantile sorting",
                     "scope": "Post-hoc development diagnosis; source evaluation and validation selection were previously observed"}


def run(root):
    started = time.perf_counter()
    root = Path(root).resolve()
    output = root / "results" / STUDY
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preserve existing closure results; this contract permits one production analysis")
    entries, contract = source_contract(root)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "analysis_contract.json", contract)
    offsets, validation_reference = {}, None
    for entry in entries:
        key = f"{entry['method']}_c{entry['corpus']}"
        path = root / entry["path"]
        with np.load(path / "predictions.npz", allow_pickle=False) as archive:
            prediction, target, quantiles, ids = (archive[name].copy() for name in
                ("val_predictions", "val_target", "quantiles", "val_episode_ids"))
        if prediction.shape != (128, 21, 16) or target.shape != (128, 16) or ids.shape != (128,):
            raise AssertionError("Source validation arrays differ from the fixed 128 x 21 x 16 contract")
        if validation_reference is None:
            validation_reference = (ids, target, quantiles)
        verify_episode_pairing(*validation_reference[:2], ids, target)
        if not np.array_equal(quantiles, validation_reference[2]):
            raise AssertionError("Validation quantile grids differ")
        native, _, _ = evaluate(prediction, target, quantiles)
        if not np.isclose(native["score"], entry["val_score"], rtol=1e-6, atol=1e-8):
            raise AssertionError("Source validation score does not reproduce saved predictions")
        fitted = fit_offsets(prediction, target, quantiles)
        offsets[key] = {"offsets": fitted.tolist(), "quantiles": quantiles.tolist(),
                        "fit_input_sha256": array_hash(prediction, target, quantiles),
                        "prediction_file_sha256": file_hash(path / "predictions.npz"),
                        "source_path": entry["path"], "calibration_episodes": 128, "calibration_labels": 2048,
                        "eval_arrays_used_for_fit": False, "reuses_prior_selection_validation": True}
    write_json(output / "offsets.json", offsets)
    rows, losses, coverages, originals, saved_arrays = [], {}, {}, {}, {}
    evaluation_reference = None
    for entry in entries:
        method, corpus = entry["method"], entry["corpus"]
        key = f"{method}_c{corpus}"
        path = root / entry["path"]
        meta = read_json(path / "result.json")
        with np.load(path / "predictions.npz", allow_pickle=False) as archive:
            prediction, target, quantiles, ids = (archive[name].copy() for name in
                ("eval_predictions", "eval_target", "quantiles", "eval_episode_ids"))
        if prediction.shape != (512, 21, 16) or target.shape != (512, 16) or ids.shape != (512,):
            raise AssertionError("Source evaluation arrays differ from the fixed 512 x 21 x 16 contract")
        if evaluation_reference is None:
            evaluation_reference = (ids, target, quantiles)
        verify_episode_pairing(*evaluation_reference[:2], ids, target)
        if not np.array_equal(quantiles, evaluation_reference[2]) or not np.array_equal(quantiles, validation_reference[2]):
            raise AssertionError("Evaluation quantile grids differ")
        native, native_loss, native_coverage = evaluate(prediction, target, quantiles)
        if not np.isclose(native["score"], meta["eval_score"], rtol=1e-6, atol=1e-8):
            raise AssertionError("Source evaluation score does not reproduce saved predictions")
        originals[(method, corpus)] = (native, native_loss, native_coverage)
        for procedure in PROCEDURES:
            corrected = sort_quantiles(prediction) if procedure == "SORT" else apply_offsets(prediction, offsets[key]["offsets"])
            metrics, episode_loss, episode_coverage = evaluate(corrected, target, quantiles)
            losses[(method, procedure, corpus)], coverages[(method, procedure, corpus)] = episode_loss, episode_coverage
            saved_arrays[f"{key}_{procedure}_predictions"] = corrected
            rows.append({"method": method, "procedure": procedure, "corpus": corpus, "lr": entry["lr"],
                         "best_step": meta["best_step"], "source_path": entry["path"], **metrics,
                         "native_score": native["score"], "native_coverage80": native["coverage80"], "native_width80": native["width80"],
                         "calibration_labels": 2048 if procedure == "QCAL" else 0,
                         "offset_parameters": 21 if procedure == "QCAL" else 0})
    if len(rows) != 24 or len(offsets) != 12:
        raise AssertionError("Incomplete 4 FM x 2 procedures x 3 corpora closure grid")
    f0 = np.stack([originals[("F0", corpus)][1] for corpus in range(3)])
    for row in rows:
        denominator = float(f0[row["corpus"]].mean())
        row["F0_denominator"] = denominator
        row["score_change_from_native_over_F0"] = (row["score"] - row["native_score"]) / denominator
    weights = bootstrap_weights()
    effects = {}
    for method in METHODS:
        stack = lambda values, procedure: np.stack([values[(method, procedure, c)] for c in range(3)])
        effects[method] = paired_effects(stack(losses, "SORT"), stack(losses, "QCAL"), f0,
                                        stack(coverages, "SORT"), stack(coverages, "QCAL"), weights)
    primary_rows = {(row["procedure"], row["corpus"]): row for row in rows if row["method"] == "ALIGN_ATTN"}
    decision = closure_decision([primary_rows[("SORT", c)]["score"] for c in range(3)],
                                [primary_rows[("QCAL", c)]["score"] for c in range(3)], f0.mean(axis=1),
                                [primary_rows[("QCAL", c)]["coverage80"] for c in range(3)])
    summary = []
    for method in METHODS:
        for procedure in PROCEDURES:
            group = [row for row in rows if (row["method"], row["procedure"]) == (method, procedure)]
            summary.append({"method": method, "procedure": procedure, **{
                field: float(np.mean([row[field] for row in group])) for field in
                ("score", "coverage80", "width80", "median_mse", "crossing")}})
    with (root / "results" / PARENT / "selected_results.csv").open(encoding="utf-8", newline="") as stream:
        old_rows = list(csv.DictReader(stream))
    references = [{key: row[key] for key in ("method", "corpus", "score", "coverage80", "width80", "median_mse")}
                  for row in old_rows if row["method"] in ("RAW", "ORACLE")]
    verify_hashes(contract["protected_hashes"])
    with (output / "all_results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    saved_arrays.update({"eval_episode_ids": evaluation_reference[0], "eval_target": evaluation_reference[1],
                         "quantiles": evaluation_reference[2]})
    np.savez_compressed(output / "calibration_predictions.npz", **saved_arrays)
    write_json(output / "summary.json", {"summary": summary, "primary_method": "ALIGN_ATTN", "decision": decision,
               "previous_diagnostic_references": references, "references_used_for_calibration": False,
               "limits": ["Same validation reused for checkpoint/LR selection and calibration; no conformal coverage guarantee",
                          "Previously observed Gaussian development evaluation, not a fresh confirmation or rejection of C",
                          "Simple closure success concerns this observed phenomenon; failure does not justify a new adapter"]})
    write_json(output / "effects.json", {"effects": effects, "replicates": BOOTSTRAP_REPLICATES,
               "seed": BOOTSTRAP_SEED, "paired_episodes": 512, "fitted_corpora": 3, "confidence": .95,
               "descriptive_intervals_only": True, "resample_training_or_validation": False,
               "multiplicity": "Four descriptive comparisons without a joint family-wise error guarantee"})
    verify_hashes(contract["protected_hashes"])
    outputs = {name: file_hash(output / name) for name in ("all_results.csv", "summary.json", "effects.json",
               "offsets.json", "calibration_predictions.npz", "analysis_contract.json")}
    verified = {"completed": True, "passed": True, "row_count": 24, "procedures": 8, "corpora": 3,
                "new_gpu_fits": 0, "new_calibration_fits": 12, "val_only_offsets": True,
                "source_prediction_scores_recomputed": True, "shared_episode_pairing_verified": True,
                "source_contract_hashes_verified_before_after": True, "analysis_sha256": file_hash(Path(__file__)),
                "protected_file_count": len(contract["protected_hashes"]), "outputs": outputs,
                "wall_seconds": time.perf_counter() - started}
    write_json(output / "verification.json", verified)
    print(json.dumps({"completed": True, "rows": 24, "seconds": verified["wall_seconds"], "output": str(output)}), flush=True)
    return verified


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    run(Path(__file__).resolve().parents[2])


if __name__ == "__main__":
    main()
