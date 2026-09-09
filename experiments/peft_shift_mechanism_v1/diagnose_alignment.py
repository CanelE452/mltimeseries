"""Exploratory frozen-model lag alignment; run under the shared CUDA guard after the study."""

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_variable] = "2"

import numpy as np


CONDITIONS = ("Q00", "Q10", "Q01", "Q11")
LAGS = (32, 48, 64)
LENGTH = 256
HORIZON = 16
BOOTSTRAP_SEED = 2026090811
BOOTSTRAP_DRAWS = 4000
CONFIDENCE = .9875


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def source_indices(lag):
    if lag not in LAGS:
        raise ValueError(f"Lag must be one of {LAGS}")
    past = np.arange(LENGTH, dtype=np.int64) - lag
    future = LENGTH + np.arange(HORIZON, dtype=np.int64) - lag
    if future.min() < 0 or future.max() >= LENGTH or past[past >= 0].max() >= LENGTH:
        raise AssertionError("Alignment would require an originally unobserved value")
    return past, future


def aligned_inputs(context, lag):
    """Only original past values enter either the retimed context or known-future tensors."""
    context = np.asarray(context, dtype=np.float32)
    if context.ndim != 3 or context.shape[1:] != (3, LENGTH) or not np.isfinite(context).all():
        raise ValueError("Expected finite N x 3 x 256 Y/U/V past context")
    past_indices, future_indices = source_indices(lag)
    shifted = context.copy()
    shifted[:, 1:, :lag] = np.nan
    shifted[:, 1:, lag:] = context[:, 1:, past_indices[lag:]]
    future = np.zeros((len(context), 3, HORIZON), dtype=np.float32)
    mask = np.zeros_like(future)
    future[:, 1:] = context[:, 1:, future_indices]
    mask[:, 1:] = 1
    return shifted, future, mask


def select_lags(validation_scores):
    if set(validation_scores) != set(CONDITIONS):
        raise ValueError("All four conditions must be scored before lag selection")
    choices = {}
    for condition in CONDITIONS:
        scores = validation_scores[condition]
        if set(scores) != set(LAGS) or not np.isfinite(list(scores.values())).all():
            raise ValueError("Every condition requires exactly three finite validation scores")
        choices[condition] = min(LAGS, key=lambda lag: (scores[lag], lag))
    return choices


def paired_comparisons(aligned_losses, f0_losses):
    aligned_losses = np.asarray(aligned_losses, dtype=np.float64)
    f0_losses = np.asarray(f0_losses, dtype=np.float64)
    if aligned_losses.shape != (4, 512) or f0_losses.shape != (4, 512):
        raise ValueError("The paired family requires 4 conditions x 512 evaluation episodes")
    if not np.isfinite(aligned_losses).all() or not np.isfinite(f0_losses).all():
        raise ValueError("Nonfinite evaluation losses")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    weights = rng.multinomial(512, np.full(512, 1 / 512), size=BOOTSTRAP_DRAWS).astype(np.float64) / 512
    tails = [(1 - CONFIDENCE) / 2, 1 - (1 - CONFIDENCE) / 2]
    results = {}
    for index, condition in enumerate(CONDITIONS):
        difference = aligned_losses[index] - f0_losses[index]
        sampled_f0 = weights @ f0_losses[index]
        if np.any(sampled_f0 <= 0):
            raise ValueError("A positive F0 denominator is required for relative effects")
        sampled_difference = weights @ difference
        results[condition] = {
            "aligned_minus_f0": float(difference.mean()),
            "aligned_minus_f0_ci": np.quantile(sampled_difference, tails).tolist(),
            "relative_improvement": float(-difference.mean() / f0_losses[index].mean()),
            "relative_improvement_ci": np.quantile(-sampled_difference / sampled_f0, tails).tolist(),
            "confidence": CONFIDENCE,
        }
    return results


def require_completed_study(root, study):
    completion = study / "completed.json"
    if not completion.exists():
        raise RuntimeError("The original study must complete before alignment inference")
    done = json.loads(completion.read_text(encoding="utf-8"))
    selected = json.loads((study / "selected.json").read_text(encoding="utf-8"))
    contract = json.loads((study / "study_contract.json").read_text(encoding="utf-8"))
    if not done.get("completed") or len(selected) != 96:
        raise RuntimeError("The original study is incomplete")
    for name, expected in contract["sources"].items():
        if file_hash(root / name) != expected:
            raise AssertionError(f"Original study source differs from its execution contract: {name}")
    if file_hash(study / "data/manifest.json") != contract["data_manifest_sha256"]:
        raise AssertionError("Original data manifest changed")
    entries = [entry for entry in selected if entry["corpus"] == 0 and entry["method"] == "F0"]
    if len(entries) != 4 or {entry["condition"] for entry in entries} != set(CONDITIONS):
        raise AssertionError("Missing or duplicate original corpus-0 F0 entries")
    paths = {}
    for entry in entries:
        condition = entry["condition"]
        canonical = f"runs/peft_shift_mechanism_v1/trials/{condition}_c0/F0/lr_1e-03"
        if entry["path"] != canonical:
            raise AssertionError("Unexpected original F0 path")
        trial = root / canonical
        status = json.loads((trial / "guard/status.json").read_text(encoding="utf-8"))
        result = json.loads((trial / "result.json").read_text(encoding="utf-8"))
        data_path = study / "data" / f"{condition}_c0.npz"
        if not status.get("completed") or status.get("returncode") != 0 or not result.get("completed"):
            raise AssertionError("Original F0 trial did not complete")
        if file_hash(data_path) != contract["data_hashes"][data_path.name] or result["data_sha256"] != file_hash(data_path):
            raise AssertionError("Original F0 and current prepared data differ")
        paths[condition] = {"data": data_path, "trial": trial}
    return paths, contract


def load_split(paths, split):
    with np.load(paths["data"], allow_pickle=False) as archive:
        context = archive["context_" + split].astype(np.float32)
        target = archive["target_" + split].astype(np.float32)
        quantiles = archive["quantiles"].astype(np.float64)
        ids = archive["episode_ids_" + split].copy()
    with np.load(paths["trial"] / "predictions.npz", allow_pickle=False) as archive:
        f0 = archive[split + "_predictions"].astype(np.float32)
        if not np.array_equal(archive[split + "_target"], target):
            raise AssertionError("Original F0 target ordering differs from prepared data")
        if not np.array_equal(archive["quantiles"], quantiles):
            raise AssertionError("Original F0 quantiles differ from prepared data")
        if not np.array_equal(archive[split + "_episode_ids"], ids):
            raise AssertionError("Original F0 episode IDs differ from prepared data")
    count = 128 if split == "val" else 512
    if context.shape != (count, 3, 256) or target.shape != (count, 16) or f0.shape != (count, 21, 16):
        raise AssertionError("Dataset split shapes differ from the original protocol")
    return context, target, quantiles, ids, f0


def run(args):
    root = Path(__file__).resolve().parents[2]
    study = root / "runs/peft_shift_mechanism_v1"
    paths, original_contract = require_completed_study(root, study)
    output = Path(args.output).resolve()
    if any((output / name).exists() for name in ("run_contract.json", "validation_selection.json", "result.json")):
        raise FileExistsError("Preserve previous alignment diagnostics; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    deadline = started + args.max_seconds

    import torch
    import chronos.chronos2.model as chronos_model
    import chronos.chronos2.layers as chronos_layers
    from . import train

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(BOOTSTRAP_SEED)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This bounded inference diagnostic requires CUDA BF16")
    torch.cuda.set_per_process_memory_fraction(.67)
    torch.cuda.reset_peak_memory_stats()
    torch.backends.cuda.matmul.allow_tf32 = False
    device = "cuda"
    checkpoint = Path(args.checkpoint) if args.checkpoint else train.DEFAULT_CHECKPOINT
    source_paths = {"diagnose_alignment.py": Path(__file__), "shift_train.py": Path(train.__file__),
                    "chronos_model.py": Path(chronos_model.__file__), "chronos_layers.py": Path(chronos_layers.__file__),
                    "scope_modeling.py": root / "experiments/peft_adaptation_scope_v1/modeling.py",
                    "diagnostic_addendum.md": root / "_docs/notes/tsfm_topics/08_peft_shift_mechanism_diagnostic_addendum_20260908.md"}
    hashes = {"source": {name: file_hash(path) for name, path in source_paths.items()},
              "data": {q: file_hash(paths[q]["data"]) for q in CONDITIONS},
              "original_f0_predictions": {q: file_hash(paths[q]["trial"] / "predictions.npz") for q in CONDITIONS},
              "original_f0_results": {q: file_hash(paths[q]["trial"] / "result.json") for q in CONDITIONS},
              "original_completed": file_hash(study / "completed.json"),
              "original_selected": file_hash(study / "selected.json"),
              "original_contract": file_hash(study / "study_contract.json"),
              "checkpoint_config": file_hash(checkpoint / "config.json"),
              "checkpoint_weights": {path.name: file_hash(path) for path in sorted(checkpoint.glob("*.safetensors"))}}
    limitations = [
        "Exploratory follow-up inference diagnostic after the original development study; not a new PEFT method or confirmatory mechanism test.",
        "The candidate set 32/48/64 uses generator knowledge and includes the true driver lag 48.",
        "The intervention jointly retimes U/V, pads the earliest context, drops the latest lag-minus-16 original driver observations, changes U/V context normalization, and uses the known-future-covariate path.",
        "Jointly retiming U/V does not add channel identity or remove the model's exact-arithmetic U/V-swap symmetry.",
        "No originally future U/V or Y values are supplied. Y past is unchanged; Y future covariates are masked out.",
        "One frozen pretrained model and one shared evaluation episode set are used; these are not three independently fitted repetitions.",
        "Paired uncertainty is conditional on the validation-selected lag and this generator family, excluding validation-selection uncertainty.",
    ]
    run_contract = {
        "kind": "frozen_causal_alignment_exploratory_diagnostic", "conditions": list(CONDITIONS),
        "corpus": 0, "candidate_lags": list(LAGS), "validation_count": 128, "evaluation_count": 512,
        "selection": "Minimize raw mean 2-pinball over validation episodes; ties choose the smaller lag",
        "validation_settings": 12, "evaluation_settings": 4, "optimizer_updates": 0,
        "device": device, "weight_dtype": "float32", "autocast": "bfloat16", "autocast_weight_cache": False,
        "tf32": False, "micro_groups": 4, "encoder_rows": 12, "torch_threads": 2, "interop_threads": 1,
        "gpu_memory_fraction": .67, "max_seconds": args.max_seconds,
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "confidence": CONFIDENCE,
                      "unit": "paired episode ID across all four conditions", "family": "four condition contrasts"},
        "source_indices": {str(lag): {"past_valid_min": 0, "past_valid_max": int(source_indices(lag)[0][-1]),
                                      "known_future_min": int(source_indices(lag)[1][0]),
                                      "known_future_max": int(source_indices(lag)[1][-1]),
                                      "dropped_latest_driver_observations": lag - HORIZON} for lag in LAGS},
        "hashes": hashes, "limitations": limitations,
    }
    write_json(output / "run_contract.json", run_contract)

    def check_deadline():
        if time.perf_counter() > deadline:
            raise TimeoutError("Alignment inference exceeded its bounded wall-time budget")
        train.check_resources()

    check_deadline()
    base = chronos_model.Chronos2Model.from_pretrained(str(checkpoint), local_files_only=True,
                                                      dtype=torch.float32, attn_implementation="sdpa").to(device)
    train.deterministic_backbone(base)
    base.requires_grad_(False)
    signature = inspect.signature(base.encode)
    if not {"future_covariates", "future_covariates_mask"}.issubset(signature.parameters):
        raise AssertionError("Installed Chronos encode does not expose the required covariate API")
    groups = torch.arange(4, device=device).repeat_interleave(3)
    calls = {"f0_identity": 0, "validation": 0, "evaluation": 0}

    def infer(context, lag, stage):
        if len(context) % 4:
            raise AssertionError("Every inference microbatch must retain four full groups")
        shifted, future, mask = aligned_inputs(context, lag) if lag is not None else (context, None, None)
        predictions = []
        with torch.no_grad():
            for start in range(0, len(context), 4):
                check_deadline()
                batch = torch.as_tensor(shifted[start:start + 4].reshape(12, 256), device=device)
                kwargs = {} if lag is None else {
                    "future_covariates": torch.as_tensor(future[start:start + 4].reshape(12, 16), device=device),
                    "future_covariates_mask": torch.as_tensor(mask[start:start + 4].reshape(12, 16), device=device),
                }
                with train.precision(device):
                    _, normalized, loc, scale = train.encode_y(base, batch, groups, **kwargs)
                    raw = train.normalized_to_raw(normalized, loc, scale, base.chronos_config.use_arcsinh)
                predictions.append(raw.float().cpu().numpy())
                calls[stage] += 1
        return np.concatenate(predictions)

    scores, validation_arrays, validation_inputs, identities = {}, {}, {}, {}
    reference_ids, reference_quantiles = None, None
    for condition in CONDITIONS:
        context, target, quantiles, ids, f0 = load_split(paths[condition], "val")
        if reference_ids is None:
            reference_ids, reference_quantiles = ids, quantiles
        if not np.array_equal(ids, reference_ids) or not np.array_equal(quantiles, reference_quantiles):
            raise AssertionError("Validation conditions are not paired by episode/grid")
        if not np.allclose(quantiles, base.chronos_config.quantiles, rtol=0, atol=1e-7):
            raise AssertionError("Prepared and model quantile grids differ")
        identity = infer(context[:4], None, "f0_identity")
        identities[condition] = float(np.max(abs(identity.astype(np.float64) - f0[:4])))
        if identities[condition] > 1e-5:
            raise AssertionError(f"Fresh four-group F0 does not reproduce original predictions: {condition}")
        f0_scores, _ = train.forecast_scores(f0, target, quantiles)
        validation_arrays[f"{condition}_target"] = target
        validation_arrays[f"{condition}_f0_predictions"] = f0
        validation_arrays[f"{condition}_episode_ids"] = ids
        scores[condition] = {}
        validation_inputs[condition] = {"context_sha256": array_hash(context), "target_sha256": array_hash(target),
                                        "episode_ids_sha256": array_hash(ids),
                                        "f0_val_score": f0_scores["raw_mean_2pinball"]}
        for lag in LAGS:
            prediction = infer(context, lag, "validation")
            metrics, losses = train.forecast_scores(prediction, target, quantiles)
            scores[condition][lag] = metrics["raw_mean_2pinball"]
            validation_arrays[f"{condition}_lag{lag}_predictions"] = prediction
            validation_arrays[f"{condition}_lag{lag}_episode_losses"] = losses
            print(json.dumps({"event": "alignment_validation", "condition": condition, "lag": lag,
                              "val_score": scores[condition][lag], "elapsed_seconds": time.perf_counter() - started}), flush=True)
    choices = select_lags(scores)
    validation_arrays["quantiles"] = reference_quantiles
    np.savez_compressed(output / "validation_predictions.npz", **validation_arrays)
    selection = {"completed": True, "stage": "validation_selection_locked", "evaluation_started": False,
                 "validation_scores": scores, "selected_lags": choices,
                 "selection_rule": run_contract["selection"], "validation_inputs": validation_inputs,
                 "validation_predictions_sha256": file_hash(output / "validation_predictions.npz"),
                 "source_hashes": hashes["source"], "data_hashes": hashes["data"],
                 "validation_forward_calls": calls["validation"], "elapsed_seconds": time.perf_counter() - started}
    selection["selection_inputs_sha256"] = hashlib.sha256(json.dumps({
        "candidate_lags": LAGS, "validation_scores": scores, "validation_inputs": validation_inputs,
        "validation_predictions_sha256": selection["validation_predictions_sha256"],
        "source_hashes": hashes["source"], "data_hashes": hashes["data"],
    }, sort_keys=True).encode()).hexdigest()
    selection_path = output / "validation_selection.json"
    write_json(selection_path, selection)
    selection_hash = file_hash(selection_path)
    if json.loads(selection_path.read_text(encoding="utf-8"))["selected_lags"] != choices:
        raise AssertionError("Persisted validation selection differs from the in-memory choice")

    evaluation_arrays, evaluation_rows, aligned_losses, baseline_losses = {}, [], [], []
    reference_eval_ids = None
    for condition in CONDITIONS:
        if file_hash(selection_path) != selection_hash:
            raise AssertionError("Lag selection changed after evaluation began")
        context, target, quantiles, ids, f0 = load_split(paths[condition], "eval")
        if reference_eval_ids is None:
            reference_eval_ids = ids
        if not np.array_equal(ids, reference_eval_ids) or not np.array_equal(quantiles, reference_quantiles):
            raise AssertionError("Evaluation conditions are not paired by episode/grid")
        prediction = infer(context, choices[condition], "evaluation")
        metrics, losses = train.forecast_scores(prediction, target, quantiles)
        f0_metrics, f0_loss = train.forecast_scores(f0, target, quantiles)
        original_result = json.loads((paths[condition]["trial"] / "result.json").read_text(encoding="utf-8"))
        if not np.isclose(f0_metrics["raw_mean_2pinball"], original_result["eval_score"], rtol=1e-7, atol=1e-9):
            raise AssertionError("Recomputed F0 score differs from original result")
        aligned_losses.append(losses)
        baseline_losses.append(f0_loss)
        evaluation_arrays.update({f"{condition}_predictions": prediction, f"{condition}_f0_predictions": f0,
                                  f"{condition}_target": target, f"{condition}_episode_ids": ids,
                                  f"{condition}_episode_losses": losses, f"{condition}_f0_episode_losses": f0_loss})
        evaluation_rows.append({"condition": condition, "selected_lag": choices[condition],
                                "val_score": scores[condition][choices[condition]],
                                "eval_score": metrics["raw_mean_2pinball"],
                                "f0_eval_score": f0_metrics["raw_mean_2pinball"],
                                "evaluation": metrics, "f0_evaluation": f0_metrics})
        print(json.dumps({"event": "alignment_evaluation_complete", "condition": condition,
                          "selected_lag": choices[condition], "elapsed_seconds": time.perf_counter() - started}), flush=True)
    comparisons = paired_comparisons(np.asarray(aligned_losses), np.asarray(baseline_losses))
    for row in evaluation_rows:
        row.update(comparisons[row["condition"]])
        if not np.isclose(row["aligned_minus_f0"], row["eval_score"] - row["f0_eval_score"], rtol=0, atol=1e-12):
            raise AssertionError("Paired point contrast does not match the score difference")
    evaluation_arrays["quantiles"] = reference_quantiles
    np.savez_compressed(output / "evaluation_predictions.npz", **evaluation_arrays)
    for name, path in source_paths.items():
        if file_hash(path) != hashes["source"][name]:
            raise AssertionError("Diagnostic source changed during execution")
    for condition in CONDITIONS:
        if file_hash(paths[condition]["data"]) != hashes["data"][condition]:
            raise AssertionError("Prepared data changed during execution")
        if file_hash(paths[condition]["trial"] / "predictions.npz") != hashes["original_f0_predictions"][condition]:
            raise AssertionError("Original F0 predictions changed during execution")
    if file_hash(selection_path) != selection_hash or calls != {"f0_identity": 4, "validation": 384, "evaluation": 512}:
        raise AssertionError("Selection or forward-call budget changed")
    check_deadline()
    result = {"completed": True, "kind": run_contract["kind"], "conditions": evaluation_rows,
              "selected_lags": choices, "validation_scores": scores,
              "validation_selection_sha256": selection_hash, "selection_saved_before_evaluation": True,
              "selection_inputs_sha256": selection["selection_inputs_sha256"],
              "diagnostic_addendum_sha256": hashes["source"]["diagnostic_addendum.md"],
              "prediction_hashes": {name: file_hash(output / name) for name in
                                    ("validation_predictions.npz", "evaluation_predictions.npz")},
              "hashes": hashes, "fresh_f0_raw_max_abs_error": identities,
              "encode_signature": str(signature), "future_covariate_api_used": True,
              "forward_calls": calls, "optimizer_updates": 0, "trainable_parameters": 0,
              "all_parameters_frozen": all(not parameter.requires_grad for parameter in base.parameters()),
              "original_future_values_supplied": 0, "wall_seconds": time.perf_counter() - started,
              "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
              "bootstrap": run_contract["bootstrap"], "limitations": limitations,
              "original_source_contract": original_contract["sources"]}
    write_json(output / "result.json", result)
    print(json.dumps({"completed": True, "selected_lags": choices, "wall_seconds": result["wall_seconds"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(Path(__file__).resolve().parents[2]
                                                / "results/peft_shift_mechanism_v1/alignment_diagnostic"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--max-seconds", type=float, default=240)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 240:
        parser.error("Use a positive wall-time limit no larger than 240 seconds")
    try:
        run(args)
    except Exception as error:
        output = Path(args.output)
        if output.exists() and not (output / "result.json").exists():
            write_json(output / "failure.json", {"completed": False, "error_type": type(error).__name__,
                                                 "error": str(error), "time": time.time()})
        raise


if __name__ == "__main__":
    main()
