"""Bounded CPU falsification gate; does not train or score a TSFM."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

for thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[thread_variable] = "2"

import numpy as np
import psutil
import torch

from .projection import FixedSupportProjection


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def max_error(a, b):
    return float((a - b).abs().max().detach())


def ratio(a, b):
    return float((torch.linalg.vector_norm(a) / torch.linalg.vector_norm(b)).detach())


def synthetic_audit():
    generator = torch.Generator().manual_seed(20260908)

    def random(*shape):
        return torch.randn(shape, generator=generator, dtype=torch.float64)

    z = torch.cat((random(96, 11), torch.ones(96, 1)), dim=1)
    z_new = torch.cat((random(23, 11), torch.ones(23, 1)), dim=1)
    x, x_new, k = random(96, 10), random(23, 10), random(6, 5)
    theta = random(10, 6).requires_grad_()
    delta, delta_new = torch.sin(x @ theta) @ k, torch.sin(x_new @ theta) @ k
    projection = FixedSupportProjection(z)
    coefficients = projection.coefficients(delta)
    w, base, target = random(12, 5), random(96, 5), random(96, 5)

    joint = z @ w + delta
    pho = z @ (w + coefficients) + projection.complement(delta)
    joint_new = z_new @ w + delta_new
    pho_new = z_new @ (w + coefficients) + projection.extend(z_new, delta_new, delta)
    candidate = z @ w + projection.complement(delta)
    reparameterized = z @ (w - coefficients) + delta
    candidate_new = z_new @ w + projection.extend(z_new, delta_new, delta)
    reparameterized_new = z_new @ (w - coefficients) + delta_new

    # Independent normal-equation baseline; synthetic Z is well conditioned.
    gram = z.T @ z
    w0 = torch.linalg.solve(gram, z.T @ (target - base))
    w_profile = torch.linalg.solve(gram, z.T @ (target - base - delta))
    pred_projected = base + z @ w0 + projection.complement(delta)
    pred_profiled = base + z @ w_profile + delta
    loss_projected = ((target - pred_projected) ** 2).mean()
    loss_profiled = ((target - pred_profiled) ** 2).mean()
    grad_projected = torch.autograd.grad(loss_projected, theta, retain_graph=True)[0]
    grad_profiled = torch.autograd.grad(loss_profiled, theta, retain_graph=True)[0]
    output_gradient = 2 * (pred_profiled - target) / target.numel()
    metrics = {
        "joint_reparameterization_fit_max_error": max_error(candidate, reparameterized),
        "joint_reparameterization_new_max_error": max_error(candidate_new, reparameterized_new),
        "pho_prediction_fit_max_error": max_error(joint, pho),
        "pho_prediction_new_max_error": max_error(joint_new, pho_new),
        "mse_profile_prediction_max_error": max_error(pred_projected, pred_profiled),
        "mse_profile_loss_abs_error": float((loss_projected - loss_profiled).abs().detach()),
        "mse_profile_gradient_max_error": max_error(grad_projected, grad_profiled),
        "optimal_mse_head_projected_cotangent_max_error": max_error(
            projection.complement(output_gradient), output_gradient),
    }
    if not all(value < 1e-9 for value in metrics.values()):
        raise AssertionError(metrics)
    return {"seed": 20260908, "dimensions": [96, 12, 5], "tolerance": 1e-9,
            "checks_passed": len(metrics), "metrics": metrics}


def geometry(z, generator):
    projection = FixedSupportProjection(z)
    probe = torch.randn((len(z), 8), dtype=torch.float64, generator=generator,
                        requires_grad=True)
    cotangent = torch.randn(probe.shape, dtype=torch.float64, generator=generator)
    residual = projection.complement(probe)
    vjp = torch.autograd.grad((residual * cotangent).sum(), probe)[0]
    s = projection.singular_values
    dual_residual_error = ratio(z.T @ residual, z) / float(torch.linalg.vector_norm(probe.detach()))
    result = {
        "shape": list(z.shape), "rank_rtol_1e_10": projection.rank,
        "rank_sensitivity_singular_value_rtol": {
            str(tol): int((s > tol * s[0]).sum()) for tol in (1e-12, 1e-10, 1e-8, 1e-6, 1e-4)
        },
        "singular_min": float(s[-1]), "singular_max": float(s[0]),
        "condition_2": float(s[0] / s[-1]),
        "random_output_residual_norm_ratio": ratio(residual, probe),
        "random_cotangent_vjp_norm_ratio": ratio(vjp, cotangent),
        "normalized_orthogonality_error": dual_residual_error,
        "coefficient_vs_basis_relative_error": ratio(
            (probe - z @ projection.coefficients(probe)) - residual, probe),
    }
    if dual_residual_error >= 1e-8 or result["coefficient_vs_basis_relative_error"] >= 1e-8:
        raise AssertionError(result)
    if projection.rank == len(z):
        if max(result["random_output_residual_norm_ratio"], result["random_cotangent_vjp_norm_ratio"]) >= 1e-8:
            raise AssertionError("A full-row-rank support must annihilate its hard complement")
    return result


def real_feature_audit(prepared, cache):
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    if not manifest["completed"] or sha256(prepared) != manifest["contract"]["data_sha256"]:
        raise AssertionError("Prepared data/cache contract mismatch")
    with np.load(prepared, allow_pickle=False) as panel:
        train_origins = panel["train_origins"].copy()
        val_origins = panel["val_origins"].copy()
        eval_origins = panel["eval_origins"].copy()
    if not np.array_equal(train_origins, manifest["contract"]["origins"]["train"]):
        raise AssertionError("Train origins mismatch")
    if np.intersect1d(train_origins, np.r_[val_origins, eval_origins]).size:
        raise AssertionError("Overlapping origin IDs")
    cache_origins = np.load(cache / "origins.npy", mmap_mode="r")
    if len(np.unique(cache_origins)) != len(cache_origins):
        raise AssertionError("Duplicate cache origins")
    row_for_origin = {int(origin): row for row, origin in enumerate(cache_origins)}
    rows = [row_for_origin[int(origin)] for origin in train_origins]
    hidden = np.load(cache / "hidden.npy", mmap_mode="r")
    fit_hidden = np.asarray(hidden[rows], dtype=np.float64)
    flat = torch.from_numpy(fit_hidden.reshape(-1, hidden.shape[-1]))
    z = torch.cat((flat, torch.ones((len(flat), 1), dtype=torch.float64)), dim=1)
    rows_per_origin = hidden.shape[1] * hidden.shape[2]
    generator = torch.Generator().manual_seed(90210)
    batches = []
    for start in (0, (len(train_origins) - 4) // 2, len(train_origins) - 4):
        item = geometry(z[start * rows_per_origin:(start + 4) * rows_per_origin], generator)
        item["train_origin_indices"] = list(range(start, start + 4))
        item["origin_values"] = train_origins[start:start + 4].tolist()
        batches.append(item)
    print("Three fit-only minibatches audited; checking full support", flush=True)
    full = geometry(z, generator)
    return {
        "dataset": "ettm2", "train_origin_count": len(train_origins),
        "validation_feature_rows_used": 0, "evaluation_feature_rows_used": 0,
        "target_values_used": False, "probe_seed": 90210,
        "note": "Random probes test projection geometry, not trained LoRA directions or forecasting quality.",
        "cache_contract": manifest["contract"],
        "hashes": {"prepared": sha256(prepared), "hidden": sha256(cache / "hidden.npy"),
                   "origins": sha256(cache / "origins.npy"), "manifest": sha256(cache / "manifest.json")},
        "minibatches": batches, "full_support": full,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(True)
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[2]
    old_source = root / "experiments/peft_adaptation_scope_v1"
    frozen_before = {name: sha256(old_source / name) for name in ("modeling.py", "train.py", "data.py")}
    synthetic = synthetic_audit()
    print(f"Synthetic equivalence checks passed: {synthetic['checks_passed']}", flush=True)
    feature = real_feature_audit(root / "runs/peft_adaptation_scope_v1/prepared/ettm2.npz",
                                root / "runs/peft_adaptation_scope_v1/cache/ettm2/09c67f2aa56dca10e847")
    frozen_after = {name: sha256(old_source / name) for name in frozen_before}
    if frozen_before != frozen_after:
        raise AssertionError("Existing S1 source changed during the audit")
    result = {
        "completed": True, "kind": "cpu_method_entry_gate", "device": "cpu", "dtype": "float64",
        "optimizer_steps": 0, "tsfm_forward_calls": 0, "new_forecasting_scores": 0,
        "torch_threads": torch.get_num_threads(), "synthetic": synthetic, "real_features": feature,
        "frozen_s1_source_hashes": frozen_after,
        "source_hashes": {name: sha256(Path(__file__).parent / name) for name in ("audit.py", "projection.py")},
        "plan_sha256": sha256(root / "_docs/notes/tsfm_topics/07_head_complement_method_gate_plan_20260908.md"),
        "elapsed_seconds": time.perf_counter() - started,
        "available_ram_gib_at_finish": psutil.virtual_memory().available / 2**30,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"completed": True, "elapsed_seconds": result["elapsed_seconds"],
                      "full_support_rank": feature["full_support"]["rank_rtol_1e_10"],
                      "minibatch_ranks": [b["rank_rtol_1e_10"] for b in feature["minibatches"]]}), flush=True)


if __name__ == "__main__":
    main()
