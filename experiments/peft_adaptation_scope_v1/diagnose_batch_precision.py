"""Inference-only batch-shape diagnostic; run under guard with a 120-second limit."""

import hashlib
import json
import os
from pathlib import Path
import time

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch

from .modeling import AdaptationModel, forecast_scores
from .train import Panel


def difference(first, second, scale):
    absolute = np.abs(np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64))
    normalized = absolute / scale[None, :, None, None]
    return {"bitwise_equal": bool(np.array_equal(first, second)),
            "raw_max_abs": float(absolute.max()), "raw_mean_abs": float(absolute.mean()),
            "fit_std_scaled_mean_abs": float(normalized.mean()),
            "fit_std_scaled_p99_abs": float(np.quantile(normalized, 0.99)),
            "different_fraction": float(np.count_nonzero(absolute) / absolute.size)}


def main():
    start = time.perf_counter()
    project = Path(__file__).resolve().parents[2]
    run = project / "runs/peft_adaptation_scope_v1"
    output = run / "batch_precision_diagnostic"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "diagnostic.json").exists():
        raise FileExistsError("Preserve the completed diagnostic")
    contract = json.loads((run / "contract.json").read_text(encoding="utf-8"))
    for name, digest in contract["source_sha256"].items():
        assert hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() == digest
    f0_folder = run / "trials/ettm2_F0_lr0_seed0"
    full_folder = run / "trials/ettm2_FULL_lr3e-06_seed1"
    f0_metrics = json.loads((f0_folder / "metrics.json").read_text())
    full_metrics = json.loads((full_folder / "metrics.json").read_text())
    assert full_metrics["selected_step"] == 0
    panel = Panel(run / "prepared/ettm2.npz")
    origins = panel.origins["eval"]
    assert len(origins) == 32
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(0.67)
    torch.cuda.reset_peak_memory_stats()
    from chronos.chronos2.model import Chronos2Model

    base = Chronos2Model.from_pretrained(contract["checkpoint"], local_files_only=True,
                                        dtype=torch.float32, attn_implementation="sdpa").to("cuda")
    model = AdaptationModel(base, "FULL", panel.count_channels)
    model.eval()
    outputs, normalized, elapsed = {}, {}, {}
    for micro_groups in (1, 4):
        raw_parts, norm_parts = [], []
        tick = time.perf_counter()
        with torch.no_grad():
            for offset in range(0, len(origins), micro_groups):
                selected = origins[offset:offset + micro_groups]
                context, _, groups = panel.batch(selected, "cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                    norm, raw, _, _ = model.from_context(context, groups, panel.horizon)
                shape = (len(selected), panel.count_channels, model.n_quantiles, panel.horizon)
                raw_parts.append(raw.float().cpu().numpy().reshape(shape))
                norm_parts.append(norm.float().cpu().numpy().reshape(shape))
        torch.cuda.synchronize()
        outputs[micro_groups] = np.concatenate(raw_parts)
        normalized[micro_groups] = np.concatenate(norm_parts)
        elapsed[micro_groups] = time.perf_counter() - tick
    assert all(parameter.grad is None for parameter in model.parameters())
    with np.load(f0_folder / "predictions.npz") as stored_f0, np.load(full_folder / "predictions.npz") as stored_full:
        f0 = stored_f0["eval_pred"]
        full = stored_full["eval_pred"]
        quantiles = stored_f0["quantiles"]
        np.testing.assert_array_equal(stored_f0["eval_origins"], origins)
        np.testing.assert_array_equal(stored_full["eval_origins"], origins)
        np.testing.assert_array_equal(quantiles, stored_full["quantiles"])
    target = panel.targets("eval")
    scores = {name: forecast_scores(pred, target, panel.fit_std, quantiles)[0]["scaled_2pinball"]
              for name, pred in [("fresh_group1", outputs[1]), ("fresh_group4", outputs[4]),
                                 ("stored_cached_F0", f0), ("stored_FULL_seed1_step0", full)]}
    comparisons = {
        "group1_vs_stored_cached_F0": difference(outputs[1], f0, panel.fit_std),
        "group4_vs_stored_FULL_step0": difference(outputs[4], full, panel.fit_std),
        "group1_vs_group4": difference(outputs[1], outputs[4], panel.fit_std),
        "stored_F0_vs_stored_FULL_step0": difference(f0, full, panel.fit_std),
    }
    norm_difference = np.abs(normalized[1] - normalized[4])
    report = {
        "completed": True, "optimizer_steps": 0, "backward_calls": 0,
        "evaluation_origins": 32, "context": panel.context, "horizon": panel.horizon,
        "channels": panel.count_channels, "micro_groups_tested": [1, 4],
        "device": torch.cuda.get_device_name(), "weights": "float32", "autocast": "bfloat16",
        "autocast_weight_cache": False, "cpu_threads": 2, "gpu_memory_fraction": 0.67,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "wall_seconds": time.perf_counter() - start, "inference_seconds_by_micro_groups": elapsed,
        "checkpoint": contract["checkpoint"], "frozen_source_sha256_verified": contract["source_sha256"],
        "source_facts": {
            "cache_generation": "train.prepare_cache calls panel.batch([origin]): one group per encoder call",
            "stored_F0_evaluation": "predict uses from_cache; micro_groups changes cache reads but does not rerun the encoder",
            "stored_FULL_evaluation": "predict uses from_context with micro_groups=4",
            "zero_identity_audit": "Only the first training origin, one group; not all evaluation origins or group batch sizes",
        },
        "scores": scores, "prediction_comparisons": comparisons,
        "native_normalized_group1_vs_group4": {"mean_abs": float(norm_difference.mean()),
                                               "max_abs": float(norm_difference.max()),
                                               "p90_abs": float(np.quantile(norm_difference, 0.9)),
                                               "p99_abs": float(np.quantile(norm_difference, 0.99))},
        "exact_reproduction": comparisons["group1_vs_stored_cached_F0"]["bitwise_equal"] and
                              comparisons["group4_vs_stored_FULL_step0"]["bitwise_equal"],
        "interpretation_limit": "A controlled change of batch shape can reproduce the difference; individual GEMM/SDPA/pointwise rounding contributions are not kernel-profiled",
    }
    np.savez_compressed(output / "predictions.npz", group1_pred=outputs[1], group4_pred=outputs[4],
                        group1_normalized=normalized[1], group4_normalized=normalized[4],
                        eval_origins=origins, quantiles=quantiles)
    (output / "diagnostic.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
