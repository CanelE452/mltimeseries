"""Read-only U/V permutation diagnostic in FP32 and BF16; run through guard."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time

from .train import DEFAULT_CHECKPOINT, Episodes, encode_y, file_hash, precision
from experiments.peft_adaptation_scope_v1.modeling import deterministic_backbone, normalized_to_raw
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    data_path = root / "runs/peft_shift_mechanism_v1/data/Q00_c0.npz"
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.67)
    torch.cuda.reset_peak_memory_stats()
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.perf_counter()
    from chronos.chronos2.model import Chronos2Model
    base = Chronos2Model.from_pretrained(DEFAULT_CHECKPOINT, local_files_only=True,
                                        dtype=torch.float32, attn_implementation="sdpa").to("cuda")
    deterministic_backbone(base)
    base.requires_grad_(False)
    data = Episodes(data_path)
    context, _, groups = data.batch("train", [0, 1, 2, 3], "cuda")
    swapped = context.reshape(4, 3, 256)[:, [0, 2, 1]].reshape(12, 256)
    results = {}
    with torch.no_grad():
        for mode in ("float32", "bfloat16"):
            with nullcontext() if mode == "float32" else precision("cuda"):
                original = encode_y(base, context, groups)
                exchanged = encode_y(base, swapped, groups)
            raw = normalized_to_raw(original[1], original[2], original[3], base.chronos_config.use_arcsinh)
            swap_raw = normalized_to_raw(exchanged[1], exchanged[2], exchanged[3], base.chronos_config.use_arcsinh)
            results[mode] = {"normalized_max_abs": float((original[1] - exchanged[1]).abs().max()),
                             "raw_max_abs": float((raw - swap_raw).abs().max()),
                             "raw_mean_abs": float((raw - swap_raw).abs().mean())}
    if results["float32"]["normalized_max_abs"] > 1e-5:
        raise AssertionError("FP32 permutation discrepancy requires further diagnosis")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"completed": True, "results": results, "optimizer_steps": 0, "fit_train_episodes": [0, 1, 2, 3],
              "data_sha256": file_hash(data_path), "source_sha256": file_hash(Path(__file__)),
              "wall_seconds": time.perf_counter() - started, "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30,
              "scope": "Pretrained model only. Exact mathematical permutation invariance is distinct from finite-precision path differences; no forecasting-risk lower bound follows from this diagnostic alone."}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
