"""Compare cached and fresh no-update forecasts without optimization."""

import json
from pathlib import Path

import numpy as np
import torch

from .modeling import deterministic_backbone, patch_to_quantiles
from .train import Panel


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(0.67)
    checkpoint = Path.home() / ".cache/huggingface/hub/models--amazon--chronos-2/snapshots/29ec3766d36d6f73f0696f85560a422f50e8498c"
    run = Path("runs/peft_adaptation_scope_v1")
    panel = Panel(run / "prepared/jena.npz", smoke=True)
    context, _, groups = panel.batch(panel.origins["train"][:1], "cuda")
    manifest = next((run / "smoke_cache/jena").glob("*/manifest.json"))
    origins = np.load(manifest.parent / "origins.npy")
    row = int(np.where(origins == panel.origins["train"][0])[0][0])
    cached = torch.tensor(np.load(manifest.parent / "base_norm.npy", mmap_mode="r")[row], device="cuda")
    from chronos.chronos2.model import Chronos2Model
    base = Chronos2Model.from_pretrained(checkpoint, local_files_only=True, dtype=torch.float32, attn_implementation="sdpa").to("cuda")
    deterministic_backbone(base)
    outputs = {}
    for use_bf16 in [True, False]:
        for requires_grad in [False, True]:
            for cast_cache in [True, False]:
                key = f"bf16={use_bf16},grad={requires_grad},cache={cast_cache}"
                base.requires_grad_(requires_grad)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16, cache_enabled=cast_cache):
                    encoded = base.encode(context=context, group_ids=groups, num_output_patches=9)[0]
                    norm = patch_to_quantiles(base.output_patch_embedding(encoded.last_hidden_state[:, -9:]), base.num_quantiles, 16).float()
                outputs[key] = norm
    reference = outputs["bf16=True,grad=False,cache=True"]
    report = {key: {"max_abs_vs_disk_cache": float((value-cached).abs().max()),
                    "mean_abs_vs_disk_cache": float((value-cached).abs().mean()),
                    "max_abs_vs_fresh_frozen": float((value-reference).abs().max())} for key,value in outputs.items()}
    report["device"] = torch.cuda.get_device_name()
    report["cache_manifest"] = str(manifest)
    destination = run / "precision_diagnostic.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
