"""DIAGNOSTIC_PROFILE_ONLY: where the per-stage inference time goes for U, C and DENSE.

Inference only. No checkpoint is loaded, no parameter is updated, no result artifact is
touched. Randomly initialized weights are fine here because the question is the cost of
the computation graph, not accuracy.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "hq_token_pilot_v1", "audit_closure_v1")

import sys
sys.path.insert(0, ROOT)
from experiments.hq_token_pilot_v1 import model as M  # noqa: E402  (model definition only)

WARM, TIMED = 100, 500
BATCH, H = 64, 336


def timed(fn, device):
    for _ in range(WARM):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    t = []
    for _ in range(TIMED):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
        t.append((time.perf_counter() - t0) * 1e3)
    return {"median_ms": float(np.median(t)), "p90_ms": float(np.percentile(t, 90))}


@torch.no_grad()
def profile(arm, B, device):
    net = M.build(arm, B, 2026090601).to(device).eval()
    x = torch.randn(BATCH, M.L, device=device)
    h = torch.full((BATCH,), float(H), device=device)

    e = net.embed(x)
    tokens, _ = net.compress(e, h)
    z = net.encoder(tokens)
    e_h = net.horizon_embed(h)

    stages = {
        "embed": timed(lambda: net.embed(x), device),
        "pool": timed(lambda: net.compress(e, h), device),
        "encoder": timed(lambda: net.encoder(tokens), device),
        "unmerge_readout_head": timed(lambda: net.decode(z, e_h), device),
        "total_forward": timed(lambda: net(x, h), device),
    }
    stages["tokens_into_encoder"] = net.B
    stages["parameters"] = net.parameter_report()["total_parameters"]
    return stages


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    res = {
        "kind": "DIAGNOSTIC_PROFILE_ONLY",
        "device": device, "batch": BATCH, "horizon": H,
        "warm_runs": WARM, "timed_runs": TIMED,
        "weights": "randomly initialized; this measures graph cost, not accuracy",
        "configs": {},
    }
    for arm, B in (("U", 32), ("C", 32), ("DENSE", 64)):
        res["configs"][f"{arm}_B{B}"] = profile(arm, B, device)
        print(f"{arm}_B{B}: total {res['configs'][f'{arm}_B{B}']['total_forward']['median_ms']:.3f} ms",
              flush=True)

    u = res["configs"]["U_B32"]
    c = res["configs"]["C_B32"]
    d = res["configs"]["DENSE_B64"]
    res["reading"] = {
        "encoder_saving_32_vs_64_tokens_ms": d["encoder"]["median_ms"] - u["encoder"]["median_ms"],
        "pool_cost_uniform_ms": u["pool"]["median_ms"],
        "pool_cost_content_scorer_ms": c["pool"]["median_ms"],
        "scorer_extra_over_uniform_ms": c["pool"]["median_ms"] - u["pool"]["median_ms"],
        "decode_cost_compressed_ms": u["unmerge_readout_head"]["median_ms"],
        "decode_cost_dense_ms": d["unmerge_readout_head"]["median_ms"],
        "note": (
            "Uniform pooling runs no scorer, so its pool stage is the pure cost of grouping, "
            "weighting and summing. Comparing that plus the extra decode work against what the "
            "shorter encoder saves shows whether compression can pay for itself at this width."),
        "scope": "one RTX-class GPU, 1.5M parameter model, batch 64. Not a statement about larger backbones.",
    }
    res["wall_seconds"] = time.time() - t0
    with open(os.path.join(OUT, "efficiency_microprofile.json"), "wb") as f:
        f.write(json.dumps(res, indent=2).encode())
    print(f"wrote efficiency_microprofile.json in {res['wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()
