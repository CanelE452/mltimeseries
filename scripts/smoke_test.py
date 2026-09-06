"""Load each forecaster and run one forecast on real ETTm2 data.

Answers a narrow question: does this machine actually run the model, and is the
output the right shape and not obviously wrong. Not an accuracy benchmark.

Needs (see _docs/history for why):
    CUDA_HOME=<any existing dir>    # xlstm reads it at import time
    TORCH_COMPILE_DISABLE=1         # TiRex-2 CPU path otherwise wants MSVC cl
"""

import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
ETTM2 = ROOT / "data" / "ETT-small" / "ETTm2.csv"
CONTEXT, HORIZON = 512, 96
results: dict[str, str] = {}


def banner(t: str) -> None:
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70, flush=True)


def load_window():
    """Last CONTEXT+HORIZON steps of ETTm2. Returns (context, truth), each (7, T)."""
    df = pd.read_csv(ETTM2)
    values = df.iloc[:, 1:].to_numpy(dtype=np.float32).T          # (7, n_steps)
    window = values[:, -(CONTEXT + HORIZON):]
    return window[:, :CONTEXT], window[:, CONTEXT:]


def naive_mae(context, truth):
    """Repeat-last-value baseline. A model far worse than this is broken."""
    return float(np.abs(truth - context[:, -1:]).mean())


def report(name, pred, truth, elapsed, extra=""):
    mae = float(np.abs(pred - truth).mean())
    print(f"  {name}: {elapsed:.1f}s  pred{tuple(pred.shape)}  MAE={mae:.4f} {extra}", flush=True)
    return mae


def median_of(forecast, quantile_levels):
    """forecast is (n_variates, n_quantiles, horizon); pull the 0.5 slice."""
    levels = [float(q) for q in quantile_levels]
    idx = levels.index(0.5)
    return np.asarray(forecast)[:, idx, :]


def run_chronos2(context, truth, device):
    from chronos import Chronos2Pipeline

    t0 = time.time()
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device)
    # (batch=1, n_variates, history) -> multivariate inference over the 7 columns
    batch = torch.tensor(context).unsqueeze(0)
    out = pipe.predict(batch, prediction_length=HORIZON)[0].float().cpu()
    median = median_of(out, pipe.quantiles)
    return report(f"chronos-2 [{device}]", median, truth, time.time() - t0), median


def run_tirex2(context, truth, device):
    from tirex2 import TimeseriesType, load_model

    t0 = time.time()
    model = load_model("NX-AI/TiRex-2", device=device)
    ts = TimeseriesType(target=torch.tensor(context), past_covariates=None,
                        future_covariates=None)
    fc = model.forecast([ts], prediction_length=HORIZON, output_type="numpy")[0]
    median = median_of(fc, model.quantiles)
    return report(f"tirex-2 [{device}]", median, truth, time.time() - t0), median


def run_patchtst(context, truth):
    """Official PatchTST, randomly initialised: shape/forward check only."""
    import sys
    from argparse import Namespace

    sys.path.insert(0, str(ROOT / "third_party" / "PatchTST" / "PatchTST_supervised"))
    from models import PatchTST

    cfg = Namespace(
        enc_in=context.shape[0], seq_len=CONTEXT, pred_len=HORIZON,
        e_layers=3, n_heads=4, d_model=16, d_ff=128, dropout=0.2, fc_dropout=0.2,
        head_dropout=0.0, individual=0, patch_len=16, stride=8, padding_patch="end",
        revin=1, affine=0, subtract_last=0, decomposition=0, kernel_size=25,
    )
    t0 = time.time()
    model = PatchTST.Model(cfg).eval()
    n_param = sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        # PatchTST takes (batch, seq_len, n_vars)
        out = model(torch.tensor(context).T.unsqueeze(0))
    pred = out.squeeze(0).T.numpy()
    mae = report("PatchTST [cpu, untrained]", pred, truth, time.time() - t0,
                 extra=f"params={n_param:,}")
    return mae, pred


def main() -> int:
    if not ETTM2.exists():
        print(f"missing {ETTM2} - run scripts/fetch_data.py first")
        return 1

    context, truth = load_window()
    base = naive_mae(context, truth)
    banner(f"ETTm2 tail window: context {context.shape} -> horizon {truth.shape}")
    print(f"repeat-last-value baseline MAE = {base:.4f}", flush=True)
    print(f"cuda available: {torch.cuda.is_available()}", flush=True)

    jobs = [
        ("chronos-2 (cuda)", lambda: run_chronos2(context, truth, "cuda")),
        ("chronos-2 (cpu)", lambda: run_chronos2(context, truth, "cpu")),
        ("tirex-2 (cpu)", lambda: run_tirex2(context, truth, "cpu")),
        ("PatchTST (cpu)", lambda: run_patchtst(context, truth)),
    ]
    for name, fn in jobs:
        banner(name)
        try:
            mae, _ = fn()
            verdict = "OK" if np.isfinite(mae) else "NON-FINITE OUTPUT"
            results[name] = f"{verdict} (MAE {mae:.4f} vs baseline {base:.4f})"
        except Exception as e:
            print("FAILED:", type(e).__name__, str(e)[:500], flush=True)
            traceback.print_exc()
            results[name] = f"FAIL ({type(e).__name__}: {str(e)[:120]})"

    banner("VERDICT")
    for k, v in results.items():
        print(f"  {k:20s} {v}", flush=True)
    n_ok = sum(1 for v in results.values() if v.startswith("OK"))
    print(f"\n{n_ok}/{len(results)} runs produced a usable forecast", flush=True)
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
