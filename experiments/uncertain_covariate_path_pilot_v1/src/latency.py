"""Inference latency and peak memory (instruction section 25).

Two costs are reported separately and never added together as if they were one
number:

  adapter_only  the arm's own forward pass, given the cached Chronos features
  end_to_end    the same, plus one Chronos-2-synth forward per example

MC pays for every scenario call and for building the mixture, because that is
what running MC actually costs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.uncertain_covariate_path_pilot_v1.src.models import ParticleForecaster
from experiments.uncertain_covariate_path_pilot_v1.src.paths import out_dir, runs_dir
from experiments.uncertain_covariate_path_pilot_v1.src.train import ARMS, SEEDS, Config, load_all

ROOT = Path(__file__).resolve().parents[3]

WARMUP, TIMED = 20, 100
BATCHES = (1, 64)


def time_call(fn, warmup: int = WARMUP, timed: int = TIMED) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(timed):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / timed * 1000.0


def main(arms=ARMS, seed: int = SEEDS[0], tag: str = "full") -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("latency is measured on GPU")
    device = "cuda"
    splits, broken, n_farms = load_all(device, tuple(arms))
    cfg = Config()
    test = splits["test"]

    from experiments.uncertain_covariate_path_pilot_v1.src.cache_chronos import (
        PREDICTION_STEPS,
        load_pipeline,
    )

    pipe = load_pipeline()
    history = np.load(
        ROOT / "data_external/ucp_path_pilot_v1/processed/panel_test.npz", allow_pickle=False
    )["history"]

    rows = []
    for arm in arms:
        source = broken if arm == "P_BROKEN" else splits
        split = source["test"]
        model = ParticleForecaster(
            arm=arm,
            n_base=split.base.shape[-1],
            n_weather=split.weather.shape[-1],
            n_farms=n_farms,
            hidden=cfg.hidden,
            n_particles=cfg.n_particles,
            dropout=cfg.dropout,
        ).to(device)
        model.load_state_dict(torch.load(runs_dir(tag) / f"{arm}_seed{seed}.pt"))
        model.eval()

        for batch in BATCHES:
            if batch > len(split):
                continue
            sl = slice(0, batch)
            base, weather, farm = split.base[sl], split.weather[sl], split.farm_idx[sl]
            ctx = torch.from_numpy(history[:batch]).unsqueeze(1)
            torch.cuda.reset_peak_memory_stats()

            if arm == "MC":
                def adapter():
                    with torch.no_grad():
                        pieces = [
                            model(base, weather[:, k : k + 1], farm)
                            for k in range(weather.shape[1])
                        ]
                        return torch.cat(pieces, dim=-1)  # mixture construction included
            else:
                def adapter():
                    with torch.no_grad():
                        return model(base, weather, farm)

            adapter_ms = time_call(adapter)
            adapter_peak = torch.cuda.max_memory_allocated() / 1e6

            torch.cuda.reset_peak_memory_stats()

            def end_to_end():
                with torch.no_grad():
                    pipe.predict_quantiles(
                        inputs=ctx, prediction_length=PREDICTION_STEPS,
                        quantile_levels=list(pipe.quantiles),
                    )
                    return adapter()

            # Chronos dominates here, so fewer repeats keep the measurement honest and short
            e2e_ms = time_call(end_to_end, warmup=3, timed=10)
            rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "batch": batch,
                    "adapter_only_ms": round(adapter_ms, 4),
                    "end_to_end_ms": round(e2e_ms, 3),
                    "adapter_peak_mb": round(adapter_peak, 1),
                    "end_to_end_peak_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1),
                    "warmup": WARMUP,
                    "timed": TIMED,
                }
            )
            print(rows[-1], flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(out_dir(tag) / "latency.csv", index=False)
    table[["arm", "batch", "adapter_peak_mb", "end_to_end_peak_mb"]].to_csv(
        out_dir(tag) / "peak_memory.csv", index=False
    )
    (out_dir(tag) / "latency_protocol.json").write_text(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "warmup": WARMUP,
                "timed_adapter": TIMED,
                "timed_end_to_end": 10,
                "synchronised": True,
                "adapter_only": "arm forward pass given cached Chronos features",
                "end_to_end": "one Chronos-2-synth forward per example plus the arm forward",
                "mc": "every scenario call and the mixture construction are inside the timed region",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
