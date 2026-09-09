"""Read-only probe: do the real covariates carry a lag worth aligning?

Direction D assumes that aligning past covariates to the target by an estimated
lag is worth doing without training. Study08's large effect came from synthetic
data with a planted lag. This measures, on the actual Bike series, whether the
cross-correlation between each covariate and the target peaks away from lag 0.
Reads the raw UCI file only.
"""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path("E:/CODING/proj/mltimeseries")
RAW = ROOT / "data_external/uci_bike_sharing/raw/hour.csv"
COVARIATES = ("temp", "atemp", "hum", "windspeed")
TARGETS = ("casual", "registered")
MAX_LAG = 48

rows = list(csv.DictReader(RAW.open()))
# The hourly file skips hours with no rides; rebuild a dense hourly grid first.
stamps = [(r["dteday"], int(r["hr"])) for r in rows]
days = sorted({d for d, _ in stamps})
day_index = {d: i for i, d in enumerate(days)}
length = len(days) * 24
series = {name: np.full(length, np.nan) for name in COVARIATES + TARGETS}
for r in rows:
    pos = day_index[r["dteday"]] * 24 + int(r["hr"])
    for name in COVARIATES + TARGETS:
        series[name][pos] = float(r[name])

report = {"source": str(RAW.relative_to(ROOT)), "grid_hours": length,
          "observed_hours": int(np.isfinite(series["cnt" if "cnt" in series else "casual"]).sum()),
          "max_lag_searched": MAX_LAG, "peaks": {}}


def best_lag(x, y, max_lag):
    """Lag L maximising |corr(x shifted by L, y)|; positive L means x leads y."""
    out = []
    for lag in range(0, max_lag + 1):
        a = x[: len(x) - lag] if lag else x
        b = y[lag:] if lag else y
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 1000:
            continue
        av, bv = a[ok], b[ok]
        if av.std() == 0 or bv.std() == 0:
            continue
        out.append((lag, float(np.corrcoef(av, bv)[0, 1])))
    peak = max(out, key=lambda t: abs(t[1]))
    zero = next(c for l, c in out if l == 0)
    return {"best_lag": peak[0], "corr_at_best_lag": peak[1], "corr_at_lag0": zero,
            "abs_gain_over_lag0": abs(peak[1]) - abs(zero)}


for target in TARGETS:
    report["peaks"][target] = {c: best_lag(series[c], series[target], MAX_LAG) for c in COVARIATES}

report["verdict_rule"] = ("A lag worth aligning would peak at a nonzero lag and beat lag 0 by a"
                          " clear margin in absolute correlation.")
report["nonzero_peak_count"] = sum(
    1 for t in TARGETS for c in COVARIATES if report["peaks"][t][c]["best_lag"] != 0)
report["max_abs_gain_over_lag0"] = max(
    report["peaks"][t][c]["abs_gain_over_lag0"] for t in TARGETS for c in COVARIATES)
print(json.dumps(report, indent=1))
