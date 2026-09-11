"""Export the local fev-cache Hospital Arrow panel to the kit's canonical NPZ.

The kit (third_party/hospital_shared_adaptation_kit) reads TSF/ZIP or a canonical
NPZ with exactly the keys values[N,84] float64, series_ids[N] str, start_months[N] str.
It has no Arrow reader, so this is the one exporter its CLI section 2(2) allows.
Order, ids and values are passed through unchanged; nothing is truncated, filled or
re-sorted. Every contract check raises instead of repairing.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.ipc as ipc

EXPECTED_SERIES, EXPECTED_MONTHS, EXPECTED_START = 767, 84, "2000-01"


def read_arrow(path):
    try:
        return ipc.open_stream(path).read_all()
    except Exception:
        return ipc.open_file(path).read_all()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrow", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    arrow = Path(args.arrow)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    table = read_arrow(str(arrow))
    d = table.to_pydict()
    ids = np.asarray(d["id"], dtype=str)
    targets, stamps = d["target"], d["timestamp"]

    if len(ids) != EXPECTED_SERIES or len(set(ids.tolist())) != len(ids) or np.any(ids == ""):
        raise ValueError(f"Expected {EXPECTED_SERIES} unique nonempty ids, got {len(ids)}")
    lengths = {len(x) for x in targets}
    if lengths != {EXPECTED_MONTHS}:
        raise ValueError(f"Expected every series to have {EXPECTED_MONTHS} months, got lengths {sorted(lengths)}")
    starts = []
    for s in stamps:
        months = [x.year * 12 + x.month for x in s]
        if any(b - a != 1 for a, b in zip(months, months[1:])):
            raise ValueError("Non-consecutive monthly calendar")
        starts.append(f"{s[0].year:04d}-{s[0].month:02d}")
    # fev monthly stamps label the month; whether they sit on the first or last day is
    # recorded in the receipt, not used to reject. The canonical key is 'YYYY-MM' only.
    stamp_pattern = sorted({(x.day, x.hour, x.minute) for s in stamps for x in s})
    starts = np.asarray(starts, dtype=str)
    if not np.all(starts == EXPECTED_START):
        raise ValueError(f"Expected all series to start {EXPECTED_START}, got {sorted(set(starts.tolist()))}")
    values = np.asarray([np.asarray(x, dtype=np.float64) for x in targets], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("Missing or nonfinite values present; no imputation is performed")
    if np.any(values < 0):
        raise ValueError("Negative values present")

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, values=values, series_ids=ids, start_months=starts)
    receipt = {
        "source_arrow": str(arrow), "source_arrow_sha256": hashlib.sha256(arrow.read_bytes()).hexdigest(),
        "source_dataset": "autogluon/fev_datasets", "source_config": "hospital",
        "source_cache_fingerprint_dir": arrow.parent.name,
        "canonical_npz": str(out), "canonical_npz_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "shape": list(values.shape), "start_month": EXPECTED_START,
        "value_min": float(values.min()), "value_max": float(values.max()),
        "all_integer_valued": bool(np.all(np.mod(values, 1) == 0)),
        "timestamp_day_hour_minute_patterns": stamp_pattern,
        "id_order": "unchanged from arrow row order", "first_ids": ids[:3].tolist(), "last_ids": ids[-3:].tolist(),
    }
    out.with_name("export_receipt.json").write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: receipt[k] for k in ("shape", "source_arrow_sha256", "canonical_npz_sha256", "first_ids")}, indent=1))


if __name__ == "__main__":
    main()
