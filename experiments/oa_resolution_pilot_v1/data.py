"""Load the two source datasets onto one shared 10-minute base grid.

Both datasets end up as the same object: a regular 10-minute grid of *interval
means* over each base bin, plus a validity mask.  Nothing downstream ever sees
the raw files again.

Jena (MPI-BGC weather station, WS Beutenberg / mpi_roof)
  The published records are already 10-minute statistics: the logger scans the
  sensors every 10 s and stores 10-minute averages, totals and maxima
  (Weatherstation.pdf, section 2).  The timestamp marks the END of the averaging
  period (Table 3).  So the record labelled 01.01.2023 00:10 is the mean over
  [00:00, 00:10).  p / T / rh are averages, and are used as the base bins as-is.

UCI Individual Household Electric Power Consumption
  The published records are one-minute averages of power, voltage and current
  (official variable descriptions).  Ten consecutive minute means average to the
  ten-minute mean of the same intensive quantity, so a base bin is the mean of
  its 10 minute records -- and only when all 10 are present.  The sub-metering
  columns are watt-hour accumulations, a different physical quantity, and are
  excluded (see measurement_semantics.json).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"

BASE_INTERVAL_MINUTES = 10

# Jena: the three variables whose 10-minute record is documented as an interval
# average (they carry no "max." prefix and are not accumulations).
JENA_CHANNELS = {
    "temperature": "T (degC)",
    "pressure": "p (mbar)",
    "relative_humidity": "rh (%)",
}
JENA_FILE_YEARS = {
    2019: ["mpi_roof_2019a.csv", "mpi_roof_2019b.csv"],
    2020: ["mpi_roof_2020a.csv", "mpi_roof_2020b.csv"],
    2021: ["mpi_roof_2021a.csv", "mpi_roof_2021b.csv"],
    2022: ["mpi_roof_2022a.csv", "mpi_roof_2022b.csv"],
    2023: ["mpi_roof_2023a.csv", "mpi_roof_2023b.csv"],
    2024: ["mpi_roof_2024.csv"],
}
# The datalogger writes -9999 where a channel produced no valid reading.  These
# are not measurements: -9999 degC / mbar / %rh are outside every instrument's
# documented range, so they are treated as missing rather than as data.
JENA_MISSING_SENTINEL = -9990.0

UCI_CHANNELS = {
    "global_active_power": "Global_active_power",
    "global_reactive_power": "Global_reactive_power",
    "voltage": "Voltage",
    "global_intensity": "Global_intensity",
}


@dataclass(frozen=True)
class BaseGrid:
    """A dataset on the shared 10-minute base grid.

    values : (n_bins, n_channels) float64, the interval mean over each base bin.
    valid  : (n_bins, n_channels) bool, True where the bin is a real measurement.
    stamps : (n_bins,) datetime64, the START of each base bin.
    """

    dataset: str
    channels: tuple[str, ...]
    values: np.ndarray
    valid: np.ndarray
    stamps: pd.DatetimeIndex
    period_start: pd.Timestamp
    period_end: pd.Timestamp
    provenance: dict

    @property
    def n_bins(self) -> int:
        return self.values.shape[0]


def _bin_starts(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Base-bin start stamps covering the half-open physical window [start, end)."""
    return pd.date_range(start, end, freq=f"{BASE_INTERVAL_MINUTES}min", inclusive="left")


# --------------------------------------------------------------------------- Jena


def _read_jena_years(years: list[int]) -> pd.DataFrame:
    frames = []
    for y in years:
        for name in JENA_FILE_YEARS[y]:
            path = DATA / "jena_mpi_roof" / name
            df = pd.read_csv(path, encoding="latin-1")
            df["_ts_end"] = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    # mpi_roof_2023b re-appends a 24 h block; every duplicated stamp carries a
    # byte-identical row, so keeping the first occurrence loses nothing.
    n_before = len(out)
    out = out.drop_duplicates(subset="_ts_end", keep="first").sort_values("_ts_end")
    out = out.reset_index(drop=True)
    out.attrs["dropped"] = n_before - len(out)
    return out


def load_jena(period_start: pd.Timestamp, period_end: pd.Timestamp) -> BaseGrid:
    years = sorted({period_start.year + k for k in range(period_end.year - period_start.year + 1)})
    years = [y for y in years if y in JENA_FILE_YEARS]
    raw = _read_jena_years(years)

    # The stamp marks the end of the averaging period; the bin it describes starts
    # one base interval earlier.
    raw["_ts_start"] = raw["_ts_end"] - pd.Timedelta(minutes=BASE_INTERVAL_MINUTES)
    raw = raw.set_index("_ts_start")

    stamps = _bin_starts(period_start, period_end)
    names = tuple(JENA_CHANNELS)
    values = np.full((len(stamps), len(names)), np.nan)
    for j, ch in enumerate(names):
        col = raw[JENA_CHANNELS[ch]].reindex(stamps)
        v = pd.to_numeric(col, errors="coerce").to_numpy(dtype=float)
        v[v <= JENA_MISSING_SENTINEL] = np.nan
        values[:, j] = v
    valid = np.isfinite(values)

    return BaseGrid(
        dataset="jena",
        channels=names,
        values=values,
        valid=valid,
        stamps=stamps,
        period_start=period_start,
        period_end=period_end,
        provenance={
            "source_files": [n for y in years for n in JENA_FILE_YEARS[y]],
            "duplicate_stamps_dropped": int(raw.attrs.get("dropped", 0)),
            "record_semantics": "native 10-minute interval mean; timestamp = end of period",
        },
    )


# ---------------------------------------------------------------------------- UCI


def _read_uci() -> pd.DataFrame:
    path = DATA / "uci_household_power" / "household_power_consumption.txt"
    df = pd.read_csv(path, sep=";", na_values=["?"], low_memory=False)
    df["_ts"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    return df.set_index("_ts")


def load_uci(period_start: pd.Timestamp, period_end: pd.Timestamp, raw: pd.DataFrame | None = None) -> BaseGrid:
    raw = _read_uci() if raw is None else raw
    minutes = pd.date_range(period_start, period_end, freq="1min", inclusive="left")
    names = tuple(UCI_CHANNELS)

    n_bins = len(minutes) // BASE_INTERVAL_MINUTES
    values = np.full((n_bins, len(names)), np.nan)
    for j, ch in enumerate(names):
        col = pd.to_numeric(raw[UCI_CHANNELS[ch]].reindex(minutes), errors="coerce")
        m = col.to_numpy(dtype=float).reshape(n_bins, BASE_INTERVAL_MINUTES)
        # All ten minute records present -> the bin is the mean of the ten
        # minute means.  One missing -> the bin is missing.  No partial means.
        ok = np.isfinite(m).all(axis=1)
        binned = np.full(n_bins, np.nan)
        binned[ok] = m[ok].mean(axis=1)
        values[:, j] = binned
    valid = np.isfinite(values)

    return BaseGrid(
        dataset="uci",
        channels=names,
        values=values,
        valid=valid,
        stamps=_bin_starts(period_start, period_end),
        period_start=period_start,
        period_end=period_end,
        provenance={
            "source_files": ["household_power_consumption.txt"],
            "record_semantics": (
                "native 1-minute interval mean; base bin = mean of 10 valid minute means, "
                "missing if any of the 10 is absent"
            ),
        },
    )


# ---------------------------------------------------------------------- selection


def jena_period_candidates() -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Declared candidate set for the Jena period, fixed before any model runs."""
    return [
        (pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y + 2}-01-01"))
        for y in (2019, 2020, 2021, 2022, 2023)
    ]


def uci_period_candidates() -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Only two full calendar-year pairs fit inside 2006-12-16 .. 2010-11-26."""
    return [
        (pd.Timestamp("2007-01-01"), pd.Timestamp("2009-01-01")),
        (pd.Timestamp("2008-01-01"), pd.Timestamp("2010-01-01")),
    ]


def coverage(grid: BaseGrid) -> float:
    """Fraction of (bin, channel) cells that carry a real measurement."""
    return float(grid.valid.mean())


def select_period(dataset: str) -> dict:
    """Apply the pre-registered rule: available, continuous 2 years, max coverage,
    ties broken towards the earlier window.  Forecast accuracy plays no part."""
    if dataset == "jena":
        cands, loader = jena_period_candidates(), load_jena
    elif dataset == "uci":
        raw = _read_uci()
        cands = uci_period_candidates()

        def loader(a, b):  # noqa: ANN001
            return load_uci(a, b, raw=raw)
    else:
        raise ValueError(dataset)

    rows = []
    for a, b in cands:
        g = loader(a, b)
        rows.append({"start": str(a.date()), "end": str(b.date()), "coverage": coverage(g),
                     "n_bins": g.n_bins})
    best = max(range(len(rows)), key=lambda i: (round(rows[i]["coverage"], 6), -i))
    return {"candidates": rows, "selected_index": best, "selected": rows[best]}


# ------------------------------------------------------------------------- splits


def chronological_split(n_bins: int) -> dict[str, tuple[int, int]]:
    """70 / 10 / 20 in time order.  One rounding rule, used everywhere."""
    n_train = int(n_bins * 0.70)
    n_val = int(n_bins * 0.10)
    return {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, n_bins),
    }


def train_scaling(grid: BaseGrid, splits: dict[str, tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and population std over the train base bins only."""
    lo, hi = splits["train"]
    v = grid.values[lo:hi]
    ok = grid.valid[lo:hi]
    mu = np.array([v[ok[:, j], j].mean() for j in range(v.shape[1])])
    sd = np.array([v[ok[:, j], j].std(ddof=0) for j in range(v.shape[1])])
    return mu, sd


def grid_fingerprint(grid: BaseGrid) -> str:
    h = hashlib.sha256()
    h.update(grid.dataset.encode())
    h.update(",".join(grid.channels).encode())
    h.update(np.ascontiguousarray(np.nan_to_num(grid.values, nan=-1e30)).tobytes())
    h.update(np.ascontiguousarray(grid.valid).tobytes())
    return h.hexdigest()


def load_dataset(dataset: str, period_start: str, period_end: str) -> BaseGrid:
    a, b = pd.Timestamp(period_start), pd.Timestamp(period_end)
    return load_jena(a, b) if dataset == "jena" else load_uci(a, b)


# ------------------------------------------------------------------------ windows

HISTORY_BINS = 288
FORECAST_BINS = 72
EVAL_ORIGIN_STRIDE = 72  # 12 hours


def eligible_origins(grid: BaseGrid, splits: dict, split: str, channel_index: int,
                     stride: int = 1) -> np.ndarray:
    """Forecast origins o whose whole window is usable.

    A window is [o-288, o) history and [o, o+72) target.  It qualifies when the
    target interval lies entirely inside `split` and every base bin it touches --
    history and target -- is a real measurement.

    Eligibility deliberately ignores r and the operation.  Requiring all 288
    history bins makes the window set identical for every report interval, every
    operation and all three arms, so no method is ever scored on a different set
    of windows than another.
    """
    lo, hi = splits[split]
    origins = np.arange(lo, hi - FORECAST_BINS + 1)
    if origins.size == 0:
        return origins
    if stride > 1:
        origins = origins[::stride]
    origins = origins[origins - HISTORY_BINS >= 0]

    csum = np.concatenate([[0], np.cumsum(grid.valid[:, channel_index])])

    def all_valid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (csum[b] - csum[a]) == (b - a)

    keep = all_valid(origins - HISTORY_BINS, origins) & all_valid(origins, origins + FORECAST_BINS)
    return origins[keep]


if __name__ == "__main__":
    for ds in ("jena", "uci"):
        print(json.dumps({ds: select_period(ds)}, indent=2, default=str), flush=True)
