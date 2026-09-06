"""Data loading, splits, normalization and the frozen evaluation grid.

Contract source: 01_forecast_query_tokenization_CLI.txt sections 4 and 5.

Everything here is deterministic given (dataset_id, seed). The training schedule is
generated up front so that every arm trained at the same seed consumes exactly the
same (update, channel, origin) sequence.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

L = 1024
P = 16
N = 64
HORIZONS = (96, 336)
H_MAX = 336
EVAL_STRIDE = 96

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(REPO_ROOT, "runs", "hq_token_pilot_v1", "_arrays")

# section 4: three datasets, paths as fetched by scripts/fetch_data.py
DATASETS = {
    "ETTm2": {
        "csv": os.path.join(REPO_ROOT, "data", "ETT-small", "ETTm2.csv"),
        "split_rule": "ett_minute_fixed",
        "expected_rows": 69680,
        "expected_channels": 7,
    },
    "weather": {
        "csv": os.path.join(REPO_ROOT, "data", "weather", "weather.csv"),
        "split_rule": "ratio_070_020",
        "expected_rows": 52696,
        "expected_channels": 21,
    },
    "electricity": {
        "csv": os.path.join(REPO_ROOT, "data", "electricity", "electricity.csv"),
        "split_rule": "ratio_070_020",
        "expected_rows": 26304,
        "expected_channels": 321,
    },
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def split_boundaries(rule: str, total_rows: int) -> dict:
    """Target split boundaries as half-open [start, end) intervals.

    ett_minute_fixed reproduces PatchTST Dataset_ETT_minute border2s
    (third_party/PatchTST/PatchTST_supervised/data_provider/data_loader.py:136-137).
    ratio_070_020 reproduces Dataset_Custom (same file, line 234-238).
    """
    if rule == "ett_minute_fixed":
        n_train = 12 * 30 * 24 * 4          # 34560
        n_val = 4 * 30 * 24 * 4             # 11520
        train = (0, n_train)
        val = (n_train, n_train + n_val)
        test = (n_train + n_val, n_train + 2 * n_val)
    elif rule == "ratio_070_020":
        n_train = int(0.7 * total_rows)
        n_test = int(0.2 * total_rows)
        n_val = total_rows - n_train - n_test
        train = (0, n_train)
        val = (n_train, n_train + n_val)
        test = (n_train + n_val, total_rows)
    else:
        raise ValueError(rule)
    return {"train": train, "validation": val, "test": test}


@dataclass
class DatasetSpec:
    dataset_id: str
    csv_path: str
    csv_sha256: str
    total_rows: int
    n_channels: int
    time_column: str
    channel_columns: list
    rows_used: int
    split_rule: str
    splits: dict
    freq_seconds: int
    time_start: str
    time_end: str
    n_nan: int
    n_inf: int
    constant_train_channels: list
    train_mean_sha256: str
    train_std_sha256: str


class Dataset:
    """One benchmark file: values, normalization and the frozen origin grids."""

    def __init__(self, dataset_id: str):
        cfg = DATASETS[dataset_id]
        self.dataset_id = dataset_id
        self.csv_path = cfg["csv"]
        self.split_rule = cfg["split_rule"]

        values, columns, time_col, stamps, csv_sha = _load_values(dataset_id, cfg)
        self.values = values                     # [T, C] float32, raw units
        self.channel_columns = columns
        self.time_column = time_col
        self.csv_sha256 = csv_sha
        self.total_rows = int(values.shape[0])
        self.n_channels = int(values.shape[1])

        if self.total_rows != cfg["expected_rows"]:
            raise AssertionError(
                f"{dataset_id}: {self.total_rows} rows, expected {cfg['expected_rows']}"
            )
        if self.n_channels != cfg["expected_channels"]:
            raise AssertionError(
                f"{dataset_id}: {self.n_channels} channels, expected {cfg['expected_channels']}"
            )

        self.splits = split_boundaries(self.split_rule, self.total_rows)
        self.rows_used = self.splits["test"][1]

        tr0, tr1 = self.splits["train"]
        train_block = self.values[tr0:tr1]
        self.n_nan = int(np.isnan(self.values[: self.rows_used]).sum())
        self.n_inf = int(np.isinf(self.values[: self.rows_used]).sum())

        mean = train_block.mean(axis=0, dtype=np.float64)
        std = train_block.std(axis=0, ddof=0, dtype=np.float64)
        self.constant_train_channels = np.nonzero(std == 0.0)[0].tolist()
        std = np.where(std == 0.0, 1.0, std)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

        # timestamps are recorded, never fed to the model
        self.time_start = str(stamps[0])
        self.time_end = str(stamps[self.total_rows - 1])
        self.freq_seconds = _infer_freq_seconds(stamps)

    @classmethod
    def from_array(cls, dataset_id: str, values: np.ndarray, split_rule: str = "ratio_070_020"):
        """Build a dataset from an in-memory array, for tests and synthetic controls.

        Bypasses the CSV contract on purpose: nothing built this way may be reported as
        a benchmark result.
        """
        self = cls.__new__(cls)
        values = np.asarray(values, dtype=np.float32)
        self.dataset_id = dataset_id
        self.csv_path = "<in-memory>"
        self.csv_sha256 = sha256_bytes(values.tobytes())
        self.split_rule = split_rule
        self.values = values
        self.channel_columns = [f"c{i}" for i in range(values.shape[1])]
        self.time_column = "<none>"
        self.total_rows = int(values.shape[0])
        self.n_channels = int(values.shape[1])
        self.splits = split_boundaries(split_rule, self.total_rows)
        self.rows_used = self.splits["test"][1]
        tr0, tr1 = self.splits["train"]
        block = values[tr0:tr1]
        mean = block.mean(axis=0, dtype=np.float64)
        std = block.std(axis=0, ddof=0, dtype=np.float64)
        self.constant_train_channels = np.nonzero(std == 0.0)[0].tolist()
        self.mean = mean.astype(np.float32)
        self.std = np.where(std == 0.0, 1.0, std).astype(np.float32)
        self.n_nan = int(np.isnan(values).sum())
        self.n_inf = int(np.isinf(values).sum())
        self.time_start = self.time_end = "<none>"
        self.freq_seconds = -1
        return self

    # ---- windows -----------------------------------------------------------

    def train_origins(self) -> np.ndarray:
        """section 5: o >= L and o + H_MAX <= train_end, shared by both horizons."""
        _, tr1 = self.splits["train"]
        return np.arange(L, tr1 - H_MAX + 1, dtype=np.int64)

    def eval_origins(self, split: str) -> np.ndarray:
        """section 5: start, start+96, ... <= split_end - H_MAX."""
        s, e = self.splits[split]
        end = e - H_MAX
        if end < s:
            return np.zeros(0, dtype=np.int64)
        return np.arange(s, end + 1, EVAL_STRIDE, dtype=np.int64)

    def inputs(self, origins: np.ndarray, channels: np.ndarray) -> np.ndarray:
        """Normalized history [n, L]. Sliced per batch, never materialized whole."""
        idx = origins[:, None] + np.arange(-L, 0)[None, :]
        raw = self.values[idx, channels[:, None]]
        return (raw - self.mean[channels][:, None]) / self.std[channels][:, None]

    def targets(self, origins: np.ndarray, channels: np.ndarray) -> np.ndarray:
        """Normalized future [n, H_MAX]. Loss masks down to the active horizon."""
        idx = origins[:, None] + np.arange(0, H_MAX)[None, :]
        raw = self.values[idx, channels[:, None]]
        return (raw - self.mean[channels][:, None]) / self.std[channels][:, None]

    # ---- manifest ----------------------------------------------------------

    def spec(self) -> DatasetSpec:
        return DatasetSpec(
            dataset_id=self.dataset_id,
            csv_path=os.path.relpath(self.csv_path, REPO_ROOT).replace("\\", "/"),
            csv_sha256=self.csv_sha256,
            total_rows=self.total_rows,
            n_channels=self.n_channels,
            time_column=self.time_column,
            channel_columns=list(self.channel_columns),
            rows_used=self.rows_used,
            split_rule=self.split_rule,
            splits={k: list(v) for k, v in self.splits.items()},
            freq_seconds=self.freq_seconds,
            time_start=self.time_start,
            time_end=self.time_end,
            n_nan=self.n_nan,
            n_inf=self.n_inf,
            constant_train_channels=self.constant_train_channels,
            train_mean_sha256=sha256_bytes(self.mean.tobytes()),
            train_std_sha256=sha256_bytes(self.std.tobytes()),
        )

    def split_key_hash(self) -> str:
        payload = json.dumps(
            {
                "dataset_id": self.dataset_id,
                "csv_sha256": self.csv_sha256,
                "splits": {k: list(v) for k, v in self.splits.items()},
                "channel_columns": list(self.channel_columns),
            },
            sort_keys=True,
        ).encode()
        return sha256_bytes(payload)

    def normalizer_hash(self) -> str:
        return sha256_bytes(self.mean.tobytes() + self.std.tobytes())


def _load_values(dataset_id, cfg):
    """Read the CSV once, cache the numeric block as .npy under runs/ (gitignored)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    csv_sha = sha256_file(cfg["csv"])
    cache = os.path.join(CACHE_DIR, f"{dataset_id}.{csv_sha[:16]}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return (
            z["values"],
            [str(c) for c in z["columns"]],
            str(z["time_column"]),
            z["stamps"],
            csv_sha,
        )

    df = pd.read_csv(cfg["csv"])
    cols = list(df.columns)
    # section 4: read the real columns, do not assume a name
    time_col = None
    for c in cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            time_col = c
            break
    if time_col is None:
        raise AssertionError(f"{dataset_id}: no non-numeric time column found in {cols}")
    channel_columns = [c for c in cols if c != time_col]
    non_numeric = [c for c in channel_columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise AssertionError(f"{dataset_id}: non-numeric target columns {non_numeric}")

    values = df[channel_columns].to_numpy(dtype=np.float32)
    stamps = df[time_col].to_numpy().astype(str)
    np.savez(
        cache,
        values=values,
        columns=np.array(channel_columns, dtype=object),
        time_column=time_col,
        stamps=stamps,
    )
    return values, channel_columns, time_col, stamps, csv_sha


def _infer_freq_seconds(stamps) -> int:
    try:
        t = pd.to_datetime(pd.Series(stamps[:3]))
        return int((t.iloc[1] - t.iloc[0]).total_seconds())
    except Exception:
        return -1


# ---- training schedule -----------------------------------------------------


def make_schedule(dataset: Dataset, seed: int, max_updates: int, batch: int) -> dict:
    """Deterministic (update, channel, origin, H) schedule shared by every arm.

    section 5: uniform over channels, uniform over eligible origins, and the true
    horizon alternates 96/336 by update index so one model learns both.
    """
    if max_updates % 2 != 0:
        raise ValueError("max_updates must be even so both horizons get equal updates")
    rng = np.random.default_rng([seed, _stable_int(dataset.dataset_id)])
    origins = dataset.train_origins()
    channels = rng.integers(0, dataset.n_channels, size=(max_updates, batch), dtype=np.int64)
    origin_idx = rng.integers(0, origins.shape[0], size=(max_updates, batch), dtype=np.int64)
    horizons = np.where(np.arange(max_updates) % 2 == 0, HORIZONS[0], HORIZONS[1]).astype(np.int64)
    return {
        "channels": channels,
        "origins": origins[origin_idx],
        "horizons": horizons,
        "n_eligible_origins": int(origins.shape[0]),
        "schedule_sha256": sha256_bytes(
            channels.tobytes() + origins[origin_idx].tobytes() + horizons.tobytes()
        ),
    }


def fake_horizon_train(seed: int, dataset_id: str, max_updates: int, batch: int) -> np.ndarray:
    """Arm R: pooling query horizon drawn independently of the true horizon."""
    rng = np.random.default_rng([seed, _stable_int(dataset_id), 2026090671])
    pick = rng.integers(0, 2, size=(max_updates, batch))
    return np.where(pick == 0, HORIZONS[0], HORIZONS[1]).astype(np.int64)


def fake_horizon_eval(dataset_id: str, split: str, origins, channels, horizons) -> np.ndarray:
    """Arm R at eval time: key-derived so the same key always gets the same fake H.

    section 7 requires the validation/test H_fake to be reproducible; deriving it from
    the logical key rather than from row order makes it invariant to batching.
    """
    origins = np.asarray(origins, dtype=np.int64)
    channels = np.asarray(channels, dtype=np.int64)
    horizons = np.asarray(horizons, dtype=np.int64)
    base = _stable_int(f"{dataset_id}|{split}|2026090671")
    mixed = (
        np.uint64(base)
        + np.uint64(1000003) * origins.astype(np.uint64)
        + np.uint64(1000033) * channels.astype(np.uint64)
        + np.uint64(1000037) * horizons.astype(np.uint64)
    )
    # splitmix64 finalizer: cheap, order-independent, reproducible
    x = mixed
    x ^= x >> np.uint64(30)
    x = x * np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    x = x * np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return np.where((x & np.uint64(1)) == 0, HORIZONS[0], HORIZONS[1]).astype(np.int64)


def _stable_int(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def build_manifest() -> dict:
    out = {"datasets": {}, "provenance": {}}
    for name in DATASETS:
        ds = Dataset(name)
        out["datasets"][name] = asdict(ds.spec())
        out["datasets"][name]["train_origins"] = int(ds.train_origins().shape[0])
        for split in ("validation", "test"):
            out["datasets"][name][f"{split}_origins"] = int(ds.eval_origins(split).shape[0])
        out["datasets"][name]["split_key_hash"] = ds.split_key_hash()
        out["datasets"][name]["normalizer_hash"] = ds.normalizer_hash()
    out["provenance"] = {
        "downloaded_by": "scripts/fetch_data.py",
        "source": "HuggingFace mirror thuml/Time-Series-Library",
        "deviation": (
            "Section 4 asks for the original linked by the PatchTST README (a Google Drive "
            "bundle). These files came from the Time-Series-Library HF mirror instead. Row "
            "counts, channel counts, sampling interval and time span were asserted against "
            "the values reported in the Autoformer/PatchTST line of papers, but the "
            "distribution channel differs from the one named in the spec."
        ),
        "split_reference": (
            "third_party/PatchTST @ 204c21efe0b39603ad6e2ca640ef5896646ab1a9, "
            "PatchTST_supervised/data_provider/data_loader.py lines 136-137 (ETT minute) "
            "and 234-238 (custom). Read only; no PatchTST code is imported."
        ),
    }
    return out


if __name__ == "__main__":
    m = build_manifest()
    for k, v in m["datasets"].items():
        print(
            f"{k}: rows={v['total_rows']} used={v['rows_used']} C={v['n_channels']} "
            f"freq={v['freq_seconds']}s nan={v['n_nan']} inf={v['n_inf']} "
            f"train_origins={v['train_origins']} val={v['validation_origins']} "
            f"test={v['test_origins']} const={len(v['constant_train_channels'])}"
        )
