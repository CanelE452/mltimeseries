"""Synthetic data generator for the PEFT shift-mechanism screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np


BASE_SEED = 20260908
L = 256
H = 16
BURN_IN = 768
A = 0.5
B = math.sqrt(0.39)
SIGMA = 0.6
DRIVER_LAG = 48
TRAIN_CORPORA = (0, 1, 2)
N_BY_SPLIT = {"train": 64, "val": 128, "eval": 512}
RAW_LAGS = np.asarray([32, 48, 64], dtype=np.int64)
QUANTILES = np.asarray(
    [
        0.01,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
        0.99,
    ],
    dtype=np.float32,
)
CONDITIONS = {
    "Q00": {"s": 32, "theta": math.pi / 12, "changed_term": "baseline"},
    "Q10": {"s": 64, "theta": math.pi / 12, "changed_term": "self_lag"},
    "Q01": {"s": 32, "theta": math.pi / 4, "changed_term": "driver_angle"},
    "Q11": {"s": 64, "theta": math.pi / 4, "changed_term": "self_lag_and_driver_angle"},
}
SPLIT_CODES = {"train": 1, "val": 2, "eval": 3, "qc": 4}


@dataclass(frozen=True)
class Condition:
    name: str
    self_lag: int
    theta: float

    @property
    def cos_theta(self) -> float:
        return math.cos(self.theta)

    @property
    def sin_theta(self) -> float:
        return math.sin(self.theta)


def condition(name: str) -> Condition:
    cfg = CONDITIONS[name]
    return Condition(name=name, self_lag=int(cfg["s"]), theta=float(cfg["theta"]))


def seed_sequence(split: str, corpus: int, stream: int = 0) -> np.random.SeedSequence:
    corpus_key = 0 if split in {"val", "eval", "qc"} else int(corpus)
    return np.random.SeedSequence([BASE_SEED, SPLIT_CODES[split], corpus_key, int(stream)])


def _innovations(split: str, corpus: int, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed_sequence(split, corpus))
    length = BURN_IN + L + H
    u = rng.standard_normal((n, length), dtype=np.float64)
    v = rng.standard_normal((n, length), dtype=np.float64)
    eps = rng.standard_normal((n, length), dtype=np.float64)
    return u, v, eps


def _series_from_innovations(cond: Condition, u: np.ndarray, v: np.ndarray, eps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.zeros_like(u)
    oracle = np.zeros_like(u)
    start = max(cond.self_lag, DRIVER_LAG)
    driver = B * (cond.cos_theta * u + cond.sin_theta * v)
    for t in range(start, u.shape[1]):
        oracle[:, t] = A * y[:, t - cond.self_lag] + driver[:, t - DRIVER_LAG]
        y[:, t] = oracle[:, t] + SIGMA * eps[:, t]
    return y, oracle


def _raw_features(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    origin = BURN_IN + L
    out = np.empty((y.shape[0], H, 9), dtype=np.float32)
    series = (y, u, v)
    for channel_index, values in enumerate(series):
        for lag_index, lag in enumerate(RAW_LAGS):
            feature_index = channel_index * len(RAW_LAGS) + lag_index
            for lead in range(H):
                out[:, lead, feature_index] = values[:, origin + lead - int(lag)]
    return out


def _contexts_targets_oracle(cond: Condition, split: str, corpus: int, n: int) -> dict[str, np.ndarray]:
    u, v, eps = _innovations(split, corpus, n)
    y, oracle = _series_from_innovations(cond, u, v, eps)
    origin = BURN_IN + L
    context = np.stack(
        [
            y[:, BURN_IN:origin],
            u[:, BURN_IN:origin],
            v[:, BURN_IN:origin],
        ],
        axis=1,
    ).astype(np.float32)
    target = y[:, origin : origin + H].astype(np.float32)
    oracle_mean = oracle[:, origin : origin + H].astype(np.float32)
    z = np.asarray([NormalDist().inv_cdf(float(q)) for q in QUANTILES], dtype=np.float32)
    oracle_quantiles = oracle_mean[:, None, :] + SIGMA * z[None, :, None]
    return {
        "context": context,
        "target": target,
        "oracle_mean": oracle_mean,
        "oracle_quantiles": oracle_quantiles.astype(np.float32),
        "raw_features": _raw_features(y, u, v),
    }


def _episode_ids(split: str, corpus: int, n: int) -> np.ndarray:
    corpus_key = 0 if split in {"val", "eval"} else int(corpus)
    return np.asarray([f"{split}_c{corpus_key:02d}_e{i:05d}" for i in range(n)], dtype="<U20")


def build_archive(condition_name: str, corpus: int) -> dict[str, np.ndarray]:
    cond = condition(condition_name)
    arrays: dict[str, np.ndarray] = {"quantiles": QUANTILES.copy()}
    for split, n in N_BY_SPLIT.items():
        pack = _contexts_targets_oracle(cond, split, corpus, n)
        suffix = "train" if split == "train" else split
        arrays[f"context_{suffix}"] = pack["context"]
        arrays[f"target_{suffix}"] = pack["target"]
        arrays[f"oracle_mean_{suffix}"] = pack["oracle_mean"]
        arrays[f"oracle_quantiles_{suffix}"] = pack["oracle_quantiles"]
        arrays[f"raw_features_{suffix}"] = pack["raw_features"]
        arrays[f"episode_ids_{suffix}"] = _episode_ids(split, corpus, n)
    return arrays


def _acf(values: np.ndarray, lag: int) -> float:
    if values.ndim == 1:
        x0 = values[:-lag]
        x1 = values[lag:]
    else:
        x0 = values[:, :-lag].reshape(-1)
        x1 = values[:, lag:].reshape(-1)
    x0 = x0 - x0.mean()
    x1 = x1 - x1.mean()
    denom = float(np.sqrt(np.dot(x0, x0) * np.dot(x1, x1)))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(x0, x1) / denom)


def _qc_for_condition(condition_name: str) -> dict[str, Any]:
    cond = condition(condition_name)
    n = 384
    pack = _contexts_targets_oracle(cond, "qc", 0, n)
    episode_y = np.concatenate([pack["context"][:, 0, :], pack["target"]], axis=1)
    combined_y = episode_y.reshape(-1)
    residual = (pack["target"] - pack["oracle_mean"]).reshape(-1)
    acf_lag1 = _acf(episode_y, 1)
    acf_self = _acf(episode_y, cond.self_lag)
    acf_double = _acf(episode_y, 2 * cond.self_lag)
    acf_other_lag = 16 if cond.self_lag != 16 else 17
    acf_other = _acf(episode_y, acf_other_lag)

    u_context = pack["context"][:, 1, :]
    v_context = pack["context"][:, 2, :]
    origin_local = L
    swap_mean = np.empty_like(pack["oracle_mean"])
    for lead in range(H):
        y_lag = pack["context"][:, 0, origin_local + lead - cond.self_lag]
        u_lag = v_context[:, origin_local + lead - DRIVER_LAG]
        v_lag = u_context[:, origin_local + lead - DRIVER_LAG]
        swap_mean[:, lead] = A * y_lag + B * (cond.cos_theta * u_lag + cond.sin_theta * v_lag)
    swap_delta = np.abs(swap_mean - pack["oracle_mean"]).mean()

    checks = {
        "mean_abs_lt_0.06": bool(abs(float(combined_y.mean())) < 0.06),
        "var_absdiff_lt_0.08": bool(abs(float(combined_y.var()) - 1.0) < 0.08),
        "acf_self_absdiff_lt_0.07": bool(abs(acf_self - A) < 0.07),
        "acf_double_absdiff_lt_0.08": bool(abs(acf_double - A**2) < 0.08),
        "acf_other_abs_lt_0.06": bool(abs(acf_other) < 0.06),
        "oracle_residual_mean_abs_lt_0.04": bool(abs(float(residual.mean())) < 0.04),
        "oracle_residual_var_absdiff_lt_0.06": bool(abs(float(residual.var()) - SIGMA**2) < 0.06),
    }
    return {
        "n_episodes": n,
        "mean_y": float(combined_y.mean()),
        "var_y": float(combined_y.var()),
        "acf_lag1": acf_lag1,
        "acf_self_lag": acf_self,
        "acf_2self_lag": acf_double,
        "acf_other_lag": acf_other_lag,
        "acf_other": acf_other,
        "oracle_residual_mean": float(residual.mean()),
        "oracle_residual_var": float(residual.var()),
        "uv_swap_oracle_mean_abs_delta": float(swap_delta),
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def _archive_manifest(condition_name: str, corpus: int, qc: dict[str, Any]) -> dict[str, Any]:
    cfg = CONDITIONS[condition_name]
    return {
        "version": "peft_shift_mechanism_v1.synthetic.20260908",
        "condition": condition_name,
        "corpus": int(corpus),
        "base_seed": BASE_SEED,
        "seed_protocol": "numpy SeedSequence([base_seed, split_code, corpus_key, stream]); val/eval corpus_key=0",
        "generator": {
            "equation": "Y[t]=0.5*Y[t-s]+sqrt(.39)*(cos(theta)*U[t-48]+sin(theta)*V[t-48])+0.6*epsilon[t]",
            "a": A,
            "b": B,
            "sigma": SIGMA,
            "driver_lag": DRIVER_LAG,
            "self_lag": int(cfg["s"]),
            "theta_radians": float(cfg["theta"]),
            "theta_label": "pi/12" if float(cfg["theta"]) == math.pi / 12 else "pi/4",
            "changed_term": cfg["changed_term"],
            "burn_in": BURN_IN,
            "context_length": L,
            "horizon": H,
            "lead_indexing": "zero_based: lead h=0..15 predicts Y[origin+h]",
        },
        "splits": {
            "train": {"episodes": N_BY_SPLIT["train"], "shared_across_corpora": False},
            "val": {"episodes": N_BY_SPLIT["val"], "shared_across_corpora": True},
            "eval": {"episodes": N_BY_SPLIT["eval"], "shared_across_corpora": True},
        },
        "input_scope": {
            "context_rows": ["Y_past", "U_past", "V_past"],
            "future_observed_inputs": [],
            "future_supervision": ["Y_future"],
            "raw_feature_order": "channels-major: Y lags 32/48/64, U lags 32/48/64, V lags 32/48/64",
            "raw_feature_alignment": "raw_features[:, h, feature] = channel[origin + h - lag]",
        },
        "theory": {
            "stationary_var_y": 1.0,
            "oracle_future_var": SIGMA**2,
            "y_only_oracle_future_var": B**2 + SIGMA**2,
            "theta45_uv_exchangeability_note": (
                "At theta=pi/4 the oracle mean is invariant to swapping U and V; "
                "at theta=pi/12 it is not. This is an interpretive confound for "
                "permutation-equivariant group models without channel identity, not a data-generation bug."
            ),
        },
        "qc": qc,
    }


def _save_npz(path: Path, arrays: dict[str, np.ndarray], manifest: dict[str, Any]) -> None:
    manifest_json = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **arrays,
        manifest_json=np.asarray(manifest_json, dtype=f"<U{len(manifest_json)}"),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    qc = {name: _qc_for_condition(name) for name in CONDITIONS}
    files: dict[str, Any] = {}
    for condition_name in CONDITIONS:
        for corpus in TRAIN_CORPORA:
            arrays = build_archive(condition_name, corpus)
            manifest = _archive_manifest(condition_name, corpus, qc[condition_name])
            path = output / f"{condition_name}_c{corpus}.npz"
            _save_npz(path, arrays, manifest)
            files[path.name] = {"sha256": _sha256(path), "bytes": path.stat().st_size}

    summary = {
        "version": "peft_shift_mechanism_v1.synthetic.20260908",
        "files": files,
        "conditions": CONDITIONS,
        "train_corpora": list(TRAIN_CORPORA),
        "n_by_split": N_BY_SPLIT,
        "quantiles": QUANTILES.tolist(),
        "qc": qc,
        "all_qc_passed": bool(all(item["passed"] for item in qc.values())),
    }
    (output / "manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = generate(args.output)
    print(
        json.dumps(
            {
                "files": len(summary["files"]),
                "all_qc_passed": summary["all_qc_passed"],
                "output": str(args.output),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
