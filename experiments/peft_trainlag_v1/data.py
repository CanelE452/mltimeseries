"""Fresh Q00 data with train-only lag selection for RAW alignment controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np


BASE_SEED = 2026090810
SOURCE_FAMILY = "peft_shift_mechanism_v1.Q00"
VERSION = "peft_trainlag_v1.synthetic.20260908"
CONDITION_NAME = "Q00"
L = 256
H = 16
BURN_IN = 768
A = 0.5
B = math.sqrt(0.39)
SIGMA = 0.6
SELF_LAG = 32
DRIVER_LAG = 48
THETA = math.pi / 12
TRAIN_CORPORA = (0, 1, 2)
N_BY_SPLIT = {"train": 64, "val": 128, "eval": 512}
CHANNELS = ("Y", "U", "V")
CANDIDATE_LAGS = np.arange(H, L // 2 + 1, dtype=np.int64)
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
SPLIT_CODES = {"train": 1, "val": 2, "eval": 3, "qc": 4}


@dataclass(frozen=True)
class LagSelection:
    selected_lags: dict[str, int]
    candidate_lags: list[int]
    squared_pearson_scores: dict[str, list[float]]
    pearson_correlations: dict[str, list[float]]
    selected_signs: dict[str, float]
    input_array_hash: str
    context_train_sha256: str
    target_train_sha256: str
    n_train_episodes: int
    n_future_labels: int
    fit_seconds: float
    selector: str
    tie_break: str
    input_scope: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "version": VERSION,
            "condition": CONDITION_NAME,
            "selected_lags": self.selected_lags,
            "candidates": self.candidate_lags,
            "candidate_lags": self.candidate_lags,
            "scores": self.squared_pearson_scores,
            "pearson_correlations": self.pearson_correlations,
            "selected_signs": self.selected_signs,
            "input_array_hash": self.input_array_hash,
            "context_train_sha256": self.context_train_sha256,
            "target_train_sha256": self.target_train_sha256,
            "n_train_episodes": self.n_train_episodes,
            "n_future_labels": self.n_future_labels,
            "fit_seconds": self.fit_seconds,
            "selector": self.selector,
            "tie_break": self.tie_break,
            "input_scope": self.input_scope,
            "known_lag_dictionary_used": False,
            "validation_or_eval_used": False,
            "oracle_or_manifest_used": False,
        }


def array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        arr = np.ascontiguousarray(array)
        digest.update(str(arr.shape).encode("utf-8"))
        digest.update(str(arr.dtype).encode("utf-8"))
        digest.update(arr.tobytes())
    return digest.hexdigest()


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


def _series_from_innovations(u: np.ndarray, v: np.ndarray, eps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.zeros_like(u)
    oracle = np.zeros_like(u)
    driver = B * (math.cos(THETA) * u + math.sin(THETA) * v)
    for t in range(max(SELF_LAG, DRIVER_LAG), u.shape[1]):
        oracle[:, t] = A * y[:, t - SELF_LAG] + driver[:, t - DRIVER_LAG]
        y[:, t] = oracle[:, t] + SIGMA * eps[:, t]
    return y, oracle


def _contexts_targets_oracle(split: str, corpus: int, n: int) -> dict[str, np.ndarray]:
    u, v, eps = _innovations(split, corpus, n)
    y, oracle = _series_from_innovations(u, v, eps)
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
    }


def _episode_ids(split: str, corpus: int, n: int) -> np.ndarray:
    corpus_key = 0 if split in {"val", "eval"} else int(corpus)
    prefix = f"trainlag_b{BASE_SEED}"
    values = [f"{prefix}_{split}_c{corpus_key:02d}_e{i:05d}" for i in range(n)]
    return np.asarray(values, dtype=f"<U{max(len(value) for value in values)}")


def _validate_train_inputs(context_train: np.ndarray, target_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    context = np.asarray(context_train)
    target = np.asarray(target_train)
    if context.ndim != 3 or context.shape[1:] != (len(CHANNELS), L):
        raise ValueError(f"context_train must have shape [N,3,{L}]")
    if target.ndim != 2 or target.shape != (context.shape[0], H):
        raise ValueError(f"target_train must have shape [N,{H}] with the same episode count")
    if not np.isfinite(context).all() or not np.isfinite(target).all():
        raise ValueError("train lag selection requires finite context and target arrays")
    return context.astype(np.float64, copy=False), target.astype(np.float64, copy=False)


def _candidate_lags(candidate_lags: np.ndarray | None) -> np.ndarray:
    lags = CANDIDATE_LAGS if candidate_lags is None else np.asarray(candidate_lags, dtype=np.int64)
    if lags.ndim != 1 or len(lags) == 0:
        raise ValueError("candidate_lags must be a non-empty one-dimensional array")
    if np.any(np.diff(lags) < 0) or len(np.unique(lags)) != len(lags):
        raise ValueError("candidate_lags must be sorted and unique")
    if int(lags[0]) < H or int(lags[-1]) > L // 2:
        raise ValueError(f"candidate_lags must stay inside the fully observed range {H}..{L // 2}")
    return lags


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x0 = x.reshape(-1).astype(np.float64)
    y0 = y.reshape(-1).astype(np.float64)
    x0 = x0 - x0.mean()
    y0 = y0 - y0.mean()
    denom = float(np.sqrt(np.dot(x0, x0) * np.dot(y0, y0)))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(x0, y0) / denom)


def select_lags_from_train(
    context_train: np.ndarray,
    target_train: np.ndarray,
    candidate_lags: np.ndarray | None = None,
) -> dict[str, Any]:
    """Select one fully observed lag per channel using only train future Y labels."""

    started = time.perf_counter()
    context, target = _validate_train_inputs(context_train, target_train)
    lags = _candidate_lags(candidate_lags)
    leads = np.arange(H, dtype=np.int64)
    correlations: dict[str, list[float]] = {}
    scores: dict[str, list[float]] = {}
    selected_lags: dict[str, int] = {}
    selected_signs: dict[str, float] = {}
    for channel_index, channel in enumerate(CHANNELS):
        channel_corrs = []
        for lag in lags:
            features = context[:, channel_index, L + leads - int(lag)]
            channel_corrs.append(_pearson(features, target))
        values = np.asarray(channel_corrs, dtype=np.float64)
        squared = values * values
        best_index = int(np.argmax(squared))
        correlations[channel] = values.tolist()
        scores[channel] = squared.tolist()
        selected_lags[channel] = int(lags[best_index])
        selected_signs[channel] = float(np.sign(values[best_index]))
    selection = LagSelection(
        selected_lags=selected_lags,
        candidate_lags=[int(value) for value in lags],
        squared_pearson_scores=scores,
        pearson_correlations=correlations,
        selected_signs=selected_signs,
        input_array_hash=array_hash(context_train, target_train),
        context_train_sha256=array_hash(context_train),
        target_train_sha256=array_hash(target_train),
        n_train_episodes=int(context.shape[0]),
        n_future_labels=int(target.size),
        fit_seconds=float(time.perf_counter() - started),
        selector="per-channel squared Pearson correlation over train future Y labels",
        tie_break="smallest lag because candidates are sorted and numpy.argmax returns the first maximum",
        input_scope={
            "context_used": "context_train[:, channel, L + h - lag]",
            "labels_used": "target_train[:, h] only",
            "candidate_lags": f"{H}..{L // 2}",
            "validation_eval_oracle_manifest_used": False,
            "context_y_used_as_extra_label": False,
        },
    )
    return selection.to_json()


def _lag_array(lags: dict[str, int] | list[int] | tuple[int, ...] | np.ndarray) -> np.ndarray:
    if isinstance(lags, dict):
        values = np.asarray([lags[channel] for channel in CHANNELS], dtype=np.int64)
    else:
        values = np.asarray(lags, dtype=np.int64)
    if values.shape != (len(CHANNELS),):
        raise ValueError("lags must provide exactly one lag for Y, U and V")
    if int(values.min()) < H or int(values.max()) > L // 2:
        raise ValueError(f"lags must stay inside the selected fully observed range {H}..{L // 2}")
    return values


def make_lag_features(
    context: np.ndarray,
    lags: dict[str, int] | list[int] | tuple[int, ...] | np.ndarray,
) -> np.ndarray:
    context = np.asarray(context)
    if context.ndim != 3 or context.shape[1:] != (len(CHANNELS), L):
        raise ValueError(f"context must have shape [N,3,{L}]")
    values = _lag_array(lags)
    leads = np.arange(H, dtype=np.int64)
    features = np.empty((context.shape[0], H, len(CHANNELS)), dtype=np.float32)
    for channel_index, lag in enumerate(values):
        features[:, :, channel_index] = context[:, channel_index, L + leads - int(lag)]
    return features


def build_archive(corpus: int) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {"quantiles": QUANTILES.copy()}
    packs: dict[str, dict[str, np.ndarray]] = {}
    for split, n in N_BY_SPLIT.items():
        pack = _contexts_targets_oracle(split, corpus, n)
        packs[split] = pack
        arrays[f"context_{split}"] = pack["context"]
        arrays[f"target_{split}"] = pack["target"]
        arrays[f"oracle_mean_{split}"] = pack["oracle_mean"]
        arrays[f"oracle_quantiles_{split}"] = pack["oracle_quantiles"]
        arrays[f"episode_ids_{split}"] = _episode_ids(split, corpus, n)
    selection = select_lags_from_train(arrays["context_train"], arrays["target_train"])
    lag_array = _lag_array(selection["selected_lags"])
    for split in N_BY_SPLIT:
        arrays[f"raw_features_{split}"] = make_lag_features(arrays[f"context_{split}"], lag_array)
    arrays["selected_lags"] = lag_array.astype(np.int64)
    arrays["candidate_lags"] = np.asarray(selection["candidate_lags"], dtype=np.int64)
    return arrays


def _acf(values: np.ndarray, lag: int) -> float:
    x0 = values[:, :-lag].reshape(-1)
    x1 = values[:, lag:].reshape(-1)
    x0 = x0 - x0.mean()
    x1 = x1 - x1.mean()
    denom = float(np.sqrt(np.dot(x0, x0) * np.dot(x1, x1)))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(x0, x1) / denom)


def _qc() -> dict[str, Any]:
    n = 384
    pack = _contexts_targets_oracle("qc", 0, n)
    episode_y = np.concatenate([pack["context"][:, 0, :], pack["target"]], axis=1)
    combined_y = episode_y.reshape(-1)
    residual = (pack["target"] - pack["oracle_mean"]).reshape(-1)
    acf_self = _acf(episode_y, SELF_LAG)
    acf_double = _acf(episode_y, 2 * SELF_LAG)
    acf_other_lag = 16
    acf_other = _acf(episode_y, acf_other_lag)
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
        "acf_self_lag": acf_self,
        "acf_2self_lag": acf_double,
        "acf_other_lag": acf_other_lag,
        "acf_other": acf_other,
        "oracle_residual_mean": float(residual.mean()),
        "oracle_residual_var": float(residual.var()),
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def _archive_manifest(corpus: int, lag_selection: dict[str, Any], qc: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": VERSION,
        "source_family": SOURCE_FAMILY,
        "fresh_same_family_not_original_source": True,
        "condition": CONDITION_NAME,
        "corpus": int(corpus),
        "base_seed": BASE_SEED,
        "episode_id_namespace": f"trainlag_b{BASE_SEED}",
        "seed_protocol": "numpy SeedSequence([base_seed, split_code, corpus_key, stream]); val/eval corpus_key=0",
        "generator": {
            "equation": "Y[t]=0.5*Y[t-32]+sqrt(.39)*(cos(pi/12)*U[t-48]+sin(pi/12)*V[t-48])+0.6*epsilon[t]",
            "a": A,
            "b": B,
            "sigma": SIGMA,
            "driver_lag": DRIVER_LAG,
            "self_lag": SELF_LAG,
            "theta_radians": THETA,
            "theta_label": "pi/12",
            "changed_term": "fresh_seed_same_q00_family",
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
        "lag_selection": {
            "file": f"Q00_c{corpus}_lags.json",
            "selected_lags": lag_selection["selected_lags"],
            "candidate_lags": f"{H}..{L // 2}",
            "selector": lag_selection["selector"],
        },
        "input_scope": {
            "context_rows": ["Y_past", "U_past", "V_past"],
            "future_observed_inputs": [],
            "future_supervision": ["Y_future"],
            "raw_feature_order": "selected train-only lag features in channel order Y,U,V",
            "raw_feature_alignment": "raw_features[:, h, channel] = context[:, channel, L + h - selected_lag[channel]]",
            "lag_selection_labels": "target_train future Y only, 64*16 labels",
            "episode_id_namespace": f"trainlag_b{BASE_SEED}",
            "context_y_used_as_extra_label": False,
            "oracle_or_metadata_used_for_lag_selection": False,
        },
        "theory": {
            "stationary_var_y": 1.0,
            "oracle_future_var": SIGMA**2,
            "y_only_oracle_future_var": B**2 + SIGMA**2,
            "true_lags_for_diagnostic_only": {"Y": SELF_LAG, "U": DRIVER_LAG, "V": DRIVER_LAG},
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
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    qc = _qc()
    files: dict[str, Any] = {}
    lag_selection_files: dict[str, Any] = {}
    selections: dict[str, Any] = {}
    for corpus in TRAIN_CORPORA:
        arrays = build_archive(corpus)
        lag_selection = select_lags_from_train(arrays["context_train"], arrays["target_train"])
        manifest = _archive_manifest(corpus, lag_selection, qc)
        path = output / f"Q00_c{corpus}.npz"
        arrays["raw_features_train"] = make_lag_features(arrays["context_train"], lag_selection["selected_lags"])
        arrays["raw_features_val"] = make_lag_features(arrays["context_val"], lag_selection["selected_lags"])
        arrays["raw_features_eval"] = make_lag_features(arrays["context_eval"], lag_selection["selected_lags"])
        arrays["selected_lags"] = _lag_array(lag_selection["selected_lags"]).astype(np.int64)
        _save_npz(path, arrays, manifest)
        data_sha256 = _sha256(path)
        lag_selection = {**lag_selection, "data_sha256": data_sha256}
        lag_path = output / f"Q00_c{corpus}_lags.json"
        lag_path.write_text(json.dumps(lag_selection, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        files[path.name] = {"sha256": data_sha256, "bytes": path.stat().st_size}
        lag_selection_files[lag_path.name] = {"sha256": _sha256(lag_path), "bytes": lag_path.stat().st_size}
        selections[f"Q00_c{corpus}"] = {
            "selected_lags": lag_selection["selected_lags"],
            "input_array_hash": lag_selection["input_array_hash"],
        }
    summary = {
        "version": VERSION,
        "source_family": SOURCE_FAMILY,
        "fresh_same_family_not_original_source": True,
        "episode_id_namespace": f"trainlag_b{BASE_SEED}",
        "files": files,
        "lag_selection_files": lag_selection_files,
        "condition": CONDITION_NAME,
        "train_corpora": list(TRAIN_CORPORA),
        "n_by_split": N_BY_SPLIT,
        "quantiles": QUANTILES.tolist(),
        "candidate_lags": [int(value) for value in CANDIDATE_LAGS],
        "lag_selection": selections,
        "qc": qc,
        "all_qc_passed": bool(qc["passed"]),
    }
    (output / "manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = generate(args.output)
    print(
        json.dumps(
            {
                "files": len(summary["files"]),
                "lag_selection_files": len(summary["lag_selection_files"]),
                "all_qc_passed": summary["all_qc_passed"],
                "output": str(args.output),
            },
            ensure_ascii=False,
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
