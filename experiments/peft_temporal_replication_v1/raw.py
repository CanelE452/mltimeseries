"""Study-13 RAW wrapper reusing the study-12 generic ridge implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from experiments.peft_external_gap_v1 import raw as delegated_raw


DEFAULT_LAMBDAS = delegated_raw.DEFAULT_LAMBDAS
sha256_file = delegated_raw.sha256_file
fit_direct_ridge = delegated_raw.fit_direct_ridge
predict_direct_ridge = delegated_raw.predict_direct_ridge
fit_candidate = delegated_raw.fit_candidate
_pinball_score = delegated_raw._pinball_score
_residual_offsets = delegated_raw._residual_offsets
_predict_quantiles = delegated_raw._predict_quantiles
_load_npz = delegated_raw._load_npz
_read_manifest = delegated_raw._read_manifest
_window_targets = delegated_raw._window_targets
_features = delegated_raw._features
_scaled_target = delegated_raw._scaled_target
_split_arrays = delegated_raw._split_arrays
_raw_target = delegated_raw._raw_target
_unscale_predictions = delegated_raw._unscale_predictions
_assert_archive_contract = delegated_raw._assert_archive_contract
_write_fixture_archives = delegated_raw._write_fixture_archives


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def run(
    fit_data: Path | str,
    holdout_data: Path | str,
    output: Path | str,
    lambdas: Sequence[float] = DEFAULT_LAMBDAS,
) -> dict:
    result = delegated_raw.run(fit_data, holdout_data, output, lambdas=lambdas)
    wrapper_path = Path(__file__)
    delegated_path = Path(delegated_raw.__file__)
    result.update({
        "study": "peft_temporal_replication_v1",
        "source_path": str(wrapper_path),
        "source_sha256": sha256_file(wrapper_path),
        "wrapper_source_path": str(wrapper_path),
        "wrapper_source_sha256": sha256_file(wrapper_path),
        "delegated_raw_source_path": str(delegated_path),
        "delegated_raw_source_sha256": sha256_file(delegated_path),
        "delegated_raw_contract": "numerical fitting/prediction helpers are re-exported unchanged from peft_external_gap_v1.raw",
    })
    _write_json(Path(output) / "result.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-data", "--fit", dest="fit_data", type=Path, required=True)
    parser.add_argument("--holdout-data", "--holdout", dest="holdout_data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lambda", dest="lambdas", type=float, action="append")
    args = parser.parse_args(argv)
    result = run(args.fit_data, args.holdout_data, args.output, lambdas=tuple(args.lambdas or DEFAULT_LAMBDAS))
    print(json.dumps({
        "completed": result["completed"],
        "dataset": result["dataset"],
        "selected_lambda": result["selected_lambda"],
        "val_score": result["val_score"],
        "result_path": str(args.output / "result.json"),
        "source_sha256": result["source_sha256"],
        "delegated_raw_source_sha256": result["delegated_raw_source_sha256"],
    }, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
