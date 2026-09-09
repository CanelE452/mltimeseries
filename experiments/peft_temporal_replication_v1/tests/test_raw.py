import json
from pathlib import Path

import numpy as np

from experiments.peft_temporal_replication_v1 import raw


def test_raw_wrapper_exports_generic_api_and_records_wrapper_source(tmp_path: Path):
    fit_path = tmp_path / "demo_fit.npz"
    holdout_path = tmp_path / "demo_holdout.npz"
    output = tmp_path / "raw"
    raw._write_fixture_archives(fit_path, holdout_path)

    result = raw.run(fit_path, holdout_path, output)

    saved = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["completed"] is True
    assert saved["source_sha256"] == raw.sha256_file(Path(raw.__file__))
    assert saved["wrapper_source_sha256"] == raw.sha256_file(Path(raw.__file__))
    assert saved["delegated_raw_source_sha256"]
    assert saved["delegated_raw_source_path"].endswith("experiments\\peft_external_gap_v1\\raw.py") or saved["delegated_raw_source_path"].endswith("experiments/peft_external_gap_v1/raw.py")
    with np.load(output / "predictions.npz", allow_pickle=False) as predictions:
        assert predictions["quantiles"].dtype == np.float64
        assert predictions["val_predictions"].shape[1] == 2
