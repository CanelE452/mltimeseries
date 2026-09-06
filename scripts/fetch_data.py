"""Download the three long-term-forecasting benchmark datasets and verify them.

Source: the HF mirror the Time-Series-Library README points to
(https://huggingface.co/datasets/thuml/Time-Series-Library). Same CSVs the
Autoformer/PatchTST line of papers uses, so numbers stay comparable.

Run:  python scripts/fetch_data.py
"""

from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "thuml/Time-Series-Library"
DATA = Path(__file__).resolve().parent.parent / "data"

# filename in the HF repo -> what the published benchmark says it should contain.
# n_rows / n_feature_cols are the values the LTSF papers report; we assert on them
# so a silently different mirror cannot slip in.
EXPECTED = {
    "ETT-small/ETTm2.csv": {"rows": 69680, "features": 7, "freq": "15min"},
    "weather/weather.csv": {"rows": 52696, "features": 21, "freq": "10min"},
    "electricity/electricity.csv": {"rows": 26304, "features": 321, "freq": "1h"},
}


def fetch(rel_path: str) -> Path:
    local = hf_hub_download(
        repo_id=REPO, filename=rel_path, repo_type="dataset", local_dir=DATA
    )
    return Path(local)


def verify(path: Path, spec: dict) -> dict:
    df = pd.read_csv(path)
    date_col = df.columns[0]
    stamps = pd.to_datetime(df[date_col])
    n_feat = df.shape[1] - 1
    step = stamps.diff().dropna().mode().iloc[0]
    return {
        "path": path,
        "rows": len(df),
        "features": n_feat,
        "start": stamps.iloc[0],
        "end": stamps.iloc[-1],
        "step": step,
        "nan": int(df.isna().sum().sum()),
        "rows_ok": len(df) == spec["rows"],
        "features_ok": n_feat == spec["features"],
        "size_mb": path.stat().st_size / 1e6,
    }


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    bad = 0
    for rel, spec in EXPECTED.items():
        print(f"\n--- {rel} ---", flush=True)
        info = verify(fetch(rel), spec)
        print(f"  saved   : {info['path']}  ({info['size_mb']:.1f} MB)")
        print(f"  rows    : {info['rows']:,}  (expected {spec['rows']:,})  "
              f"{'OK' if info['rows_ok'] else 'MISMATCH'}")
        print(f"  features: {info['features']}  (expected {spec['features']})  "
              f"{'OK' if info['features_ok'] else 'MISMATCH'}")
        print(f"  range   : {info['start']}  ..  {info['end']}   step={info['step']}")
        print(f"  NaN     : {info['nan']}")
        if not (info["rows_ok"] and info["features_ok"]):
            bad += 1
    print(f"\nVERDICT: {len(EXPECTED) - bad}/{len(EXPECTED)} datasets match the published shape")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
