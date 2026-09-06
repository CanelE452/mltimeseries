"""Turn the ENS archives into per-origin weather tensors of shape [K, T, D].

The archive is read as a gzip/tar stream and each netCDF member is opened from
memory, so no 54 GB of .nc files are ever written to disk. A truncated stream is
normal here: the same archive is still downloading, so the extractor takes what
has arrived, records how far it got, and can be re-run to pick up the rest.

Member identity is never touched. For a given origin the member axis is carried
through the lead axis unchanged, which is the whole point of the pilot.
"""

from __future__ import annotations

import gzip
import io
import json
import re
import sys
import tarfile
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data_external/ucp_path_pilot_v1"
RESULTS = ROOT / "results/uncertain_covariate_path_pilot_v1"
PROCESSED = DATA / "processed"

LEADS_H = tuple(range(6, 73, 6))  # T = 12
RAW_VARS = ("u10", "v10", "t2m")  # the author's TIGGE set
FEATURES = ("u10", "v10", "t2m", "ws10")  # ws10 derived on the grid, then interpolated
FNAME = re.compile(r"tigge-GB_(\d{10})\.nc$")
ARCHIVES = {
    "ens_2019_2020": "ecmwf_ens_2019-2020.tar.gz",
    "ens_2021": "ecmwf_ens_2021.tar.gz",
}


def bilinear_weights(coord: np.ndarray, value: float):
    """Index pair and weight for one axis, handling ascending or descending order."""
    ascending = coord[1] > coord[0]
    axis = coord if ascending else coord[::-1]
    if not (axis[0] <= value <= axis[-1]):
        raise ValueError(f"{value} outside grid {axis[0]}..{axis[-1]}")
    hi = int(np.searchsorted(axis, value, side="left"))
    lo = max(hi - 1, 0)
    hi = min(max(hi, 1), len(axis) - 1)
    span = axis[hi] - axis[lo]
    w = 0.0 if span == 0 else float((value - axis[lo]) / span)
    if ascending:
        return lo, hi, w
    n = len(coord) - 1
    return n - lo, n - hi, w


def read_origin(data: bytes, farms: pd.DataFrame, name: str):
    """One netCDF member -> (origin, array[n_farms, K, T, D])."""
    ds = nc.Dataset(name, mode="r", memory=data)
    try:
        times = pd.to_datetime(
            np.array(ds["time"][:]).astype("int64"), unit="h", origin="1900-01-01"
        )
        origin = times[0]
        wanted = [origin + pd.Timedelta(hours=h) for h in LEADS_H]
        pos = {t: i for i, t in enumerate(times)}
        missing = [t for t in wanted if t not in pos]
        if missing:
            raise ValueError(f"{name}: lead timestamps missing {missing[:2]}")
        idx = [pos[t] for t in wanted]  # timestamp join, never a positional guess

        lat = np.array(ds["latitude"][:], dtype=np.float64)
        lon = np.array(ds["longitude"][:], dtype=np.float64)
        n_members = ds.dimensions["number"].size

        # (n_raw_vars, T, K, lat, lon) restricted to the leads we use
        cube = np.stack([np.array(ds[v][idx, :, :, :], dtype=np.float32) for v in RAW_VARS])
        ws = np.sqrt(cube[0] ** 2 + cube[1] ** 2)  # magnitude on the grid, then interpolate
        cube = np.concatenate([cube, ws[None]], axis=0)

        out = np.full((len(farms), n_members, len(LEADS_H), len(FEATURES)), np.nan, np.float32)
        for f, row in enumerate(farms.itertuples()):
            i0, i1, wy = bilinear_weights(lat, row.lat)
            j0, j1, wx = bilinear_weights(lon, row.lon)
            corner = cube[:, :, :, [i0, i0, i1, i1], [j0, j1, j0, j1]]  # (D, T, K, 4)
            weights = np.array(
                [(1 - wy) * (1 - wx), (1 - wy) * wx, wy * (1 - wx), wy * wx], np.float32
            )
            out[f] = np.einsum("dtkc,c->ktd", corner, weights)
        return origin, out
    finally:
        ds.close()


def stream_archive(key: str, farms: pd.DataFrame, save_every: int = 200):
    """Read as much of the archive as has arrived; save what was decoded."""
    path = DATA / "raw" / ARCHIVES[key]
    store = PROCESSED / f"weather_{key}.npz"
    PROCESSED.mkdir(parents=True, exist_ok=True)

    done: dict[pd.Timestamp, np.ndarray] = {}
    if store.exists():
        cached = np.load(store, allow_pickle=False)
        for t, block in zip(pd.to_datetime(cached["origins"]), cached["weather"]):
            done[pd.Timestamp(t)] = block
        print(f"[{key}] {len(done)} origins already stored", flush=True)

    def save():
        order = sorted(done)
        np.savez_compressed(
            store,
            origins=np.array([str(t) for t in order]),
            weather=np.stack([done[t] for t in order]).astype(np.float32),
            farms=np.array(farms["farm_id"].tolist()),
            leads_h=np.array(LEADS_H),
            features=np.array(FEATURES),
        )

    seen = truncated = 0
    try:
        with gzip.open(path, "rb") as gz:
            with tarfile.open(fileobj=gz, mode="r|") as tf:
                for member in tf:
                    m = FNAME.search(member.name)
                    if not (member.isfile() and m):
                        continue
                    seen += 1
                    stamp = pd.Timestamp(
                        f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:8]} {m.group(1)[8:10]}:00"
                    )
                    if stamp in done:
                        continue
                    payload = tf.extractfile(member).read()
                    origin, block = read_origin(payload, farms, member.name)
                    if origin != stamp:
                        raise ValueError(f"{member.name}: filename {stamp} != first valid time {origin}")
                    done[origin] = block
                    if len(done) % save_every == 0:
                        save()
                        print(f"[{key}] {len(done)} origins", flush=True)
    except (EOFError, tarfile.ReadError, gzip.BadGzipFile) as exc:
        truncated = 1
        print(f"[{key}] stream ends early ({type(exc).__name__}); archive still downloading", flush=True)

    save()
    print(f"[{key}] {len(done)} origins stored, {seen} members seen, truncated={bool(truncated)}", flush=True)
    return done, bool(truncated)


def main(keys: list[str]) -> int:
    farms = pd.read_csv(RESULTS / "farm_selection.csv")[["farm_id", "lat", "lon", "type"]]
    manifest = {}
    for key in keys:
        done, truncated = stream_archive(key, farms)
        order = sorted(done)
        manifest[key] = {
            "n_origins": len(done),
            "first_origin": str(order[0]) if order else None,
            "last_origin": str(order[-1]) if order else None,
            "archive_truncated": truncated,
            "shape_per_origin": ["n_farms", "K", "T", "D"],
            "n_farms": len(farms),
            "members": int(next(iter(done.values())).shape[1]) if done else None,
            "leads_h": list(LEADS_H),
            "features": list(FEATURES),
        }
        print(json.dumps({key: manifest[key]}, indent=2), flush=True)
    fp = RESULTS / "preprocessing_manifest.json"
    existing = json.loads(fp.read_text()) if fp.exists() else {}
    existing.update(manifest)
    fp.write_text(json.dumps(existing, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or list(ARCHIVES)))
