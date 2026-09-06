"""Zenodo source contract for UCP-PATH-PILOT-v1.

Records, checksums and download/extract driver. Sizes and md5 values are the
ones Zenodo reported on 2026-09-06; every download is verified against them.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tarfile
import time
from pathlib import Path

import certifi
import requests

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data_external" / "ucp_path_pilot_v1"
RAW = DATA / "raw"
EXTRACTED = DATA / "extracted"

SOURCES = {
    "ens_2019_2020": {
        "record": "13255991",
        "doi": "10.5281/zenodo.13255991",
        "title": "ECMWF/ENS - 2019 to 2020 - Great Britain",
        "filename": "ecmwf_ens_2019-2020.tar.gz",
        "size": 36034983151,
        "md5": "5bc1122fb5a64bdd9ac980b9f1db2e64",
        "license": "cc-by-4.0",
    },
    "ens_2021": {
        "record": "13256007",
        "doi": "10.5281/zenodo.13256007",
        "title": "ECMWF/ENS - 2021 - Great Britain",
        "filename": "ecmwf_ens_2021.tar.gz",
        "size": 18003586760,
        "md5": "24caeb6b55d5f2038b913855f5085b0b",
        "license": "cc-by-4.0",
    },
    "bmra": {
        "record": "13256014",
        "doi": "10.5281/zenodo.13256014",
        "title": "Raw and Preprocessed BMRA Wind Power Data",
        "filename": "BMRA_Data.tar.gz",
        "size": 330104976,
        "md5": "16967e6f7b695109a14088c044b02118",
        "license": "not stated in Zenodo metadata; access_right=open",
    },
    "forecasting_software": {
        "record": "13309948",
        "doi": "10.5281/zenodo.13309948",
        "title": 'Forecasting Software for "Seamless short- to mid-term probabilistic wind power forecasting"',
        "filename": "forecasting_software.tar.gz",
        "size": 117912,
        "md5": "6c69551cfd53e5d37a495acdf0ccf535",
        "license": "cc-by-4.0",
    },
    "read_bmra": {
        "record": "13309890",
        "doi": "10.5281/zenodo.13309890",
        "title": "Code for preprocessing of BMRA Wind Power Data",
        "filename": "read_bmra.tar.gz",
        "size": 239259,
        "md5": "37139e81262879d35b4fe16962fc123f",
        "license": "cc-by-4.0",
    },
}


def url_for(key: str) -> str:
    s = SOURCES[key]
    return f"https://zenodo.org/records/{s['record']}/files/{s['filename']}?download=1"


def md5_of(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download(key: str, retries: int = 5) -> Path:
    """Download with HTTP range resume; verified against the recorded md5."""
    s = SOURCES[key]
    RAW.mkdir(parents=True, exist_ok=True)
    dest = RAW / s["filename"]

    if dest.exists() and dest.stat().st_size == s["size"]:
        print(f"[{key}] already complete, verifying md5", flush=True)
        got = md5_of(dest)
        if got == s["md5"]:
            print(f"[{key}] md5 OK {got}", flush=True)
            return dest
        print(f"[{key}] md5 MISMATCH {got}, restarting", flush=True)
        dest.unlink()

    for attempt in range(1, retries + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if have >= s["size"]:
            break
        headers = {"Range": f"bytes={have}-"} if have else {}
        print(f"[{key}] attempt {attempt}: resuming at {have/1e9:.2f} / {s['size']/1e9:.2f} GB", flush=True)
        try:
            with requests.get(url_for(key), headers=headers, stream=True,
                              verify=certifi.where(), timeout=(30, 300)) as r:
                if have and r.status_code == 200:
                    # server ignored Range: start over
                    have = 0
                    dest.unlink(missing_ok=True)
                r.raise_for_status()
                mode = "ab" if have else "wb"
                last = time.time()
                with open(dest, mode) as fh:
                    for chunk in r.iter_content(chunk_size=8 << 20):
                        fh.write(chunk)
                        have += len(chunk)
                        if time.time() - last > 30:
                            print(f"[{key}] {have/1e9:.2f} / {s['size']/1e9:.2f} GB"
                                  f" ({100*have/s['size']:.1f}%)", flush=True)
                            last = time.time()
        except Exception as exc:  # network drop -> resume on next attempt
            print(f"[{key}] transfer error: {exc!r}", flush=True)
            time.sleep(5 * attempt)

    size = dest.stat().st_size if dest.exists() else 0
    if size != s["size"]:
        raise RuntimeError(f"[{key}] DOWNLOAD_INCOMPLETE {size} != {s['size']}")
    got = md5_of(dest)
    if got != s["md5"]:
        raise RuntimeError(f"[{key}] CHECKSUM_MISMATCH {got} != {s['md5']}")
    print(f"[{key}] download complete, md5 OK {got}", flush=True)
    return dest


def extract(key: str) -> Path:
    s = SOURCES[key]
    out = EXTRACTED / key
    marker = out / ".extract_complete"
    if marker.exists():
        print(f"[{key}] already extracted", flush=True)
        return out
    out.mkdir(parents=True, exist_ok=True)
    src = RAW / s["filename"]
    print(f"[{key}] extracting {src.name}", flush=True)
    n = 0
    with tarfile.open(src, "r:gz") as tf:
        for member in tf:
            tf.extract(member, out, filter="data")
            n += 1
            if n % 500 == 0:
                print(f"[{key}] {n} members", flush=True)
    marker.write_text(f"{n} members\n")
    print(f"[{key}] extracted {n} members to {out}", flush=True)
    return out


# The ENS archives are consumed as gzip/tar streams by extract_ens.py, so unpacking
# them would write about 38 GB of .nc files that nothing reads.
STREAM_ONLY = {"ens_2019_2020", "ens_2021"}


def main(keys: list[str]) -> int:
    manifest = {}
    for key in keys:
        path = download(key)
        out = None if key in STREAM_ONLY else extract(key)
        manifest[key] = {
            **{k: v for k, v in SOURCES[key].items()},
            "raw_path": str(path),
            "extracted_path": None if out is None else str(out),
            "consumed_as_stream": key in STREAM_ONLY,
            "verified_md5": True,
        }
    RES = ROOT / "results" / "uncertain_covariate_path_pilot_v1"
    RES.mkdir(parents=True, exist_ok=True)
    fp = RES / "data_sources.json"
    existing = json.loads(fp.read_text()) if fp.exists() else {}
    existing.update(manifest)
    fp.write_text(json.dumps(existing, indent=2))
    print(f"wrote {fp}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or list(SOURCES)))
