"""Download the two OA-RESOLUTION-PILOT-v1 source datasets from their official hosts.

Dataset A -- Max-Planck-Institute for Biogeochemistry weather station, Jena
  https://www.bgc-jena.mpg.de/wetter/weather_data.html
  Station "WS Beutenberg" (mpi_roof).  Half-year ZIPs up to 2023, full-year from 2024.

Dataset B -- UCI Individual Household Electric Power Consumption
  https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

Nothing here interprets the numbers.  It downloads, records a sha256 of every
archive and every extracted file, and stops.  Measurement semantics come from the
official documentation, not from this script.

Run:  python scripts/fetch_oa_resolution_data.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import certifi
import requests

DATA = Path(__file__).resolve().parent.parent / "data"

JENA_BASE = "https://www.bgc-jena.mpg.de/wetter/"
JENA_FILES = ["mpi_roof_2023a.zip", "mpi_roof_2023b.zip", "mpi_roof_2024.zip"]
JENA_DIR = DATA / "jena_mpi_roof"

UCI_URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)
UCI_DIR = DATA / "uci_household_power"

UA = "Mozilla/5.0 (compatible; mltimeseries-research/1.0)"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached  : {dest.name} ({dest.stat().st_size/1e6:.1f} MB)", flush=True)
        return dest
    print(f"  GET     : {url}", flush=True)
    # requests + certifi: the stdlib opener has no usable CA store on this box.
    with requests.get(url, headers={"User-Agent": UA}, timeout=300, stream=True,
                      verify=certifi.where()) as resp:
        resp.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(1 << 20):
                fh.write(chunk)
    print(f"  saved   : {dest.name} ({dest.stat().st_size/1e6:.1f} MB)", flush=True)
    return dest


def extract(archive: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            target = out_dir / Path(name).name
            if not target.exists():
                with zf.open(name) as src, target.open("wb") as dst:
                    dst.write(src.read())
            written.append(target)
    return written


def main() -> int:
    manifest: dict[str, object] = {"archives": [], "files": []}

    print("--- Dataset A: Jena MPI-BGC weather station (mpi_roof) ---", flush=True)
    for name in JENA_FILES:
        arch = download(JENA_BASE + name, JENA_DIR / "_zip" / name)
        manifest["archives"].append(
            {"dataset": "jena", "url": JENA_BASE + name, "file": str(arch.relative_to(DATA)),
             "bytes": arch.stat().st_size, "sha256": sha256(arch)}
        )
        for f in extract(arch, JENA_DIR):
            manifest["files"].append(
                {"dataset": "jena", "file": str(f.relative_to(DATA)),
                 "bytes": f.stat().st_size, "sha256": sha256(f)}
            )
            print(f"  extract : {f.name} ({f.stat().st_size/1e6:.1f} MB)", flush=True)

    print("--- Dataset B: UCI Individual Household Electric Power Consumption ---", flush=True)
    arch = download(UCI_URL, UCI_DIR / "_zip" / "household_power_consumption.zip")
    manifest["archives"].append(
        {"dataset": "uci", "url": UCI_URL, "file": str(arch.relative_to(DATA)),
         "bytes": arch.stat().st_size, "sha256": sha256(arch)}
    )
    for f in extract(arch, UCI_DIR):
        manifest["files"].append(
            {"dataset": "uci", "file": str(f.relative_to(DATA)),
             "bytes": f.stat().st_size, "sha256": sha256(f)}
        )
        print(f"  extract : {f.name} ({f.stat().st_size/1e6:.1f} MB)", flush=True)

    out = DATA / "oa_resolution_pilot_v1_source_hashes.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nhashes  : {out}", flush=True)
    print(f"archives: {len(manifest['archives'])}   files: {len(manifest['files'])}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
