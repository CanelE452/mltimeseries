"""Fetch pinned BDG2 files and run the predeclared train-only entry check."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

for _thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_var] = "2"

ROOT = Path(__file__).resolve().parents[2]
PLAN = "_docs/notes/tsfm_topics/17_coarse_supervision_data_entry_plan_20260908.md"
STUDY = "peft_coarse_supervision_v1"
DATA = ROOT / "data_external/bdg2_coarse_supervision_v1"
REVISION = "9b97ccbe90096aff42ed4fd6493bf7ae692d7118"
BASE_URL = f"https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/{REVISION}/data/"
SOURCES = {
    "metadata": {"url": BASE_URL + "metadata/metadata.csv", "filename": "metadata.csv", "bytes": 272024,
                 "sha256": "992d0b29f24f96ad4332bc4dbb534b7bdd7dd2689aad093f94e93068ecddca02"},
    "electricity": {"url": BASE_URL + "meters/raw/electricity.csv", "filename": "electricity.csv", "bytes": 174239039,
                    "sha256": "039d909d8981e2d69eaeb366144e6ab7e84fa5e7e216aee42bddd95384a66418"},
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_once(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != record:
            raise FileExistsError(f"Preserve existing different record: {path}")
        return
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def contract():
    from experiments.peft_observation_operator_entry_v1 import prepare as previous
    parent = previous.validate(ROOT)
    final_path = ROOT / "results/peft_observation_operator_entry_v1/final_audit.json"
    final = read(final_path)
    if not final["passed"]:
        raise AssertionError("Complete study16 first")
    protected = dict(parent["protected_hashes"])
    protected.update(final["artifact_hashes"])
    protected[final_path.relative_to(ROOT).as_posix()] = sha(final_path)
    for item in parent["source_hashes"].values():
        protected[item["path"]] = item["sha256"]
    protected[parent["plan_path"]] = parent["plan_sha256"]
    for name, expected in protected.items():
        if sha(ROOT / name) != expected:
            raise AssertionError(f"Protected evidence changed: {name}")
    files = [ROOT / "experiments" / STUDY / name for name in ("__init__.py", "fetch.py", "data_qc.py")]
    return {"stage": "prefetch_and_train_only_qc", "revision": REVISION,
            "plan_path": PLAN, "plan_sha256": sha(ROOT / PLAN), "sources": SOURCES,
            "source_hashes": {p.relative_to(ROOT).as_posix(): sha(p) for p in files},
            "protected_hashes": protected,
            "limits": {"request_seconds": 30, "whole_seconds": 300,
                       "file_bytes": 200 * 1024 ** 2, "total_bytes": 201 * 1024 ** 2},
            "selection": {"sites": ["Eagle", "Lamb"], "primaryspaceusage": "Office",
                          "selection_year": 2016, "expected_candidate_counts": {"Eagle": 40, "Lamb": 17},
                          "complete_nonnegative_train_hours": 8784, "positive_annual_sum": True,
                          "per_site_each_role": "min(8, eligible_count // 2)", "minimum_each_role": 4,
                          "order": "sorted building_id; donor then target", "evaluation_values_used": False},
            "new_training_updates": 0, "known_pretraining_manifest_overlap": "UNKNOWN"}


def fetch_one(key, source, deadline):
    path = DATA / "raw" / source["filename"]
    receipt = DATA / "receipts" / f"{key}.json"
    part = path.with_name(path.name + ".part")
    if part.exists():
        raise FileExistsError(f"Preserve partial download before any recovery: {part}")
    if path.exists():
        if path.stat().st_size != source["bytes"] or sha(path) != source["sha256"]:
            raise AssertionError(f"Existing raw does not match the fixed LFS object: {path}")
        return {"verified_fixed_object_reused": True, "receipt_available": receipt.exists(), **source}
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = 0
    digest = hashlib.sha256()
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Whole data-entry deadline exceeded")
    print(f"Fetching {key}: {source['bytes']} bytes", flush=True)
    with urllib.request.urlopen(source["url"], timeout=min(30, remaining)) as response:
        length = response.headers.get("Content-Length")
        if length is not None and int(length) != source["bytes"]:
            raise AssertionError(f"Unexpected Content-Length for {key}: {length}")
        with part.open("xb") as stream:
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError("Whole data-entry deadline exceeded; partial preserved")
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                seen += len(chunk)
                if seen > source["bytes"] or seen > 200 * 1024 ** 2:
                    raise AssertionError("Exceeded exact pinned object size; partial preserved")
                stream.write(chunk)
                digest.update(chunk)
                if seen % (32 * 1024 ** 2) == 0:
                    print(f"{key}: {seen} bytes", flush=True)
        if seen != source["bytes"] or digest.hexdigest() != source["sha256"]:
            raise AssertionError(f"Pinned LFS size/SHA mismatch for {key}; partial preserved")
        record = {**source, "verified_fixed_object_reused": False, "received_bytes": seen,
                  "response_url": response.geturl(), "status": response.status,
                  "content_length_header": length, "last_modified": response.headers.get("Last-Modified"),
                  "received_at_utc": datetime.now(timezone.utc).isoformat()}
    part.replace(path)
    write_once(receipt, record)
    return record


def run(prepare_only=False):
    frozen = contract()
    write_once(DATA / "contract.json", frozen)
    if prepare_only:
        return {"prepared": True, "protected": len(frozen["protected_hashes"])}
    output = DATA / "qc.json"
    if output.exists():
        raise FileExistsError("Completed or partial QC already exists; preserve it")
    if sum(s["bytes"] for s in SOURCES.values()) > frozen["limits"]["total_bytes"]:
        raise AssertionError("Total size cap violated")
    started = time.monotonic()
    deadline = started + frozen["limits"]["whole_seconds"]
    receipts = {key: fetch_one(key, source, deadline) for key, source in SOURCES.items()}
    from .data_qc import compute_qc
    qc = compute_qc(DATA / "raw/metadata.csv", DATA / "raw/electricity.csv")
    if time.monotonic() > deadline:
        raise TimeoutError("Data-entry deadline exceeded during QC")
    if contract() != frozen:
        raise AssertionError("Data-entry contract changed during execution")
    record = {"completed": True, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
              "contract_sha256": sha(DATA / "contract.json"), "source_receipts": receipts,
              "qc": qc, "wall_seconds": time.monotonic() - started, "new_training_updates": 0}
    write_once(output, record)
    return {"completed": True, "output": str(output), "wall_seconds": record["wall_seconds"],
            "decision": qc.get("decision", "INSPECT_QC")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.prepare_only)), flush=True)
    except Exception as exc:
        failure = DATA / "failure.json"
        if not failure.exists():
            write_once(failure, {"completed": False, "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                                 "exception": type(exc).__name__, "message": str(exc),
                                 "partial_files_preserved": True})
        raise
