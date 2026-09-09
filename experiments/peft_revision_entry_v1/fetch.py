"""Fetch the two preregistered ALFRED archives through official download forms."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
from html.parser import HTMLParser
import http.cookiejar
import json
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "_docs/notes/tsfm_topics/19_peft_revision_entry_plan_20260908.md"
SERIES = ("PAYEMS", "INDPRO")
MAX_ZIP = 16 * 1024**2
MAX_UNPACKED = 128 * 1024**2


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


class FormParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.select = None
        self.dates = []
        self.fields = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in {"select", "input", "button"} and attrs.get("name"):
            self.fields.add(attrs["name"])
        if tag == "select":
            self.select = attrs.get("id")
        if tag == "option" and self.select == "form_selected_vintage_dates":
            self.dates.append(date.fromisoformat(attrs["value"]))

    def handle_endtag(self, tag):
        if tag == "select":
            self.select = None


def request_to_file(opener, request, path: Path, cap: int) -> dict:
    started = time.monotonic()
    receipt = {"requested_at": utc(), "request_url": request.full_url, "method": request.get_method()}
    try:
        response = opener.open(request, timeout=30)
    except urllib.error.HTTPError as error:
        response = error
    with response, path.open("xb") as output:
        receipt.update(status=response.status, final_url=response.geturl(), headers={
            key: response.headers.get(key) for key in ("Content-Type", "Content-Length", "Content-Disposition", "Last-Modified")
        })
        count = 0
        while True:
            block = response.read(min(65536, cap - count + 1))
            if not block:
                break
            output.write(block)
            count += len(block)
            if count > cap:
                raise RuntimeError(f"Response exceeds byte cap: {path}")
            if time.monotonic() - started > 30:
                raise TimeoutError(f"Response exceeds 30-second elapsed cap: {path}")
    receipt.update(received_at=utc(), bytes=path.stat().st_size, sha256=sha256(path), elapsed_seconds=time.monotonic()-started)
    if receipt["status"] != 200:
        raise RuntimeError(f"HTTP {receipt['status']}; response retained at {path}")
    return receipt


def fetch(raw_dir: Path, diagnostic_vintages: bool = False) -> dict:
    raw_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = raw_dir / "fetch_receipt.json"
    if receipt_path.exists():
        raise FileExistsError(f"Refusing to overwrite previous fetch: {receipt_path}")
    receipt = {"status": "RUNNING", "started_at": utc(), "plan": str(PLAN), "plan_sha256": sha256(PLAN),
               "source_sha256": {str(Path(__file__).relative_to(ROOT)): sha256(Path(__file__)),
                                 "experiments/peft_adaptation_scope_v1/guard.py": sha256(ROOT/"experiments/peft_adaptation_scope_v1/guard.py")},
               "series": {}, "diagnostic_only": diagnostic_vintages,
               "limits": {"zip_bytes_each": MAX_ZIP, "unpacked_bytes_total": MAX_UNPACKED, "request_timeout_seconds": 30}}
    if diagnostic_vintages:
        receipt["scope_change"] = "Transport diagnosis only: PAYEMS, first/last selected official vintage plus unchanged boundaries. Endpoint, observation dates, units, format and limits unchanged. Excluded from analysis."
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    (raw_dir / "fetch_source.py").write_bytes(Path(__file__).read_bytes())
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    headers = {"User-Agent": "mltimeseries-research/1.0", "Accept-Encoding": "identity"}
    unpacked = 0
    try:
        for series in SERIES[:1] if diagnostic_vintages else SERIES:
            entry = receipt["series"][series] = {}
            url = f"https://alfred.stlouisfed.org/series/downloaddata?seid={series}"
            form_path = raw_dir / f"{series}.form.html"
            entry["form"] = request_to_file(opener, urllib.request.Request(url, headers=headers), form_path, 2*1024**2)
            parser = FormParser()
            parser.feed(form_path.read_text(encoding="utf-8"))
            if not parser.dates:
                raise ValueError("Official vintage selector missing")
            required = {"form[units]", "form[obs_start_date]", "form[obs_end_date]", "form[selected_vintage_dates][]", "form[entered_vintage_dates]", "form[file_type]", "form[file_format]", "form[download_data]"}
            if not required.issubset(parser.fields):
                raise ValueError(f"Official form fields changed: {required-parser.fields}")
            selected = sorted(set(d for d in parser.dates if date(1990,1,1) <= d <= date(2025,6,30)))
            if not selected:
                raise ValueError("No preregistered vintage dates available")
            if diagnostic_vintages:
                selected = sorted({selected[0], selected[-1]})
            fields = [("form[units]", "lin"), ("form[obs_start_date]", "1989-12-01"), ("form[obs_end_date]", "2024-12-01"),
                      ("form[entered_vintage_dates]", "1990-01-01 2025-06-30"), ("form[file_type]", "1"),
                      ("form[file_format]", "csv"), ("form[download_data]", "")]
            fields += [("form[selected_vintage_dates][]", d.isoformat()) for d in selected]
            entry.update(request_fields=fields, selector_count=len(parser.dates), selector_first=min(parser.dates).isoformat(),
                         selector_last=max(parser.dates).isoformat(), selected_vintage_count=len(selected))
            body = urllib.parse.urlencode(fields).encode("ascii")
            response_path = raw_dir / f"{series}.response"
            entry["archive"] = request_to_file(opener, urllib.request.Request(url, data=body, headers={**headers,"Content-Type":"application/x-www-form-urlencoded","Referer":url}), response_path, MAX_ZIP)
            if not zipfile.is_zipfile(response_path):
                raise ValueError(f"Official response is not ZIP; retained at {response_path}")
            archive_path = raw_dir / f"{series}.zip"
            response_path.rename(archive_path)
            entry["archive"]["path"] = str(archive_path.relative_to(ROOT))
            with zipfile.ZipFile(archive_path) as archive:
                members = archive.infolist()
                unpacked += sum(member.file_size for member in members)
                if unpacked > MAX_UNPACKED:
                    raise ValueError("Combined uncompressed archive size exceeds contract")
                entry["members"] = []
                extraction = raw_dir / series
                extraction.mkdir(exist_ok=False)
                for member in members:
                    if member.is_dir():
                        continue
                    relative = Path(member.filename)
                    if relative.is_absolute() or ".." in relative.parts or ":" in member.filename or "\\" in member.filename:
                        raise ValueError(f"Unsafe archive member path: {member.filename}")
                    target = extraction / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source, target.open("xb") as output:
                        size = 0
                        while chunk := source.read(65536):
                            size += len(chunk)
                            if size > member.file_size:
                                raise ValueError("Archive member exceeds declared size")
                            output.write(chunk)
                    entry["members"].append({"name":member.filename,"bytes":size,"sha256":sha256(target)})
            print(f"{series}: ZIP {entry['archive']['bytes']} bytes; {entry['members']}", flush=True)
        receipt.update(status="PASS", unpacked_bytes=unpacked)
    except Exception as error:
        receipt.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt["finished_at"] = utc()
        receipt["retained_files"] = [{"name":p.name,"bytes":p.stat().st_size,"sha256":sha256(p)} for p in raw_dir.iterdir() if p.is_file() and p!=receipt_path]
        receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    return receipt


def import_browser_archives(raw_dir: Path, manifest: Path) -> dict:
    """Verify files downloaded through the official UI; this does not POST."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = raw_dir / "browser_fetch_receipt.json"
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    ui = json.loads(manifest.read_text(encoding="utf-8"))
    receipt = {"status": "RUNNING", "transport": "official Chrome form UI; Python imports and verifies local files only",
               "started_at": utc(), "browser_manifest": ui, "source_sha256": sha256(Path(__file__)),
               "plan_sha256": {p.name: sha256(p) for p in PLAN.parent.glob("19*revision*20260908.md")}, "series": {}}
    (raw_dir / "fetch_import_source.py").write_bytes(Path(__file__).read_bytes())
    opener = urllib.request.build_opener()
    total = 0
    try:
        for series in SERIES:
            info = ui[series]
            entry = receipt["series"][series] = {"browser_event": info}
            archive_path = raw_dir / f"{series}.zip"
            source = Path(info["source_download"])
            if source.stat().st_size > MAX_ZIP:
                raise ValueError("Browser archive exceeds contracted byte cap")
            if archive_path.exists():
                raise FileExistsError(archive_path)
            shutil.copyfile(source, archive_path)
            entry["archive"] = {"name": archive_path.name, "bytes": archive_path.stat().st_size, "sha256": sha256(archive_path)}
            with zipfile.ZipFile(archive_path) as archive:
                total += sum(item.file_size for item in archive.infolist())
                if total > MAX_UNPACKED:
                    raise ValueError("Combined browser archives exceed uncompressed byte cap")
                entry["members"] = []
                for item in archive.infolist():
                    if item.is_dir():
                        continue
                    relative = Path(item.filename)
                    if relative.is_absolute() or ".." in relative.parts or ":" in item.filename or "\\" in item.filename:
                        raise ValueError(f"Unsafe ZIP member: {item.filename}")
                    target = raw_dir / series / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(item) as src, target.open("xb") as dst:
                        size = 0
                        while block := src.read(65536):
                            size += len(block)
                            if size > item.file_size:
                                raise ValueError("Member exceeds declared size")
                            dst.write(block)
                    entry["members"].append({"name": item.filename, "bytes": size, "sha256": sha256(target)})
            url = f"https://alfred.stlouisfed.org/series/downloaddata?seid={series}"
            form = raw_dir / f"{series}.form.html"
            entry["metadata_get"] = request_to_file(opener, urllib.request.Request(url), form, 2*1024**2)
            parser = FormParser()
            parser.feed(form.read_text(encoding="utf-8"))
            dates = sorted(set(d.isoformat() for d in parser.dates if date(1990,1,1) <= d <= date(2025,6,30)))
            if (len(dates), dates[0], dates[-1]) != (info["selected_count"], info["selected_first"], info["selected_last"]):
                raise ValueError("Captured official selector differs from browser verification")
            entry["request_fields"] = {"form[units]": "lin", "form[obs_start_date]": "1989-12-01", "form[obs_end_date]": "2024-12-01",
                                       "form[selected_vintage_dates][]": dates, "form[entered_vintage_dates]": "1990-01-01 2025-06-30",
                                       "form[file_type]": "1", "form[file_format]": "csv", "form[download_data]": ""}
            print(f"{series}: verified browser ZIP {entry['archive']['bytes']} bytes, {entry['members']}", flush=True)
        receipt.update(status="PASS", unpacked_bytes=total)
    except Exception as error:
        receipt.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt["finished_at"] = utc()
        receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT/"data_external/alfred_revision_entry_v1")
    parser.add_argument("--diagnostic-vintages", action="store_true")
    parser.add_argument("--import-browser-manifest", type=Path)
    arguments = parser.parse_args()
    if arguments.import_browser_manifest:
        import_browser_archives(arguments.raw_dir.resolve(), arguments.import_browser_manifest)
    else:
        fetch(arguments.raw_dir.resolve(), arguments.diagnostic_vintages)
