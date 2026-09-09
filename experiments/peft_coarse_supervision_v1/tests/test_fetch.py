import hashlib
import time

import pytest

from experiments.peft_coarse_supervision_v1 import fetch


class Response:
    status = 200

    def __init__(self, pieces, expected_size):
        self.pieces = iter(pieces)
        self.headers = {"Content-Length": str(expected_size)}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def geturl(self):
        return "https://example.test/pinned"

    def read(self, size):
        item = next(self.pieces, b"")
        if isinstance(item, Exception):
            raise item
        return item


def source(payload):
    return {"url": "https://example.test/pinned", "filename": "data.csv", "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest()}


def test_complete_download_verifies_pinned_hash_and_reuses_without_network(tmp_path, monkeypatch):
    payload = b"verified data\n"
    monkeypatch.setattr(fetch, "DATA", tmp_path)
    monkeypatch.setattr(fetch.urllib.request, "urlopen", lambda *a, **kw: Response([payload], len(payload)))
    receipt = fetch.fetch_one("meter", source(payload), time.monotonic() + 5)
    assert receipt["received_bytes"] == len(payload)
    assert (tmp_path / "raw/data.csv").read_bytes() == payload
    monkeypatch.setattr(fetch.urllib.request, "urlopen", lambda *a, **kw: pytest.fail("Do not refetch matching object"))
    assert fetch.fetch_one("meter", source(payload), time.monotonic() + 5)["verified_fixed_object_reused"]


@pytest.mark.parametrize("pieces", [[b"wrong data\n"], [b"short"], [b"part", OSError("lost connection")]])
def test_failure_preserves_partial_and_never_promotes_wrong_object(tmp_path, monkeypatch, pieces):
    expected = b"right data\n"
    monkeypatch.setattr(fetch, "DATA", tmp_path)
    monkeypatch.setattr(fetch.urllib.request, "urlopen", lambda *a, **kw: Response(pieces, len(expected)))
    with pytest.raises((AssertionError, OSError)):
        fetch.fetch_one("meter", source(expected), time.monotonic() + 5)
    part = tmp_path / "raw/data.csv.part"
    assert part.exists()
    assert not (tmp_path / "raw/data.csv").exists()
    assert not (tmp_path / "receipts/meter.json").exists()
    preserved = part.read_bytes()
    with pytest.raises(FileExistsError):
        fetch.fetch_one("meter", source(expected), time.monotonic() + 5)
    assert part.read_bytes() == preserved
