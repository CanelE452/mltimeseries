import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
import pytest

from experiments.peft_adaptation_scope_v1 import guard


def sample(**changes):
    resource = {
        "timestamp": "test", "available_ram_gib": 20, "available_commit_gib": 20,
        "child_tree_rss_gib": 0.01, "git_process_count": 0, "gpus": None,
    }
    resource.update(changes)
    return resource


@pytest.fixture
def safe_resources(monkeypatch):
    monkeypatch.setattr(guard, "_sample_resources", lambda child, gpu: sample(gpus=gpu))


def test_child_completes_and_logs_without_pipes(tmp_path, safe_resources):
    status = guard.run_guarded(
        [sys.executable, "-c", "import sys; print('output'); print('error', file=sys.stderr)"],
        tmp_path / "guard", tmp_path, require_gpu=False,
    )
    assert status["completed"] and status["returncode"] == 0
    assert (tmp_path / "guard" / "stdout.log").read_text().strip() == "output"
    assert (tmp_path / "guard" / "stderr.log").read_text().strip() == "error"
    assert not (tmp_path / "runs" / "peft_adaptation_scope_v1" / ".guard.lock").exists()


def test_nonzero_child_is_not_reported_complete(tmp_path, safe_resources):
    status = guard.run_guarded([sys.executable, "-c", "raise SystemExit(7)"], tmp_path / "guard", tmp_path, require_gpu=False)
    assert not status["completed"]
    assert status["returncode"] == 7
    assert status["state"] == "child_failed"


def test_gpu_unavailable_fails_before_child_launch(tmp_path, safe_resources, monkeypatch):
    monkeypatch.setattr(guard, "_gpu_resources", lambda: (_ for _ in ()).throw(FileNotFoundError("nvidia-smi")))
    marker = tmp_path / "should_not_exist"
    status = guard.run_guarded([sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"], tmp_path / "guard", tmp_path)
    assert not status["completed"] and "child_pid" not in status
    assert not marker.exists()
    assert (tmp_path / "guard" / "safety_stop.json").exists()


def test_git_limit_stops_before_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(guard, "_sample_resources", lambda child, gpu: sample(git_process_count=33))
    status = guard.run_guarded([sys.executable, "-c", "raise RuntimeError('must not launch')"], tmp_path / "guard", tmp_path, require_gpu=False)
    assert status["reasons"] == ["git_process_count_above_limit"]
    assert "child_pid" not in status


def test_active_memory_limit_stops_owned_child_only(tmp_path, monkeypatch):
    monkeypatch.setattr(guard, "_sample_resources", lambda child, gpu: sample(available_commit_gib=4 if child else 20))
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"])
    try:
        status = guard.run_guarded([sys.executable, "-c", "import time; time.sleep(20)"], tmp_path / "guard", tmp_path, require_gpu=False)
        assert "available_commit_below_limit" in status["reasons"]
        assert not status["completed"]
        assert not psutil.pid_exists(status["child_pid"])
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_timeout_preserves_child_artifacts(tmp_path, safe_resources):
    marker = tmp_path / "checkpoint.txt"
    script = f"from pathlib import Path; import time; Path({str(marker)!r}).write_text('saved'); time.sleep(20)"
    status = guard.run_guarded([sys.executable, "-c", script], tmp_path / "guard", tmp_path, timeout_seconds=0.7, require_gpu=False)
    assert status["reasons"] == ["timeout"]
    assert marker.read_text() == "saved"
    assert not psutil.pid_exists(status["child_pid"])


def test_live_lock_rejects_second_guard(tmp_path):
    lock = tmp_path / "lock"
    with guard.StudyLock(lock):
        with pytest.raises(RuntimeError, match="Another guard"):
            with guard.StudyLock(lock):
                pytest.fail("Concurrent guard acquired lock")
    assert not lock.exists()


def test_stale_pid_reuse_lock_is_recovered(tmp_path):
    lock = tmp_path / "lock"
    lock.write_text(json.dumps({"pid": os.getpid(), "create_time": psutil.Process().create_time() - 10}))
    with guard.StudyLock(lock):
        assert json.loads(lock.read_text())["create_time"] == psutil.Process().create_time()
    assert not lock.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object lifecycle")
def test_parent_exit_cleans_remaining_descendant(tmp_path, safe_resources):
    marker = tmp_path / "descendant.pid"
    child_script = (
        "import subprocess,sys; from pathlib import Path; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        f"Path({str(marker)!r}).write_text(str(p.pid))"
    )
    status = guard.run_guarded([sys.executable, "-c", child_script], tmp_path / "guard", tmp_path, require_gpu=False)
    descendant = int(marker.read_text())
    try:
        for _ in range(30):
            if not psutil.pid_exists(descendant):
                break
            time.sleep(0.05)
        assert status["completed"]
        assert not psutil.pid_exists(descendant)
    finally:
        if psutil.pid_exists(descendant):
            psutil.Process(descendant).kill()


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object guard-termination guarantee")
def test_os_cleans_worker_if_guard_is_killed(tmp_path):
    marker = tmp_path / "worker.pid"
    worker = f"import os,time; from pathlib import Path; Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(30)"
    launcher = (
        "from pathlib import Path; from experiments.peft_adaptation_scope_v1.guard import run_guarded,Thresholds; "
        f"run_guarded({[sys.executable, '-c', worker]!r},Path({str(tmp_path / 'guard')!r}),Path({str(tmp_path)!r}),"
        "require_gpu=False,thresholds=Thresholds(min_available_ram_gib=0,min_available_commit_gib=0))"
    )
    outer = subprocess.Popen([sys.executable, "-c", launcher], cwd=Path(__file__).resolve().parents[1])
    worker_pid = None
    try:
        for _ in range(100):
            if marker.exists():
                worker_pid = int(marker.read_text())
                break
            if outer.poll() is not None:
                pytest.fail("Outer guard exited before worker started")
            time.sleep(0.05)
        assert worker_pid is not None
        outer.terminate()
        outer.wait(timeout=5)
        for _ in range(60):
            if not psutil.pid_exists(worker_pid):
                break
            time.sleep(0.05)
        assert not psutil.pid_exists(worker_pid)
    finally:
        if outer.poll() is None:
            outer.terminate()
            outer.wait(timeout=5)
        if worker_pid is not None and psutil.pid_exists(worker_pid):
            psutil.Process(worker_pid).kill()
