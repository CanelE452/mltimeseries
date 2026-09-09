"""Run one study job with resource limits and ownership-scoped cleanup."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import psutil


GIB = 1024**3


@dataclass(frozen=True)
class Thresholds:
    min_available_ram_gib: float = 5.0
    min_available_commit_gib: float = 6.0
    max_child_rss_gib: float | None = 8.0
    max_git_processes: int = 32
    max_gpu_memory_mib: float = 10500.0
    max_gpu_temperature_c: float = 85.0


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, data: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


class StudyLock:
    def __init__(self, path: Path):
        self.path = path
        self.record = json.dumps({"pid": os.getpid(), "create_time": psutil.Process().create_time()})
        self.acquired = False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(2):
            try:
                descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                if attempt:
                    raise RuntimeError(f"Another guard holds {self.path}")
                before = self.path.stat()
                raw = self.path.read_text(encoding="utf-8")
                try:
                    owner = json.loads(raw)
                    process = psutil.Process(int(owner["pid"]))
                    live = abs(process.create_time() - float(owner["create_time"])) < 0.001
                except psutil.NoSuchProcess:
                    live = False
                except (KeyError, TypeError, ValueError, psutil.AccessDenied) as exc:
                    raise RuntimeError(f"Cannot verify lock owner: {self.path}") from exc
                if live:
                    raise RuntimeError(f"Another guard is running (PID {owner['pid']})")
                after = self.path.stat()
                if (before.st_ino, before.st_mtime_ns) != (after.st_ino, after.st_mtime_ns):
                    raise RuntimeError("Guard lock changed during stale-lock validation")
                if self.path.read_text(encoding="utf-8") != raw:
                    raise RuntimeError("Guard lock changed during stale-lock validation")
                self.path.unlink()
            else:
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    stream.write(self.record)
                self.acquired = True
                return self
        raise RuntimeError("Could not acquire study lock")

    def __exit__(self, *_):
        if self.acquired:
            with contextlib.suppress(FileNotFoundError):
                if self.path.read_text(encoding="utf-8") == self.record:
                    self.path.unlink()


def _available_commit() -> int:
    if os.name != "nt":
        return psutil.virtual_memory().available + psutil.swap_memory().free
    from ctypes import wintypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
            (name, ctypes.c_ulonglong)
            for name in ("total_phys", "avail_phys", "total_page", "avail_page", "total_virtual", "avail_virtual", "avail_extended")
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(status.avail_page)


def _gpu_resources() -> list[dict]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,temperature.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=3, check=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    devices = []
    for index, line in enumerate(result.stdout.splitlines()):
        if line.strip():
            memory, temperature = (float(part.strip()) for part in line.split(","))
            devices.append({"index": index, "memory_used_mib": memory, "temperature_c": temperature})
    if not devices:
        raise RuntimeError("nvidia-smi returned no usable GPU measurements")
    return devices


def _sample_resources(child: psutil.Process | None, gpu: list[dict] | None) -> dict:
    rss = 0
    if child is not None:
        try:
            processes = [child] + child.children(recursive=True)
        except psutil.NoSuchProcess:
            processes = []
        for process in processes:
            with contextlib.suppress(psutil.NoSuchProcess):
                rss += process.memory_info().rss
    git_count = sum(
        1 for process in psutil.process_iter(["name"])
        if (process.info["name"] or "").lower() in {"git", "git.exe"}
    )
    return {
        "timestamp": _utc(), "available_ram_gib": psutil.virtual_memory().available / GIB,
        "available_commit_gib": _available_commit() / GIB,
        "child_tree_rss_gib": rss / GIB, "git_process_count": git_count, "gpus": gpu,
    }


def _violations(sample: dict, limits: Thresholds) -> list[str]:
    reasons = []
    if sample["available_ram_gib"] < limits.min_available_ram_gib:
        reasons.append("available_ram_below_limit")
    if sample["available_commit_gib"] < limits.min_available_commit_gib:
        reasons.append("available_commit_below_limit")
    if limits.max_child_rss_gib is not None and sample["child_tree_rss_gib"] > limits.max_child_rss_gib:
        reasons.append("child_tree_rss_above_limit")
    if sample["git_process_count"] > limits.max_git_processes:
        reasons.append("git_process_count_above_limit")
    for gpu in sample["gpus"] or []:
        if gpu["memory_used_mib"] > limits.max_gpu_memory_mib:
            reasons.append(f"gpu_{gpu['index']}_memory_above_limit")
        if gpu["temperature_c"] >= limits.max_gpu_temperature_c:
            reasons.append(f"gpu_{gpu['index']}_temperature_at_limit")
    return reasons


class WindowsJob:
    """The OS kills this job's descendants even if the guard itself is killed."""

    def __init__(self):
        from ctypes import wintypes

        class BasicLimits(ctypes.Structure):
            _fields_ = [
                ("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64), ("flags", wintypes.DWORD),
                ("min_working_set", ctypes.c_size_t), ("max_working_set", ctypes.c_size_t),
                ("active_processes", wintypes.DWORD), ("affinity", ctypes.c_size_t),
                ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD),
            ]

        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in ("reads", "writes", "others", "read_bytes", "write_bytes", "other_bytes")]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("basic", BasicLimits), ("io", IoCounters), ("process_memory", ctypes.c_size_t),
                ("job_memory", ctypes.c_size_t), ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t),
            ]

        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.kernel.SetInformationJobObject.restype = wintypes.BOOL
        self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenThread.restype = wintypes.HANDLE
        self.kernel.ResumeThread.argtypes = [wintypes.HANDLE]
        self.kernel.ResumeThread.restype = wintypes.DWORD
        self.kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.kernel.TerminateJobObject.restype = wintypes.BOOL
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def assign_and_resume(self, pid: int) -> None:
        handle = self.kernel.OpenProcess(0x100 | 0x1, False, pid)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not self.kernel.AssignProcessToJobObject(self.handle, handle):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            self.kernel.CloseHandle(handle)
        for thread in psutil.Process(pid).threads():
            handle = self.kernel.OpenThread(0x2, False, thread.id)
            if not handle:
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                if self.kernel.ResumeThread(handle) == 0xFFFFFFFF:
                    raise ctypes.WinError(ctypes.get_last_error())
            finally:
                self.kernel.CloseHandle(handle)

    def terminate(self) -> None:
        if self.handle and not self.kernel.TerminateJobObject(self.handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self) -> None:
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def _stop_owned(child: subprocess.Popen | None, job: WindowsJob | None) -> None:
    if job is not None:
        try:
            job.terminate()
        finally:
            job.close()
    if child is None:
        return
    if os.name != "nt":
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGTERM)
    if child.poll() is None:
        child.terminate()
    try:
        child.wait(timeout=3)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
        child.kill()
        child.wait(timeout=3)


def run_guarded(
    command: list[str], output: Path, cwd: Path, timeout_seconds: float = 7200,
    *, require_gpu: bool = True, thresholds: Thresholds | None = None,
) -> dict:
    if not command or timeout_seconds <= 0:
        raise ValueError("A command and positive timeout are required")
    output, cwd = Path(output).resolve(), Path(cwd).resolve()
    limits = thresholds or Thresholds()
    lock = cwd / "runs" / "peft_adaptation_scope_v1" / ".guard.lock"
    with StudyLock(lock):
        output.mkdir(parents=True, exist_ok=True)
        # Remove only a previous stop marker in this exact guard output directory.
        with contextlib.suppress(FileNotFoundError):
            (output / "safety_stop.json").unlink()
        started = time.monotonic()
        status = {
            "started_at": _utc(), "guard_pid": os.getpid(), "command": command,
            "cwd": str(cwd), "require_gpu": require_gpu, "thresholds": asdict(limits),
            "completed": False, "returncode": None, "state": "preflight",
        }
        child = None
        job = None
        reasons = []
        last_sample = None
        with (output / "resource_log.jsonl").open("w", encoding="utf-8", buffering=1) as log, \
             (output / "stdout.log").open("w", encoding="utf-8") as stdout, \
             (output / "stderr.log").open("w", encoding="utf-8") as stderr:
            try:
                gpu = _gpu_resources() if require_gpu else None
                last_sample = _sample_resources(None, gpu)
                log.write(json.dumps(last_sample) + "\n")
                reasons = _violations(last_sample, limits)
                if not reasons:
                    job = WindowsJob() if os.name == "nt" else None
                    child = subprocess.Popen(
                        command, cwd=cwd, stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL,
                        creationflags=(subprocess.CREATE_NO_WINDOW | 0x4) if os.name == "nt" else 0,
                        start_new_session=os.name != "nt",
                    )
                    if job is not None:
                        job.assign_and_resume(child.pid)
                    child_process = psutil.Process(child.pid)
                    status.update(state="running", child_pid=child.pid, child_create_time=child_process.create_time())
                    _write_json(output / "status.json", status)
                    last_gpu_check = last_log = time.monotonic()
                    while child.poll() is None:
                        now = time.monotonic()
                        if now - started >= timeout_seconds:
                            reasons = ["timeout"]
                            break
                        if require_gpu and now - last_gpu_check >= 10:
                            gpu = _gpu_resources()
                            last_gpu_check = time.monotonic()
                        last_sample = _sample_resources(child_process, gpu)
                        reasons = _violations(last_sample, limits)
                        if reasons or now - last_log >= 10:
                            log.write(json.dumps(last_sample) + "\n")
                            last_log = now
                        if reasons:
                            break
                        time.sleep(min(2.0, max(0.01, timeout_seconds - (time.monotonic() - started))))
                status["state"] = "safety_stop" if reasons else "completed"
            except KeyboardInterrupt:
                reasons = ["interrupted"]
                status["state"] = "interrupted"
            except Exception as exc:
                reasons = ["guard_or_sensor_error"]
                status.update(state="error", error=f"{type(exc).__name__}: {exc}")
            finally:
                if reasons:
                    _write_json(output / "safety_stop.json", {
                        "timestamp": _utc(), "reasons": reasons, "last_sample": last_sample,
                        "artifacts_preserved": True, "resume_requires_checkpoint": True,
                        "error": status.get("error"),
                    })
                try:
                    _stop_owned(child, job)
                except Exception as exc:
                    reasons.append("cleanup_error")
                    status.update(state="error", cleanup_error=f"{type(exc).__name__}: {exc}")
                status.update(
                    finished_at=_utc(), elapsed_seconds=round(time.monotonic() - started, 3),
                    returncode=child.returncode if child is not None else None, reasons=reasons,
                )
                status["completed"] = child is not None and child.returncode == 0 and not reasons
                if not status["completed"] and status["state"] == "completed":
                    status["state"] = "child_failed"
                log.write(json.dumps({"timestamp": _utc(), "event": "finish", "state": status["state"]}) + "\n")
                _write_json(output / "status.json", status)
        return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--timeout-seconds", type=float, default=7200)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        status = run_guarded(command, args.output, args.cwd, args.timeout_seconds, require_gpu=not args.cpu_only)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Guard did not launch: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(status, ensure_ascii=False))
    return 0 if status["completed"] else 130 if status["state"] == "interrupted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
