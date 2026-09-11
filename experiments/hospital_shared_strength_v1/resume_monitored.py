"""Resume the unchanged Hospital kit with per-job admission and private-memory logs."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import ctypes
from ctypes import wintypes
import json
import sys
import threading
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
KIT = ROOT / 'third_party/hospital_shared_adaptation_kit'
RUN = ROOT / 'runs/hospital_shared_strength_v1_run2'


class MemoryStatus(ctypes.Structure):
    _fields_ = [('length', wintypes.DWORD), ('load', wintypes.DWORD)] + [
        (key, ctypes.c_ulonglong) for key in ('totalPhys', 'availPhys', 'totalPageFile',
        'availPageFile', 'totalVirtual', 'availVirtual', 'availExtendedVirtual')]


def memory():
    state = MemoryStatus()
    state.length = ctypes.sizeof(state)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
        raise ctypes.WinError()
    return {'available_ram_gib': state.availPhys / 2**30,
            'available_commit_gib': state.availPageFile / 2**30,
            'commit_limit_gib': state.totalPageFile / 2**30}


def admission(log, stage, timeout=600):
    started = time.monotonic()
    consecutive = 0
    while True:
        sample = memory()
        ok = sample['available_commit_gib'] >= 13 and sample['available_ram_gib'] >= 5
        consecutive = consecutive + 1 if ok else 0
        event = {'utc': datetime.now(timezone.utc).isoformat(), 'stage': stage,
                 'admitted': consecutive >= 2, **sample}
        with (log / 'admission.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(event) + '\n')
        if consecutive >= 2:
            print(json.dumps(event), flush=True)
            return time.monotonic() - started
        if time.monotonic() - started >= timeout:
            raise RuntimeError('Admission timed out: need 13 GiB commit; no threshold relaxation')
        time.sleep(5)


def monitor(log, stop):
    with (log / 'private_memory.jsonl').open('a', encoding='utf-8') as f:
        while not stop.is_set():
            try:
                owner = psutil.Process()
                children = {p.pid for p in owner.children(recursive=True)} | {owner.pid}
                processes = []
                owned = []
                for p in psutil.process_iter(['pid', 'name']):
                    try:
                        m = p.memory_info()
                        row = {'pid': p.pid, 'name': p.info['name'],
                               'private_gib': m.private / 2**30, 'rss_gib': m.rss / 2**30}
                        processes.append(row)
                        if p.pid in children:owned.append(row)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                event = {'utc': datetime.now(timezone.utc).isoformat(), **memory(),
                         'owned_tree': owned,
                         'top_private': sorted(processes, key=lambda x:x['private_gib'], reverse=True)[:12]}
                f.write(json.dumps(event) + '\n');f.flush()
            except Exception as exc:
                f.write(json.dumps({'monitor_error': str(exc)}) + '\n');f.flush()
            stop.wait(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        print(json.dumps(memory()));return
    log = ROOT / 'runs' / ('hospital_shared_strength_v1_resume_monitor_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    log.mkdir()
    print(f'Monitor: {log}', flush=True)
    stopped = threading.Event()
    thread = threading.Thread(target=monitor, args=(log, stopped), daemon=True)
    thread.start()
    try:
        admission(log, 'before_controller_import')
        sys.path.insert(0, str(KIT))
        from hospital_pilot import runner
        job_graph_original = runner.job_graph

        def admitted_jobs(cfg):
            for name, action, extras in job_graph_original(cfg):
                receipts = list((RUN/'stages'/name).glob('attempt_*/receipt.json'))
                if action in ('s0', 'fit', 'forecast_gate', 'forecast_eval') and not receipts:
                    # Wait before the original runner allocates an attempt directory.
                    admission(log, name)
                yield name, action, extras

        # Scheduling/telemetry only. Kit and guard source bytes and scientific contract stay fixed.
        runner.job_graph = admitted_jobs
        contract = json.loads((RUN / 'run_contract.json').read_text())
        stages = runner.run(ROOT, contract['checkpoint'], contract['data'], RUN,
                            resume=True, retry_failed=True)
        guard, _ = runner.import_guard(ROOT)
        audit_out = RUN / 'independent_audit.json'
        if not audit_out.exists():
            command = [sys.executable, str(KIT/'scripts/independent_audit.py'),
                       '--data', contract['data'], '--gate-forecast', stages['forecast_gate'],
                       '--eval-forecast', stages['forecast_eval'], '--policies', stages['select_gates'],
                       '--analysis', stages['analyse'], '--out', str(audit_out)]
            result = guard.run_guarded(command, log/'independent_audit_guard', ROOT,
                                       timeout_seconds=180, require_gpu=False)
            if not result.get('completed'):raise RuntimeError('Independent audit did not complete')
        audit = json.loads(audit_out.read_text())
        if not audit.get('passed'):raise RuntimeError('Independent audit failed')
        (log/'completed.json').write_text(json.dumps({'completed': True, 'audit': audit,
            'analysis': stages['analyse']}, indent=2), encoding='utf-8')
        print(json.dumps({'completed': True, 'analysis': stages['analyse'], 'audit': audit}), flush=True)
    except Exception as exc:
        (log/'stopped.json').write_text(json.dumps({'completed': False,'error':repr(exc)},indent=2),encoding='utf-8')
        raise
    finally:
        stopped.set();thread.join(timeout=5)


if __name__ == '__main__':
    main()
