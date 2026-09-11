"""Verify closure artifacts and preservation without importing any experiment."""
import ast
import csv
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
NOTES = ROOT / '_docs/notes/tsfm_topics/07_research_direction'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    before = read_json(OUT / 'tracked_files_before.json')
    permitted = {'_docs/history/2026-09-11.md',
                 '_docs/notes/tsfm_topics/07_research_direction/README.md'}
    changed = [p for p, h in before.items()
               if not (ROOT / p).is_file() or sha(ROOT / p) != h]
    assert set(changed) <= permitted, changed
    evidence = read_json(OUT / 'evidence_reconstruction.json')
    assert evidence['status'] == 'PASS_CORE_RECONSTRUCTION'
    assert all(c['status'] != 'EVIDENCE_MISMATCH' for c in evidence['checks'])
    assert all(sha(ROOT / p) == item['sha256']
               for p, item in evidence['provenance'].items())
    prereg = read_json(OUT / 'preregistered_stage_a.json')
    assert prereg['status'] == 'DRAFT_NOT_FROZEN' and not prereg['executable']
    assert prereg['execution_caps']['actual_fit_count'] == 0
    assert prereg['execution_caps']['actual_forecast_count'] == 0
    assert prereg['arms']['WIDE']['expected_trainable_parameters'] == prereg['arms']['JOINT']['expected_trainable_parameters']
    for script in OUT.glob('*.py'):
        ast.parse(script.read_text(encoding='utf-8-sig'))
    with (OUT / 'data_exposure_ledger.csv').open(encoding='utf-8-sig', newline='') as stream:
        ledger = list(csv.DictReader(stream))
    assert ledger
    ledger_refs = sorted({p.strip() for row in ledger
                          for p in row['evidence_refs'].split(';') if p.strip()})
    assert all((ROOT / p).is_file() for p in ledger_refs)
    assert all(row['final_holdout_candidate'] != 'YES' for row in ledger)
    inventory = read_json(OUT / 'data_exposure_dir_inventory.json')
    actual_dirs = {p.relative_to(ROOT).as_posix() for area in ('experiments', 'results')
                   for p in (ROOT / area).glob('peft_*') if p.is_dir()}
    recorded_dirs = {row['peft_dir'].replace('\\', '/') for row in inventory}
    assert actual_dirs == recorded_dirs, (actual_dirs - recorded_dirs, recorded_dirs - actual_dirs)
    original = OUT / 'data_exposure_ledger.original_20260911_181304.csv'
    assert sha(original) == 'b4b0edca0df8b74ca088e3e27fced2568270158767ec45aa7fc1b0f0a8103bd9'
    novelty = (NOTES / '37_novelty_boundary.md').read_text(encoding='utf-8')
    papers = re.split(r'^## \d+\. ', novelty, flags=re.M)[1:]
    assert len(papers) >= 11
    assert all(all(re.search(rf'^{i}\. ', p, re.M) for i in range(1, 14)) for p in papers)
    documents = list(OUT.glob('*.md')) + list(NOTES.glob('37_*.md'))
    missing_links = []
    for document in documents:
        content = document.read_text(encoding='utf-8-sig')
        for target in re.findall(r'\]\(([^)]+)\)', content):
            if '://' in target or target.startswith('#'):
                continue
            local = target.split('#')[0].strip('<>')
            if (document.parent / local).resolve() == (OUT / 'verification.json').resolve():
                continue  # This receipt is created after the remaining checks pass.
            if local and not (document.parent / local).exists():
                missing_links.append({'document': str(document), 'target': target})
    assert not missing_links, missing_links
    readiness = read_json(OUT / 'paper_readiness.json')
    allowed = {'READY_FOR_CHARACTERIZATION_PAPER', 'READY_FOR_METHOD_DEVELOPMENT',
               'READY_FOR_METHOD_PAPER', 'MORE_INDEPENDENT_EVIDENCE_REQUIRED',
               'NOVELTY_GATE_FAIL', 'PHENOMENON_NOT_REPLICATED', 'DATA_HOLDOUT_BLOCKED'}
    assert readiness['verdict'] in allowed
    assert readiness['GPU_fits'] == readiness['GPU_forecasts'] == 0
    diff = subprocess.run(['git', 'diff', '--check'], cwd=ROOT, capture_output=True, text=True)
    assert diff.returncode == 0, diff.stdout + diff.stderr
    report = {'status': 'PASS', 'verified_utc': datetime.now(timezone.utc).isoformat(),
              'protected_tracked_files': len(before), 'permitted_changed_files': changed,
              'evidence_numeric_checks': len(evidence['checks']),
              'evidence_hashed_inputs_unchanged': len(evidence['provenance']),
              'ledger_rows_not_independent_units': len(ledger),
              'peft_directories_in_coverage_inventory': len(inventory),
              'ledger_source_hashes': {p: sha(ROOT / p) for p in ledger_refs},
              'original_untracked_ledger_preserved_sha256': sha(original),
              'novelty_papers_with_13_fields': len(papers), 'local_links_missing': missing_links,
              'prereg_status': prereg['status'], 'verdict': readiness['verdict'],
              'git_diff_check_exit': diff.returncode, 'GPU_fits': 0, 'GPU_forecasts': 0,
              'scope': 'Artifact consistency and hashes; not a fresh model result or full literature replication.',
              'verification_note': 'An earlier ad-hoc AST check hit Windows cp949 decoding; explicit UTF-8 corrected the reader only.'}
    (OUT / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    assert (OUT / 'verification.json').is_file()
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
