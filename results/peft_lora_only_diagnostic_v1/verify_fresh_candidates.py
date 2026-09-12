"""Read-only, streaming validation of fresh manifest hashes and BDG2 origin coverage."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(1024*1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    manifest = json.loads((ROOT/'results/peft_paper_closure_v1/fresh_stage_a_candidate_manifest.json').read_text(encoding='utf-8'))
    verified = []
    for candidate in manifest['candidates']:
        source = candidate['source_provenance']
        for key in source:
            if key.endswith('_path'):
                assert digest(ROOT/source[key]) == source[key.replace('_path', '_sha256')]
                verified.append(source[key])
        bounds = candidate['proposed_period']['boundaries']
        for split, count in candidate['proposed_period']['origin_counts'].items():
            start, end = [np.datetime64(t, 'h') for t in bounds[split]]
            assert int((end-start)/np.timedelta64(1, 'h'))-48 == (count-1)*24
        assert bounds['train'][1] <= bounds['val'][0] < bounds['val'][1] <= bounds['e1'][0] < bounds['e1'][1] <= bounds['e2'][0]
    bdg = next(c for c in manifest['candidates'] if c['status'] == 'READY')
    channels = bdg['series']['channels']
    times, values = [], []
    with (ROOT/bdg['source_provenance']['electricity_path']).open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        time_key = reader.fieldnames[0]
        for row in reader:
            times.append(np.datetime64(row[time_key], 'h'))
            values.append([float(row[c]) if row[c].strip() else np.nan for c in channels])
    times, values = np.asarray(times), np.asarray(values)
    assert np.all(np.diff(times) == np.timedelta64(1, 'h'))
    checks = {}
    for split in ('train', 'val', 'e1', 'e2'):
        start, end = [np.datetime64(t, 'h') for t in bdg['proposed_period']['boundaries'][split]]
        block = values[(times >= start) & (times < end)]
        fractions = [np.isfinite(block[o:o+48, :2]).mean(axis=0).tolist() for o in range(0, len(block)-47, 24)]
        assert len(fractions) == bdg['proposed_period']['origin_counts'][split]
        assert min(min(v) for v in fractions) >= .7
        checks[split] = {'origins': len(fractions), 'minimum_per_target_finite_fraction': min(min(v) for v in fractions)}
    payload = {'status': 'PASS', 'source_hashes_verified': verified, 'BDG2_per_target_window_checks': checks,
               'both_split_lengths_verified': True, 'target_performance_computed': False,
               'household_limit': 'Raw hash and split geometry independently checked; missing-window scan remains manifest preparer evidence.'}
    (OUT/'fresh_manifest_verification.json').write_text(json.dumps(payload, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(payload))


if __name__ == '__main__':
    main()
