"""Use the existing causal hourly preparation with fixed new dates."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
from datetime import datetime
import argparse
import json
from pathlib import Path
from experiments.peft_fullft_reference_v3 import data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    data.PANEL_STARTS = {'bike': datetime(2012, 4, 6), 'household': datetime(2008, 8, 30)}
    data.STUDY = 'peft_overlap_transfer_v1'
    data.VERSION = 'peft_overlap_transfer_v1.prepare.20260910'
    manifest = data.prepare(Path(args.output))
    assert manifest['all_qc_passed']
    ends = {'bike': ['2012-09-14T00:00:00', '2012-12-04T00:00:00'], 'household': ['2009-02-07T00:00:00', '2009-04-29T00:00:00']}
    old = json.loads((data.ROOT / 'runs/peft_optimization_control_v1/contract.json').read_text())
    for ds, spec in manifest['datasets'].items():
        assert spec['boundaries']['eval'] == ends[ds]
        assert spec['boundaries']['eval'][0] >= old['reference']['datasets'][ds]['boundaries']['eval'][1]
        assert spec['origin_counts'] == {'train': 90, 'val': 30, 'cal': 20, 'eval': 80}
    (Path(args.output) / 'result.json').write_text(json.dumps({'completed': True, 'manifest_sha256': manifest['manifest_sha256'], 'new_E_target_times_disjoint': True}, indent=2))
    print(json.dumps({'completed': True, 'new_E_target_times_disjoint': True}), flush=True)


if __name__ == '__main__':
    main()
