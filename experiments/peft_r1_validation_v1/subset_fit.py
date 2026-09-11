"""Use the frozen matched-head implementation with an explicit train-origin subset."""
import json
import sys
import numpy as np
from experiments.peft_mechanism_diagnostics_v1 import controlled_fit as original


def main():
    job = json.loads(sys.argv[sys.argv.index('--job') + 1])
    parent = original.shared.Panel

    class SubsetPanel(parent):
        def __init__(self, path, stage, smoke=False):
            super().__init__(path, stage, smoke)
            if stage == 'fit' and not smoke:
                rows = np.asarray(job['training_rows'], dtype=int)
                assert len(self.origins['train']) == 90
                assert len(rows) == 30 and len(np.unique(rows)) == 30
                assert rows.min() >= 0 and rows.max() < 90 and np.all(np.diff(rows) > 0)
                self.origins['train'] = self.origins['train'][rows]

    original.shared.Panel = SubsetPanel
    original.main()


if __name__ == '__main__':
    main()
