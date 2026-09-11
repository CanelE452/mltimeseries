"""Fit-archive-only F0 rolling forecasts, with no parameter adaptation."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
from experiments.peft_mechanism_diagnostics_v1 import controlled_fit as core


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--contract', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    started = time.perf_counter()
    contract = json.loads(Path(args.contract).read_text())
    for path, digest in contract['source_hashes'].items():
        assert core.sha(core.ROOT/path) == digest
    spec = contract['reference']['datasets'][args.dataset]
    path = core.ROOT/spec['fit_data_path']
    assert core.sha(path) == spec['fit_data_sha256']
    panel = core.shared.Panel(path, 'fit')
    runtime = SimpleNamespace(checkpoint=contract['reference']['checkpoint'], device='cuda', min_free_ram_gib=5.)
    core.check(runtime)
    model = core.Controlled(core.shared.load_base(runtime), {'seed':25000,'head':'mlp','blocks':[]}, panel.count_channels)
    before = core.parameter_digest(model)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    scores = {}
    for split in ('train', 'val'):
        prediction, score = core.predict(model, panel, split, runtime)
        np.savez_compressed(out/(split+'.npz'), prediction=prediction, target=panel.targets(split),
                            origins=panel.origins[split], scale=panel.fit_std[panel.target_indices], quantiles=panel.quantiles)
        scores[split] = score
    assert before == core.parameter_digest(model)
    core.save(out/'result.json', {'completed':True,'dataset':args.dataset,'contract_sha256':core.sha(args.contract),
                               'scores':scores,'parameters_unchanged':True,'holdout_opened':False,'seconds':time.perf_counter()-started})


if __name__ == '__main__':
    main()
