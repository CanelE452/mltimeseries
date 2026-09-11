"""Zero-initialized residual head forecast, verified identical to native F0."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
import torch
from experiments.peft_mechanism_diagnostics_v1.controlled_fit import Controlled, check, predict, sha, save, shared, native

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contract', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    contract = json.loads(Path(args.contract).read_text())
    for path, digest in contract['source_hashes'].items():
        assert sha(ROOT / path) == digest, path
    spec = contract['reference']['datasets'][args.dataset]
    assert sha(ROOT / spec['holdout_data_path']) == spec['holdout_data_sha256']
    panel = shared.Panel(ROOT / spec['holdout_data_path'], 'forecast', False)
    runtime = SimpleNamespace(checkpoint=contract['reference']['checkpoint'], device='cuda', min_free_ram_gib=5.)
    check(runtime)
    base = shared.load_base(runtime)
    job = {'head': 'mlp', 'seed': 26000, 'blocks': []}
    origins, _ = shared.padded_origins(panel.origins['eval'][:4])
    context, _, groups = panel.batch(origins, 'cuda')
    with torch.no_grad(), shared.legacy.precision(runtime):
        enc, _, _, _ = base.encode(context=context, group_ids=groups, num_output_patches=3)
        reference = native.patch_to_quantiles(base.output_patch_embedding(enc.last_hidden_state[:, -3:]), 21, 16).float()
    model = Controlled(base, job, panel.count_channels)
    with torch.no_grad(), shared.legacy.precision(runtime):
        _, initial, _, _ = model.encode(context, groups, 48)
    error = float((initial - reference).abs().max())
    assert error < 1e-5
    del enc, reference, initial, context, groups
    prediction, score = predict(model, panel, 'eval', runtime)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / 'predictions.npz', prediction=prediction, target=panel.targets('eval'), origins=panel.origins['eval'], quantiles=panel.quantiles, scale=panel.fit_std[panel.target_indices])
    save(out / 'result.json', {'completed': True, 'dataset': args.dataset, 'score': score, 'seconds': time.perf_counter()-started, 'identity_error': error, 'contract_sha256': sha(args.contract)})


if __name__ == '__main__':
    main()
