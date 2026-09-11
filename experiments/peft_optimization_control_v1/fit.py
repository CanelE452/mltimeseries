"""Immutable matched-head model with two prespecified checkpoint budgets."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch
from experiments.peft_mechanism_diagnostics_v1.controlled_fit import (
    Controlled, check, predict, sha, save, shared, native,
    parameter_digest, snapshot_trainable, restore_trainable,
)

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contract', required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--forecast-fit')
    parser.add_argument('--regime', choices=['UPDATE', 'EXPOSURE'])
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    started = time.perf_counter()
    contract = json.loads(Path(args.contract).read_text())
    job = json.loads(args.job)
    for path, digest in contract['source_hashes'].items():
        assert sha(ROOT / path) == digest, path
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    spec = contract['reference']['datasets'][job['dataset']]
    forecast = args.forecast_fit is not None
    data_path = ROOT / spec['holdout_data_path' if forecast else 'fit_data_path']
    assert sha(data_path) == spec['holdout_data_sha256' if forecast else 'fit_data_sha256']
    panel = shared.Panel(data_path, 'forecast' if forecast else 'fit', False)
    if not forecast:
        rows = np.asarray(job['training_rows'], dtype=int)
        assert len(panel.origins['train']) == 90
        assert len(rows) in (30, 90) and np.all(np.diff(rows) > 0)
        assert rows.min() >= 0 and rows.max() < 90
        panel.origins['train'] = panel.origins['train'][rows]
    runtime = SimpleNamespace(checkpoint=contract['reference']['checkpoint'], device='cuda', min_free_ram_gib=5.)
    check(runtime)
    base = shared.load_base(runtime)
    origins, _ = shared.padded_origins(next(iter(panel.origins.values()))[:4])
    context, _, groups = panel.batch(origins, 'cuda')
    with torch.no_grad(), shared.legacy.precision(runtime):
        enc, _, _, _ = base.encode(context=context, group_ids=groups, num_output_patches=3)
        reference = native.patch_to_quantiles(base.output_patch_embedding(enc.last_hidden_state[:, -3:]), 21, 16).float()
    model = Controlled(base, job, panel.count_channels)
    with torch.no_grad(), shared.legacy.precision(runtime):
        _, initial, _, _ = model.encode(context, groups, 48)
    identity = float((initial - reference).abs().max())
    assert identity < 1e-5
    del enc, reference, initial, context, groups
    frozen = parameter_digest(model, trainable=False)
    params = [p for p in model.parameters() if p.requires_grad]
    count = sum(p.numel() for p in params)
    assert count == (1768949 if job['arm'] == 'ALL' else 589301)
    if forecast:
        fit = Path(args.forecast_fit)
        record = json.loads((fit / 'result.json').read_text())
        assert record['job'] == job and record['contract_sha256'] == sha(args.contract)
        chosen = record['regimes'][args.regime]
        checkpoint = fit / (args.regime + '.pt')
        assert sha(checkpoint) == chosen['checkpoint_sha256']
        restore_trainable(model, torch.load(checkpoint, map_location='cpu', weights_only=True))
        prediction, score = predict(model, panel, 'eval', runtime)
        assert parameter_digest(model, trainable=False) == frozen
        np.savez_compressed(out / 'predictions.npz', prediction=prediction, target=panel.targets('eval'), origins=panel.origins['eval'], quantiles=panel.quantiles, scale=panel.fit_std[panel.target_indices])
        save(out / 'result.json', {'completed': True, 'job': job, 'regime': args.regime, 'score': score, 'fit': str(fit), 'seconds': time.perf_counter()-started, 'trainable': count, 'contract_sha256': sha(args.contract), 'exposed_E': True})
        return
    schedules = contract['schedules'][job['condition']]
    if args.smoke:
        schedules = {name: [0, 1, 3] for name in schedules}
    steps = 3 if args.smoke else contract['steps']
    opt_groups = [{'params': list(model.probe.parameters()), 'lr': job['head_lr']}]
    adapter = [p for p in base.parameters() if p.requires_grad]
    if adapter:
        opt_groups.append({'params': adapter, 'lr': job['lora_lr']})
    optimizer = torch.optim.AdamW(opt_groups, weight_decay=0., foreach=False)
    assert {id(p) for g in optimizer.param_groups for p in g['params']} == {id(p) for p in params}
    samples = np.random.default_rng(job['seed']).integers(len(panel.origins['train']), size=(steps, 8))
    q = torch.as_tensor(panel.quantiles, dtype=torch.float32, device='cuda')
    initial_pred, initial_score = predict(model, panel, 'val', runtime)
    winners = {name: {'score': initial_score, 'step': 0, 'prediction': initial_pred, 'state': snapshot_trainable(model)} for name in schedules}
    initial_trainable = parameter_digest(model, trainable=True)
    history = [{'step': 0, 'score': initial_score}]
    minimum = check(runtime)
    checkpoints = set().union(*map(set, schedules.values()))
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for offset in (0, 4):
            origins = panel.origins['train'][samples[step-1, offset:offset+4]]
            context, target, groups = panel.batch(origins, 'cuda')
            with shared.legacy.precision(runtime):
                _, norm, loc, scale = model.encode(context, groups, 48)
                loss = native.native_pinball(norm, target, loc, scale, q, model.use_arcsinh)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite loss')
            (loss / 2).backward()
            minimum = min(minimum, check(runtime))
            del context, target, groups, norm, loc, scale, loss
        grad = float(torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True))
        if step == 1:
            assert grad > 0
        optimizer.step()
        if step in checkpoints:
            prediction, score = predict(model, panel, 'val', runtime)
            history.append({'step': step, 'score': score})
            for name, schedule in schedules.items():
                if step in schedule and score < winners[name]['score']:
                    winners[name] = {'score': score, 'step': step, 'prediction': prediction, 'state': snapshot_trainable(model)}
            save(out / 'progress.json', {'step': step, 'elapsed': time.perf_counter()-started})
        minimum = min(minimum, check(runtime))
    assert initial_trainable != parameter_digest(model, trainable=True)
    assert parameter_digest(model, trainable=False) == frozen
    records = {}
    for name, winner in winners.items():
        restore_trainable(model, winner['state'])
        replay, replay_score = predict(model, panel, 'val', runtime)
        np.testing.assert_allclose(replay, winner['prediction'], rtol=0, atol=0)
        assert abs(replay_score - winner['score']) < 1e-10
        torch.save(winner['state'], out / (name + '.pt'))
        np.savez_compressed(out / (name + '_val.npz'), prediction=winner['prediction'], target=panel.targets('val'))
        records[name] = {'best_score': winner['score'], 'best_step': winner['step'], 'checkpoint_sha256': sha(out / (name + '.pt')), 'replay_verified': True}
    save(out / 'result.json', {'completed': True, 'job': job, 'regimes': records, 'history': history, 'trainable': count, 'initial_head_hash': model.initial_head_hash, 'identity_error': identity, 'frozen_hash': frozen, 'frozen_verified': True, 'steps': steps, 'minimum_commit_gib': minimum, 'max_cuda_allocated_gib': torch.cuda.max_memory_allocated()/2**30, 'seconds': time.perf_counter()-started, 'contract_sha256': sha(args.contract), 'holdout_opened': False, 'sample_index_sha256': __import__('hashlib').sha256(samples.tobytes()).hexdigest()})
    print(json.dumps({'done': job, 'regimes': records, 'seconds': time.perf_counter()-started}), flush=True)


if __name__ == '__main__':
    main()
