"""One immutable initial-adaptation trajectory, or sealed D forecast."""
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
from experiments.peft_contribution_freeze_v1.fit import (
    Controlled, check, predict, shared, native, digest, snapshot, restore, sha, save,
)
from experiments.peft_capacity_probe_v1.model import Controlled as WideControlled
from experiments.peft_decision_transfer_v1.panel import Panel

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--index', required=True, type=int)
    parser.add_argument('--output', required=True)
    parser.add_argument('--stage', choices=['fit', 'forecast'], default='fit')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    started = time.perf_counter()
    plan = json.loads(Path(args.plan).read_text())
    entry = plan['jobs'][args.index]; job = entry['job']
    for path, expected in plan['source_hashes'].items():
        assert sha(ROOT/path) == expected, path
    run = ROOT/plan['run_dir']; spec = plan['data'][job['dataset']]
    forecast = args.stage == 'forecast'
    role = 'holdout' if forecast else 'fit'
    if forecast:
        seal = json.loads((run/'selection_sealed.json').read_text())
        assert seal['plan_sha256'] == sha(args.plan)
        assert args.index in seal['forecast_indices']
    path = ROOT/spec[role+'_path']
    assert sha(path) == spec[role+'_sha256']
    panel = Panel(path, 'forecast' if forecast else 'fit', args.smoke)
    if forecast:
        panel.origins['eval'] = panel.origins['eval'][::4]
    elif not args.smoke:
        panel.origins['train'] = panel.origins['train'][job['training_rows']]
    runtime = SimpleNamespace(checkpoint=plan['checkpoint'], device='cuda', min_free_ram_gib=5.)
    check(runtime)
    model = (WideControlled if job['family'] == 'WIDE' else Controlled)(shared.load_base(runtime), job, panel.count_channels)
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    adapters = {n for n in names if 'lora_' in n}
    heads = names-adapters
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert count == (589301 if job['family'] == 'HEAD' else 1768949)
    assert bool(adapters) == (job['family'] == 'JOINT')
    frozen_hash = digest(model, frozen)
    initial_head_hash = digest(model, heads)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)

    def export(name, prediction, split):
        np.savez_compressed(out/(name+'.npz'), prediction=prediction,
            target=panel.targets(split), scale=panel.fit_std[panel.target_indices],
            quantiles=panel.quantiles, origins=panel.origins[split])

    if forecast:
        source = run/'fit'/entry['key']/'output'
        record = json.loads((source/'result.json').read_text())
        assert sha(source/'result.json') == seal['fit_result_hashes'][entry['key']]
        assert record['job'] == job and record['plan_sha256'] == sha(args.plan)
        assert sha(source/'best.pt') == record['checkpoint_sha256']
        if job['family'] == 'HEAD':
            prediction, f0 = predict(model, panel, 'eval', runtime)
            export('F0', prediction, 'eval')
        state = torch.load(source/'best.pt', map_location='cpu', weights_only=True)
        assert set(state) == names
        restore(model, state)
        assert digest(model, names) == record['selected_hash']
        prediction, score = predict(model, panel, 'eval', runtime)
        export('selected', prediction, 'eval')
        assert digest(model, frozen) == frozen_hash
        save(out/'result.json', {'completed': True, 'job': job, 'D_score': score,
            'plan_sha256': sha(args.plan), 'selected_hash': digest(model, names),
            'fit_result_sha256': sha(source/'result.json'), 'seconds': time.perf_counter()-started})
        return

    schedule = [0, 1, 2, 3] if args.smoke else plan['schedules'][job['condition']]
    samples = np.random.default_rng(job['seed']).integers(len(panel.origins['train']), size=(schedule[-1], 8))
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [{'params': list(model.probe.parameters()), 'lr': job['head_lr']}]
    if adapters:
        groups.append({'params': [p for n, p in model.named_parameters() if n in adapters], 'lr': job['lora_lr']})
    optimizer = torch.optim.AdamW(groups, weight_decay=0., foreach=False)
    q = torch.as_tensor(panel.quantiles, dtype=torch.float32, device='cuda')
    initial_prediction, initial_score = predict(model, panel, 'val', runtime)
    assert initial_score > 0
    export('F0_val', initial_prediction, 'val')
    # Training predictions for low-dimensional correction are exported only once per cell.
    if job['family'] == 'HEAD' and job['recipe'] == 0:
        prediction, _ = predict(model, panel, 'train', runtime)
        export('F0_train', prediction, 'train')
    best, best_step = initial_score, 0
    best_prediction = initial_prediction
    state = snapshot(model, names)
    history = [{'step': 0, 'V': initial_score}]
    torch.cuda.synchronize(); trajectory_start = time.perf_counter()
    gradients = []
    for step in range(1, schedule[-1]+1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        for offset in (0, 4):
            context, target, ids = panel.batch(panel.origins['train'][samples[step-1, offset:offset+4]], 'cuda')
            with shared.legacy.precision(runtime):
                _, norm, loc, scale = model.encode(context, ids, 48)
                loss = native.native_pinball(norm, target, loc, scale, q, model.use_arcsinh)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite loss')
            (loss/2).backward(); check(runtime)
            del context, target, ids, norm, loc, scale, loss
        grad = float(torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True))
        gradients.append(grad); optimizer.step()
        if step in schedule:
            prediction, score = predict(model, panel, 'val', runtime)
            history.append({'step': step, 'V': score})
            if score < best:
                best, best_step, best_prediction = score, step, prediction
                state = snapshot(model, names)
            save(out/'progress.json', {'step': step, 'cap': schedule[-1], 'best_step': best_step,
                                      'seconds': time.perf_counter()-started})
    torch.cuda.synchronize(); trajectory_seconds = time.perf_counter()-trajectory_start
    final_hash = digest(model, names)
    assert digest(model, frozen) == frozen_hash
    restore(model, state)
    replay, replay_score = predict(model, panel, 'val', runtime)
    np.testing.assert_array_equal(replay, best_prediction)
    assert abs(replay_score-best) < 1e-12
    torch.save(state, out/'best.pt'); export('selected_val', replay, 'val')
    save(out/'result.json', {'completed': True, 'job': job, 'plan_sha256': sha(args.plan),
        'initial_V': initial_score, 'best_V': best, 'best_step': best_step, 'history': history,
        'trainable': count, 'selected_hash': digest(model, names), 'final_hash': final_hash,
        'initial_head_hash': initial_head_hash, 'frozen_hash': frozen_hash, 'frozen_verified': True,
        'restore_exact': True, 'checkpoint_sha256': sha(out/'best.pt'), 'D_opened': False,
        'sample_sha256': __import__('hashlib').sha256(samples.tobytes()).hexdigest(),
        'gradient_norm_min': min(gradients), 'gradient_norm_max': max(gradients),
        'clipped_steps': sum(x > 1 for x in gradients), 'steps': schedule[-1],
        'trajectory_seconds': trajectory_seconds, 'seconds': time.perf_counter()-started,
        'smoke': args.smoke})
    print(json.dumps({'completed': entry['key'], 'best_step': best_step,
                      'seconds': time.perf_counter()-started}), flush=True)


if __name__ == '__main__':
    main()
