"""Native-path LoRA-only; Study35 loss, panels, precision and optimizer contract."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch
from peft import LoraConfig, inject_adapter_in_model
from peft.tuners.lora.layer import LoraLayer
from experiments.peft_contribution_freeze_v1.fit import check, predict, shared, native, digest, snapshot, restore, sha, save
from experiments.peft_decision_transfer_v1.panel import Panel
from experiments.peft_initial_headroom_v1.analyse import score_npz

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--stage', choices=['gate', 'fit', 'forecast'], required=True)
    parser.add_argument('--budget', default='L720', choices=['S180', 'L720'])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    plan_path = ROOT/'experiments/peft_lora_only_diagnostic_v1/plan.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    for path, expected in (plan['source_hashes'] | plan['input_hashes']).items():
        assert sha(ROOT/path) == expected, path
    job = plan['jobs'][args.index]
    run = ROOT/plan['run_dir']
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    forecast = args.stage == 'forecast'
    if forecast:
        seal = json.loads((run/'selection.json').read_text())
        assert seal['plan_sha256'] == sha(plan_path)
        assert [args.index, args.budget] in seal['forecasts']
    spec = plan['data'][job['dataset']]
    path = ROOT/spec['holdout_path' if forecast else 'fit_path']
    panel = Panel(path, 'forecast' if forecast else 'fit')
    if forecast:
        panel.origins['eval'] = panel.origins['eval'][::plan['eval_stride']]
    runtime = SimpleNamespace(checkpoint=plan['checkpoint'], device='cuda', min_free_ram_gib=5.)
    check(runtime)
    torch.manual_seed(job['seed'])
    model = native.AdaptationModel(shared.load_base(runtime), 'F0', panel.count_channels)
    split = 'eval' if forecast else 'val'
    f0, f0_score = predict(model, panel, split, runtime)
    modules = [f'encoder.block.{b}.layer.{layer}.self_attention.{part}'
               for b in range(12) for layer in (0, 1) for part in ('q', 'k', 'v', 'o')]
    assert all(n in dict(model.base.named_modules()) for n in modules)
    torch.manual_seed(job['seed'] + 1000)
    inject_adapter_in_model(LoraConfig(r=8, lora_alpha=16, lora_dropout=0., bias='none',
                                      target_modules=modules), model.base)
    native.deterministic_backbone(model.base)
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    frozen = {n for n, p in model.named_parameters() if not p.requires_grad}
    audit = [{'name': n, 'shape': list(p.shape), 'count': p.numel(), 'dtype': str(p.dtype)}
             for n, p in model.named_parameters() if p.requires_grad]
    assert names and all('lora_' in n for n in names)
    assert not hasattr(model, 'probe')
    assert all(not p.requires_grad for p in model.base.output_patch_embedding.parameters())
    actual_modules = [n for n, m in model.base.named_modules() if isinstance(m, LoraLayer)]
    assert set(actual_modules) == set(modules)
    assert all(torch.count_nonzero(p).item() == 0 for n, p in model.named_parameters() if 'lora_B' in n)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert count == 1179648 and len(names) == 192
    initial, initial_v = predict(model, panel, split, runtime)
    error = float(np.max(np.abs(initial - f0)))
    gate = {'step0_max_abs_error': error, 'F0_score': f0_score, 'initial_score': initial_v,
            'trainable_count': count, 'trainable_parameters': audit, 'actual_modules': actual_modules,
            'no_probe': True, 'native_head_frozen': True,
            'native_encode_inherited': type(model).encode is native.AdaptationModel.encode,
            'initialization_source': inspect.getsource(LoraLayer.reset_lora_parameters),
            'initialization_source_file_sha256': sha(inspect.getfile(LoraLayer)),
            'passed': error <= 1e-6 and abs(initial_v - f0_score) <= 1e-12}
    save(out/'gate.json', gate)
    assert gate['passed'], gate['step0_max_abs_error']
    frozen_hash = digest(model, frozen)

    def export(name, prediction, which):
        np.savez_compressed(out/f'{name}.npz', prediction=prediction, target=panel.targets(which),
                            scale=panel.fit_std[panel.target_indices], quantiles=panel.quantiles,
                            origins=panel.origins[which])

    export('F0' if forecast else 'initial_val', initial, split)
    assert abs(score_npz(out/('F0.npz' if forecast else 'initial_val.npz')) - initial_v) < 1e-12
    if args.stage == 'gate':
        save(out/'result.json', {'completed': True, 'stage': 'gate', 'optimizer_updates': 0,
             'seconds': time.perf_counter()-started, 'plan_sha256': sha(plan_path), 'gate': gate})
        return
    if forecast:
        source = run/'fit'/job['key']/'output'
        record = json.loads((source/'result.json').read_text())
        assert sha(source/'result.json') == seal['fit_hashes'][job['key']]
        assert record['plan_sha256'] == sha(plan_path) and record['job'] == job
        selected = record['budgets'][args.budget]
        assert sha(source/f'{args.budget}.pt') == selected['checkpoint_sha256']
        state = torch.load(source/f'{args.budget}.pt', map_location='cpu', weights_only=True)
        assert set(state) == names
        restore(model, state)
        assert digest(model, names) == selected['selected_hash']
        prediction, score = predict(model, panel, 'eval', runtime)
        export('selected', prediction, 'eval')
        assert abs(score_npz(out/'selected.npz')-score) < 1e-12
        assert digest(model, frozen) == frozen_hash
        save(out/'result.json', {'completed': True, 'job': job, 'budget': args.budget,
             'D_score': score, 'plan_sha256': sha(plan_path), 'seconds': time.perf_counter()-started,
             'frozen_verified': True, 'peak_allocated': torch.cuda.max_memory_allocated(),
             'peak_reserved': torch.cuda.max_memory_reserved()})
        return
    old_selection = json.loads((ROOT/'runs/peft_head_convergence_v1/selection_A.json').read_text())
    old = old_selection['cells'][f"{job['dataset']}/s{job['seed']}"]['L720']['HEAD']
    old_folder = ROOT/'runs/peft_head_convergence_v1/fit'/old['key']/'output'
    with np.load(old_folder/'initial_val.npz') as z:
        np.testing.assert_array_equal(initial, z['prediction'])
    samples = np.random.default_rng(job['seed']).integers(len(panel.origins['train']), size=(720, 8))
    sample_hash = hashlib.sha256(samples.tobytes()).hexdigest()
    assert sample_hash == json.loads((old_folder/'result.json').read_text())['sample_sha256']
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=job['lora_lr'], weight_decay=0., foreach=False)
    q = torch.as_tensor(panel.quantiles, dtype=torch.float32, device='cuda')
    best = {b: {'V': initial_v, 'step': 0, 'state': snapshot(model, names), 'prediction': initial.copy()}
            for b in ('S180', 'L720')}
    history, gradients = [], []
    torch.cuda.synchronize()
    trajectory_start = time.perf_counter()

    def observe(step, prediction, v):
        _, train_score = predict(model, panel, 'train', runtime)
        assert np.isfinite(v) and np.isfinite(train_score)
        export(f'point_{step}', prediction, 'val')
        history.append({'step': step, 'V': v, 'train_score': train_score})
        for budget, cap in (('S180', 180), ('L720', 720)):
            if step <= cap and v < best[budget]['V']:
                best[budget] = {'V': v, 'step': step, 'state': snapshot(model, names), 'prediction': prediction.copy()}

    observe(0, initial, initial_v)
    for step in range(1, 721):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for offset in (0, 4):
            context, target, ids = panel.batch(panel.origins['train'][samples[step-1, offset:offset+4]], 'cuda')
            with shared.legacy.precision(runtime):
                _, norm, loc, scale = model.encode(context, ids, 48)
                loss = native.native_pinball(norm, target, loc, scale, q, model.use_arcsinh)
            assert torch.isfinite(loss), 'nonfinite loss'
            (loss/2).backward()
            check(runtime)
            del context, target, ids, norm, loc, scale, loss
        gradients.append(float(torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)))
        optimizer.step()
        if step in plan['schedule']:
            prediction, v = predict(model, panel, 'val', runtime)
            observe(step, prediction, v)
            save(out/'progress.json', {'step': step, 'seconds': time.perf_counter()-started})
    torch.cuda.synchronize()
    trajectory_seconds = time.perf_counter()-trajectory_start
    assert digest(model, frozen) == frozen_hash
    selected = {}
    for budget, b in best.items():
        restore(model, b['state'])
        replay, score = predict(model, panel, 'val', runtime)
        np.testing.assert_array_equal(replay, b['prediction'])
        assert abs(score-b['V']) < 1e-12
        torch.save(b['state'], out/f'{budget}.pt')
        export(f'{budget}_val', replay, 'val')
        selected[budget] = {'V': score, 'step': b['step'], 'selected_hash': digest(model, names),
                            'checkpoint_sha256': sha(out/f'{budget}.pt')}
    assert digest(model, frozen) == frozen_hash
    save(out/'result.json', {'completed': True, 'job': job, 'plan_sha256': sha(plan_path),
         'history': history, 'budgets': selected, 'trainable': count, 'sample_sha256': sample_hash,
         'frozen_verified': True, 'frozen_hash': frozen_hash, 'restore_exact': True, 'D_opened': False,
         'steps': 720, 'trajectory_seconds': trajectory_seconds, 'seconds': time.perf_counter()-started,
         'gradient_max': max(gradients), 'clipped_steps': sum(g > 1 for g in gradients),
         'peak_allocated': torch.cuda.max_memory_allocated(), 'peak_reserved': torch.cuda.max_memory_reserved()})


if __name__ == '__main__':
    main()
