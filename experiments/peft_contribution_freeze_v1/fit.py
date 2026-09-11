"""Guarded single-trajectory adaptation with immutable parameter ownership."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch
from peft.tuners.tuners_utils import BaseTunerLayer
from experiments.peft_mechanism_diagnostics_v1.controlled_fit import (
    Controlled, check, predict, sha, save, shared, native,
)
from experiments.peft_contribution_freeze_v1.policy import contribution_plateau, early_stop
from experiments.peft_contribution_freeze_v1.panel import Panel

ROOT = Path(__file__).resolve().parents[2]


def digest(model, names):
    h = hashlib.sha256()
    for n, p in model.named_parameters():
        if n in names:
            v = p.detach().cpu().contiguous()
            h.update(n.encode()); h.update(str((tuple(v.shape), v.dtype)).encode()); h.update(v.numpy().tobytes())
    return h.hexdigest()


def snapshot(model, names):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if n in names}


def restore(model, state):
    with torch.no_grad():
        named = dict(model.named_parameters())
        for n, v in state.items():
            named[n].copy_(v.to(named[n]))


@contextmanager
def adapters_off(model):
    layers = [m for m in model.modules() if isinstance(m, BaseTunerLayer)]
    assert len(layers) == 96 and all(not m.merged and not m.disable_adapters for m in layers)
    flags = {n: p.requires_grad for n, p in model.named_parameters()}
    try:
        for m in layers:
            m.enable_adapters(False)
        yield
    finally:
        for m in layers:
            m.enable_adapters(True)
        for n, p in model.named_parameters():
            p.requires_grad_(flags[n])
        assert all(not m.disable_adapters and not m.merged for m in layers)


def freeze(model, optimizer, adapter_names):
    for n, p in model.named_parameters():
        if n in adapter_names:
            p.requires_grad_(False); p.grad = None
            optimizer.state.pop(p, None)
    optimizer.param_groups = [g for g in optimizer.param_groups if any(p.requires_grad for p in g['params'])]
    assert len(optimizer.param_groups) == 1
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 589301
    assert all(not m.disable_adapters for m in model.modules() if isinstance(m, BaseTunerLayer))


def halves(prediction, panel):
    target = panel.targets('val'); scale = panel.fit_std[panel.target_indices]
    return [shared.scores(prediction[s], target[s], scale, panel.quantiles)[0]['score']
            for s in (slice(0, 14), slice(16, 30))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contract', required=True); parser.add_argument('--job', required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--forecast-fit')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(); started = time.perf_counter()
    contract = json.loads(Path(args.contract).read_text()); job = json.loads(args.job)
    for path, expected in contract['source_hashes'].items():
        assert sha(ROOT / path) == expected, path
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    forecast = args.forecast_fit is not None
    spec = contract['reference']['datasets'][job['dataset']]
    role = 'holdout' if forecast else 'fit'
    data_path = ROOT / spec[f'{role}_data_path']
    assert sha(data_path) == spec[f'{role}_data_sha256']
    panel = Panel(data_path, 'forecast' if forecast else 'fit', False)
    if not forecast:
        rows = np.asarray(job['training_rows'], dtype=int)
        assert len(panel.origins['train']) == 90 and len(rows) in (30, 90)
        assert np.all(np.diff(rows) > 0) and rows.min() >= 0 and rows.max() < 90
        panel.origins['train'] = panel.origins['train'][rows]
    runtime = SimpleNamespace(checkpoint=contract['reference']['checkpoint'], device='cuda', min_free_ram_gib=5.)
    check(runtime); base = shared.load_base(runtime)
    origins, _ = shared.padded_origins(next(iter(panel.origins.values()))[:4])
    context, _, groups = panel.batch(origins, 'cuda')
    with torch.no_grad(), shared.legacy.precision(runtime):
        enc, _, _, _ = base.encode(context=context, group_ids=groups, num_output_patches=3)
        reference = native.patch_to_quantiles(base.output_patch_embedding(enc.last_hidden_state[:, -3:]), 21, 16).float()
    model = Controlled(base, job, panel.count_channels)
    with torch.no_grad(), shared.legacy.precision(runtime):
        _, initial, _, _ = model.encode(context, groups, 48)
    identity = float((initial-reference).abs().max()); assert identity < 1e-5
    del enc, reference, initial, context, groups
    names = {n for n,p in model.named_parameters() if p.requires_grad}
    frozen_names = {n for n,p in model.named_parameters() if not p.requires_grad}
    adapter_names = {n for n in names if 'lora_' in n}
    head_names = names-adapter_names
    frozen_digest = digest(model, frozen_names)
    params = [p for p in model.parameters() if p.requires_grad]
    count = sum(p.numel() for p in params); assert count == 1768949
    if forecast:
        fit = Path(args.forecast_fit); record = json.loads((fit/'result.json').read_text())
        assert record['job'] == job and record['contract_sha256'] == sha(args.contract)
        assert sha(fit/'best.pt') == record['checkpoint_sha256']
        state = torch.load(fit/'best.pt', map_location='cpu', weights_only=True)
        assert set(state) == names
        restore(model, state); del state
        prediction, score = predict(model, panel, 'eval', runtime)
        assert digest(model, frozen_names) == frozen_digest
        np.savez_compressed(out/'predictions.npz', prediction=prediction, target=panel.targets('eval'), origins=panel.origins['eval'], quantiles=panel.quantiles, scale=panel.fit_std[panel.target_indices])
        save(out/'result.json', {'completed':True,'job':job,'score':score,'fit':str(fit),'seconds':time.perf_counter()-started,'contract_sha256':sha(args.contract)})
        return
    schedule = [0,1,2,3] if args.smoke else contract['schedules'][job['condition']]
    steps = schedule[-1]
    groups = [{'params':list(model.probe.parameters()),'lr':job['head_lr']},
              {'params':[p for n,p in model.named_parameters() if n in adapter_names],'lr':job['lora_lr']}]
    optimizer = torch.optim.AdamW(groups, weight_decay=0., foreach=False)
    assert {id(p) for g in groups for p in g['params']} == {id(p) for p in params}
    samples = np.random.default_rng(job['seed']).integers(len(panel.origins['train']), size=(steps,8))
    q = torch.as_tensor(panel.quantiles, dtype=torch.float32, device='cuda')
    initial_pred, initial_score = predict(model, panel, 'val', runtime)
    assert initial_score > 0
    best, best_step, best_pred, state = initial_score, 0, initial_pred, snapshot(model,names)
    initial_digest = digest(model,names)
    history = [{'step':0,'score':initial_score,'contribution_halves':[0.,0.],'utc':datetime.now(timezone.utc).isoformat()}]
    np.savez_compressed(out/'point_0.npz', prediction=initial_pred, target=panel.targets('val'), scale=panel.fit_std[panel.target_indices], quantiles=panel.quantiles)
    frozen_step = None; freeze_digest = None; head_at_freeze = None
    minimum = check(runtime); stop_reason = 'cap'; toggle_checks = 0
    for step in range(1,steps+1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        for offset in (0,4):
            origins = panel.origins['train'][samples[step-1,offset:offset+4]]
            context,target,group_ids = panel.batch(origins,'cuda')
            with shared.legacy.precision(runtime):
                _,norm,loc,scale = model.encode(context,group_ids,48)
                loss = native.native_pinball(norm,target,loc,scale,q,model.use_arcsinh)
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
            (loss/2).backward(); minimum = min(minimum,check(runtime))
            del context,target,group_ids,norm,loc,scale,loss
        grad = float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
        if step == 1: assert grad > 0
        optimizer.step()
        if step in schedule:
            prediction,score = predict(model,panel,'val',runtime)
            record = {'step':step,'score':score,'active_trainable':sum(p.numel() for p in model.parameters() if p.requires_grad)}
            arrays = {'prediction':prediction,'target':panel.targets('val'),'scale':panel.fit_std[panel.target_indices],'quantiles':panel.quantiles}
            if frozen_step is None and step < steps and (job['arm']=='CONTRIB_FREEZE' or args.smoke):
                before = digest(model,names)
                with adapters_off(model): off,off_score = predict(model,panel,'val',runtime)
                assert digest(model,names) == before
                assert {n for n,p in model.named_parameters() if p.requires_grad} == names
                if args.smoke:
                    replay,_ = predict(model,panel,'val',runtime)
                    np.testing.assert_array_equal(replay,prediction)
                toggle_checks += 1
                on_halves,off_halves = halves(prediction,panel),halves(off,panel)
                record.update(off_score=off_score,contribution=100*(off_score-score)/initial_score,
                              contribution_halves=[100*(b-a)/initial_score for a,b in zip(on_halves,off_halves)])
                arrays['off_prediction'] = off
            record['utc'] = datetime.now(timezone.utc).isoformat(); history.append(record)
            np.savez_compressed(out/f'point_{step}.npz',**arrays)
            if score < best: best,best_step,best_pred,state = score,step,prediction,snapshot(model,names)
            should_freeze = args.smoke and step == 1
            if not args.smoke and frozen_step is None and step < steps:
                if job['arm']=='FIXED_FREEZE': should_freeze = step >= steps//3
                elif job['arm']=='CONTRIB_FREEZE':
                    should_freeze, details = contribution_plateau(history,steps//3)
                    record['plateau_decision'] = should_freeze; record['slope_details'] = details
            if frozen_step is None and should_freeze:
                freeze_digest = digest(model,adapter_names); head_at_freeze = digest(model,head_names)
                freeze(model,optimizer,adapter_names); frozen_step = step
            save(out/'progress.json',{'step':step,'frozen_step':frozen_step,'elapsed':time.perf_counter()-started})
            if job['arm']=='ES2' and not args.smoke and early_stop(history):
                stop_reason='validation_patience2'; break
        minimum = min(minimum,check(runtime))
    assert digest(model,frozen_names) == frozen_digest and digest(model,names) != initial_digest
    tail_verified = frozen_step is not None
    if tail_verified:
        assert digest(model,adapter_names) == freeze_digest
        assert digest(model,head_names) != head_at_freeze
    restore(model,state)
    replay,replay_score = predict(model,panel,'val',runtime)
    np.testing.assert_array_equal(replay,best_pred); assert abs(replay_score-best)<1e-10
    assert set(state)==names
    torch.save(state,out/'best.pt')
    np.savez_compressed(out/'val_predictions.npz',prediction=best_pred,target=panel.targets('val'),scale=panel.fit_std[panel.target_indices],quantiles=panel.quantiles)
    save(out/'result.json',{'completed':True,'job':job,'best_score':best,'best_step':best_step,'history':history,
         'steps':step,'cap':steps,'stop_reason':stop_reason,'frozen_step':frozen_step,'adapter_updates':frozen_step or step,
         'frozen_tail_verified':tail_verified,'toggle_checks':toggle_checks,'trainable':count,'snapshot_names':sorted(names),
         'initial_head_hash':model.initial_head_hash,'identity_error':identity,'frozen_hash':frozen_digest,'frozen_verified':True,
         'checkpoint_sha256':sha(out/'best.pt'),'replay_verified':True,'minimum_commit_gib':minimum,
         'max_cuda_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'seconds':time.perf_counter()-started,
         'contract_sha256':sha(args.contract),'holdout_opened':False,
         'sample_index_sha256':hashlib.sha256(samples[:step].tobytes()).hexdigest()})
    print(json.dumps({'done':job,'best_step':best_step,'freeze_step':frozen_step,'steps':step,'seconds':time.perf_counter()-started}),flush=True)


if __name__=='__main__': main()
