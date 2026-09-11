"""Same-state continuation forks with checkpoint-selection and clipping controls."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch
from experiments.peft_contribution_freeze_v1.fit import (
    Controlled, check, predict, shared, native, Panel, digest, snapshot, restore,
    adapters_off, freeze, sha, save,
)

ROOT = Path(__file__).resolve().parents[2]


def cpu_copy(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list): return [cpu_copy(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def state_hash(value):
    h = hashlib.sha256()
    def add(v):
        if isinstance(v, torch.Tensor):
            x = v.detach().cpu().contiguous()
            h.update(str((tuple(x.shape), x.dtype)).encode()); h.update(x.numpy().tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=repr): h.update(repr(k).encode()); add(v[k])
        elif isinstance(v, (tuple, list)):
            h.update(type(v).__name__.encode())
            for x in v: add(x)
        else: h.update(repr(v).encode())
    add(value)
    return h.hexdigest()


def rng_state():
    return {'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def optimizer_for(model, adapter_names, job):
    return torch.optim.AdamW([
        {'params': list(model.probe.parameters()), 'lr': job['head_lr']},
        {'params': [p for n,p in model.named_parameters() if n in adapter_names], 'lr': job['lora_lr']},
    ], weight_decay=0., foreach=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True); parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(); started = time.perf_counter()
    plan = json.loads(Path(args.plan).read_text()); entry = plan['jobs'][args.index]; job = entry['job']
    for p,h in plan['source_hashes'].items(): assert sha(ROOT/p) == h, p
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    data = ROOT/plan['fit_data'][job['dataset']]['path']
    assert sha(data) == plan['fit_data'][job['dataset']]['sha256']
    panel = Panel(data, 'fit', False)
    assert panel.origins['val'][13]+panel.horizon <= panel.origins['val'][16]
    panel.origins['train'] = panel.origins['train'][np.asarray(job['training_rows'], dtype=int)]
    runtime = SimpleNamespace(checkpoint=plan['checkpoint'], device='cuda', min_free_ram_gib=5.)
    check(runtime); model = Controlled(shared.load_base(runtime), job, panel.count_channels)
    names = {n for n,p in model.named_parameters() if p.requires_grad}
    native_names = {n for n,p in model.named_parameters() if not p.requires_grad}
    adapter_names = {n for n in names if 'lora_' in n}; head_names = names-adapter_names
    assert len(names) == 196 and len(adapter_names) == 192
    native_hash = digest(model, native_names)
    params = [p for p in model.parameters() if p.requires_grad]
    schedule = [0,1,2,3] if args.smoke else plan['schedules'][job['condition']]
    cap = schedule[-1]; forks = [cap//3, 2*cap//3]
    samples = np.random.default_rng(job['seed']).integers(len(panel.origins['train']), size=(cap,8))
    q = torch.as_tensor(panel.quantiles, dtype=torch.float32, device='cuda')
    target = panel.targets('val'); scale = panel.fit_std[panel.target_indices]
    old = ROOT/entry['reference_fit']/'output'
    saved = {}; first_joint = {}; results = []; prefix_history = []

    def scores(pred):
        return {label: shared.scores(pred[sl], target[sl], scale, panel.quantiles)[0]['score']
                for label,sl in (('S',slice(0,14)),('D',slice(16,30)))}

    def evaluate(folder, step, observe=False):
        folder.mkdir(parents=True, exist_ok=True)
        pred,_ = predict(model, panel, 'val', runtime)
        point = {'step': step, **scores(pred)}
        arrays = {'prediction': pred, 'target': target, 'quantiles': panel.quantiles, 'scale': scale}
        if observe:
            before = digest(model, names)
            with adapters_off(model): off,_ = predict(model, panel, 'val', runtime)
            assert digest(model,names) == before
            assert {n for n,p in model.named_parameters() if p.requires_grad} == names
            point['S_off'] = scores(off)['S']; arrays['off_prediction'] = off
        np.savez_compressed(folder/f'point_{step}.npz', **arrays)
        point['model_hash'] = digest(model,names)
        return point, pred

    def step_update(optimizer, step, mode):
        model.train(); optimizer.zero_grad(set_to_none=True)
        for offset in (0,4):
            origins = panel.origins['train'][samples[step-1,offset:offset+4]]
            context,y,group_ids = panel.batch(origins,'cuda')
            with shared.legacy.precision(runtime):
                _,norm,loc,sc = model.encode(context,group_ids,48)
                loss = native.native_pinball(norm,y,loc,sc,q,model.use_arcsinh)
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
            (loss/2).backward(); check(runtime)
            del context,y,group_ids,norm,loc,sc,loss
        grad = float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
        if mode == 'MASKED_UPDATE':
            for n,p in model.named_parameters():
                if n in adapter_names: p.grad = None
        optimizer.step()
        return grad

    optimizer = optimizer_for(model,adapter_names,job)
    point,pred = evaluate(out/'joint',0)
    assert point['S'] > 0 and point['D'] > 0
    with np.load(old/'point_0.npz', allow_pickle=False) as z: np.testing.assert_array_equal(pred,z['prediction'])
    point['S_off'] = point['S']; prefix_history.append(point)
    first = dict(point)
    for step in range(1,cap+1):
        grad = step_update(optimizer,step,'JOINT')
        if step-1 in forks:
            first_joint[step-1] = {'grad_norm':grad,'head_hash':digest(model,head_names)}
        if step in schedule:
            point,pred = evaluate(out/'joint',step,observe=step<cap)
            if not args.smoke:
                with np.load(old/f'point_{step}.npz', allow_pickle=False) as z:
                    np.testing.assert_array_equal(pred,z['prediction'])
            prefix_history.append(point)
            if step in forks:
                state = {'weights':snapshot(model,names),'optimizer':cpu_copy(optimizer.state_dict()),'rng':cpu_copy(rng_state())}
                state['head_optimizer_hash'] = state_hash({n:optimizer.state[p] for n,p in model.named_parameters() if n in head_names})
                state['model_hash'] = digest(model,names); state['optimizer_hash'] = state_hash(state['optimizer'])
                state['rng_hash'] = state_hash(state['rng']); saved[step] = state
                torch.save(state,out/f'fork_{step}.pt')
            save(out/'progress.json',{'stage':'JOINT','step':step,'seconds':time.perf_counter()-started})

    for fork in forks:
        prefix = [h for h in prefix_history if h['step'] <= fork]
        fork_record = {'fork':fork,'prefix_model_hash':saved[fork]['model_hash'],
                       'optimizer_hash':saved[fork]['optimizer_hash'],'rng_hash':saved[fork]['rng_hash'],
                       'head_optimizer_hash':saved[fork]['head_optimizer_hash'],
                       'suffix_sample_sha256':hashlib.sha256(samples[fork:].tobytes()).hexdigest(),
                       'branches':{'JOINT':{'history':prefix_history,'selected':min(prefix_history,key=lambda h:h['S']),
                                           'final':prefix_history[-1]}}}
        for mode in (('HEAD_ONLY','MASKED_UPDATE','JOINT_REPLAY') if args.smoke else ('HEAD_ONLY','MASKED_UPDATE')):
            for n,p in model.named_parameters(): p.requires_grad_(n in names); p.grad = None
            state = saved[fork]; restore(model,state['weights'])
            optimizer = optimizer_for(model,adapter_names,job)
            optimizer.load_state_dict(cpu_copy(state['optimizer']))
            torch.set_rng_state(state['rng']['cpu']); torch.cuda.set_rng_state_all(state['rng']['cuda'])
            assert digest(model,names) == state['model_hash']
            assert state_hash(optimizer.state_dict()) == state['optimizer_hash']
            assert state_hash(rng_state()) == state['rng_hash']
            adapter_before = digest(model,adapter_names); head_before = digest(model,head_names)
            if mode == 'HEAD_ONLY': freeze(model,optimizer,adapter_names)
            assert state_hash({n:optimizer.state[p] for n,p in model.named_parameters() if n in head_names}) == state['head_optimizer_hash']
            folder = out/f'fork_{fork}'/mode
            point,pred = evaluate(folder,fork)
            with np.load(out/'joint'/f'point_{fork}.npz',allow_pickle=False) as z: np.testing.assert_array_equal(pred,z['prediction'])
            history = copy.deepcopy(prefix); clipped = 0; first_update = None
            for step in range(fork+1,cap+1):
                grad = step_update(optimizer,step,mode); clipped += grad>1.
                if step == fork+1:
                    first_update = {'grad_norm':grad,'head_hash':digest(model,head_names)}
                    if mode in ('MASKED_UPDATE','JOINT_REPLAY'): assert first_update == first_joint[fork]
                if step in schedule:
                    point,pred = evaluate(folder,step)
                    if mode == 'JOINT_REPLAY':
                        with np.load(out/'joint'/f'point_{step}.npz',allow_pickle=False) as z: np.testing.assert_array_equal(pred,z['prediction'])
                    history.append(point)
                    save(out/'progress.json',{'stage':mode,'fork':fork,'step':step,'seconds':time.perf_counter()-started})
            if mode != 'JOINT_REPLAY':
                assert digest(model,adapter_names) == adapter_before
                assert digest(model,head_names) != head_before
            assert digest(model,native_names) == native_hash
            selected = min(history,key=lambda h:h['S'])
            branch = {'history':history,'selected':selected,'final':history[-1],
                      'clipped_steps':clipped,'first_update':first_update,'restore_verified':True,
                      'adapter_unchanged':mode!='JOINT_REPLAY','head_changed':True}
            fork_record['branches'][mode] = branch
        results.append(fork_record)
    assert digest(model,native_names) == native_hash
    save(out/'result.json',{'completed':True,'job':job,'smoke':args.smoke,'plan_sha256':sha(args.plan),
         'cap':cap,'S_rows':list(range(14)),'D_rows':list(range(16,30)),'initial':first,'forks':results,
         'prefix_exact_study31':not args.smoke,'native_frozen_hash':native_hash,'holdout_opened':False,
         'sample_index_sha256':hashlib.sha256(samples.tobytes()).hexdigest(),
         'seconds':time.perf_counter()-started,'finished_utc':datetime.now(timezone.utc).isoformat()})
    print(json.dumps({'completed':True,'dataset':job['dataset'],'condition':job['condition'],'seed':job['seed']}),flush=True)


if __name__ == '__main__': main()
