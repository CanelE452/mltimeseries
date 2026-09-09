"""Guarded study20 fits; holdout files are never opened in this module."""

import argparse
import gc
import json
import math
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_variable] = "2"

import numpy as np
import psutil
import torch

from experiments.peft_external_gap_v1 import train as shared
from .contract import ROOT, STUDY, digest, read_contract, save_json
from . import model as modeling


def absolute(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT/path).resolve()


def new_output(path):
    output = absolute(path)
    if not output.is_relative_to(ROOT/"runs"/STUDY):
        raise ValueError("Outputs must remain inside this study's runs directory")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Preserve existing complete or partial trial: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def runtime(contract):
    return SimpleNamespace(checkpoint=str(absolute(contract["checkpoint"])), device="cuda",
                           min_free_ram_gib=5.)


def predict(model, panel, split, args):
    model.eval()
    blocks = []
    with torch.no_grad():
        for index in range(0, len(panel.origins[split]), 4):
            shared.legacy.check_resources(args)
            selected, count = shared.padded_origins(panel.origins[split][index:index+4])
            context, _, groups = panel.batch(selected, args.device)
            with shared.legacy.precision(args):
                _, raw, _, _ = modeling.forward(model, context, groups, panel.horizon)
            value = raw.float().cpu().numpy().reshape(4, panel.count_channels, 21, 48)
            blocks.append(value[:count, panel.target_indices])
    unsorted = np.concatenate(blocks)
    if not np.isfinite(unsorted).all():
        raise FloatingPointError("Nonfinite native forecast")
    return np.sort(unsorted, axis=2), unsorted


def first_update_deltas(model, initial_state):
    squares = {"encoder": 0., "head": 0., "other": 0.}
    for name, parameter in model.named_parameters():
        if name not in initial_state:
            continue
        key = "encoder" if name.startswith("base.encoder.") else "head" if name.startswith("base.output_patch_embedding.") else "other"
        difference = parameter.detach().cpu().double() - initial_state[name].double()
        squares[key] += float(torch.sum(difference.square()))
    return {name: math.sqrt(value) for name, value in squares.items()}


def validate_fit(args, contract):
    settings = contract["settings"]
    if args.dataset not in contract["datasets"] or args.arm not in ("HEAD_ONLY", "LORA", "FULL_FT"):
        raise ValueError("Unknown frozen fit job")
    if args.seed not in settings["seeds"]:
        raise ValueError("Seed is outside the frozen grid")
    rates = [settings["smoke_lr"][args.arm]] if args.smoke else settings["lr_grids"][args.arm]
    if args.lr not in rates or (args.smoke and args.seed != settings["seeds"][0]):
        raise ValueError("Learning rate or smoke seed is outside the frozen grid")
    for field, value in {"steps":200,"val_every":40,"effective_batch":8,"micro_batch":4,"rank":8,"alpha":16,"threads":2}.items():
        if settings[field] != value:
            raise ValueError(f"Reference implementation requires {field}={value}")
    return settings["smoke_steps"] if args.smoke else settings["steps"], settings["smoke_steps"] if args.smoke else settings["val_every"]


def execute(args):
    started = time.perf_counter()
    contract_path = absolute(args.contract)
    contract = read_contract(contract_path, verify="core")
    steps, interval = validate_fit(args, contract)
    output = new_output(args.output)
    device_args = runtime(contract)
    shared.legacy.check_resources(device_args)
    dataset = contract["datasets"][args.dataset]
    fit_path = absolute(dataset["fit_data_path"])
    save_json(output/"trial_contract.json", {"contract_sha256":digest(contract_path), "dataset":args.dataset,
              "arm":args.arm,"seed":args.seed,"lr":args.lr,"smoke":args.smoke,
              "fit_data_path":str(fit_path),"fit_data_sha256":digest(fit_path),"holdout_file_opened":False})
    panel = shared.Panel(fit_path,"fit",args.smoke)
    if panel.dataset != args.dataset:
        raise ValueError("Fit archive dataset differs from requested job")
    timings = {}
    rss_max = psutil.Process().memory_info().rss
    tick = time.perf_counter()
    base = shared.load_base(device_args)
    timings["load"] = time.perf_counter()-tick
    quantiles = np.asarray(base.chronos_config.quantiles,dtype=np.float64)
    if not np.array_equal(quantiles,panel.quantiles):
        raise AssertionError("Model and prepared quantiles differ")
    initial_native_count = sum(p.numel() for p in base.parameters())
    if initial_native_count != modeling.BASE_PARAMETERS:
        raise AssertionError("Unexpected native parameter count")
    audits = shared.information_audit(base,panel,device_args) if args.smoke else {}
    probe_origins, _ = shared.padded_origins(panel.origins["train"][:4])
    probe_context, _, probe_groups = panel.batch(probe_origins,device_args.device)
    with torch.no_grad(), shared.legacy.precision(device_args):
        encoded, (loc,scale), _, _ = base.encode(context=probe_context,group_ids=probe_groups,num_output_patches=3)
        native_norm = shared.native.patch_to_quantiles(base.output_patch_embedding(encoded.last_hidden_state[:,-3:]),21,16).float()
    model = modeling.construct(base,args.arm,args.seed,panel.count_channels)
    scope = modeling.audit_scope(model,args.arm)
    with torch.no_grad(), shared.legacy.precision(device_args):
        actual_norm, _, _, _ = modeling.forward(model,probe_context,probe_groups)
    identity_error = float((native_norm-actual_norm).abs().max())
    if not math.isfinite(identity_error) or identity_error > 1e-5:
        raise AssertionError("Adaptation initialization differs from F0")
    step0_probe_hash = shared.array_hash(native_norm.cpu().numpy())
    del native_norm, actual_norm, encoded, probe_context, probe_groups
    initial_hash = modeling.parameter_digest(model,trainable=True)
    initial_all_hash = modeling.parameter_digest(model)
    frozen_hash = modeling.parameter_digest(model,trainable=False) if scope["frozen_parameters"] else None
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters,lr=args.lr,weight_decay=0.,foreach=False)
    if {id(p) for g in optimizer.param_groups for p in g["params"]} != {id(p) for p in parameters}:
        raise AssertionError("Optimizer registration differs from exact update scope")
    samples = np.random.default_rng(args.seed).integers(len(panel.origins["train"]),size=(steps,8))
    sampler_hash = shared.array_hash(panel.origins["train"][samples])
    q = torch.as_tensor(quantiles,dtype=torch.float32,device=device_args.device)
    target = panel.targets("val")
    tick = time.perf_counter()
    prediction, unsorted = predict(model,panel,"val",device_args)
    metrics, _, _ = shared.scores(prediction,target,panel.fit_std[panel.target_indices],quantiles)
    timings["validation"] = time.perf_counter()-tick
    best, best_step = metrics["score"], 0
    step0_val_hash = shared.array_hash(prediction)
    best_prediction = prediction.copy()
    best_state = modeling.snapshot_trainable(model)
    history = [{"step":0,"val_score":best}]
    shared.log(output,"validation",**history[-1])
    durations, gradients, losses = [], [], []
    for step in range(1,steps+1):
        shared.legacy.check_resources(device_args)
        rss_max = max(rss_max,psutil.Process().memory_info().rss)
        tick = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.
        for offset in (0,4):
            origins = panel.origins["train"][samples[step-1,offset:offset+4]]
            context, batch_target, groups = panel.batch(origins,device_args.device)
            with shared.legacy.precision(device_args):
                norm, _, loc, scale = modeling.forward(model,context,groups)
                loss = shared.native.native_pinball(norm,batch_target,loc,scale,q,model.use_arcsinh)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite native training loss")
            (loss/2).backward()
            total_loss += float(loss.detach())/2
        gradient = torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        gradients.append(float(gradient))
        optimizer.step()
        torch.cuda.synchronize()
        durations.append(time.perf_counter()-tick)
        losses.append(total_loss)
        if step == 1:
            delta = first_update_deltas(model,best_state)
            if args.arm == "FULL_FT" and (delta["encoder"] <= 0 or delta["head"] <= 0):
                raise AssertionError("FULL_FT did not actually update encoder and head")
            audits["first_update_l2"] = delta
        if step % interval == 0 or step == steps:
            tick = time.perf_counter()
            prediction, unsorted = predict(model,panel,"val",device_args)
            metrics, _, _ = shared.scores(prediction,target,panel.fit_std[panel.target_indices],quantiles)
            if metrics["score"] < best:
                best, best_step = metrics["score"], step
                best_prediction = prediction.copy()
                del best_state
                best_state = modeling.snapshot_trainable(model)
            timings["validation"] += time.perf_counter()-tick
            history.append({"step":step,"val_score":metrics["score"],"train_loss":total_loss,"gradient_norm":float(gradient)})
            shared.log(output,"validation",**history[-1],best_step=best_step)
    final_hash = modeling.parameter_digest(model,trainable=True)
    if max(gradients) <= 0 or final_hash == initial_hash:
        raise AssertionError("No finite nonzero training update")
    if frozen_hash is not None and modeling.parameter_digest(model,trainable=False) != frozen_hash:
        raise AssertionError("Frozen parameters changed")
    optimizer.zero_grad(set_to_none=True)
    del optimizer, parameters, context, batch_target, groups, norm, loss
    gc.collect()
    tick = time.perf_counter()
    checkpoint = output/"best_trainable.pt"
    modeling.write_checkpoint(best_state,checkpoint)
    del best_state
    reloaded = torch.load(checkpoint,map_location="cpu",weights_only=True)
    modeling.restore_trainable(model,reloaded)
    del reloaded
    timings["checkpoint_write_restore"] = time.perf_counter()-tick
    restored_hash = modeling.parameter_digest(model,trainable=True)
    tick = time.perf_counter()
    prediction, unsorted = predict(model,panel,"val",device_args)
    metrics, sums, counts = shared.scores(prediction,target,panel.fit_std[panel.target_indices],quantiles)
    if not np.allclose(prediction,best_prediction,rtol=1e-6,atol=1e-6) or not np.isclose(metrics["score"],best,rtol=1e-6,atol=1e-8):
        raise AssertionError("Restored checkpoint differs from selected validation predictions")
    timings["restore_validation"] = time.perf_counter()-tick
    predictions_path = output/"Vpredictions.npz"
    np.savez_compressed(predictions_path,predictions=prediction,unsorted_predictions=unsorted,target=target,
                        origins=panel.origins["val"],loss_sums=sums,valid_counts=counts)
    rss_max = max(rss_max,psutil.Process().memory_info().rss)
    audits.update(zero_update_identity=True,zero_update_max_abs=identity_error,trainable_map_verified=True,
                  optimizer_exact_parameter_set=True,finite_nonzero_gradient_verified=True,checkpoint_reload_verified=True,
                  frozen_parameters_verified=True if frozen_hash is not None else None,
                  frozen_check_applicable=frozen_hash is not None,full_model_update_scope_verified=args.arm=="FULL_FT")
    timings.update(optimizer=sum(durations),total=time.perf_counter()-started)
    result = {"completed":True,"study":STUDY,"stage":"fit","dataset":args.dataset,"arm":args.arm,
              "seed":args.seed,"lr":args.lr,"smoke":args.smoke,"steps_completed":steps,"best_step":best_step,
              "val_score":metrics["score"],"val_metrics":metrics,"trainable":scope["trainable"],
              "total_parameters":scope["total_parameters"],"native_parameters":initial_native_count,
              "trainable_names":scope["trainable_names"],"module_map":model.module_map,"sampler_sha256":sampler_hash,
              "checkpoint_sha256":digest(checkpoint),"Vpredictions_sha256":digest(predictions_path),
              "fit_data_sha256":digest(fit_path),"contract_sha256":digest(contract_path),"holdout_file_opened":False,
              "step0_predictions_sha256":step0_val_hash,"step0_probe_predictions_sha256":step0_probe_hash,
              "step0_parameters_sha256":initial_all_hash,"initial_trainable_sha256":initial_hash,
              "final_training_parameters_sha256":final_hash,"restored_trainable_sha256":restored_hash,
              "frozen_parameters_sha256":frozen_hash,"history":history,"training_losses":losses,"gradient_norms":gradients,
              "seconds":timings,"optimizer_seconds_per_step":float(np.mean(durations)),
              "peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_cuda_reserved_bytes":torch.cuda.max_memory_reserved(),
              "rss_sample_max_bytes":rss_max,"rss_sampling":"child process RSS sampled at step boundaries and final; not continuous peak",
              "audits":audits,"checkpoint_policy":"single final write of CPU best trainable state; no optimizer/RNG resume state"}
    save_json(output/"result.json",result)
    shared.log(output,"fit_completed",best_step=best_step,val_score=metrics["score"])
    return result


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract",type=Path,required=True)
    parser.add_argument("--dataset",choices=("bike","household"),required=True)
    parser.add_argument("--arm",choices=("HEAD_ONLY","LORA","FULL_FT"),required=True)
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--lr",type=float,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--smoke",action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = cli()
    try:
        execute(arguments)
    except Exception as error:
        failure = absolute(arguments.output)/"failure.json"
        if failure.parent.exists() and failure.parent.is_relative_to(ROOT/"runs"/STUDY) and not failure.exists():
            save_json(failure,{"completed":False,"error":f"{type(error).__name__}: {error}","traceback":traceback.format_exc()})
        raise
