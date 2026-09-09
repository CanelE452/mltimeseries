"""Selection-gated C/E forecasts with no optimizer or model updates."""

import argparse
import gc
import json
import os
from pathlib import Path
import time
import traceback

for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_variable] = "2"

import numpy as np
import psutil
import torch

from experiments.peft_external_gap_v1 import train as shared
from .contract import ROOT, STUDY, digest, read_contract, save_json
from . import model as modeling
from .train import absolute, new_output, predict, runtime


def authorize_selection(contract_path, contract, dataset, arm, fit_dir):
    selection_path = contract_path.parent/"selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (selection.get("completed") is not True or selection.get("global_choices_frozen") is not True
            or selection.get("contract_sha256") != digest(contract_path)):
        raise AssertionError("Global model selection is not frozen under this contract")
    if dataset not in contract["datasets"]:
        raise ValueError("Unknown dataset")
    if arm == "F0":
        if fit_dir is not None:
            raise ValueError("F0 cannot load an adapted checkpoint")
        return selection_path,None,None
    if fit_dir is None:
        raise ValueError("Adapted forecasts require the selected fit directory")
    fit_dir = absolute(fit_dir)
    if not fit_dir.is_relative_to(ROOT/"runs"/STUDY):
        raise ValueError("Fit directory must belong to study20")
    matches = [entry for entry in selection["selected"] if entry["dataset"] == dataset and entry["arm"] == arm
               and absolute(entry["fit_dir"]) == fit_dir]
    if len(matches) != 1:
        raise AssertionError("Requested fit is not a unique globally selected trial")
    entry = matches[0]
    fit_result_path, checkpoint = fit_dir/"result.json",fit_dir/"best_trainable.pt"
    if digest(fit_result_path) != entry["fit_result_sha256"] or digest(checkpoint) != entry["checkpoint_sha256"]:
        raise AssertionError("Selected fit artifacts changed")
    fit = json.loads(fit_result_path.read_text(encoding="utf-8"))
    if (fit.get("completed") is not True or fit.get("smoke") is not False or fit.get("stage") != "fit"
            or fit.get("contract_sha256") != digest(contract_path) or fit.get("holdout_file_opened") is not False
            or (fit["dataset"],fit["arm"],fit["seed"]) != (dataset,arm,entry["seed"])
            or fit["steps_completed"] != contract["settings"]["steps"]
            or fit["lr"] not in contract["settings"]["lr_grids"][arm]
            or fit["seed"] not in contract["settings"]["seeds"]):
        raise AssertionError("Selected fit result violates the training contract")
    for flag in ("zero_update_identity","trainable_map_verified","optimizer_exact_parameter_set",
                 "finite_nonzero_gradient_verified","checkpoint_reload_verified"):
        if fit["audits"].get(flag) is not True:
            raise AssertionError(f"Selected fit failed audit: {flag}")
    if arm == "FULL_FT" and fit["audits"].get("full_model_update_scope_verified") is not True:
        raise AssertionError("Selected full-FT run lacks whole-model scope verification")
    if arm in ("HEAD_ONLY","LORA") and (fit["audits"].get("frozen_parameters_verified") is not True
                                         or fit["audits"].get("frozen_check_applicable") is not True):
        raise AssertionError("Selected PEFT run lacks frozen-backbone verification")
    return selection_path,entry,fit


def execute(args):
    started = time.perf_counter()
    contract_path = absolute(args.contract)
    contract = read_contract(contract_path,verify="core")
    selection_path, entry, fit = authorize_selection(contract_path,contract,args.dataset,args.arm,args.fit_dir)
    output = new_output(args.output)
    holdout_path = absolute(contract["datasets"][args.dataset]["holdout_data_path"])
    selection_hash = digest(selection_path)
    save_json(output/"forecast_contract.json",{"contract_sha256":digest(contract_path),"selection_sha256":selection_hash,
              "dataset":args.dataset,"arm":args.arm,"selected_fit":entry,"selection_verified_before_holdout_load":True})
    if digest(holdout_path) != contract["datasets"][args.dataset]["holdout_data_sha256"]:
        raise AssertionError("Holdout archive differs from frozen contract")
    panel = shared.Panel(holdout_path,"forecast")
    if panel.dataset != args.dataset:
        raise AssertionError("Holdout archive differs from requested dataset")
    device_args = runtime(contract)
    shared.legacy.check_resources(device_args)
    seed = contract["settings"]["seeds"][0] if fit is None else fit["seed"]
    timings = {}
    rss_max = psutil.Process().memory_info().rss
    tick = time.perf_counter()
    base = shared.load_base(device_args)
    if sum(p.numel() for p in base.parameters()) != modeling.BASE_PARAMETERS:
        raise AssertionError("Native model parameter count differs")
    model = modeling.construct(base,args.arm,seed,panel.count_channels)
    timings["load"] = time.perf_counter()-tick
    quantiles = np.asarray(base.chronos_config.quantiles,dtype=np.float64)
    if not np.array_equal(quantiles,panel.quantiles):
        raise AssertionError("Model and data quantile grids differ")
    if fit is not None:
        tick = time.perf_counter()
        state = torch.load(absolute(args.fit_dir)/"best_trainable.pt",map_location="cpu",weights_only=True,mmap=True)
        modeling.restore_trainable(model,state)
        del state
        gc.collect()
        if modeling.parameter_digest(model,trainable=True) != fit["restored_trainable_sha256"]:
            raise AssertionError("Forecast checkpoint differs from fitted best state")
        timings["checkpoint_restore"] = time.perf_counter()-tick
    initial_hash = modeling.parameter_digest(model)
    metrics = {}
    prediction_hashes = {}
    for split, letter in (("cal","C"),("eval","E")):
        tick = time.perf_counter()
        prediction,unsorted = predict(model,panel,split,device_args)
        target = panel.targets(split)
        score,sums,counts = shared.scores(prediction,target,panel.fit_std[panel.target_indices],quantiles)
        path = output/f"{letter}predictions.npz"
        np.savez_compressed(path,predictions=prediction,unsorted_predictions=unsorted,target=target,
                            origins=panel.origins[split],loss_sums=sums,valid_counts=counts)
        timings[f"{split}_forecast"] = time.perf_counter()-tick
        prediction_hashes[f"{letter}predictions_sha256"] = digest(path)
        metrics[split] = score
        rss_max = max(rss_max,psutil.Process().memory_info().rss)
    if modeling.parameter_digest(model) != initial_hash:
        raise AssertionError("Model changed during forecast")
    if digest(selection_path) != selection_hash:
        raise AssertionError("Global selection changed during forecast")
    timings["total"] = time.perf_counter()-started
    scope = modeling.audit_scope(model,args.arm)
    result = {"completed":True,"study":STUDY,"stage":"forecast","dataset":args.dataset,"arm":args.arm,"seed":seed,
              "optimizer_steps":0,"trainable":scope["trainable"],"total_parameters":scope["total_parameters"],
              "fit_dir":str(absolute(args.fit_dir)) if fit else None,"fit_result_sha256":entry["fit_result_sha256"] if entry else None,
              "checkpoint_sha256":entry["checkpoint_sha256"] if entry else None,"best_step":fit["best_step"] if fit else 0,
              "val_score":fit["val_score"] if fit else None,"cal_score":metrics["cal"]["score"],"eval_score":metrics["eval"]["score"],
              "cal_metrics":metrics["cal"],"eval_metrics":metrics["eval"],"selection_sha256":selection_hash,
              "contract_sha256":digest(contract_path),"holdout_data_sha256":digest(holdout_path),**prediction_hashes,
              "seconds":timings,"peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(),
              "peak_cuda_reserved_bytes":torch.cuda.max_memory_reserved(),"rss_sample_max_bytes":rss_max,
              "rss_sampling":"child RSS sampled at load and after each split; not continuous peak",
              "audits":{"model_unchanged":True,"model_parameters_sha256":initial_hash,
                        "selection_verified_before_holdout_load":True,"checkpoint_reload_verified":fit is not None,
                        "C_used_for_model_selection":False,"E_used_for_model_selection":False},
              "primary":"SORT21 native quantiles; C saved only for common secondary calibration"}
    save_json(output/"result.json",result)
    shared.log(output,"forecast_completed",eval_score=result["eval_score"])
    return result


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract",type=Path,required=True)
    parser.add_argument("--dataset",choices=("bike","household"),required=True)
    parser.add_argument("--arm",choices=modeling.ARMS,required=True)
    parser.add_argument("--fit-dir",type=Path)
    parser.add_argument("--output",type=Path,required=True)
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
