"""Save every evaluation forecast without opening the evaluation truth file."""

import argparse
from pathlib import Path
import time
import traceback

from .cache import (ROOT, STUDY, validate_stage, read_json, fresh_output, native_args, load_base,
                    check_resources, precision, parameter_hash, atomic_json, file_hash)
from .model import PointModel
from .ridge import _coarse_features
from .train import SEED, load_trainable

import numpy as np
import torch


def run(args):
    from .run_study import verify_selection

    started = time.perf_counter()
    contract_path, contract = validate_stage(args)
    selection = verify_selection(contract)
    selection_sha = file_hash(contract["paths"]["selection"])
    output = Path(args.output).resolve()
    if output != Path(contract["paths"]["forecast_result"]).parent:
        raise ValueError("Forecast output differs from the contract")
    output = fresh_output(output)
    try:
        with np.load(contract["data"]["evaluation_inputs"]["path"], allow_pickle=False) as z:
            if set(z.files) != {"context", "horizon", "scale", "site", "target_id", "month", "profile"}:
                raise AssertionError("Evaluation inputs contain unexpected information")
            panel = {key: z[key].copy() for key in z.files}
        if panel["context"].shape != (48, 512) or set(panel["month"].tolist()) != {"2017-04", "2017-05", "2017-06"}:
            raise AssertionError("Evaluation panel does not match the frozen months")
        if not np.isfinite(panel["context"]).all() or not np.isfinite(panel["scale"]).all() or np.any(panel["scale"] <= 0):
            raise ValueError("Invalid evaluation inputs")
        with np.load(contract["paths"]["ridge_head"], allow_pickle=False) as z:
            weight, bias = z["weight"].copy(), z["bias"].copy()
        with np.load(contract["paths"]["ridge_coarse"], allow_pickle=False) as z:
            coarse_weight = z["weight"].copy()
        settings = native_args(contract)
        torch.manual_seed(SEED)
        base = load_base(settings)
        model = PointModel(base, "HEAD").load_point_head(weight, bias)
        frozen = parameter_hash(model.named_parameters())
        predictions = {arm: np.full((48, 744), np.nan, dtype=np.float64) for arm in contract["arms"]}
        predictions["PROFILE"] = panel["profile"].astype(np.float64)
        groups = torch.zeros(1, dtype=torch.long, device=settings.device)
        with torch.no_grad():
            for index, horizon in enumerate(panel["horizon"]):
                check_resources(settings)
                h = int(horizon)
                context = torch.as_tensor(panel["context"][index:index + 1], device=settings.device)
                with precision(settings):
                    hidden, point, _, scale = model.encode(context, groups, h)
                with torch.autocast(device_type="cuda", enabled=False):
                    residual = model.point_head(hidden.float()).flatten(1)
                predictions["F0"][index, :h] = point[0, :h].float().cpu().numpy()
                predictions["FROZEN_HEAD"][index, :h] = (point + scale * residual)[0, :h].float().cpu().numpy()
        if parameter_hash(model.named_parameters()) != frozen:
            raise AssertionError("Baseline inference changed weights")
        features = _coarse_features(panel, predictions["F0"])
        for index, horizon in enumerate(panel["horizon"]):
            h = int(horizon)
            mean = float(features[index] @ coarse_weight[int(panel["site"][index])]) * panel["scale"][index]
            predictions["COARSE_LIFT"][index, :h] = predictions["F0"][index, :h] + mean - predictions["F0"][index, :h].mean()
        del model
        model = PointModel(base, "ATTN_LORA_FIXED_HEAD", seed=SEED).load_point_head(weight, bias)
        load_trainable(model, selection["best_trainable_path"], settings.device)
        restored = parameter_hash(model.named_parameters())
        with torch.no_grad():
            for index, horizon in enumerate(panel["horizon"]):
                check_resources(settings)
                h = int(horizon)
                context = torch.as_tensor(panel["context"][index:index + 1], device=settings.device)
                with precision(settings):
                    point = model(context, groups, h)
                predictions["ATTN_LORA"][index, :h] = point[0].float().cpu().numpy()
        if parameter_hash(model.named_parameters()) != restored:
            raise AssertionError("LoRA inference changed weights")
        for arm, values in predictions.items():
            for i, h in enumerate(panel["horizon"]):
                if not np.isfinite(values[i, :int(h)]).all():
                    raise FloatingPointError(f"Nonfinite evaluation forecast: {arm}/{i}")
        np.savez(output / "predictions.npz", **predictions,
                 **{key: panel[key] for key in ("horizon", "scale", "site", "target_id", "month")})
        if validate_stage(args)[1] != contract or verify_selection(contract) != selection:
            raise AssertionError("Forecast inputs or selection changed")
        result = {"completed": True, "stage": "forecast", "contract_sha256": file_hash(contract_path),
                  "selection_sha256": selection_sha, "predictions_sha256": file_hash(output / "predictions.npz"),
                  "evaluation_inputs_sha256": contract["data"]["evaluation_inputs"]["sha256"],
                  "evaluation_truth_opened": False, "evaluation_rows": 48, "arms": contract["arms"],
                  "wall_seconds": time.perf_counter() - started,
                  "peak_cuda_gib": torch.cuda.max_memory_allocated() / 1024**3,
                  "selected_checkpoint_parameter_sha256": restored, "weights_unchanged": True}
        atomic_json(output / "result.json", result)
        return result
    except Exception as error:
        atomic_json(output / "result.json", {"completed": False, "error": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract")
    parser.add_argument("--output", required=True)
    print(run(parser.parse_args()))
