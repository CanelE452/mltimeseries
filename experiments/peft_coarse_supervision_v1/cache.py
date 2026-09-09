"""Frozen point features from the train and coarse-validation inputs only."""

import argparse
import json
import math
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "2"

import numpy as np
import torch

from experiments.peft_external_gap_v1.train import (
    array_hash, file_hash, atomic_json, load_base, parameter_hash,
)
from experiments.peft_adaptation_scope_v1.train import check_resources, precision
from .model import PointModel


ROOT = Path(__file__).resolve().parents[2]
STUDY = "peft_coarse_supervision_v1"
IDENTITIES = ("horizon", "site", "target_id", "month")
PADDED_HORIZON = 752
PATCHES = 47


class Episodes:
    def __init__(self, path):
        self.path = Path(path).resolve()
        required = ("context", "horizon", "scale", "total", "label_valid", "site", "target_id", "month", "profile")
        with np.load(path, allow_pickle=False) as archive:
            self.arrays = {key: archive[key].copy() for key in required}
        self.n = len(self.arrays["horizon"])
        for key in required:
            if len(self.arrays[key]) != self.n:
                raise ValueError(f"Episode count differs for {key}")
        if self.arrays["context"].shape != (self.n, 512) or not np.isfinite(self.arrays["context"]).all():
            raise ValueError("Every proxy must contain 512 finite past hours")
        for key in ("horizon", "scale", "total", "label_valid", "site", "target_id", "month"):
            if self.arrays[key].shape != (self.n,):
                raise ValueError(f"Expected one {key} per episode")
        horizon, scale = self.arrays["horizon"], self.arrays["scale"]
        if not np.isin(horizon, (672, 696, 720, 744)).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("Invalid calendar horizon or coarse-only target scale")
        valid = self.arrays["label_valid"].astype(bool)
        if not np.isfinite(self.arrays["total"][valid]).all():
            raise ValueError("An available monthly total is not finite")
        self.identity_sha256 = array_hash(*(self.arrays[key] for key in IDENTITIES))

    def context(self, index, device):
        return torch.as_tensor(self.arrays["context"][index:index + 1], dtype=torch.float32, device=device)

    def __len__(self):
        return self.n


def validation_eligibility(panel):
    arrays = panel.arrays
    sites, targets, months = (arrays[key] for key in ("site", "target_id", "month"))
    if len(panel) != 48 or len(np.unique(sites)) != 2 or set(months.tolist()) != {"2017-01", "2017-02", "2017-03"}:
        raise AssertionError("Validation must preserve 48 cells from the fixed two-site, three-month panel")
    counts = {}
    for site in np.unique(sites):
        selected_site = sites == site
        names = np.unique(targets[selected_site])
        if len(names) != 8 or int(selected_site.sum()) != 24:
            raise AssertionError("Each validation site must preserve eight targets and 24 cells")
        for name in names:
            selected = selected_site & (targets == name)
            if selected.sum() != 3 or len(np.unique(months[selected])) != 3 or not arrays["label_valid"][selected].any():
                raise AssertionError("Each target needs all three validation cells and at least one complete monthly label")
        count = int(arrays["label_valid"][selected_site].sum())
        if count < 16:
            raise AssertionError("Validation eligibility requires at least sixteen complete monthly labels per site")
        counts[str(site)] = count
    return counts


def native_args(contract):
    return SimpleNamespace(checkpoint=contract["checkpoint"], device="cuda", min_free_ram_gib=5.)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_stage(args):
    from . import prepare

    contract_path = Path(args.contract or ROOT / "runs" / STUDY / "study_contract.json").resolve()
    contract = prepare.validate(root=ROOT, contract_path=contract_path)
    return contract_path, contract


def fresh_output(output):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing or partial stage: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def encode_arrays(model, panels, args):
    arrays, hidden_dtypes = {}, set()
    model.eval()
    with torch.no_grad():
        for split, panel in panels.items():
            hidden = np.zeros((len(panel), PATCHES, 768), dtype=np.float32)
            point = np.zeros((len(panel), PADDED_HORIZON), dtype=np.float32)
            scale = np.zeros(len(panel), dtype=np.float32)
            for index, horizon in enumerate(panel.arrays["horizon"]):
                check_resources(args)
                horizon = int(horizon)
                with precision(args):
                    h, p, _, s = model.encode(panel.context(index, args.device),
                                               torch.zeros(1, dtype=torch.long, device=args.device), horizon)
                hidden_dtypes.add(str(h.dtype))
                patches = math.ceil(horizon / 16)
                hidden[index, :patches] = h[0].float().cpu().numpy()
                point[index, :horizon] = p[0, :horizon].float().cpu().numpy()
                scale[index] = s.item()
                if (not np.isfinite(hidden[index]).all() or not np.isfinite(point[index]).all()
                        or not np.isfinite(scale[index]) or scale[index] <= 0):
                    raise FloatingPointError(f"Nonfinite frozen features at {split}:{index}")
            arrays.update({f"{split}_hidden": hidden, f"{split}_base_point": point,
                           f"{split}_native_scale": scale})
            arrays.update({f"{split}_{key}": panel.arrays[key] for key in IDENTITIES})
    return arrays, sorted(hidden_dtypes)


def load_verified_cache(contract, contract_sha256, panels):
    path, result_path = Path(contract["paths"]["cache"]), Path(contract["paths"]["cache_result"])
    result = read_json(result_path)
    if (not result.get("completed") or result.get("contract_sha256") != contract_sha256
            or result.get("cache_sha256") != file_hash(path)):
        raise AssertionError("Frozen cache provenance is incomplete or has changed")
    expected_inputs = {key: contract["data"][key]["sha256"] for key in ("train", "validation")}
    if result["data_input_hashes"] != expected_inputs:
        raise AssertionError("Frozen cache belongs to different train/validation inputs")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    for split, panel in panels.items():
        for key in IDENTITIES:
            if not np.array_equal(arrays[f"{split}_{key}"], panel.arrays[key]):
                raise AssertionError(f"Frozen cache episode ordering changed: {split}/{key}")
        shapes = {"hidden": (len(panel), PATCHES, 768), "base_point": (len(panel), PADDED_HORIZON),
                  "native_scale": (len(panel),)}
        for key, shape in shapes.items():
            value = arrays[f"{split}_{key}"]
            if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
                raise AssertionError(f"Invalid frozen cache array: {split}/{key}")
        for index, horizon in enumerate(panel.arrays["horizon"]):
            if np.any(arrays[f"{split}_base_point"][index, int(horizon):]) or np.any(
                    arrays[f"{split}_hidden"][index, math.ceil(int(horizon) / 16):]):
                raise AssertionError("Unused cached point/hidden padding must be zero")
    return arrays, result


def run(args):
    started = time.perf_counter()
    contract_path, contract = validate_stage(args)
    contract_sha256 = file_hash(contract_path)
    output = Path(args.output).resolve()
    if output != Path(contract["paths"]["cache"]).resolve().parent:
        raise ValueError("Cache output does not match the frozen stage path")
    output = fresh_output(output)
    try:
        inputs = {key: file_hash(contract["data"][key]["path"]) for key in ("train", "validation")}
        if inputs != {key: contract["data"][key]["sha256"] for key in inputs}:
            raise AssertionError("Cache input hashes changed")
        panels = {key: Episodes(contract["data"][key]["path"]) for key in inputs}
        validation_counts = validation_eligibility(panels["validation"])
        settings = native_args(contract)
        torch.manual_seed(2026090817)
        model = PointModel(load_base(settings), "HEAD")
        before = parameter_hash(model.named_parameters())
        arrays, dtypes = encode_arrays(model, panels, settings)
        if parameter_hash(model.named_parameters()) != before:
            raise AssertionError("Frozen cache generation changed model parameters")
        temporary = output / "cache.partial.npz"
        np.savez(temporary, **arrays)
        os.replace(temporary, output / "cache.npz")
        _, after = validate_stage(args)
        if after != contract or file_hash(contract_path) != contract_sha256:
            raise AssertionError("The cache contract changed during inference")
        result = {"completed": True, "study": STUDY, "stage": "cache", "contract_sha256": contract_sha256,
                  "source_hashes": contract["source_hashes"], "data_input_hashes": inputs,
                  "cache_sha256": file_hash(output / "cache.npz"), "wall_seconds": time.perf_counter() - started,
                  "peak_cuda_gib": torch.cuda.max_memory_allocated() / 1024**3,
                  "episode_counts": {key: len(value) for key, value in panels.items()},
                  "validation_valid_counts_by_site": validation_counts,
                  "identity_hashes": {key: value.identity_sha256 for key, value in panels.items()},
                  "native_hidden_dtypes": dtypes, "stored_dtype": "float32", "encoder_rows": 1,
                  "context": 512, "max_patches": PATCHES, "max_padded_horizon": PADDED_HORIZON,
                  "weight_dtype": "float32", "autocast": "bfloat16", "point_head_dtype": "float32",
                  "autocast_cache_enabled": False, "tf32": False, "optimizer_steps": 0,
                  "model_unchanged": True, "evaluation_data_opened": False,
                  "model_parameter_sha256": before, "array_hashes": {key: array_hash(value) for key, value in arrays.items()}}
        atomic_json(output / "result.json", result)
        return result
    except Exception as error:
        atomic_json(output / "result.json", {"completed": False, "study": STUDY, "stage": "cache",
                    "error": str(error), "traceback": traceback.format_exc(), "contract_sha256": contract_sha256})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
