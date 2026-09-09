"""Process-local module deletion controls using the unchanged shift-study training loop."""

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

from experiments.peft_shift_mechanism_v1 import train as shared


NEW_METHODS = ("OUT_ONLY", "ATTN_ONLY")
CLI_METHODS = (*NEW_METHODS, "OFF_LORA")
OUTPUT_MODULE = "output_patch_embedding.output_layer"
COUNTS = {"OUT_ONLY": 27264, "ATTN_ONLY": 1179648, "OFF_LORA": 1206912}
PLAN_NAME = "_docs/notes/tsfm_topics/09_peft_module_ablation_plan_20260908.md"


def expected_map(method):
    attention = [f"encoder.block.{block}.layer.{layer}.self_attention.{part}"
                 for block in range(12) for layer in (0, 1) for part in ("q", "k", "v", "o")]
    if method == "OUT_ONLY":
        return [OUTPUT_MODULE]
    if method == "ATTN_ONLY":
        return attention
    if method == "OFF_LORA":
        return [*attention, OUTPUT_MODULE]
    raise ValueError(method)


def subset_targets(base, method, original_targets):
    if method not in NEW_METHODS:
        return original_targets(base, method)
    full, rank = original_targets(base, "OFF_LORA")
    if full != expected_map("OFF_LORA") or rank != 8:
        raise AssertionError("Original OFF_LORA module/rank contract changed")
    wanted = set(expected_map(method))
    selected = [name for name in full if name in wanted]
    if selected != expected_map(method):
        raise AssertionError("Ablation is not the declared original module subset")
    return selected, rank


def canonical_seed(seed, name):
    return int.from_bytes(hashlib.sha256(f"{seed}:{name}:rank=8".encode()).digest()[:8],
                          "little") % (2**63 - 1)


def initialization_audit(names, parameters, module_seeds, seed):
    expected_seeds = {name: canonical_seed(seed, name) for name in names}
    if module_seeds != expected_seeds:
        raise AssertionError("Module initialization seeds differ from original OFF_LORA")
    a_hashes, b_hashes = {}, {}
    for name in names:
        a = parameters[f"base.{name}.lora_A.default.weight"].detach().cpu()
        b = parameters[f"base.{name}.lora_B.default.weight"].detach().cpu()
        generator = shared.torch.Generator(device="cpu").manual_seed(expected_seeds[name])
        expected_a = shared.torch.empty(a.shape, dtype=shared.torch.float32, device="cpu")
        shared.nn.init.kaiming_uniform_(expected_a, a=math.sqrt(5), generator=generator)
        if not shared.torch.equal(a, expected_a) or shared.torch.count_nonzero(b).item() != 0:
            raise AssertionError(f"Retained module A/B initialization changed: {name}")
        a_hashes[name] = hashlib.sha256(a.contiguous().numpy().tobytes()).hexdigest()
        b_hashes[name] = hashlib.sha256(b.contiguous().numpy().tobytes()).hexdigest()
    return {"canonical_a_bitwise_verified": True, "zero_b_verified": True,
            "module_a_initial_sha256": a_hashes, "module_b_initial_sha256": b_hashes,
            "module_seeds": expected_seeds}


def model_audit(model, method, seed):
    if model.module_map != expected_map(method) or model.rank != 8 or model.adaptive_count != COUNTS[method]:
        raise AssertionError("Constructed model differs from the deletion contract")
    if model.probe is not None or model.head_names:
        raise AssertionError("Module deletion controls must not add a residual head")
    expected_names = {f"base.{name}.lora_{side}.default.weight"
                      for name in expected_map(method) for side in ("A", "B")}
    parameters = dict(model.named_parameters())
    trainable_names = {name for name, parameter in parameters.items() if parameter.requires_grad}
    if trainable_names != expected_names or set(model.adaptive_names) != expected_names:
        raise AssertionError("An undeclared parameter is trainable")
    return {"module_subset_verified": True, "residual_head_absent": True,
            "native_pretrained_parameters_frozen": True,
            "adaptive_count": model.adaptive_count, "module_map": list(model.module_map),
            **initialization_audit(model.module_map, parameters, model.module_seeds, seed)}


@contextmanager
def registration(output, audits, cache_loader=None):
    """Replace module attributes temporarily, without mutating their original objects or files."""
    keys = ("METHODS", "INTERNAL_METHODS", "EXPECTED_PARAMETERS", "lora_targets", "ShiftModel", "atomic_json",
            "prepare_cache")
    original = {key: getattr(shared, key) for key in keys}
    result_path = (Path(output) / "result.json").resolve()

    def targets(base, method):
        return subset_targets(base, method, original["lora_targets"])

    def construct(base, method, seed):
        model = original["ShiftModel"](base, method, seed)
        audits.append(model_audit(model, method, seed))
        return model

    def pending_json(path, value):
        if Path(path).resolve() == result_path:
            value = {**value, "inner_training_completed": bool(value.get("completed")),
                     "completed": False, "wrapped_completed": False}
        return original["atomic_json"](path, value)

    try:
        shared.METHODS = (*original["METHODS"], *NEW_METHODS)
        shared.INTERNAL_METHODS = original["INTERNAL_METHODS"] | set(NEW_METHODS)
        shared.EXPECTED_PARAMETERS = {**original["EXPECTED_PARAMETERS"], **COUNTS}
        shared.lora_targets = targets
        shared.ShiftModel = construct
        shared.atomic_json = pending_json
        if cache_loader is not None:
            shared.prepare_cache = cache_loader
        yield
    finally:
        for key, value in original.items():
            setattr(shared, key, value)


def validate_args(args):
    if args.method not in CLI_METHODS or (args.method == "OFF_LORA" and not args.smoke):
        raise ValueError("Use OUT_ONLY/ATTN_ONLY; OFF_LORA is available only for the S0 wrapper control")
    if args.lr not in (3e-5, 1e-4) or args.steps != 200 or args.val_every != 40:
        raise ValueError("Retain the planned LR grid, 200 updates, and validation interval 40")
    if not args.device.startswith("cuda") or args.lp_fraction != .4:
        raise ValueError("Retain the original CUDA precision and unused default LP fraction")


def original_inputs(args, root):
    study = root / "runs/peft_shift_mechanism_v1"
    done = json.loads((study / "completed.json").read_text(encoding="utf-8"))
    selected = json.loads((study / "selected.json").read_text(encoding="utf-8"))
    contract = json.loads((study / "study_contract.json").read_text(encoding="utf-8"))
    if not done.get("completed") or len(selected) != 96:
        raise RuntimeError("The original shift study must already be complete")
    protected = [study / name for name in ("completed.json", "selected.json", "selection.json", "study_contract.json")]
    protected.extend((study / "data/manifest.json", root / PLAN_NAME, Path(__file__)))
    for name, expected in contract["sources"].items():
        path = root / name
        if shared.file_hash(path) != expected:
            raise AssertionError(f"Original source changed: {name}")
        protected.append(path)
    original_plan = root / "_docs/notes/tsfm_topics/08_peft_shift_mechanism_plan_20260908.md"
    if shared.file_hash(original_plan) != contract["plan_hash"]:
        raise AssertionError("Original study plan changed")
    protected.append(original_plan)
    if shared.file_hash(study / "data/manifest.json") != contract["data_manifest_sha256"]:
        raise AssertionError("Original data manifest changed")
    with shared.np.load(args.data, allow_pickle=False) as data:
        metadata = json.loads(data["manifest_json"].item())
    corpus = metadata["corpus"]
    if metadata["condition"] != "Q00" or corpus not in (0, 1, 2) or args.seed != 7100 + corpus:
        raise ValueError("Use the original Q00 corpus0/1/2 and its paired optimizer seed")
    original_data = study / "data" / f"Q00_c{corpus}.npz"
    if Path(args.data).resolve() != original_data.resolve() or shared.file_hash(args.data) != contract["data_hashes"][original_data.name]:
        raise AssertionError("Prepared data differs from the original execution contract")
    protected.append(original_data)
    baseline = {}
    for method in ("F0", "OFF_LORA"):
        entries = [entry for entry in selected if (entry["condition"], entry["corpus"], entry["method"])
                   == ("Q00", corpus, method)]
        if len(entries) != 1:
            raise AssertionError("Missing or duplicate original baseline")
        expected_lr = .001 if method == "F0" else .00003
        canonical = f"runs/peft_shift_mechanism_v1/trials/Q00_c{corpus}/{method}/lr_{expected_lr:.0e}"
        if entries[0]["path"] != canonical or entries[0]["lr"] != expected_lr:
            raise AssertionError("Original baseline does not use the intended fixed learning rate")
        path = root / canonical
        status = json.loads((path / "guard/status.json").read_text(encoding="utf-8"))
        result = json.loads((path / "result.json").read_text(encoding="utf-8"))
        if not status.get("completed") or status.get("returncode") != 0 or not result.get("completed"):
            raise AssertionError("Original baseline did not complete successfully")
        if result["data_sha256"] != shared.file_hash(args.data) or result["seed"] != args.seed:
            raise AssertionError("Original baseline data or optimizer seed differs")
        baseline[method] = result
        protected.extend(path / name for name in ("result.json", "predictions.npz", "guard/status.json"))
    cache_path = Path(baseline["F0"]["cache"]).resolve()
    manifest = json.loads((cache_path / "manifest.json").read_text(encoding="utf-8"))
    cache_contract = manifest["contract"]
    if not manifest.get("completed") or cache_contract["data_sha256"] != shared.file_hash(args.data):
        raise AssertionError("Original feature cache is incomplete or mismatched")
    if cache_contract["source_sha256"] != shared.file_hash(Path(shared.__file__)):
        raise AssertionError("Original cache uses a different training source")
    if Path(args.checkpoint).resolve() != Path(cache_contract["checkpoint"]).resolve():
        raise AssertionError("Do not change the original pretrained checkpoint")
    if shared.file_hash(Path(args.checkpoint) / "config.json") != cache_contract["config_sha256"]:
        raise AssertionError("Checkpoint configuration changed")
    for filename, expected in manifest["checkpoint_weights"].items():
        if shared.file_hash(Path(args.checkpoint) / filename) != expected:
            raise AssertionError("Pretrained checkpoint weights changed")
    protected.append(cache_path / "manifest.json")
    protected.extend(sorted(cache_path.glob("*.npy")))
    if args.smoke:
        if not Path(args.cache).resolve().is_relative_to(root / "runs/peft_module_ablation_v1"):
            raise ValueError("S0 must use its own cache inside the new ablation study")
    elif Path(args.cache).resolve() != cache_path.parent:
        raise ValueError(f"Production must read the original cache parent: {cache_path.parent}")
    hashes = {str(path.resolve()): shared.file_hash(path) for path in protected}
    return {"corpus": corpus, "manifest": manifest, "cache_path": cache_path,
            "protected_hashes": hashes, "baseline": baseline}


def read_only_cache_loader(inputs):
    def load(base, episodes, args, output):
        manifest = inputs["manifest"]
        contract = manifest["contract"]
        observed = {"data_sha256": shared.file_hash(args.data), "metadata": episodes.metadata,
                    "counts": {split: len(values) for split, values in episodes.targets.items()},
                    "context": 256, "horizon": 16, "micro_groups": 4, "encoder_rows": 12,
                    "probe_rows": 4, "channel_order": ["Y", "U", "V"], "smoke": False,
                    "weight_dtype": "float32", "autocast": "bfloat16", "autocast_weight_cache": False,
                    "dropout": 0.0, "source_sha256": shared.file_hash(Path(shared.__file__))}
        if any(contract.get(key) != value for key, value in observed.items()):
            raise AssertionError("Original cache does not satisfy the current input/precision contract")
        if Path(args.cache).resolve() != inputs["cache_path"].parent or base.model_dim != 768 or base.num_quantiles != 21:
            raise AssertionError("Cache model or path differs from the original study")
        return shared.Cache(inputs["cache_path"], manifest), False
    return load


def finish_result(output, result, inputs, audits, wrapper_contract, started):
    pending = json.loads((output / "result.json").read_text(encoding="utf-8"))
    if pending.get("completed") or pending.get("wrapped_completed") or not pending.get("inner_training_completed"):
        raise AssertionError("The inner result bypassed the wrapper completion gate")
    if not result.get("completed") or len(audits) != 1:
        raise AssertionError("The original training loop did not produce one audited model")
    if not result["smoke"] and result["cache_created_this_trial"]:
        raise AssertionError("Production must not create or rewrite the original cache")
    if result["trainable_parameters"] != COUNTS[result["method"]] or result["module_map"] != expected_map(result["method"]):
        raise AssertionError("Saved result has an unexpected module map/count")
    if result["head_lr"] is not None or result["head_only_updates"] != 0 or result["phase_transitions"]:
        raise AssertionError("A residual-head or phased training path was used")
    expected_steps = 5 if result["smoke"] else 200
    count = 8 if result["smoke"] else 64
    samples = shared.np.random.default_rng(result["seed"]).integers(count, size=(expected_steps, 8))
    expected_sampler = hashlib.sha256(samples.astype(shared.np.int64).tobytes()).hexdigest()
    if result["steps_completed"] != expected_steps or result["sampler_sha256"] != expected_sampler:
        raise AssertionError("The original update/sampler contract changed")
    if not result["smoke"] and result["sampler_sha256"] != inputs["baseline"]["OFF_LORA"]["sampler_sha256"]:
        raise AssertionError("Sampler differs from the paired original OFF_LORA trial")
    for path, expected in inputs["protected_hashes"].items():
        if shared.file_hash(path) != expected:
            raise AssertionError(f"Protected input/source/artifact changed during execution: {path}")
    if not result["audits"].get("frozen_parameters_verified") or not result["audits"].get("checkpoint_reload_verified"):
        raise AssertionError("Original frozen/checkpoint audits did not pass")
    result.update({"completed": True, "wrapped_completed": True, "inner_training_completed": True,
                   "wrapper_contract": wrapper_contract, "wrapper_audits": audits[0],
                   "wrapper_wall_seconds": time.perf_counter() - started})
    result["source_hashes"]["experiments/peft_module_ablation_v1/train.py"] = wrapper_contract["source_sha256"]
    shared.atomic_json(output / "result.json", result)
    return result


def run(args):
    started = time.perf_counter()
    validate_args(args)
    root = Path(__file__).resolve().parents[2]
    output = Path(args.output).resolve()
    if not output.is_relative_to(root / "runs/peft_module_ablation_v1") or (output / "result.json").exists():
        raise ValueError("Use a new trial directory inside runs/peft_module_ablation_v1")
    inputs = original_inputs(args, root)
    output.mkdir(parents=True, exist_ok=True)
    wrapper_contract = {
        "kind": "Q00 native LoRA module deletion retraining ablation",
        "scope": "Exploratory follow-up; module capacities differ and this is not an internal causal-mechanism identification",
        "source_sha256": shared.file_hash(Path(__file__)), "original_train_sha256": shared.file_hash(Path(shared.__file__)),
        "plan_sha256": shared.file_hash(root / PLAN_NAME), "method": args.method,
        "module_map": expected_map(args.method), "trainable_parameters": COUNTS[args.method],
        "rank": 8, "alpha": 16, "new_residual_head": False,
        "attention_output_maps_disjoint": not bool(set(expected_map("OUT_ONLY")) & set(expected_map("ATTN_ONLY"))),
        "union_is_original_off_map": set(expected_map("OUT_ONLY")) | set(expected_map("ATTN_ONLY")) == set(expected_map("OFF_LORA")),
        "corpus": inputs["corpus"], "protected_original_hashes": inputs["protected_hashes"],
        "original_cache_read_only": not args.smoke, "original_cache": str(inputs["cache_path"]),
        "requested_arguments": vars(args).copy(),
        "common_training": {"steps": 5 if args.smoke else 200, "val_every": 1 if args.smoke else 40,
                            "effective_groups": 8, "micro_groups": 4, "gradient_clip_norm": 1.0,
                            "weight_decay": 0.0, "weight_dtype": "float32", "autocast": "bfloat16",
                            "autocast_weight_cache": False, "tf32": False, "target": "Y only",
                            "loss": "Unchanged shared.native_pinball", "new_training_loop": False},
    }
    shared.atomic_json(output / "wrapper_contract.json", wrapper_contract)
    audits = []
    cache_loader = None if args.smoke else read_only_cache_loader(inputs)
    with registration(output, audits, cache_loader):
        result = shared.run(argparse.Namespace(**vars(args)))
    return finish_result(output, result, inputs, audits, wrapper_contract, started)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--cache", required=True)
    result.add_argument("--method", choices=CLI_METHODS, required=True)
    result.add_argument("--lr", type=float, default=1e-4)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--steps", type=int, default=200)
    result.add_argument("--val-every", type=int, default=40)
    result.add_argument("--checkpoint", default=str(shared.DEFAULT_CHECKPOINT))
    result.add_argument("--device", default="cuda")
    result.add_argument("--lp-fraction", type=float, default=.4)
    result.add_argument("--smoke", action="store_true")
    return result


def main():
    args = parser().parse_args()
    try:
        run(args)
    except Exception as error:
        output = Path(args.output)
        if output.exists() and output.resolve().is_relative_to(Path(__file__).resolve().parents[2]
                                                               / "runs/peft_module_ablation_v1"):
            shared.atomic_json(output / "failure.json", {"completed": False, "wrapped_completed": False,
                               "error_type": type(error).__name__, "error": str(error),
                               "traceback": traceback.format_exc(), "time": time.time()})
        raise


if __name__ == "__main__":
    main()
