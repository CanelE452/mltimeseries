"""Model adapters.

Each adapter calls the model's own official inference entry point and receives
exactly the inputs the track allows. No per-model tuning: batch size and device
are infrastructure choices, everything that affects the forecast is left at the
vendor default.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import torch

from .tracks import WindowInputs

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ForecastResult:
    quantiles: np.ndarray  # (n_units, n_variates, horizon, n_quantiles)
    point: np.ndarray  # (n_units, n_variates, horizon)
    seconds: float
    peak_memory_mb: float
    notes: dict


class Adapter:
    model_id: str = ""
    hf_repo: str = ""
    family: str = ""

    def load(self) -> None:
        raise NotImplementedError

    def unload(self) -> None:
        self._model = None
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def supports(self, track: str) -> bool:
        return True

    def forecast(self, inputs: WindowInputs, quantile_levels: list[float]) -> ForecastResult:
        raise NotImplementedError

    @staticmethod
    def _reset_memory() -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @staticmethod
    def _peak_memory_mb() -> float:
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**2
        return float("nan")


class Chronos2Adapter(Adapter):
    """amazon/chronos-2 and autogluon/chronos-2-synth via Chronos2Pipeline."""

    family = "transformer_encoder"

    def __init__(self, model_id: str, hf_repo: str, batch_size: int = 256):
        self.model_id = model_id
        self.hf_repo = hf_repo
        self.batch_size = batch_size
        self._model = None

    def load(self) -> None:
        from chronos import BaseChronosPipeline

        self._model = BaseChronosPipeline.from_pretrained(self.hf_repo, device_map=DEVICE)

    def forecast(self, inputs: WindowInputs, quantile_levels: list[float]) -> ForecastResult:
        payload = []
        for i in range(inputs.n_items):
            item: dict = {"target": inputs.targets[i]}
            if inputs.future_covariates[i] is not None:
                history = inputs.past_covariate_history[i]
                future = inputs.future_covariates[i]
                item["past_covariates"] = {
                    name: history[j] for j, name in enumerate(inputs.known_columns)
                }
                item["future_covariates"] = {
                    name: future[j] for j, name in enumerate(inputs.known_columns)
                }
            payload.append(item)
        self._reset_memory()
        start = time.perf_counter()
        quantile_list, mean_list = self._model.predict_quantiles(
            payload,
            prediction_length=inputs.horizon,
            quantile_levels=list(quantile_levels),
            batch_size=self.batch_size,
        )
        seconds = time.perf_counter() - start
        # (n_variates, horizon, n_quantiles) per item
        quantiles = np.stack([q.float().cpu().numpy() for q in quantile_list], axis=0)
        point = np.stack([m.float().cpu().numpy() for m in mean_list], axis=0)
        return ForecastResult(
            quantiles=quantiles,
            point=point,
            seconds=seconds,
            peak_memory_mb=self._peak_memory_mb(),
            notes={
                "entry_point": "Chronos2Pipeline.predict_quantiles",
                "batch_size": self.batch_size,
                "default_context_length": int(self._model.default_context_length),
            },
        )


class TiRex2Adapter(Adapter):
    """NX-AI TiRex-2 checkpoints via tirex2.ForecastModel.forecast.

    On this Windows host the fused FlashRNN/Triton kernels cannot be built (no
    MSVC toolchain, no nvcc), so the pure-PyTorch kernels the package ships for
    CPU and Metal are used on CUDA instead. They compute the same function; the
    smoke report records the CPU-vs-CUDA agreement actually measured.
    """

    family = "recurrent_xlstm"

    def __init__(self, model_id: str, hf_repo: str, batch_size: int = 256):
        self.model_id = model_id
        self.hf_repo = hf_repo
        self.batch_size = batch_size
        self._model = None
        self.native_kernels = False

    def load(self) -> None:
        import tirex2.model.component.flashrnn_slstm as flashrnn_slstm
        import tirex2.model.component.mlstm_block as mlstm_block

        if not getattr(mlstm_block, "_tsfm_gap_patched", False):
            original = mlstm_block._mlstm_backend_config
            flashrnn_slstm._flashrnn_backend = lambda device: "vanilla"
            mlstm_block._mlstm_backend_config = lambda config, device: original(config, "cpu")
            mlstm_block._tsfm_gap_patched = True
        self.native_kernels = True

        from tirex2 import load_model

        self._model = load_model(self.hf_repo, device=DEVICE)

    def forecast(self, inputs: WindowInputs, quantile_levels: list[float]) -> ForecastResult:
        from tirex2 import TimeseriesType

        series = []
        for i in range(inputs.n_items):
            future = inputs.future_covariates[i]
            history = inputs.past_covariate_history[i]
            if future is not None:
                combined = torch.tensor(
                    np.concatenate([history, future], axis=1), dtype=torch.float32
                )
            else:
                combined = None
            series.append(
                TimeseriesType(
                    target=torch.tensor(inputs.targets[i], dtype=torch.float32),
                    past_covariates=None,
                    future_covariates=combined,
                )
            )
        self._reset_memory()
        start = time.perf_counter()
        forecasts = self._model.forecast(
            series,
            prediction_length=inputs.horizon,
            output_type="numpy",
            batch_size=self.batch_size,
        )
        seconds = time.perf_counter() - start
        native_levels = self._model._quantile_levels()
        if list(native_levels) != list(quantile_levels):
            raise SystemExit(
                f"METRIC_MISMATCH: TiRex-2 emits {native_levels}, task asks {quantile_levels}"
            )
        # (n_variates, n_quantiles, horizon) -> (n_variates, horizon, n_quantiles)
        quantiles = np.stack([np.asarray(f).transpose(0, 2, 1) for f in forecasts], axis=0)
        median_index = list(quantile_levels).index(0.5)
        point = quantiles[..., median_index]
        return ForecastResult(
            quantiles=quantiles,
            point=point,
            seconds=seconds,
            peak_memory_mb=self._peak_memory_mb(),
            notes={
                "entry_point": "tirex2.ForecastModel.forecast",
                "batch_size": self.batch_size,
                "native_pytorch_kernels": self.native_kernels,
                "point_forecast": "median quantile (model emits no separate mean)",
            },
        )


class TimesFM3Adapter(Adapter):
    """google/timesfm-3.0-pytorch via TimesFM3Evaluator.predict_batch."""

    family = "transformer_patch_variate"

    def __init__(self, model_id: str, hf_repo: str, batch_size: int = 32):
        self.model_id = model_id
        self.hf_repo = hf_repo
        self.batch_size = batch_size
        self._model = None
        self._quantiles: list[float] = []

    def load(self) -> None:
        from timesfm3 import ModelConfig, TimesFM3Evaluator

        config = ModelConfig(
            checkpoint_path=self.hf_repo,
            per_core_batch_size=self.batch_size,
            device=DEVICE,
        )
        self._model = TimesFM3Evaluator(config)
        self._quantiles = list(config.quantiles)

    def forecast(self, inputs: WindowInputs, quantile_levels: list[float]) -> ForecastResult:
        if list(self._quantiles) != list(quantile_levels):
            raise SystemExit(
                f"METRIC_MISMATCH: TimesFM-3 emits {self._quantiles}, task asks {quantile_levels}"
            )
        contexts = [inputs.targets[i] for i in range(inputs.n_items)]
        if inputs.track == "C":
            past_future = []
            for i in range(inputs.n_items):
                history = inputs.past_covariate_history[i]
                future = inputs.future_covariates[i]
                past_future.append(np.concatenate([history, future], axis=1))
        else:
            past_future = None
        self._reset_memory()
        start = time.perf_counter()
        outputs = list(
            self._model.predict_batch(
                contexts=contexts,
                horizon=inputs.horizon,
                past_only_covariates=None,
                past_future_covariates=past_future,
                return_quantiles=True,
                univariate=(inputs.track == "U"),
            )
        )
        seconds = time.perf_counter() - start
        n_quantiles = len(quantile_levels)
        quantiles, point = [], []
        for index, output in enumerate(outputs):
            # A univariate series comes back as (horizon, n_quantiles); a multivariate
            # one as (n_variates, horizon, n_quantiles). Reshape explicitly rather than
            # with atleast_3d, which would append the axis instead of prepending it.
            q = np.asarray(output.quantiles, dtype=np.float64)
            if q.ndim == 2:
                q = q[None, :, :]
            if q.ndim != 3 or q.shape[1] != inputs.horizon or q.shape[2] != n_quantiles:
                raise SystemExit(
                    f"unexpected TimesFM quantile shape {q.shape}; expected "
                    f"(n_variates, {inputs.horizon}, {n_quantiles})"
                )
            f = np.asarray(output.forecast, dtype=np.float64)
            if f.ndim == 1:
                f = f[None, :]
            if f.shape != q.shape[:2]:
                raise SystemExit(f"TimesFM point shape {f.shape} does not match quantiles {q.shape}")
            expected_variates = inputs.targets[index].shape[0] if inputs.track != "U" else 1
            if q.shape[0] != expected_variates:
                raise SystemExit(
                    f"TimesFM returned {q.shape[0]} variates, task unit has {expected_variates}"
                )
            quantiles.append(q)
            point.append(f)
        return ForecastResult(
            quantiles=np.stack(quantiles, axis=0),
            point=np.stack(point, axis=0),
            seconds=seconds,
            peak_memory_mb=self._peak_memory_mb(),
            notes={
                "entry_point": "TimesFM3Evaluator.predict_batch",
                "per_core_batch_size": self.batch_size,
                "univariate_flag": inputs.track == "U",
            },
        )


REGISTRY: dict[str, dict] = {
    "chronos-2": {"cls": Chronos2Adapter, "hf_repo": "amazon/chronos-2", "role": "primary"},
    "tirex-2": {
        "cls": TiRex2Adapter,
        "hf_repo": "NX-AI/TiRex-2",
        "role": "primary",
    },
    "timesfm-3.0": {
        "cls": TimesFM3Adapter,
        "hf_repo": "google/timesfm-3.0-pytorch",
        "role": "primary",
    },
    "chronos-2-synth": {
        "cls": Chronos2Adapter,
        "hf_repo": "autogluon/chronos-2-synth",
        "role": "diagnostic",
    },
}

PRIMARY_MODELS = [name for name, spec in REGISTRY.items() if spec["role"] == "primary"]
DIAGNOSTIC_MODELS = [name for name, spec in REGISTRY.items() if spec["role"] == "diagnostic"]


def build(model_id: str) -> Adapter:
    spec = REGISTRY[model_id]
    return spec["cls"](model_id=model_id, hf_repo=spec["hf_repo"])


def configure_environment() -> None:
    """Settings this host needs before any model is imported."""
    os.environ.setdefault("CUDA_HOME", "C:/dummy" if os.name == "nt" else "/usr")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
