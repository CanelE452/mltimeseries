"""Study33 panels: validated L336/H48 four-channel Jena/BMRA archives."""
import json
from pathlib import Path

import numpy as np
import torch

from experiments.peft_external_gap_v1.train import array_hash


class Panel:
    def __init__(self, path, stage, smoke=False):
        self.path, self.stage, self.smoke = Path(path), stage, smoke
        with np.load(path, allow_pickle=False) as archive:
            self.context_values = archive["context_values"].astype(np.float32)
            self.target_values = archive["target_values"].astype(np.float32)
            self.channels = [str(value) for value in archive["channels"]]
            self.target_indices = archive["target_indices"].astype(np.int64)
            self.quantiles = archive["quantiles"].astype(np.float64)
            observed_mask = archive["observed_mask"].astype(bool)
            loss_mask = archive["target_loss_mask"].astype(bool)
            self.timestamps = archive["timestamps"].copy()
            self.fit_mean = archive["fit_mean"].astype(np.float64)
            self.fit_std = archive["fit_std"].astype(np.float64)
            self.fit_median = archive["fit_median"].astype(np.float64)
            self.context, self.horizon = int(archive["context"]), int(archive["horizon"])
            self.metadata = json.loads(archive["manifest_json"].item())
            names = ("train", "val") if stage == "fit" else ("cal", "eval")
            if smoke:
                if stage != "fit":
                    raise ValueError("S0 is fit-only")
                origins = archive["train_origins"].astype(np.int64)
                if len(origins) < 14:
                    raise ValueError("S0 needs fourteen train origins for its isolated pseudo-validation")
                self.origins = {"train": origins[:8], "val": origins[10:14]}
                self.smoke_train_origin_hash = array_hash(origins)
            else:
                self.origins = {name: archive[name + "_origins"].astype(np.int64) for name in names}
            forbidden = ("cal_origins", "eval_origins") if stage == "fit" else ("train_origins", "val_origins")
            if any(name in archive for name in forbidden):
                raise AssertionError("Prepared archive mixes fit and holdout origin contracts")
        self.count_channels = len(self.channels)
        self.dataset = self.metadata["dataset"]
        expected_channels = {"jena": 4, "bmra": 4}
        if self.dataset not in expected_channels or self.count_channels != expected_channels[self.dataset]:
            raise ValueError("Use the declared Jena/BMRA four-channel panels")
        if (self.context, self.horizon) != (336, 48) or self.target_indices.shape != (2,) or len(np.unique(self.target_indices)) != 2:
            raise ValueError("Expected L336/H48 and exactly two target channels")
        if np.any(self.target_indices < 0) or np.any(self.target_indices >= self.count_channels):
            raise ValueError("Target indices are outside the input channel map")
        if self.context_values.ndim != 2 or self.context_values.shape != self.target_values.shape or self.context_values.shape[1] != self.count_channels:
            raise ValueError("Context and original target arrays must have matching [T,C] shapes")
        expected_loss_mask = np.zeros_like(self.target_values, dtype=bool)
        expected_loss_mask[:, self.target_indices] = np.isfinite(self.target_values[:, self.target_indices])
        if not np.array_equal(observed_mask, np.isfinite(self.target_values)) or not np.array_equal(loss_mask, expected_loss_mask):
            raise AssertionError("Prepared observed/target loss masks disagree with original labels")
        if self.quantiles.shape != (21,) or np.any(np.diff(self.quantiles) <= 0):
            raise ValueError("Prepared quantile grid must contain 21 increasing levels")
        if not np.isfinite(self.context_values).all() or not np.isfinite(self.fit_std).all() or np.any(self.fit_std <= 0):
            raise ValueError("Past-only imputation or train scaling is invalid")
        if len(self.timestamps) != len(self.context_values):
            raise ValueError("Timestamp and data row counts differ")
        for split, origins in self.origins.items():
            if origins.ndim != 1 or len(origins) == 0 or len(np.unique(origins)) != len(origins):
                raise ValueError("Every split needs distinct chronological origins")
            if np.any(np.diff(origins) <= 0) or origins.min() < self.context or origins.max() + self.horizon > len(self.context_values):
                raise ValueError("Origin violates the legal context/target boundaries")
            if "boundary_indices" in self.metadata:
                left, right = self.metadata["boundary_indices"]["train" if smoke else split]
                if origins.min() < left or origins.max() + self.horizon > right:
                    raise AssertionError("Target horizon crosses its declared split boundary")
        if smoke and self.origins["train"][-1] + self.horizon > self.origins["val"][0]:
            raise AssertionError("S0 pseudo-validation labels overlap its optimizer origins")
        self.stats_hash = array_hash(self.fit_mean, self.fit_std, self.fit_median, self.target_indices)

    def batch(self, origins, device):
        if len(origins) != 4:
            raise ValueError("Every encoder call must contain four complete isolated groups")
        context = np.stack([self.context_values[o - self.context:o].T for o in origins])
        target = np.stack([self.target_values[o:o + self.horizon].T for o in origins])
        inactive = np.ones(self.count_channels, dtype=bool)
        inactive[self.target_indices] = False
        target[:, inactive] = np.nan
        groups = np.repeat(np.arange(4), self.count_channels)
        return (
            torch.as_tensor(context.reshape(4 * self.count_channels, self.context), device=device),
            torch.as_tensor(target.reshape(4 * self.count_channels, self.horizon), device=device),
            torch.as_tensor(groups, dtype=torch.long, device=device),
        )

    def targets(self, split):
        return np.stack([self.target_values[o:o + self.horizon, self.target_indices].T for o in self.origins[split]])
