"""The shared small model and the six pooling arms.

Contract source: 01_forecast_query_tokenization_CLI.txt sections 6 and 7.

This is NOT official PatchTST. It borrows the patching and channel-independent idea
and is otherwise a minimal pilot model built from standard PyTorch modules.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

L = 1024
P = 16
N = 64
D = 128
H_MAX = 336

ARMS = ("U", "I", "C", "R", "H_STATIC", "DENSE")
POOLING_ARMS = ("U", "I", "C", "R", "H_STATIC")


def horizon_features(h: torch.Tensor) -> torch.Tensor:
    """u_H = [log1p(H)/log1p(336), H/1024]  (section 6)."""
    h = h.to(torch.float32)
    return torch.stack([torch.log1p(h) / math.log1p(H_MAX), h / L], dim=-1)


def sinusoidal_positions() -> torch.Tensor:
    """Fixed encoding on the center index of the original 64 patches (section 6)."""
    centers = torch.arange(N, dtype=torch.float32) * P + P / 2.0
    i = torch.arange(D // 2, dtype=torch.float32)
    angle = centers[:, None] / torch.pow(10000.0, 2.0 * i[None, :] / D)
    return torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)


class HorizonEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, D))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(horizon_features(h))


class ContentScorer(nn.Module):
    """concat(e_i, mean_i e_i, query) -> Linear(384,64) -> GELU -> Linear(64,1).

    The MLP is what makes the content-horizon interaction nonlinear, which section 7
    requires: a score of the form g_i + s(H) would cancel inside the group softmax.
    No dropout here, so arms consume identical dropout randomness.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3 * D, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, e: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        b, n, _ = e.shape
        summary = e.mean(dim=1, keepdim=True).expand(-1, n, -1)
        q = query[:, None, :].expand(-1, n, -1)
        return self.net(torch.cat([e, summary, q], dim=-1)).squeeze(-1)


class PilotModel(nn.Module):
    def __init__(self, arm: str, B: int = 32, dropout: float = 0.1):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        if arm == "DENSE":
            B = N
        if N % B != 0:
            raise ValueError(f"N={N} not divisible by B={B}")
        self.arm = arm
        self.B = B
        self.r = N // B

        self.patch_embed = nn.Linear(P, D)
        self.horizon_embed = HorizonEmbed()

        # Only arms that actually score tokens own a scorer. U and DENSE do not get a
        # dummy module, so their parameter counts are reported as they really are.
        self.scorer = ContentScorer() if arm in ("I", "C", "R") else None
        # H_STATIC: trainable logits [2 horizons, B bins, r positions], init 0.
        self.static_logits = (
            nn.Parameter(torch.zeros(2, self.B, self.r)) if arm == "H_STATIC" else None
        )

        layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=4,
            dim_feedforward=256,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=3, enable_nested_tensor=False)

        self.readout = nn.Linear(N * D, D)
        self.readout_drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * D, H_MAX)

        self.register_buffer("positions", sinusoidal_positions(), persistent=False)

    # ---- pieces ------------------------------------------------------------

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """[batch, 1024] -> e_i [batch, 64, 128] (content + fixed position)."""
        b = x.shape[0]
        patches = x.reshape(b, N, P)
        return self.patch_embed(patches) + self.positions

    def pool_weights(self, e: torch.Tensor, h_pool: torch.Tensor) -> torch.Tensor:
        """Within-group softmax weights [batch, B, r], rows summing to 1."""
        b = e.shape[0]
        if self.arm == "U":
            return e.new_full((b, self.B, self.r), 1.0 / self.r)
        if self.arm == "H_STATIC":
            idx = (h_pool.reshape(-1) > 96).long()          # 0 -> H=96, 1 -> H=336
            return self.static_logits[idx].softmax(dim=-1)
        if self.arm == "I":
            query = e.new_zeros((b, D))
        else:                                                # C and R
            query = self.horizon_embed(h_pool)
        scores = self.scorer(e, query).reshape(b, self.B, self.r)
        return scores.softmax(dim=-1)

    def compress(self, e: torch.Tensor, h_pool: torch.Tensor):
        """e -> tokens [batch, B, 128] plus the weights used."""
        if self.arm == "DENSE":
            return e, None
        w = self.pool_weights(e, h_pool)
        grouped = e.reshape(e.shape[0], self.B, self.r, D)
        return (grouped * w.unsqueeze(-1)).sum(dim=2), w

    def decode(self, z: torch.Tensor, e_h_true: torch.Tensor) -> torch.Tensor:
        """Unmerge by fixed group correspondence, then the shared readout."""
        if self.arm != "DENSE":
            z = z.repeat_interleave(self.r, dim=1)
        flat = z.reshape(z.shape[0], N * D)
        hidden = self.readout_drop(torch.nn.functional.gelu(self.readout(flat)))
        return self.head(torch.cat([hidden, e_h_true], dim=-1))

    def forward(
        self,
        x: torch.Tensor,
        h_true: torch.Tensor,
        h_pool: torch.Tensor | None = None,
        return_weights: bool = False,
    ):
        """x [batch, 1024] -> yhat [batch, 336]. Always 336; the loss masks to H."""
        if h_pool is None:
            h_pool = h_true
        e = self.embed(x)
        tokens, w = self.compress(e, h_pool)
        z = self.encoder(tokens)
        yhat = self.decode(z, self.horizon_embed(h_true))
        if return_weights:
            return yhat, w
        return yhat

    # ---- diagnostics -------------------------------------------------------

    def group_bounds(self):
        """Patch index range covered by each group, for the section 13 plots."""
        return [(g * self.r, g * self.r + self.r - 1) for g in range(self.B)]

    def weighted_centers(self, w: torch.Tensor) -> torch.Tensor:
        """Weighted mean patch center (in samples) per group: [batch, B]."""
        local = torch.arange(self.r, device=w.device, dtype=w.dtype)
        starts = torch.arange(self.B, device=w.device, dtype=w.dtype) * self.r
        patch_idx = starts[None, :, None] + local[None, None, :]
        return ((patch_idx * w).sum(-1)) * P + P / 2.0

    def parameter_report(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        scorer = sum(p.numel() for p in self.scorer.parameters()) if self.scorer else 0
        static = self.static_logits.numel() if self.static_logits is not None else 0
        return {
            "arm": self.arm,
            "B": self.B,
            "r": self.r,
            "total_parameters": total,
            "tokenizer_parameters": scorer + static,
            "encoder_parameters": sum(p.numel() for p in self.encoder.parameters()),
            "decoder_parameters": sum(p.numel() for p in self.readout.parameters())
            + sum(p.numel() for p in self.head.parameters()),
        }


def build(arm: str, B: int, seed: int) -> PilotModel:
    """Seeded construction. I, C and R have identical structure, so the same seed
    gives them byte-identical initial parameters (section 7)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return PilotModel(arm=arm, B=B)


def masked_mse(yhat: torch.Tensor, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Mean squared error over the first H lead times only (section 9)."""
    steps = torch.arange(H_MAX, device=yhat.device)[None, :]
    mask = (steps < h[:, None]).to(yhat.dtype)
    err = (yhat - y) ** 2 * mask
    return err.sum() / mask.sum()
