"""The three core arms.

R  operator-aware reconstruction to the base grid, then a base-grid forecaster.
M  the observed tokens with generic centre-time metadata.
O  the same tokens with the interval-integrated time basis.

M and O are literally the same nn.Module class with the same shapes and the same
initialisation order; the only thing that differs is the `time_basis` array fed in
from operators.token_features, which carries no trainable parameter.  That keeps
the parameter counts equal and the initial states byte-identical under one seed.
"""

from __future__ import annotations

import torch
from torch import nn

from .operators import FORECAST_BINS, FOURIER_K, HISTORY_BINS

D_MODEL = 128
N_HEADS = 4
FFN = 256
DROPOUT = 0.1
N_ENCODER_LAYERS = 2
BASIS_DIM = 2 * FOURIER_K


class CrossAttentionDecoder(nn.Module):
    """One pre-norm cross-attention block: queries read the encoder memory.  No
    self-attention among the queries, so the 72 future positions stay independent
    given the history."""

    def __init__(self) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(D_MODEL)
        self.norm_kv = nn.LayerNorm(D_MODEL)
        self.attn = nn.MultiheadAttention(D_MODEL, N_HEADS, dropout=DROPOUT, batch_first=True)
        self.norm_ffn = nn.LayerNorm(D_MODEL)
        self.ffn = nn.Sequential(
            nn.Linear(D_MODEL, FFN), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(FFN, D_MODEL)
        )
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, q: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        h = self.norm_q(q)
        m = self.norm_kv(memory)
        q = q + self.drop(self.attn(h, m, m, need_weights=False)[0])
        return q + self.drop(self.ffn(self.norm_ffn(q)))


def _encoder() -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=FFN, dropout=DROPOUT,
        activation="gelu", batch_first=True, norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=N_ENCODER_LAYERS)


class TokenForecaster(nn.Module):
    """M and O.  Token = value + centre-or-integrated time basis + width + operation."""

    def __init__(self) -> None:
        super().__init__()
        self.value_proj = nn.Linear(1, D_MODEL)
        self.time_proj = nn.Linear(BASIS_DIM, D_MODEL)
        self.width_proj = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, D_MODEL))
        self.op_embed = nn.Embedding(2, D_MODEL)
        self.token_norm = nn.LayerNorm(D_MODEL)

        self.encoder = _encoder()

        self.query_time = nn.Linear(BASIS_DIM, D_MODEL)
        self.query_norm = nn.LayerNorm(D_MODEL)
        self.decoder = CrossAttentionDecoder()
        self.head = nn.Linear(D_MODEL, 1)

    def forward(
        self,
        values: torch.Tensor,      # (B, T)
        time_basis: torch.Tensor,  # (B, T, BASIS_DIM)
        width: torch.Tensor,       # (B, T, 1)
        op_id: torch.Tensor,       # (B, T)
        query_basis: torch.Tensor,  # (B, 72, BASIS_DIM)
    ) -> torch.Tensor:
        tok = (
            self.value_proj(values.unsqueeze(-1))
            + self.time_proj(time_basis)
            + self.width_proj(width)
            + self.op_embed(op_id)
        )
        memory = self.encoder(self.token_norm(tok))
        q = self.query_norm(self.query_time(query_basis))
        return self.head(self.decoder(q, memory)).squeeze(-1)


class BaseGridForecaster(nn.Module):
    """R's forecaster.  Reads the 288 reconstructed base bins, which already carry
    their own regular positions, and writes the same 72 future bins."""

    def __init__(self) -> None:
        super().__init__()
        self.value_proj = nn.Linear(1, D_MODEL)
        self.time_proj = nn.Linear(BASIS_DIM, D_MODEL)
        self.token_norm = nn.LayerNorm(D_MODEL)

        self.encoder = _encoder()

        self.query_time = nn.Linear(BASIS_DIM, D_MODEL)
        self.query_norm = nn.LayerNorm(D_MODEL)
        self.decoder = CrossAttentionDecoder()
        self.head = nn.Linear(D_MODEL, 1)

    def forward(
        self,
        history: torch.Tensor,      # (B, 288)
        time_basis: torch.Tensor,   # (B, 288, BASIS_DIM)
        query_basis: torch.Tensor,  # (B, 72, BASIS_DIM)
    ) -> torch.Tensor:
        tok = self.value_proj(history.unsqueeze(-1)) + self.time_proj(time_basis)
        memory = self.encoder(self.token_norm(tok))
        q = self.query_norm(self.query_time(query_basis))
        return self.head(self.decoder(q, memory)).squeeze(-1)


def build(arm: str) -> nn.Module:
    if arm in ("M", "O"):
        return TokenForecaster()
    if arm == "R":
        return BaseGridForecaster()
    raise ValueError(arm)


def parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def hourly_mean(pred_10min: torch.Tensor) -> torch.Tensor:
    """72 ten-minute values -> 12 hourly means.  The only way a 60-minute number is
    ever produced, for predictions and for targets alike."""
    b = pred_10min.shape[0]
    return pred_10min.reshape(b, FORECAST_BINS // 6, 6).mean(dim=-1)


def primary_loss(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """0.5 * MSE_10min + 0.5 * MSE_60min, each averaged over its own lead axis
    first so the 72-step term does not silently outweigh the 12-step one."""
    mse10 = ((pred - target) ** 2).mean(dim=-1).mean()
    mse60 = ((hourly_mean(pred) - hourly_mean(target)) ** 2).mean(dim=-1).mean()
    return 0.5 * mse10 + 0.5 * mse60, mse10, mse60


assert HISTORY_BINS == 288 and FORECAST_BINS == 72
