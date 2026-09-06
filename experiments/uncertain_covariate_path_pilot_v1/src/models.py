"""Arms for UCP-PATH-PILOT-v1.

Every arm shares the same base feature MLP, fusion MLP and particle head. The
arms differ only in how the weather tensor C of shape (B, K, T, D) becomes a
per-lead 64-d weather code:

  H        no weather at all (the weather code is exactly zero)
  M        mean over members first, then phi, then the temporal encoder
  S        per-lead summary statistics, then phi_s, then the temporal encoder
  D        z_t = mean_k phi(C[:, k, t, :]),  u = TemporalEncoder(z)     pool -> temporal
  P        e_k  = TemporalEncoder(phi(C[:, k])), u_t = mean_k e_k,t     temporal -> pool
  MC       the P encoder applied to one member at a time (K = 1)

D and P instantiate exactly the same submodules in the same order, so their
parameter counts are identical by construction (test A12) and the only
difference is where the member pooling happens.
"""

from __future__ import annotations

import torch
import torch.nn as nn

SUMMARY_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
N_SUMMARY = 2 + len(SUMMARY_QUANTILES)  # mean, std, q10, q25, q50, q75, q90


def mlp(sizes: list[int], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class TemporalEncoder(nn.Module):
    """2-layer 1-D temporal conv over the lead axis, kernel 3, residual."""

    def __init__(self, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, T, hidden) -> (N, T, hidden)."""
        h = self.drop(self.act(self.conv1(self.norm1(x).transpose(1, 2)).transpose(1, 2)))
        x = x + h
        h = self.drop(self.act(self.conv2(self.norm2(x).transpose(1, 2)).transpose(1, 2)))
        return x + h


def summarise_members(weather: torch.Tensor) -> torch.Tensor:
    """(B, K, T, D) -> (B, T, N_SUMMARY * D) of per-lead member statistics."""
    mean = weather.mean(dim=1)
    std = weather.std(dim=1, unbiased=False)
    qs = torch.quantile(
        weather,
        torch.tensor(SUMMARY_QUANTILES, device=weather.device, dtype=weather.dtype),
        dim=1,
    )  # (n_q, B, T, D)
    qs = qs.permute(1, 2, 3, 0).flatten(-2)  # (B, T, D * n_q)
    return torch.cat([mean, std, qs], dim=-1)


class WeatherEncoder(nn.Module):
    """Weather branch for one arm. `mode` selects where members are pooled."""

    def __init__(self, mode: str, n_weather: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        if mode not in {"H", "M", "S", "D", "P", "MC"}:
            raise ValueError(f"unknown weather mode {mode!r}")
        self.mode = mode
        self.hidden = hidden
        if mode == "H":
            return
        n_in = N_SUMMARY * n_weather if mode == "S" else n_weather
        self.phi = mlp([n_in, hidden, hidden], dropout)
        self.temporal = TemporalEncoder(hidden, dropout)

    def forward(self, weather: torch.Tensor) -> torch.Tensor:
        """weather: (B, K, T, D) -> (B, T, hidden)."""
        B, K, T, _ = weather.shape
        if self.mode == "H":
            return weather.new_zeros(B, T, self.hidden)
        if self.mode == "M":
            return self.temporal(self.phi(weather.mean(dim=1)))
        if self.mode == "S":
            return self.temporal(self.phi(summarise_members(weather)))
        if self.mode == "D":
            # pool members per lead first, then read the pooled path in time
            return self.temporal(self.phi(weather).mean(dim=1))
        # P and MC: read each member's own path in time first, then pool
        per_member = self.temporal(self.phi(weather).reshape(B * K, T, self.hidden))
        return per_member.reshape(B, K, T, self.hidden).mean(dim=1)


class ParticleForecaster(nn.Module):
    """Base features + weather code -> R empirical particles per lead."""

    def __init__(
        self,
        arm: str,
        n_base: int,
        n_weather: int,
        n_farms: int,
        hidden: int = 64,
        farm_emb: int = 8,
        n_particles: int = 32,
        particle_emb: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.arm = arm
        self.n_particles = n_particles
        weather_mode = "P" if arm == "P_BROKEN" else arm
        self.farm_embedding = nn.Embedding(n_farms, farm_emb)
        self.base_mlp = mlp([n_base + farm_emb, hidden, hidden], dropout)
        self.weather_encoder = WeatherEncoder(weather_mode, n_weather, hidden, dropout)
        self.fusion = mlp([2 * hidden, 2 * hidden, hidden], dropout)
        self.particle_embedding = nn.Parameter(torch.randn(n_particles, particle_emb) * 0.02)
        self.head = mlp([hidden + particle_emb, hidden, 1], dropout)

    def forward(self, base: torch.Tensor, weather: torch.Tensor, farm_idx: torch.Tensor) -> torch.Tensor:
        """base: (B, T, n_base), weather: (B, K, T, D), farm_idx: (B,) -> (B, T, R)."""
        B, T, _ = base.shape
        farm = self.farm_embedding(farm_idx).unsqueeze(1).expand(B, T, -1)
        base_code = self.base_mlp(torch.cat([base, farm], dim=-1))
        weather_code = self.weather_encoder(weather)
        fused = self.fusion(torch.cat([base_code, weather_code], dim=-1))  # (B, T, hidden)
        fused_r = fused.unsqueeze(2).expand(B, T, self.n_particles, fused.shape[-1])
        pe_r = self.particle_embedding.view(1, 1, self.n_particles, -1).expand(
            B, T, self.n_particles, -1
        )
        return self.head(torch.cat([fused_r, pe_r], dim=-1)).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
