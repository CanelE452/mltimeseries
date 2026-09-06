"""Observation operators, the two time bases, and the R reconstruction solve.

Everything here works in *base-bin units*: a window is 288 base bins of history
followed by 72 base bins of target, and time is measured relative to the forecast
origin o so that history covers tau in [-1, 0) after dividing by 288.

The two controlled observation types, both defined on the base grid:

  END_BIN        the single last base bin of a report interval [a, b)
                 -> value x[b-1], support [b-1, b)
  INTERVAL_MEAN  the mean of every base bin in [a, b)
                 -> value mean(x[a:b]), support [a, b)

END_BIN is not a mathematical point: its support is one whole base bin, and the
integrated basis treats it that way.
"""

from __future__ import annotations

import numpy as np

HISTORY_BINS = 288      # 48 h at 10 min
FORECAST_BINS = 72      # 12 h at 10 min
OPS = ("END_BIN", "INTERVAL_MEAN")
OP_INDEX = {op: i for i, op in enumerate(OPS)}

FOURIER_K = 16
# Geometric frequencies from 1 to 128 cycles per 48-hour history window.  The
# base grid Nyquist is 144 cycles per window, so every frequency is resolvable.
FOURIER_CYCLES = 2.0 ** (np.arange(FOURIER_K) * (7.0 / (FOURIER_K - 1)))
FOURIER_OMEGA = 2.0 * np.pi * FOURIER_CYCLES


def report_blocks(r: int) -> np.ndarray:
    """Report-interval [a, b) bounds in local history coordinates, aligned backwards
    from the forecast origin so the last block always ends exactly at the origin."""
    if HISTORY_BINS % r:
        raise ValueError(f"r={r} does not divide the {HISTORY_BINS}-bin history")
    ends = np.arange(HISTORY_BINS, 0, -r)[::-1]        # ..., 288
    return np.stack([ends - r, ends], axis=1)          # (n_tokens, 2)


def observation_support(r: int, op: str) -> np.ndarray:
    """Support [a, b) of every observation token, in local history coordinates."""
    blocks = report_blocks(r)
    if op == "INTERVAL_MEAN":
        return blocks
    if op == "END_BIN":
        return np.stack([blocks[:, 1] - 1, blocks[:, 1]], axis=1)
    raise ValueError(op)


def observe(history: np.ndarray, r: int, op: str) -> np.ndarray:
    """Apply the observation operator to a batch of base-grid histories.

    history : (..., 288) base-bin values
    returns : (..., 288 // r) observed values
    """
    blocks = report_blocks(r)
    n = len(blocks)
    if op == "INTERVAL_MEAN":
        return history[..., : n * r].reshape(*history.shape[:-1], n, r).mean(axis=-1)
    if op == "END_BIN":
        return history[..., blocks[:, 1] - 1]
    raise ValueError(op)


def observation_matrix(r: int, op: str) -> np.ndarray:
    """The linear operator A with v = A x, x the 288-bin base history."""
    sup = observation_support(r, op)
    A = np.zeros((len(sup), HISTORY_BINS))
    for i, (a, b) in enumerate(sup):
        A[i, a:b] = 1.0 / (b - a)
    return A


# ------------------------------------------------------------------ time bases


def _tau(bounds: np.ndarray) -> np.ndarray:
    """Support bounds in base-bin coordinates -> tau relative to the forecast
    origin, where the 288-bin history occupies tau in [-1, 0)."""
    return (bounds - HISTORY_BINS) / HISTORY_BINS


def _sinc0(x: np.ndarray) -> np.ndarray:
    """sin(x)/x with the removable singularity at 0 filled in."""
    out = np.ones_like(x)
    nz = np.abs(x) > 1e-12
    out[nz] = np.sin(x[nz]) / x[nz]
    return out


def centre_basis(bounds: np.ndarray) -> np.ndarray:
    """phi_M: the Fourier basis evaluated at the centre of each support.

    bounds : (n, 2) supports [a, b) in base-bin coordinates
    returns: (n, 2 * FOURIER_K)
    """
    t = _tau(bounds.astype(float))
    c = 0.5 * (t[:, 0] + t[:, 1])
    ang = np.outer(c, FOURIER_OMEGA)
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=1)


def integrated_basis(bounds: np.ndarray) -> np.ndarray:
    """phi_O: the Fourier basis averaged over each support,

        1/(b-a) * integral_a^b phi(t) dt

    in closed form.  Using  mean_[a,b] sin(w t) = sin(w c) * sinc(w d / 2)  and
    the matching cosine identity, with c the centre and d = b - a.  No quadrature
    at run time and no trainable parameter.
    """
    t = _tau(bounds.astype(float))
    c = 0.5 * (t[:, 0] + t[:, 1])
    d = t[:, 1] - t[:, 0]
    ang = np.outer(c, FOURIER_OMEGA)
    atten = _sinc0(0.5 * np.outer(d, FOURIER_OMEGA))
    return np.concatenate([np.sin(ang) * atten, np.cos(ang) * atten], axis=1)


def future_query_basis() -> np.ndarray:
    """Query features for the 72 future base bins.  Identical for M and O: the
    target grid never changes with r, so integrating the query could not carry
    resolution information, and keeping it fixed leaves the observation-side
    representation as the single difference between the two models."""
    bounds = np.stack(
        [np.arange(HISTORY_BINS, HISTORY_BINS + FORECAST_BINS),
         np.arange(HISTORY_BINS + 1, HISTORY_BINS + FORECAST_BINS + 1)], axis=1
    )
    return centre_basis(bounds)


def token_features(r: int, op: str, arm: str) -> dict[str, np.ndarray]:
    """Everything about the observation tokens that does not depend on the values.

    M and O receive the same width, the same operation id and the same report
    interval.  They differ only in `time_basis`.
    """
    sup = observation_support(r, op)
    t = _tau(sup.astype(float))
    basis = integrated_basis(sup) if arm == "O" else centre_basis(sup)
    return {
        "time_basis": basis.astype(np.float32),
        "width": (t[:, 1] - t[:, 0]).astype(np.float32)[:, None],
        "op_id": np.full(len(sup), OP_INDEX[op], dtype=np.int64),
        "support": sup,
    }


# --------------------------------------------------------- R: operator-aware solve


def second_difference_matrix(n: int = HISTORY_BINS) -> np.ndarray:
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return D


class ReconstructionSolver:
    """x_hat = argmin ||A x - v||^2 + lambda ||D2 x||^2.

    The normal equations depend only on (r, op, lambda), so the Cholesky factor is
    built once per combination and reused for every window.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[int, str, float], tuple[np.ndarray, np.ndarray]] = {}

    def factor(self, r: int, op: str, lam: float) -> tuple[np.ndarray, np.ndarray]:
        key = (r, op, float(lam))
        if key not in self._cache:
            A = observation_matrix(r, op)
            D = second_difference_matrix()
            M = A.T @ A + lam * (D.T @ D)
            self._cache[key] = (np.linalg.cholesky(M), A)
        return self._cache[key]

    def reconstruct(self, v: np.ndarray, r: int, op: str, lam: float) -> np.ndarray:
        """v : (batch, n_tokens) observed values -> (batch, 288) base history."""
        L, A = self.factor(r, op, lam)
        rhs = v @ A                                   # (batch, 288) == (A^T v)^T
        y = np.linalg.solve(L, rhs.T)
        return np.linalg.solve(L.T, y).T


def reconstruction_operator(r: int, op: str, lam: float) -> np.ndarray:
    """The dense map from observations to the reconstructed base history,

        x_hat = P v,   P = (A^T A + lambda D2^T D2)^{-1} A^T

    Built once per (r, op, lambda) and applied as a single matmul afterwards.
    """
    A = observation_matrix(r, op)
    D = second_difference_matrix()
    return np.linalg.solve(A.T @ A + lam * (D.T @ D), A.T)


def base_bin_bounds() -> np.ndarray:
    """Support of every base bin in the 288-bin history."""
    return np.stack([np.arange(HISTORY_BINS), np.arange(1, HISTORY_BINS + 1)], axis=1)
