"""Fixed fit-support residualization prototype, not a novel PEFT algorithm."""

import torch


class FixedSupportProjection:
    def __init__(self, z: torch.Tensor, rtol: float = 1e-10):
        self.z = z.detach()
        u, s, vh = torch.linalg.svd(self.z, full_matrices=False)
        keep = s > rtol * s[0]
        self.singular_values = s
        self.rank = int(keep.sum())
        self.u = u[:, keep]
        self.coefficient_map = (vh[keep].T / s[keep]) @ self.u.T

    def coefficients(self, delta: torch.Tensor) -> torch.Tensor:
        return self.coefficient_map @ delta

    def complement(self, delta: torch.Tensor) -> torch.Tensor:
        return delta - self.u @ (self.u.T @ delta)

    def extend(self, z_new: torch.Tensor, delta_new: torch.Tensor,
               fit_delta: torch.Tensor) -> torch.Tensor:
        return delta_new - z_new @ self.coefficients(fit_delta)
