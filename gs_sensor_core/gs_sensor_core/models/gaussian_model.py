"""In-memory 2D Gaussian Splat model: raw optimized tensors + the activation
functions that turn them into renderable quantities.

Deliberately holds only what a *trained* model needs at render time -- no
optimizer state, no densification, no training-only bookkeeping. Activation
functions (exp / sigmoid / normalize) match the 2D-GS paper's own model
(scale is stored log-space, opacity as an inverse-sigmoid logit, rotation as
a raw un-normalized quaternion) -- this is the PLY file format's contract,
not a choice specific to any particular viewer implementation.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GaussianModel:
    xyz: torch.Tensor              # [N, 3]
    raw_opacity: torch.Tensor      # [N, 1], pre-sigmoid
    raw_scaling: torch.Tensor      # [N, 3], pre-exp (log-space)
    raw_rotation: torch.Tensor     # [N, 4], un-normalized quaternion (w, x, y, z)
    features_dc: torch.Tensor      # [N, 1, 3]
    features_rest: torch.Tensor    # [N, K, 3]
    active_sh_degree: int

    @property
    def get_xyz(self) -> torch.Tensor:
        return self.xyz

    @property
    def get_opacity(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_opacity)

    @property
    def get_scaling(self) -> torch.Tensor:
        return torch.exp(self.raw_scaling)

    @property
    def get_rotation(self) -> torch.Tensor:
        return torch.nn.functional.normalize(self.raw_rotation)

    @property
    def get_features(self) -> torch.Tensor:
        return torch.cat((self.features_dc, self.features_rest), dim=1)

    @property
    def num_points(self) -> int:
        return self.xyz.shape[0]
