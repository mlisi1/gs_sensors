"""In-memory 2D Gaussian Splat model: raw optimized tensors + the activation
functions that turn them into renderable quantities.

Deliberately holds only what a *trained* model needs at render time -- no
optimizer state, no densification, no training-only bookkeeping. Activation
functions (exp / sigmoid / normalize) match the 2D-GS paper's own model
(scale is stored log-space, opacity as an inverse-sigmoid logit, rotation as
a raw un-normalized quaternion) -- this is the PLY file format's contract,
not a choice specific to any particular viewer implementation.

Raw tensors may be float16 (compression_level >= 1, see compression.py) to
halve resting VRAM and the memory traffic masking/gathering has to move --
but every accessor below always hands back float32: the rasterizer CUDA
kernel is hardcoded float32 (third_party/diff-surfel-rasterization has no
dtype dispatch at all), so fp16 storage is an implementation detail of this
class, never something a caller needs to know about. `.float()` is a no-op
(returns self, no copy) when a tensor is already float32, so this costs
nothing at compression_level 0.
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
        return self.xyz.float()

    @property
    def get_opacity(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_opacity.float())

    @property
    def get_scaling(self) -> torch.Tensor:
        return torch.exp(self.raw_scaling.float())

    @property
    def get_rotation(self) -> torch.Tensor:
        return torch.nn.functional.normalize(self.raw_rotation.float())

    @property
    def get_features(self) -> torch.Tensor:
        return torch.cat((self.features_dc.float(), self.features_rest.float()), dim=1)

    @property
    def num_points(self) -> int:
        return self.xyz.shape[0]

    def render_fields(self, mask: torch.Tensor | None = None):
        """(xyz, opacity, scaling, rotation, features), activated, restricted
        to `mask` if given. Indexes the *raw* tensors before activating them
        rather than activating-then-indexing (what naive `get_opacity[mask]`
        etc. does) -- sigmoid/exp/normalize/cat are nontrivial per-point
        ops, so doing them over the full model and discarding most of the
        result makes their cost independent of how much culling actually
        removes. This is why it belongs on the model, not the caller: only
        this class knows which raw field maps to which activation.

        Masks *before* upcasting to float32, so the masking/gather itself
        moves fp16-sized data when the model is stored that way, but
        activation math (sigmoid/exp/normalize) still always runs in
        float32 on the upcast result -- numerically the same as computing
        everything in float32 on a once-fp16-rounded input, just touching
        less memory to get there."""
        if mask is None:
            return self.get_xyz, self.get_opacity, self.get_scaling, self.get_rotation, self.get_features
        return (
            self.xyz[mask].float(),
            torch.sigmoid(self.raw_opacity[mask].float()),
            torch.exp(self.raw_scaling[mask].float()),
            torch.nn.functional.normalize(self.raw_rotation[mask].float()),
            torch.cat((self.features_dc[mask].float(), self.features_rest[mask].float()), dim=1),
        )
