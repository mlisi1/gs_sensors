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

    @staticmethod
    def _activate(xyz, raw_opacity, raw_scaling, raw_rotation, features_dc, features_rest):
        """Shared by render_fields (boolean-mask path) and
        GaussianRasterizerWrapper's contiguous-slice-gather path (see
        reorder_) -- same activation formulas, different way of arriving at
        the (already-reduced-to-candidates) raw tensors fed in here."""
        return (
            xyz.float(),
            torch.sigmoid(raw_opacity.float()),
            torch.exp(raw_scaling.float()),
            torch.nn.functional.normalize(raw_rotation.float()),
            torch.cat((features_dc.float(), features_rest.float()), dim=1),
        )

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
        less memory to get there.

        Boolean-mask indexing here is O(N) regardless of mask sparsity (it
        has to scan/compact the full N-length mask) -- fine for the
        brute-force (mask=None) and no-octree cases this is still used for,
        but GaussianRasterizerWrapper's main per-frame path no longer calls
        this with a mask; see its _gather_leaf_slices, which reaches
        _activate directly with contiguous-slice-gathered (not
        boolean-masked) raw tensors instead, for exactly this reason."""
        if mask is None:
            return self._activate(self.xyz, self.raw_opacity, self.raw_scaling,
                                   self.raw_rotation, self.features_dc, self.features_rest)
        return self._activate(
            self.xyz[mask], self.raw_opacity[mask], self.raw_scaling[mask],
            self.raw_rotation[mask], self.features_dc[mask], self.features_rest[mask],
        )

    def reorder_(self, perm: torch.Tensor) -> None:
        """In-place permutation of every raw per-splat tensor -- called
        once at load time (not per-frame) to put the model into an
        octree's leaf-contiguous order (perm = that octree's
        flat_indices), so a leaf's points become directly sliceable as
        model.xyz[node_offsets[j]:node_offsets[j+1]] with no further
        indirection. This is what makes GaussianRasterizerWrapper's
        contiguous-slice gather possible instead of boolean-mask indexing
        -- see culling.py's module docstring and Kestrel's renderer.py
        (the prior art this mirrors) for why that distinction matters:
        boolean masking is O(N) to compact regardless of how few points
        survive, contiguous slicing of a pre-sorted array is not."""
        self.xyz = self.xyz[perm].contiguous()
        self.raw_opacity = self.raw_opacity[perm].contiguous()
        self.raw_scaling = self.raw_scaling[perm].contiguous()
        self.raw_rotation = self.raw_rotation[perm].contiguous()
        self.features_dc = self.features_dc[perm].contiguous()
        self.features_rest = self.features_rest[perm].contiguous()
