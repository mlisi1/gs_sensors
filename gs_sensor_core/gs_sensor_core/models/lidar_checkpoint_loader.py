"""Loads a trained GS-LiDAR checkpoint (`ckpt/chkpnt<N>.pth` +
`ckpt/lidar_raydrop_prior_chkpnt<N>.pth`) into a `LidarGaussianModel` +
`RayDropPrior`.

Unlike the camera branch's PLY loader, there is no PLY-equivalent portable
file format here -- GS-LiDAR's own checkpoint is a `torch.save`'d tuple in
the exact field order its `GaussianModel.capture()`/`RayDropPrior.capture()`
produce (`~/GS-LiDAR/scene/gaussian_model.py:84-106`,
`~/GS-LiDAR/scene/raydrop_prior.py:12-16`). This module `torch.load`s that
tuple directly and unpacks it into gs_sensor_core's own classes -- it does
NOT import GS-LiDAR's `scene.gaussian_model`/`scene.raydrop_prior` at
runtime (same zero-runtime-dependency rule as the rest of gs_sensor_core;
GS-LiDAR is prior art/vendored-for-its-CUDA-kernel-only, not a Python
dependency). If GS-LiDAR's checkpoint tuple layout ever changes upstream,
this is the one place that needs updating.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from gs_sensor_core.models.lidar_gaussian_model import LidarGaussianModel

# Positional indices into GaussianModel.capture()'s tuple -- see this
# module's docstring for the source. Everything past _INTENSITY is
# training-only bookkeeping (grad accumulators, optimizer state) and is
# never unpacked.
(_SH_DEGREE, _XYZ, _FEATURES_DC, _FEATURES_REST, _SCALING, _ROTATION,
 _OPACITY, _T, _SCALING_T, _VELOCITY, _INTENSITY, _MAX_RADII2D,
 _XYZ_GRAD_ACCUM, _XYZ_GRAD_ACCUM_ABS, _T_GRAD_ACCUM, _DENOM,
 _OPTIMIZER_STATE, _SPATIAL_LR_SCALE, _CYCLE_T, _VELOCITY_DECAY) = range(20)


@dataclass
class RayDropPrior:
    """Envmap-shaped learned raydrop bias, paired 1:1 with a specific
    checkpoint iteration. Ported from `~/GS-LiDAR/scene/raydrop_prior.py`'s
    `forward()` -- a plain function here rather than an `nn.Module` since no
    optimizer/training state is needed at inference."""
    prior: torch.Tensor  # [1, h, w * 2] -- forward half then backward half along width

    def __call__(self, towards: str) -> torch.Tensor:
        w = self.prior.shape[-1] // 2
        if towards == "forward":
            half = self.prior[:, :, :w]
        elif towards == "backward":
            half = self.prior[:, :, w:]
        else:
            raise ValueError(f"towards must be 'forward' or 'backward', got {towards!r}")
        return torch.sigmoid(half)

    def resize(self, h: int, w_per_half: int) -> "RayDropPrior":
        """Bilinear-resizes the envmap to a new (h, w_per_half) resolution --
        ported from GS-LiDAR's own `RayDropPrior.upscale()`
        (`~/GS-LiDAR/scene/raydrop_prior.py:39-42`, used there when its own
        training schedule steps resolution up). Needed whenever a caller
        renders at a resolution other than the one this checkpoint's envmap
        was saved at (e.g. `scripts/validate_lidar.py`'s native-resolution
        QA comparison vs. a profile's training-matched publish resolution)
        -- this is a fixed per-pixel learned bias tied to a pixel grid, not
        a function of ray direction, so it can't just be sliced/broadcast
        against a differently-sized render."""
        resized = torch.nn.functional.interpolate(
            self.prior[None], size=(h, w_per_half * 2), mode="bilinear", align_corners=True,
        )[0]
        return RayDropPrior(prior=resized)


def load_lidar_gaussian_model(path: str | Path, device: str = "cuda") -> LidarGaussianModel:
    model_args, iteration = torch.load(str(path), map_location=device, weights_only=False)
    model = LidarGaussianModel(
        xyz=model_args[_XYZ].to(device),
        raw_opacity=model_args[_OPACITY].to(device),
        raw_scaling=model_args[_SCALING].to(device),
        raw_rotation=model_args[_ROTATION].to(device),
        features_dc=model_args[_FEATURES_DC].to(device),
        features_rest=model_args[_FEATURES_REST].to(device),
        raw_t=model_args[_T].to(device),
        raw_scaling_t=model_args[_SCALING_T].to(device),
        velocity=model_args[_VELOCITY].to(device),
        active_sh_degree=int(model_args[_SH_DEGREE]),
        T=float(model_args[_CYCLE_T]),
        velocity_decay=float(model_args[_VELOCITY_DECAY]),
    )
    print(f"[gs_sensor_core] Loaded {model.num_points:,} LiDAR splats "
          f"(SH degree {model.active_sh_degree}, iteration {iteration}) from {path}")
    return model


def load_raydrop_prior(path: str | Path, device: str = "cuda") -> RayDropPrior:
    (prior_tensor, _optimizer_state), _iteration = torch.load(
        str(path), map_location=device, weights_only=False)
    return RayDropPrior(prior=prior_tensor.to(device))
