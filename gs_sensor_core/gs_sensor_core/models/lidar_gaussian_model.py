"""In-memory GS-LiDAR splat model: raw optimized tensors + the activation
functions that turn them into renderable quantities. Parallels
`gaussian_model.py`'s `GaussianModel`, but for GS-LiDAR's panoramic/
time-varying representation instead of 2D-GS surfels -- see CLAUDE.md's
"Scope for this iteration" for why this is a separate class rather than an
extension of the camera-branch one (different upstream format, different
vendored rasterizer, different channel semantics).

Two real differences from `GaussianModel`:

- `features_dc`/`features_rest` are 4-channel, not 3 -- GS-LiDAR reuses the
  SH machinery to carry an intensity channel alongside RGB (`[N, K, 4]`,
  channel index 3 = intensity), evaluated by the vendored kernel into the
  `intensity_sh` render output. Confirmed against
  `~/GS-LiDAR/scene/gaussian_model.py`'s `create_from_pcd` (`features =
  torch.zeros((N, 4, K))`, comment aside -- the RGB channels are unused for
  a LiDAR-only capture and always end up ~0, only channel 3 carries signal).
- Time-varying fields (`t`, `scaling_t`, `velocity`, `T`, `velocity_decay`)
  for GS-LiDAR's periodic-motion model (`get_xyz_SHM`/`get_marginal_t`
  below, ported from the same file). `Crosslab_lidar` itself was trained
  with `dynamic: False` (see `setting.txt`), so at `t=0` `get_marginal_t`
  evaluates to 1 everywhere and `get_xyz_SHM(0)` reduces to plain `xyz` --
  but the fields are still loaded and the formulas still run, rather than
  special-cased away, so a future dynamic-captured model works without a
  second model class.

Deliberately NOT carried (training-only, dropped at load like
`GaussianModel` drops optimizer state): `max_radii2D`, gradient-accumulation
buffers, `denom`, optimizer state, `spatial_lr_scale`, `max_sh_degree`
(`active_sh_degree` is all rendering needs), and the raw scalar `_intensity`
tensor GS-LiDAR's checkpoint carries -- confirmed unused by grepping GS-LiDAR
for `get_intensity`: defined but never called from `render()`/
`render_range_map()`/`train.py`, dead in the actual render path. The
per-pixel intensity gs_sensors publishes is the SH-evaluated `intensity_sh`
channel above, not this field.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class LidarGaussianModel:
    xyz: torch.Tensor              # [N, 3]
    raw_opacity: torch.Tensor      # [N, 1], pre-sigmoid
    raw_scaling: torch.Tensor      # [N, 3], pre-exp (log-space) -- 3 columns: GS-LiDAR's
                                    # vendored kernel is a 3D-GS-family fork, not a 2D surfel one
    raw_rotation: torch.Tensor     # [N, 4], un-normalized quaternion (w, x, y, z)
    features_dc: torch.Tensor      # [N, 1, 4]  (r, g, b, intensity)
    features_rest: torch.Tensor    # [N, K, 4]
    raw_t: torch.Tensor            # [N, 1], splat's temporal center
    raw_scaling_t: torch.Tensor    # [N, 1], pre-exp (log-space) temporal extent
    velocity: torch.Tensor         # [N, 3]
    active_sh_degree: int
    T: float                       # motion period ("cycle" in GS-LiDAR training args)
    velocity_decay: float

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
    def get_scaling_t(self) -> torch.Tensor:
        return torch.exp(self.raw_scaling_t.float())

    @property
    def get_rotation(self) -> torch.Tensor:
        return torch.nn.functional.normalize(self.raw_rotation.float())

    @property
    def get_features(self) -> torch.Tensor:
        return torch.cat((self.features_dc.float(), self.features_rest.float()), dim=1)

    @property
    def num_points(self) -> int:
        return self.xyz.shape[0]

    def get_xyz_SHM(self, t: float) -> torch.Tensor:
        """Splat position at time `t`, following a sinusoidal motion model
        with period `T` -- ported verbatim from
        `~/GS-LiDAR/scene/gaussian_model.py:151-152` (`get_xyz_SHM`). At
        `t == raw_t` (or for a static capture, everywhere) this reduces to
        `xyz`."""
        a = 1.0 / self.T * np.pi * 2.0
        return self.xyz.float() + self.velocity.float() * torch.sin((t - self.raw_t.float()) * a) / a

    def get_marginal_t(self, t: float) -> torch.Tensor:
        """Gaussian-in-time visibility weight, ported from
        `get_marginal_t` in the same file. Evaluates to ~1 for every splat
        when `t` matches training's fixed time (the `dynamic: False` case)."""
        return torch.exp(-0.5 * (self.raw_t.float() - t) ** 2 / self.get_scaling_t ** 2)

    @property
    def get_inst_velocity(self) -> torch.Tensor:
        return self.velocity.float() * torch.exp(
            -self.get_scaling_t / self.T / 2.0 * self.velocity_decay
        )
