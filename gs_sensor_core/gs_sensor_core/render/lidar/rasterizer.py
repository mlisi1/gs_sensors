"""Wraps the vendored panoramic kernel (`_kernel.py`): renders the
forward+backward pass pair for one LiDAR pose, composites the raydrop-prior
envmap, and stitches both into one full panorama.

Ported faithfully from `~/GS-LiDAR/gaussian_renderer/__init__.py`'s
`render()` (per-pass render + opacity/marginal_t prefilter mask) and
`render_range_map()` (the two-pass loop, depth mean/median variance-gated
blend, and the `breaks`-index stitch) -- see those functions for the
reference this mirrors. Simplifications made deliberately for an inference-
only renderer (not training/eval): only the final blended "mix" depth
channel is kept (GS-LiDAR's own `export_ply.py` only ever uses
`depth_pano[[0]]`, i.e. this same channel, for anything downstream of
training); `sky_depth`'s harmonic-mean far-plane blend is wired through as a
flag (off by default, matching `Crosslab_lidar`'s `sky_depth: False`) rather
than hardcoded out, since a future model could train with it on.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from gs_sensor_core.models.lidar_checkpoint_loader import RayDropPrior
from gs_sensor_core.models.lidar_gaussian_model import LidarGaussianModel
from gs_sensor_core.render.lidar._kernel import LidarRasterizationSettings, rasterize
from gs_sensor_core.render.lidar.camera import LidarRenderCamera, build_lidar_cameras

# color/feature output channel layout: GS-LiDAR's SH features are 4-channel
# (r, g, b, <repurposed channel>) -- for a LiDAR-only capture the RGB
# channels carry no real signal, only channels 2 and 3 are used downstream.
# Confirmed against `~/GS-LiDAR/gaussian_renderer/__init__.py`'s render():
# `_, rendered_intensity_sh, rendered_raydrop = rendered_image.split([2, 1, 1], dim=0)`.
_INTENSITY_SH_CHANNEL = 2
_RAYDROP_CHANNEL = 3

# GS-LiDAR's kernel takes tanfovx/tanfovy but does its own equirectangular
# projection from vfov/hfov -- training used `neg_fov: True` (base.yaml,
# unmodified by Crosslab_lidar's config), which sets both to tan(-0.5)
# unconditionally rather than deriving them from any real FOV. Kept as a
# literal constant, not derived, to match what the model was evaluated
# against -- deriving "the correct" tanfov here would silently diverge from
# training-time tile-binning behavior for no benefit.
_NEG_FOV_TANFOV = math.tan(-0.5)

# Background for the 4-channel composite: (r, g, b, raydrop). Raydrop's
# background is 1.0, not 0.0 -- rays that hit nothing (infinite depth) must
# default to "dropped", per GS-LiDAR's own `train.py:65` comment
# ("infinite distance's raydrop probability is 1") and confirmed against
# `Crosslab_lidar`'s own `setting.txt` (`white_background: False`, so
# `bg_color = [0, 0, 0, 1]`, not all-zero).
_BACKGROUND = torch.tensor([0.0, 0.0, 0.0, 1.0])


@dataclass
class LidarPanorama:
    depth: torch.Tensor       # [1, h, w*2], GS-training-space units
    intensity: torch.Tensor   # [1, h, w*2], ~[0, 1]
    raydrop: torch.Tensor     # [1, h, w*2], ~[0, 1] probability (not yet thresholded)
    num_rendered: int


def _prefilter_mask(opacity: torch.Tensor, marginal_t: torch.Tensor | None,
                     dynamic: bool) -> torch.Tensor:
    mask = opacity[:, 0] > (1.0 / 255.0)
    if dynamic and marginal_t is not None:
        mask = mask & (marginal_t[:, 0] > 0.05)
    return mask


def _render_one_pass(model: LidarGaussianModel, cam: LidarRenderCamera,
                      timestamp: float, dynamic: bool, scale_factor: float,
                      raydrop_prior: RayDropPrior, sky_depth: bool, sky_depth_m: float):
    means3D = model.get_xyz_SHM(timestamp)
    marginal_t = model.get_marginal_t(timestamp)
    opacity = model.get_opacity
    if dynamic:
        opacity = opacity * marginal_t
    mask = _prefilter_mask(opacity, marginal_t, dynamic)

    settings = LidarRasterizationSettings(
        image_height=cam.height, image_width=cam.width,
        tanfovx=_NEG_FOV_TANFOV, tanfovy=_NEG_FOV_TANFOV,
        bg=_BACKGROUND.to(means3D.device),
        scale_modifier=1.0,
        viewmatrix=cam.world_view_transform,
        projmatrix=cam.world_view_transform,  # no separate projection matrix, see camera.py
        sh_degree=model.active_sh_degree,
        campos=cam.camera_center,
        prefiltered=False, debug=False,
        vfov=cam.vfov, hfov=cam.hfov, scale_factor=scale_factor,
    )
    _contrib, color, _feature, depth4, alpha, _radii = rasterize(
        means3D, opacity, model.get_features, model.get_scaling, model.get_rotation,
        settings, mask=mask,
    )

    depth_mean, depth_median = depth4[[0]], depth4[[1]]
    depth_square = depth4[[3]]
    depth_var = depth_square - depth_mean ** 2
    var_quantile = depth_var.median() * 10.0
    depth_mix = torch.where(depth_var > var_quantile, depth_median, depth_mean)

    if sky_depth:
        # Harmonic-mean far-plane blend, ported from render()'s
        # `depth_blend_mode == 0` branch -- the only mode Crosslab_lidar's
        # training config would have used if sky_depth had been on (it
        # wasn't: `sky_depth: False`), so this path is untested against
        # real data. Kept for a future model that trains with it.
        eps = 1e-5
        depth_mix = depth_mix / alpha.clamp_min(eps)
        depth_mix = 1.0 / (alpha / depth_mix.clamp_min(eps) + (1.0 - alpha) / sky_depth_m).clamp_min(eps)

    intensity_sh = color[[_INTENSITY_SH_CHANNEL]]
    raydrop_render = color[[_RAYDROP_CHANNEL]]
    prior = raydrop_prior(cam.towards)
    # Composite on the raw (unclamped) kernel output, clamp only the final
    # result -- matches reference order exactly (`gaussian_renderer/__init__.py`
    # composites first, `.clamp(0, 1)`s only at the return). Clamping
    # `raydrop_render` before compositing would change the result whenever
    # the raw SH-evaluated channel goes outside [0, 1], which it can.
    raydrop = (prior + (1.0 - prior) * raydrop_render).clamp(0.0, 1.0)

    return depth_mix, intensity_sh, raydrop, int(mask.sum().item())


def render_lidar_panorama(
    model: LidarGaussianModel, raydrop_prior: RayDropPrior, cam_forward: LidarRenderCamera,
    cam_backward: LidarRenderCamera, timestamp: float = 0.0, dynamic: bool = False,
    scale_factor: float = 1.0, sky_depth: bool = False, sky_depth_m: float = 900.0,
) -> LidarPanorama:
    h, w = cam_forward.height, cam_forward.width
    depth_pano = torch.zeros((1, h, w * 2), device=model.xyz.device)
    intensity_pano = torch.zeros((1, h, w * 2), device=model.xyz.device)
    raydrop_pano = torch.zeros((1, h, w * 2), device=model.xyz.device)

    breaks = (0, w // 2, w + w // 2, w * 2)  # (0, w/2, 3w/2, 2w), see module docstring
    n_rendered = 0
    for cam in (cam_forward, cam_backward):
        depth, intensity, raydrop, n = _render_one_pass(
            model, cam, timestamp, dynamic, scale_factor, raydrop_prior, sky_depth, sky_depth_m)
        n_rendered += n
        if cam.towards == "forward":
            depth_pano[:, :, breaks[1]:breaks[2]] = depth
            intensity_pano[:, :, breaks[1]:breaks[2]] = intensity
            raydrop_pano[:, :, breaks[1]:breaks[2]] = raydrop
        else:
            depth_pano[:, :, breaks[2]:breaks[3]] = depth[:, :, 0:(breaks[3] - breaks[2])]
            depth_pano[:, :, breaks[0]:breaks[1]] = depth[:, :, (w - breaks[1] + breaks[0]):w]
            intensity_pano[:, :, breaks[2]:breaks[3]] = intensity[:, :, 0:(breaks[3] - breaks[2])]
            intensity_pano[:, :, breaks[0]:breaks[1]] = intensity[:, :, (w - breaks[1] + breaks[0]):w]
            raydrop_pano[:, :, breaks[2]:breaks[3]] = raydrop[:, :, 0:(breaks[3] - breaks[2])]
            raydrop_pano[:, :, breaks[0]:breaks[1]] = raydrop[:, :, (w - breaks[1] + breaks[0]):w]

    return LidarPanorama(depth=depth_pano, intensity=intensity_pano, raydrop=raydrop_pano,
                          num_rendered=n_rendered)
