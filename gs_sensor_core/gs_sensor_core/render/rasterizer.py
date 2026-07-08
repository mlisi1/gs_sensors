"""Wraps diff-surfel-rasterization directly: SH evaluation + the CUDA
tile-sort/alpha-composite kernel + extraction of depth from its extra-
channels output buffer.

The channel layout of the rasterizer's second output tensor (7 channels) is
fixed by the CUDA kernel itself (cuda_rasterizer/auxiliary.h): DEPTH_OFFSET=0
(accumulated depth*weight), ALPHA_OFFSET=1, NORMAL_OFFSET=2..4,
MIDDEPTH_OFFSET=5, DISTORTION_OFFSET=6 -- confirmed against that header, not
assumed. Only depth (accumulated depth / alpha, i.e. the expected depth) is
extracted here; phase 1 doesn't publish normals or distortion.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from gs_sensor_core.culling import Octree, visible_point_mask
from gs_sensor_core.models.gaussian_model import GaussianModel
from gs_sensor_core.render.camera import RenderCamera
from gs_sensor_core.sh_utils import C0, eval_sh

_DEPTH_OFFSET = 0
_ALPHA_OFFSET = 1


@dataclass
class RenderOutput:
    rgb: torch.Tensor       # [3, H, W], float, ~[0, 1]
    depth: torch.Tensor     # [H, W], float, GS-training-space units
    num_rendered: int       # splats actually passed to the rasterizer this frame


class GaussianRasterizerWrapper:
    """One instance per loaded model -- the model is loaded once, this is
    reused every frame with a new camera. `octree`/`culling_enabled` are
    optional: with no octree, every splat is rendered every frame."""

    def __init__(self, model: GaussianModel, device: str = "cuda",
                 octree: Octree | None = None, culling_enabled: bool = True):
        from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        self._settings_cls = GaussianRasterizationSettings
        self._rasterizer_cls = GaussianRasterizer

        self.model = model
        self.device = device
        self.octree = octree
        self.culling_enabled = culling_enabled
        self.background = torch.zeros(3, dtype=torch.float32, device=device)
        self.last_visible_count = model.num_points

    def _visible_mask(self, camera: RenderCamera) -> torch.Tensor | None:
        if not (self.culling_enabled and self.octree is not None):
            return None
        full_proj_np = camera.full_proj_transform.detach().cpu().numpy()
        mask_np = visible_point_mask(self.octree, full_proj_np, self.model.num_points)
        return torch.from_numpy(mask_np).to(self.device)

    def _compute_colors(self, means3D: torch.Tensor, shs: torch.Tensor, camera: RenderCamera) -> torch.Tensor:
        degree = self.model.active_sh_degree
        if degree > 0:
            dirs = means3D - camera.camera_center
            dirs = dirs / (dirs.norm(dim=1, keepdim=True) + 1e-8)
            sh_dim = (degree + 1) ** 2
            colors = eval_sh(degree, shs.transpose(1, 2)[:, :, :sh_dim], dirs)
            return torch.clamp_min(colors + 0.5, 0.0)
        return torch.clamp_min(C0 * shs[:, 0, :] + 0.5, 0.0)

    def render(self, camera: RenderCamera) -> RenderOutput:
        model = self.model
        mask = self._visible_mask(camera)

        if mask is not None:
            means3D = model.get_xyz[mask]
            opacity = model.get_opacity[mask]
            scales = model.get_scaling[mask]
            rotations = model.get_rotation[mask]
            shs = model.get_features[mask]
        else:
            means3D = model.get_xyz
            opacity = model.get_opacity
            scales = model.get_scaling
            rotations = model.get_rotation
            shs = model.get_features

        self.last_visible_count = int(means3D.shape[0])
        means2D = torch.zeros_like(means3D)
        colors = self._compute_colors(means3D, shs, camera)

        raster_settings = self._settings_cls(
            image_height=int(camera.height),
            image_width=int(camera.width),
            tanfovx=math.tan(camera.fov_x * 0.5),
            tanfovy=math.tan(camera.fov_y * 0.5),
            bg=self.background,
            scale_modifier=1.0,
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            sh_degree=model.active_sh_degree,
            campos=camera.camera_center,
            prefiltered=False,
            debug=False,
        )
        rasterizer = self._rasterizer_cls(raster_settings=raster_settings)

        rendered_image, _radii, allmap = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=colors,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
        )

        alpha = allmap[_ALPHA_OFFSET:_ALPHA_OFFSET + 1]
        depth = torch.nan_to_num(
            allmap[_DEPTH_OFFSET:_DEPTH_OFFSET + 1] / alpha, nan=0.0, posinf=0.0, neginf=0.0
        )
        return RenderOutput(rgb=rendered_image, depth=depth.squeeze(0), num_rendered=self.last_visible_count)
