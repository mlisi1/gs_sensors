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
import time
from dataclasses import dataclass

import torch

from gs_sensor_core.culling import (
    Octree,
    point_leaf_ids,
    visible_leaf_mask_torch,
    visible_point_mask,
    visible_point_mask_exact_torch,
)
from gs_sensor_core.models.gaussian_model import GaussianModel
from gs_sensor_core.render.camera import RenderCamera
from gs_sensor_core.sh_utils import C0, eval_sh

_DEPTH_OFFSET = 0
_ALPHA_OFFSET = 1
_CULLING_BACKENDS = ("cpu", "gpu")


@dataclass
class RenderOutput:
    rgb: torch.Tensor       # [3, H, W], float, ~[0, 1]
    depth: torch.Tensor     # [H, W], float, GS-training-space units
    num_rendered: int       # splats actually passed to the rasterizer this frame
    timings: dict[str, float] | None = None  # stage -> ms, only populated when profile=True


class GaussianRasterizerWrapper:
    """One instance per loaded model -- the model is loaded once, this is
    reused every frame with a new camera. `octree`/`culling_enabled` are
    optional: with no octree, every splat is rendered every frame.

    `culling_backend`: "cpu" (default) runs the existing numpy octree test
    and uploads the resulting point mask; "gpu" runs the same Gribb-Hartmann
    test in torch directly on the camera's projection matrix, with no
    GPU->CPU->GPU round trip. Left as a flag rather than a flat switch
    because which one is actually faster depends on splat count vs. leaf
    count vs. PCIe bandwidth on the target machine -- see the benchmark note
    on `visible_point_mask` in culling.py. Measure with `profile=True`
    before assuming either is a win."""

    def __init__(self, model: GaussianModel, device: str = "cuda",
                 octree: Octree | None = None, culling_enabled: bool = True,
                 culling_backend: str = "cpu", culling_narrow_phase: bool = False,
                 culling_margin: float = 0.0):
        from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        self._settings_cls = GaussianRasterizationSettings
        self._rasterizer_cls = GaussianRasterizer

        if culling_backend not in _CULLING_BACKENDS:
            raise ValueError(f"culling_backend must be one of {_CULLING_BACKENDS}, got {culling_backend!r}")

        self.model = model
        self.device = device
        self.octree = octree
        self.culling_enabled = culling_enabled
        self.culling_backend = culling_backend
        self.culling_narrow_phase = culling_narrow_phase
        self.culling_margin = culling_margin
        self.background = torch.zeros(3, dtype=torch.float32, device=device)
        self.last_visible_count = model.num_points

        self._node_aabbs_gpu = None
        self._point_leaf_id_gpu = None
        if culling_enabled and octree is not None and culling_backend == "gpu":
            self._node_aabbs_gpu = torch.from_numpy(octree.node_aabbs).to(device)
            self._point_leaf_id_gpu = torch.from_numpy(point_leaf_ids(octree)).to(device)

    def _visible_mask(self, camera: RenderCamera) -> torch.Tensor | None:
        if not (self.culling_enabled and self.octree is not None):
            return None
        if self.culling_backend == "gpu":
            leaf_vis = visible_leaf_mask_torch(self._node_aabbs_gpu, camera.full_proj_transform)
            return leaf_vis[self._point_leaf_id_gpu]
        full_proj_np = camera.full_proj_transform.detach().cpu().numpy()
        mask_np = visible_point_mask(self.octree, full_proj_np, self.model.num_points)
        return torch.from_numpy(mask_np).to(self.device)

    def _narrow_phase_mask(self, mask: torch.Tensor, camera: RenderCamera) -> torch.Tensor:
        """Exact per-point refinement of the leaf-level broad-phase `mask`,
        restricted to points that already passed it -- the leaf test keeps
        a whole leaf's points if the leaf's AABB merely touches the
        frustum, so at scene edges this can be a real over-count. By
        construction this can only ever remove points, never add any (it's
        AND'd onto the broad-phase result), so it can't render something
        the broad phase had already excluded."""
        idx = torch.nonzero(mask, as_tuple=True)[0]
        if idx.numel() == 0:
            return mask
        xyz_candidates = self.model.get_xyz[idx]
        exact = visible_point_mask_exact_torch(xyz_candidates, camera.full_proj_transform, margin=self.culling_margin)
        refined = torch.zeros_like(mask)
        refined[idx] = exact
        return refined

    def _compute_colors(self, means3D: torch.Tensor, shs: torch.Tensor, camera: RenderCamera) -> torch.Tensor:
        degree = self.model.active_sh_degree
        if degree > 0:
            dirs = means3D - camera.camera_center
            dirs = dirs / (dirs.norm(dim=1, keepdim=True) + 1e-8)
            sh_dim = (degree + 1) ** 2
            colors = eval_sh(degree, shs.transpose(1, 2)[:, :, :sh_dim], dirs)
            return torch.clamp_min(colors + 0.5, 0.0)
        return torch.clamp_min(C0 * shs[:, 0, :] + 0.5, 0.0)

    def _sync(self) -> None:
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    def render(self, camera: RenderCamera, profile: bool = False) -> RenderOutput:
        # Stage timings cost a torch.cuda.synchronize() each -- normally the
        # GPU pipeline overlaps across stages, so measuring per-stage wall
        # time requires forcing sync points that wouldn't otherwise happen.
        # Skipped entirely unless profile=True, so it doesn't tax the hot path.
        timings: dict[str, float] | None = {} if profile else None
        t = time.perf_counter()

        def lap(name: str) -> None:
            nonlocal t
            if timings is None:
                return
            self._sync()
            now = time.perf_counter()
            timings[name] = (now - t) * 1000.0
            t = now

        model = self.model
        mask = self._visible_mask(camera)
        lap("cull")

        if mask is not None and self.culling_narrow_phase:
            mask = self._narrow_phase_mask(mask, camera)
        lap("narrow_cull")

        # render_fields masks the raw tensors before activating them, so
        # this scales with visible count instead of total model size -- see
        # GaussianModel.render_fields.
        means3D, opacity, scales, rotations, shs = model.render_fields(mask)
        lap("gather")

        self.last_visible_count = int(means3D.shape[0])
        means2D = torch.zeros_like(means3D)
        colors = self._compute_colors(means3D, shs, camera)
        lap("sh_eval")

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
        lap("rasterize")

        alpha = allmap[_ALPHA_OFFSET:_ALPHA_OFFSET + 1]
        depth = torch.nan_to_num(
            allmap[_DEPTH_OFFSET:_DEPTH_OFFSET + 1] / alpha, nan=0.0, posinf=0.0, neginf=0.0
        )
        lap("depth_extract")
        return RenderOutput(
            rgb=rendered_image, depth=depth.squeeze(0),
            num_rendered=self.last_visible_count, timings=timings,
        )
