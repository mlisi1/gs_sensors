"""Narrow public interface: pose in, RGB (+ depth) out. No ROS types, no
image-processing decisions (JPEG quality, distortion, etc.) -- see CLAUDE.md
"Rendering core". Thin adapter over gsplat2d_rendering.render.pipeline.Renderer
(vendored at gs_sensor_core/third_party/gsplat2d-rendering) -- the actual
rasterization/culling/LOD implementation lives there now, this class only
translates Pose/CameraProfile into a gsplat2d_rendering Camera (_build_camera
below) and converts the library's model-space-units depth into this
project's metric meters via `gs_scale` (see frames.py's GSFrameTransform).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gsplat2d_rendering.camera import Camera, Intrinsics
from gsplat2d_rendering.culling import Octree
from gsplat2d_rendering.math_utils.rotations import quat_to_rotmat
from gsplat2d_rendering.model import GaussianModel
from gsplat2d_rendering.render.pipeline import Renderer

from gs_sensor_core.camera_profiles.schema import CameraProfile
from gs_sensor_core.frames import Pose


@dataclass
class RenderResult:
    rgb: np.ndarray             # (H, W, 3) uint8
    depth: np.ndarray | None    # (H, W) float32, metric meters
    num_rendered: int           # splats actually rendered this frame (post-culling)
    timings: dict[str, float] | None = None  # stage -> ms, only populated when profile=True


def _build_camera(pose_gs: Pose, profile: CameraProfile, device: str) -> Camera:
    """`pose_gs` must already be in GS-training space and the optical-frame
    axis convention (x-right, y-down, z-forward) -- see frames.py. See
    gsplat2d_rendering.camera's module docstring for the exact matrix/axis
    convention this builds against, and camera_profiles/schema.py's
    docstring for why cx/cy aren't modeled here (the upstream CUDA
    rasterizer's projection matrix has no principal-point term)."""
    r_c2w = quat_to_rotmat(pose_gs.orientation)
    intrinsics = Intrinsics(width=profile.width, height=profile.height, fx=profile.fx, fy=profile.fy)
    return Camera.from_c2w(r_c2w, pose_gs.position, intrinsics, device=device)


class CameraRasterizer:
    """One instance per simulated camera. Holds the loaded model + this
    camera's intrinsics profile; `render()` is the per-frame entry point."""

    def __init__(self, model: GaussianModel, profile: CameraProfile,
                 gs_scale: float = 1.0, publish_depth: bool = True, device: str = "cuda",
                 octree: Octree | None = None, culling_enabled: bool = True,
                 culling_narrow_phase: bool = False, culling_margin: float = 0.0,
                 screen_size_culling: bool = False, screen_size_min_pixels: float = 1.0,
                 octree_lod: bool = False, lod_leaf_pixel_threshold: float = 16.0):
        self.profile = profile
        self.gs_scale = gs_scale
        self.publish_depth = publish_depth
        self.device = device
        self._renderer = Renderer(
            model, device=device, with_depth=publish_depth, octree=octree, culling_enabled=culling_enabled,
            culling_narrow_phase=culling_narrow_phase,
            culling_margin=culling_margin, screen_size_culling=screen_size_culling,
            screen_size_min_pixels=screen_size_min_pixels, octree_lod=octree_lod,
            lod_leaf_pixel_threshold=lod_leaf_pixel_threshold)

    def render(self, pose_gs: Pose, profile: bool = False) -> RenderResult:
        """`pose_gs` must already be in GS-training space (see frames.py) and
        in the optical-frame axis convention. `profile=True` breaks down
        render time by stage (see RenderResult.timings) at the cost of extra
        torch.cuda.synchronize() calls -- wire it to the same debug flag
        that already gates the total-time log line, not a hot-path default."""
        camera = _build_camera(pose_gs, self.profile, device=self.device)
        # enable_profiling()/disable_profiling() are just a bool flag flip on
        # the underlying SplatRenderer -- cheap enough to toggle every frame
        # rather than requiring the caller to manage it separately.
        if profile:
            self._renderer.enable_profiling()
        else:
            self._renderer.disable_profiling()
        output = self._renderer.render(camera)
        depth = (output.depth / self.gs_scale) if output.depth is not None else None
        timings = self._renderer.get_last_timings() if profile else None
        return RenderResult(rgb=output.rgb, depth=depth, num_rendered=output.num_rendered, timings=timings)
