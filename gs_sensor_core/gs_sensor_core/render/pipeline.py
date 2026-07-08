"""Narrow public interface: pose in, RGB (+ depth) out. No ROS types, no
image-processing decisions (JPEG quality, distortion, etc.) -- see CLAUDE.md
"Rendering core".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gs_sensor_core.camera_profiles.schema import CameraProfile
from gs_sensor_core.culling import Octree
from gs_sensor_core.frames import Pose
from gs_sensor_core.models.gaussian_model import GaussianModel
from gs_sensor_core.render.camera import build_camera
from gs_sensor_core.render.rasterizer import GaussianRasterizerWrapper


@dataclass
class RenderResult:
    rgb: np.ndarray             # (H, W, 3) uint8
    depth: np.ndarray | None    # (H, W) float32, metric meters
    num_rendered: int           # splats actually rendered this frame (post-culling)


class CameraRasterizer:
    """One instance per simulated camera. Holds the loaded model + this
    camera's intrinsics profile; `render()` is the per-frame entry point."""

    def __init__(self, model: GaussianModel, profile: CameraProfile,
                 gs_scale: float = 1.0, publish_depth: bool = True, device: str = "cuda",
                 octree: Octree | None = None, culling_enabled: bool = True):
        self.profile = profile
        self.gs_scale = gs_scale
        self.publish_depth = publish_depth
        self.device = device
        self._rasterizer = GaussianRasterizerWrapper(
            model, device=device, octree=octree, culling_enabled=culling_enabled)

    def render(self, pose_gs: Pose) -> RenderResult:
        """`pose_gs` must already be in GS-training space (see frames.py) and
        in the optical-frame axis convention."""
        import torch

        camera = build_camera(pose_gs, self.profile, device=self.device)
        with torch.no_grad():
            output = self._rasterizer.render(camera)
            rgb = (output.rgb.clamp(0., 1.)
                   .permute(1, 2, 0).mul(255).byte().cpu().numpy())
            depth = None
            if self.publish_depth:
                depth = (output.depth / self.gs_scale).cpu().numpy().astype(np.float32)
        return RenderResult(rgb=rgb, depth=depth, num_rendered=output.num_rendered)
