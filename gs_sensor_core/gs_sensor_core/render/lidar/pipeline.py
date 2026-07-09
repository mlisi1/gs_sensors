"""Narrow public interface: pose (+ timestamp) in, xyz + intensity points
out. No ROS types, no PointCloud2 packing, no raydrop-threshold policy
decisions beyond the default -- mirrors `render/pipeline.py`'s
`CameraRasterizer` shape, see CLAUDE.md "Rendering core".
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from gs_sensor_core.frames import Pose
from gs_sensor_core.lidar_profiles.schema import LidarProfile
from gs_sensor_core.models.lidar_checkpoint_loader import RayDropPrior
from gs_sensor_core.models.lidar_gaussian_model import LidarGaussianModel
from gs_sensor_core.render.lidar.camera import build_lidar_cameras
from gs_sensor_core.render.lidar.pointcloud import pano_to_points
from gs_sensor_core.render.lidar.rasterizer import render_lidar_panorama
from gs_sensor_core.render.lidar.refine import RaydropRefineUNet, refine_raydrop


@dataclass
class LidarRenderResult:
    points_xyz: np.ndarray       # [K, 3] float32, metric meters, LiDAR-local frame
    intensity: np.ndarray        # [K] float32, ~[0, 1]
    num_returned: int            # points surviving the raydrop threshold
    num_rendered: int            # splats actually rendered this frame (post-prefilter)
    timings: dict[str, float] | None = None


class LidarRasterizer:
    """One instance per simulated LiDAR. Holds the loaded model + raydrop
    prior (+ optional refine UNet) + this sensor's profile; `render()` is
    the per-frame entry point."""

    def __init__(self, model: LidarGaussianModel, raydrop_prior: RayDropPrior,
                 profile: LidarProfile, refine_unet: RaydropRefineUNet | None = None,
                 gs_scale: float = 1.0, dynamic: bool = False, device: str = "cuda",
                 raydrop_threshold: float = 0.5):
        self.model = model
        self.raydrop_prior = raydrop_prior
        self.profile = profile
        self.refine_unet = refine_unet
        self.gs_scale = gs_scale
        self.dynamic = dynamic
        self.device = device
        self.raydrop_threshold = raydrop_threshold

    def render(self, pose_gs: Pose, timestamp: float = 0.0, profile: bool = False) -> LidarRenderResult:
        """`pose_gs` must already be in GS-training space (see frames.py)
        and the LiDAR's optical-like frame convention -- see
        `render/lidar/camera.py`/`pointcloud.py` docstrings for the
        caveats on that convention. `timestamp` only matters for a
        dynamic-captured model (see `LidarGaussianModel.get_xyz_SHM`);
        `Crosslab_lidar` was trained static, so `0.0` is the only value
        exercised against real data so far."""
        timings: dict[str, float] | None = {} if profile else None
        t0 = time.perf_counter()

        def lap(name: str) -> None:
            nonlocal t0
            if timings is None:
                return
            if self.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            now = time.perf_counter()
            timings[name] = (now - t0) * 1000.0
            t0 = now

        cam_forward, cam_backward = build_lidar_cameras(pose_gs, self.profile, device=self.device)
        lap("build_camera")

        with torch.no_grad():
            pano = render_lidar_panorama(
                self.model, self.raydrop_prior, cam_forward, cam_backward,
                timestamp=timestamp, dynamic=self.dynamic,
                scale_factor=self.profile.scale_factor,
            )
            lap("rasterize")

            raydrop = pano.raydrop
            if self.refine_unet is not None:
                raydrop = refine_raydrop(self.refine_unet, pano.raydrop, pano.intensity, pano.depth)
                lap("refine")

            depth = pano.depth * (raydrop <= self.raydrop_threshold).float()
            points_xyz, points_intensity = pano_to_points(depth, pano.intensity, vfov=self.profile.vfov)
            lap("unproject")

            points_xyz = (points_xyz / self.gs_scale).cpu().numpy().astype(np.float32)
            points_intensity = points_intensity.clamp(0.0, 1.0).cpu().numpy().astype(np.float32)
            lap("copy_to_cpu")

        return LidarRenderResult(
            points_xyz=points_xyz, intensity=points_intensity,
            num_returned=points_xyz.shape[0], num_rendered=pano.num_rendered,
            timings=timings,
        )
