"""Camera pose -> view/projection matrices, in the exact convention
diff-surfel-rasterization's CUDA kernel expects: row-major matrices,
transposed relative to the textbook column-vector form
(`world_view_transform = w2c.T`, `full_proj_transform = (w2c @ P).T`) --
this is the standard Gaussian-Splatting-family camera convention, required
for the kernel to interpret the matrices correctly, not a choice made here.

Principal-point offset is NOT modeled: the projection matrix is built from
fov alone (derived from fx/fy), implicitly assuming cx=width/2, cy=height/2.
This is a limitation of the upstream CUDA rasterizer itself (its projection
matrix has no cx/cy term) -- see docs/coordinate_frames.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from gs_sensor_core.camera_profiles.schema import CameraProfile
from gs_sensor_core.frames import Pose
from gs_sensor_core.rotations import quat_to_rotmat

ZNEAR = 0.01
ZFAR = 100.0


@dataclass
class RenderCamera:
    width: int
    height: int
    fov_x: float
    fov_y: float
    world_view_transform: torch.Tensor   # [4, 4]
    full_proj_transform: torch.Tensor    # [4, 4]
    camera_center: torch.Tensor          # [3]


def _world_to_view_transposed(r_w2c: np.ndarray, t_w2c: np.ndarray) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = r_w2c
    m[:3, 3] = t_w2c
    return m.T


def _projection_transposed(fov_x: float, fov_y: float) -> np.ndarray:
    tan_half_x = math.tan(fov_x * 0.5)
    tan_half_y = math.tan(fov_y * 0.5)
    top, bottom = tan_half_y * ZNEAR, -tan_half_y * ZNEAR
    right, left = tan_half_x * ZNEAR, -tan_half_x * ZNEAR

    p = np.zeros((4, 4), dtype=np.float64)
    p[0, 0] = 2.0 * ZNEAR / (right - left)
    p[1, 1] = 2.0 * ZNEAR / (top - bottom)
    p[0, 2] = (right + left) / (right - left)
    p[1, 2] = (top + bottom) / (top - bottom)
    p[3, 2] = 1.0
    p[2, 2] = ZFAR / (ZFAR - ZNEAR)
    p[2, 3] = -(ZFAR * ZNEAR) / (ZFAR - ZNEAR)
    return p.T


def build_camera(pose_gs: Pose, profile: CameraProfile, device: str = "cuda") -> RenderCamera:
    """`pose_gs` must already be in GS-training space and the optical-frame
    axis convention (x-right, y-down, z-forward) -- see frames.py."""
    r_c2w = quat_to_rotmat(pose_gs.orientation)
    r_w2c = r_c2w.T
    t_w2c = -r_w2c @ pose_gs.position

    fov_x = 2.0 * math.atan(profile.width / (2.0 * profile.fx))
    fov_y = 2.0 * math.atan(profile.height / (2.0 * profile.fy))

    world_view = _world_to_view_transposed(r_w2c, t_w2c)
    full_proj = world_view @ _projection_transposed(fov_x, fov_y)

    return RenderCamera(
        width=profile.width,
        height=profile.height,
        fov_x=fov_x,
        fov_y=fov_y,
        world_view_transform=torch.tensor(world_view, dtype=torch.float32, device=device),
        full_proj_transform=torch.tensor(full_proj, dtype=torch.float32, device=device),
        # Camera center in world = camera position (see docs/coordinate_frames.md
        # for the derivation from inv(world_view_transform)).
        camera_center=torch.tensor(pose_gs.position, dtype=torch.float32, device=device),
    )
