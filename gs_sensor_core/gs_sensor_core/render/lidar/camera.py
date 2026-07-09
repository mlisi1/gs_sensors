"""LiDAR pose -> the pair of panoramic "cameras" GS-LiDAR's kernel renders
and stitches into one 360-degree scan.

A spinning LiDAR's full 360-degree azimuth range is rendered as two
opposing 180-degree (`hfov=(-90, 90)`) panoramic passes -- "forward" (the
pose's own heading) and "backward" (the same position, yaw-rotated 180
degrees) -- then stitched (see `rasterizer.py`). This split is GS-LiDAR's
own training-data convention (`~/GS-LiDAR/scene/kitti360_loader.py` builds
paired forward/backward `Camera`s per LiDAR pose), not a choice made here.

The forward/backward relationship (`R_backward = R_forward @ diag(-1, 1,
-1)`, i.e. negate the local x and z axes) is derived from
`~/GS-LiDAR/scene/cameras.py`'s `get_local_directions_panorama`: its
`towards == 'backward'` branch negates exactly those two direction
components relative to `'forward'`.

Same view-matrix convention as the camera branch's `render/camera.py`
(`world_view_transform = (W2C).T`, no separate projection matrix -- GS-
LiDAR's own `Camera.__init__` sets `projection_matrix = eye(4)`, i.e.
`full_proj_transform == world_view_transform`; confirmed by reading
`~/GS-LiDAR/scene/cameras.py`, not assumed, since the vendored kernel takes
vfov/hfov directly and does its own equirectangular projection instead of
using a projection matrix).

`pose_gs` must already be in GS-training space (see `frames.py`) and in the
same optical-frame axis convention as the camera branch (x-right, y-down,
z-forward) -- **not yet independently verified for LiDAR** (derived from
`get_local_directions_panorama`'s theta/phi formula matching that
convention, but this project's own validation script is what actually
confirms it against `test_data/gs_lidar_source/qa/`, see TODO.md).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from gs_sensor_core.frames import Pose
from gs_sensor_core.lidar_profiles.schema import LidarProfile
from gs_sensor_core.rotations import quat_to_rotmat

_FLIP_LOCAL_XZ = np.diag([-1.0, 1.0, -1.0])


@dataclass
class LidarRenderCamera:
    height: int
    width: int
    vfov: tuple[float, float]
    hfov: tuple[float, float]
    towards: str  # 'forward' or 'backward'
    world_view_transform: torch.Tensor  # [4, 4], == full_proj_transform (see module docstring)
    camera_center: torch.Tensor         # [3]


def _world_to_view_transposed(r_w2c: np.ndarray, t_w2c: np.ndarray) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = r_w2c
    m[:3, 3] = t_w2c
    return m.T


def build_lidar_cameras(
    pose_gs: Pose, profile: LidarProfile, device: str = "cuda",
) -> tuple[LidarRenderCamera, LidarRenderCamera]:
    """Returns `(cam_forward, cam_backward)` for one LiDAR pose."""
    r_c2w_fwd = quat_to_rotmat(pose_gs.orientation)

    cams = []
    for towards, r_c2w in (("forward", r_c2w_fwd), ("backward", r_c2w_fwd @ _FLIP_LOCAL_XZ)):
        r_w2c = r_c2w.T
        t_w2c = -r_w2c @ pose_gs.position
        world_view = _world_to_view_transposed(r_w2c, t_w2c)
        cams.append(LidarRenderCamera(
            height=profile.hw[0], width=profile.hw[1],
            vfov=tuple(profile.vfov), hfov=tuple(profile.hfov), towards=towards,
            world_view_transform=torch.tensor(world_view, dtype=torch.float32, device=device),
            camera_center=torch.tensor(pose_gs.position, dtype=torch.float32, device=device),
        ))
    return cams[0], cams[1]
