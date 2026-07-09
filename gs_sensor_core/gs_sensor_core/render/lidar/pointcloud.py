"""Stitched range panorama -> xyz point cloud, in the LiDAR's own local
(optical-like) frame -- ported from
`~/GS-LiDAR/utils/graphics_utils.py:pano_to_lidar`.

Unprojects using the *full* stitched azimuth range `hfov=(-180, 180)`
(matching `~/GS-LiDAR/scripts/export_ply.py`'s own call:
`pano_to_lidar(pred_depth, args.vfov, (-180, 180))`), not the per-half
`(-90, 90)` each individual render pass used -- the stitched panorama built
by `rasterizer.py` already spans the full circle.

No camera-pose rotation/translation is applied here: the returned points
are directly in the frame implied by the theta/phi formula below (derived
to be REP103-like -- z-forward, y-down, x-right -- from
`get_local_directions_panorama`'s axis layout, but **not yet independently
verified for LiDAR**; the validation script is what actually confirms this
against `test_data/gs_lidar_source/qa/`). This matches ROS's own
`sensor_msgs/PointCloud2` convention of publishing points relative to the
sensor's own frame_id and letting TF place them in world/map -- so
`lidar_debug_node.py` does not need to re-apply the pose transform GS-
LiDAR's kernel already used to select which splats landed where.
"""
from __future__ import annotations

import torch


def pano_to_points(range_image: torch.Tensor, intensity_image: torch.Tensor,
                    vfov: tuple[float, float], hfov: tuple[float, float] = (-180.0, 180.0)):
    """`range_image`/`intensity_image`: `[1, h, w]`. Returns `(xyz [K, 3],
    intensity [K])` for the `range_image > 0` pixels only."""
    valid = range_image > 0
    h, w = range_image.shape[-2:]
    device = range_image.device
    theta, phi = torch.meshgrid(
        torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")

    vertical_range = vfov[1] - vfov[0]
    theta = (90.0 - vfov[1] + theta / h * vertical_range) * torch.pi / 180.0

    horizontal_range = hfov[1] - hfov[0]
    phi = (hfov[0] + phi / w * horizontal_range) * torch.pi / 180.0

    dx = torch.sin(theta) * torch.sin(phi)
    dz = torch.sin(theta) * torch.cos(phi)
    dy = -torch.cos(theta)
    directions = torch.nn.functional.normalize(torch.stack([dx, dy, dz], dim=0), dim=0)

    points_xyz = (directions * range_image)[:, valid[0]].permute(1, 0)
    points_intensity = intensity_image[valid]
    return points_xyz, points_intensity
