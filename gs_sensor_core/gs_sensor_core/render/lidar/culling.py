"""LiDAR-specific broad-phase visibility/LOD tests, built on the same
generic octree the camera branch uses (`gs_sensor_core.culling.Octree`/
`build_octree`/`load_or_build_octree` -- imported here, not duplicated).

A camera-style frustum test doesn't apply: a LiDAR pose's two panoramic
passes (`render/lidar/camera.py`) together already cover the full
360-degree azimuth, so excluding splats *by direction* has limited payoff
-- the only genuinely "invisible" region is outside the (narrow) vertical
FOV band, tested by `visible_leaf_mask_lidar_torch` below. The dominant
rendering cost instead comes from splat *count*: at this sensor's actual
resolution (tens of thousands of rays per frame, vs. a camera's millions of
pixels), most splats subtend an angle far smaller than a single ray's own
angular pitch -- redundant detail the sensor can't resolve, not something
invisible. `angular_size_mask_torch` is the level-of-detail decision that
addresses that (paired with `gs_sensor_core.lod.build_leaf_proxies`, which
is already fully generic and needs no LiDAR-specific version).
"""
from __future__ import annotations


def visible_leaf_mask_lidar_torch(node_aabbs, sensor_position, r_w2c,
                                   vfov_min_deg: float, vfov_max_deg: float,
                                   margin_deg: float = 5.0):
    """Broad-phase vertical-FOV-band test, vectorized over all leaves ->
    [L] bool. For each leaf, computes the elevation angle (in the sensor's
    own local optical frame, matching the vendored kernel's own
    `theta = atan2(sqrt(x^2+z^2), -y)` convention, re-expressed here as
    `elevation_deg = 90 - degrees(theta)` so +90 = straight up, -90 =
    straight down, 0 = horizon -- the same sign convention `LidarProfile`'s
    own `vfov` already uses) at each of the leaf's 8 AABB corners, and
    keeps the leaf if that corner-derived [min, max] elevation range
    overlaps `[vfov_min_deg - margin_deg, vfov_max_deg + margin_deg]`.

    `r_w2c`: the FORWARD pass's world-to-sensor-local rotation (3x3, e.g.
    from `render/lidar/camera.py`'s own `r_w2c_fwd`) -- NOT camera-specific
    per pass: the backward pass's local frame only flips the local X/Z axes
    (`_FLIP_LOCAL_XZ` in camera.py), which leaves elevation (a function of
    local Y and sqrt(x^2+z^2), both invariant to X/Z sign flips) identical
    for both passes, so one shared test correctly covers both.

    Corner-based min/max is a standard, but not perfectly exact, bound
    (elevation is a nonlinear function of position; an AABB whose interior
    passes directly over/under the sensor could in principle have an
    interior extremum no corner captures) -- the same category of
    approximation `culling.py`'s own Gribb-Hartmann corner test makes for
    camera frustum culling, not treated as exact there either. `margin_deg`
    is the deliberate conservative buffer for this, tune from what's
    actually visible at the vfov edges, not derived from splat scale."""
    import torch

    device = node_aabbs.device
    aabb_min = node_aabbs[:, :3]  # [L, 3]
    aabb_max = node_aabbs[:, 3:]  # [L, 3]

    # 8 corners per leaf via a fixed (min/max)-per-axis selector, [8, 3] bool.
    sel = torch.tensor(
        [[i, j, k] for i in range(2) for j in range(2) for k in range(2)],
        device=device, dtype=torch.bool,
    )
    corners = torch.where(
        sel.unsqueeze(0), aabb_max.unsqueeze(1), aabb_min.unsqueeze(1)
    )  # [L, 8, 3]

    delta = corners - sensor_position.view(1, 1, 3)  # [L, 8, 3]
    local = torch.einsum('ij,lkj->lki', r_w2c, delta)  # r_w2c @ delta per corner
    lx, ly, lz = local[..., 0], local[..., 1], local[..., 2]
    theta = torch.atan2(torch.sqrt(lx * lx + lz * lz), -ly)  # [L, 8], radians
    elevation_deg = 90.0 - torch.rad2deg(theta)

    leaf_elev_min = elevation_deg.amin(dim=1)  # [L]
    leaf_elev_max = elevation_deg.amax(dim=1)  # [L]

    return (leaf_elev_max >= vfov_min_deg - margin_deg) & (leaf_elev_min <= vfov_max_deg + margin_deg)


def angular_size_mask_torch(leaf_center, leaf_radius, sensor_position,
                             ray_angular_pitch_rad: float, cutoff: float = 1.0):
    """LOD decision: `True` (keep full per-splat detail) if a leaf's
    angular size, as seen from `sensor_position`, is at least
    `cutoff * ray_angular_pitch_rad`; `False` (use its precomputed LOD
    proxy instead, see `lod.py`'s `build_leaf_proxies`) if the leaf is
    smaller than the sensor's own per-ray angular resolution can usefully
    tell apart from its neighbors.

    Small-angle approximation (`angular_size_rad ~= leaf_radius /
    distance`) -- same style as `culling.py`'s
    `visible_point_mask_screen_size_torch` pinhole approximation for the
    camera branch, just angular instead of pixel-projected. Deliberately
    has no near/far range term: unlike range-based culling (which would
    need to replicate the vendored kernel's own `near_n`/`far_n` CUDA
    constants in Python, a vendored-internals duplication risk not worth
    taking here), angular size shrinks with distance on its own, so nearby
    content naturally stays fine-detail without a separate check.

    `leaf_center`: [L, 3]. `leaf_radius`: [L, 1] (half the leaf's AABB
    diagonal, same convention `GaussianRasterizerWrapper.__init__` already
    computes for the camera branch's own LOD). `ray_angular_pitch_rad`: the
    sensor's own per-ray angular pitch (radians), the tighter of its
    horizontal/vertical resolution -- computed by the caller from
    `LidarProfile.hfov`/`vfov`/`hw`, not this function's concern."""
    import torch

    dist = torch.norm(leaf_center - sensor_position.unsqueeze(0), dim=-1).clamp_min(1e-6)
    angular_size_rad = leaf_radius.squeeze(-1) / dist
    return angular_size_rad >= (cutoff * ray_angular_pitch_rad)
