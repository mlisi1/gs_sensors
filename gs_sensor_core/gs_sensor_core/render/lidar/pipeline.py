"""Narrow public interface: pose (+ timestamp) in, xyz + intensity points
out. No ROS types, no PointCloud2 packing, no raydrop-threshold policy
decisions beyond the default -- mirrors `render/pipeline.py`'s
`CameraRasterizer` shape, see CLAUDE.md "Rendering core".
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from gsplat2d_rendering.culling import Octree
from gsplat2d_rendering.math_utils.rotations import quat_to_rotmat

from gs_sensor_core.frames import Pose
from gs_sensor_core.lidar_profiles.schema import LidarProfile
from gs_sensor_core.models.lidar_checkpoint_loader import RayDropPrior
from gs_sensor_core.models.lidar_gaussian_model import LidarGaussianModel
from gs_sensor_core.render.lidar.camera import build_lidar_cameras
from gs_sensor_core.render.lidar.culling import angular_size_mask_torch, visible_leaf_mask_lidar_torch
from gs_sensor_core.render.lidar.pointcloud import pano_to_points
from gs_sensor_core.render.lidar.rasterizer import render_lidar_panorama
from gs_sensor_core.render.lidar.refine import RaydropRefineUNet, refine_raydrop

# Positional order matching LidarGaussianModel's raw-tensor fields -- used by
# _gather_leaf_slices/_append_proxies below, same "raw tuple" convention
# gsplat2d_rendering's render/rasterizer.py (the camera-branch equivalent) uses.
_RAW_FIELDS = (
    "xyz", "raw_opacity", "raw_scaling", "raw_rotation",
    "features_dc", "features_rest", "raw_t", "raw_scaling_t", "velocity",
)


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
    the per-frame entry point.

    PRECONDITION when an octree is supplied: `model` must already be
    permuted into that octree's leaf-contiguous order via
    `model.reorder_(octree.flat_indices)` -- caller's job (see
    lidar_debug_node.py), same contract as the camera branch's
    `GaussianRasterizerWrapper`.

    `octree_lod`/`lod_ray_pitch_cutoff`: level-of-detail, not visibility
    culling -- a LiDAR pose's two panoramic passes already cover the full
    360-degree azimuth, so excluding splats by direction has limited
    payoff (see `render/lidar/culling.py`'s module docstring). What
    actually matches this sensor's real bottleneck (tens of thousands of
    rays vs. a camera's millions of pixels, so most splats are finer
    detail than any ray can resolve) is merging distant/small splats into
    precomputed proxies, same mechanism as the camera branch's
    `octree_lod` (`lod.py`'s `build_leaf_proxies`), just gated by angular
    size instead of screen-pixel size.

    Known limitation: LOD proxies always render as static (zero velocity,
    `raw_t=0`) -- `build_leaf_proxies` doesn't compute a merged velocity/
    temporal-center for a proxy, so combining `dynamic=True` with
    `octree_lod=True` would silently render proxy regions as non-moving
    while real (non-proxied) splats move -- `__init__` warns if both are
    set, since the one real trained model available (`Crosslab_lidar`) is
    static (`dynamic=False`) and hasn't exercised this combination."""

    def __init__(self, model: LidarGaussianModel, raydrop_prior: RayDropPrior,
                 profile: LidarProfile, refine_unet: RaydropRefineUNet | None = None,
                 gs_scale: float = 1.0, dynamic: bool = False, device: str = "cuda",
                 raydrop_threshold: float = 0.5,
                 range_noise_stddev_m: float = 0.0, intensity_noise_stddev: float = 0.0,
                 octree: Octree | None = None, culling_enabled: bool = True,
                 culling_margin_deg: float = 5.0,
                 octree_lod: bool = False, lod_ray_pitch_cutoff: float = 1.0):
        """`range_noise_stddev_m`/`intensity_noise_stddev`: synthetic
        per-frame measurement noise -- 0.0 (default) disables each
        independently, same "empty/zero sentinel = off" convention as
        `refine_unet_path`. The trained field itself is smooth/deterministic
        (a continuous function fit by gradient descent across many real
        frames converges toward the mean return per ray direction, not the
        raw per-shot noise a real sensor has) -- see CLAUDE.md/TODO.md for
        why this doesn't come "for free" the way it might from a real
        capture. `range_noise_stddev_m` perturbs the *range* (radially,
        before unprojection), matching how real LiDAR accuracy specs are
        quoted (e.g. VLP-16's "+/-3cm"), not an isotropic 3D xyz jitter,
        which would distort the unprojection geometry unrealistically."""
        self.model = model
        self.raydrop_prior = raydrop_prior
        self.profile = profile
        self.refine_unet = refine_unet
        self.gs_scale = gs_scale
        self.dynamic = dynamic
        self.device = device
        self.raydrop_threshold = raydrop_threshold
        self.range_noise_stddev_m = range_noise_stddev_m
        self.intensity_noise_stddev = intensity_noise_stddev
        self.culling_margin_deg = culling_margin_deg
        self.octree_lod = octree_lod
        self.lod_ray_pitch_cutoff = lod_ray_pitch_cutoff

        if dynamic and octree_lod:
            import warnings
            warnings.warn(
                "octree_lod=True with dynamic=True: LOD proxies always render "
                "as static (build_leaf_proxies doesn't compute a merged "
                "velocity/raw_t) -- proxy regions won't move while real "
                "splats do. Untested combination, see LidarRasterizer's "
                "class docstring.")

        self._has_octree = culling_enabled and octree is not None
        self.octree = octree
        self._node_aabbs_gpu = None
        self._node_offsets_gpu = None
        self._leaf_center_gpu = None
        self._leaf_radius_gpu = None
        self._proxy_xyz_gpu = None
        self._proxy_scale_gpu = None
        self._proxy_rotation_gpu = None
        self._proxy_opacity_gpu = None
        self._proxy_features_dc_gpu = None
        if self._has_octree:
            self._node_aabbs_gpu = torch.from_numpy(octree.node_aabbs).to(device)
            self._node_offsets_gpu = torch.from_numpy(octree.node_offsets).to(device)
            self._leaf_center_gpu = (self._node_aabbs_gpu[:, :3] + self._node_aabbs_gpu[:, 3:]) * 0.5
            self._leaf_radius_gpu = (
                (self._node_aabbs_gpu[:, 3:] - self._node_aabbs_gpu[:, :3]) * 0.5
            ).amax(dim=-1, keepdim=True)
            if octree_lod and octree.has_lod:
                self._proxy_xyz_gpu = torch.from_numpy(octree.proxy_xyz).to(device)
                self._proxy_scale_gpu = torch.from_numpy(octree.proxy_scale).to(device)
                self._proxy_rotation_gpu = torch.from_numpy(octree.proxy_rotation).to(device)
                self._proxy_opacity_gpu = torch.from_numpy(octree.proxy_opacity).to(device)
                self._proxy_features_dc_gpu = torch.from_numpy(octree.proxy_features_dc).to(device)

        # Ray angular pitch (radians), tighter of the two axes -- see
        # culling.py's angular_size_mask_torch docstring for why this needs
        # no near/far range term.
        h_range_rad = math.radians(profile.hfov[1] - profile.hfov[0])
        v_range_rad = math.radians(profile.vfov[1] - profile.vfov[0])
        self._ray_pitch_rad = min(h_range_rad / profile.hw[1], v_range_rad / profile.hw[0])

    def _gather_leaf_slices(self, leaf_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """Same single-index contiguous gather as the camera branch's
        `_gather_leaf_slices` (gsplat2d_rendering's render/rasterizer.py) --
        touches only the
        visible K points, never the full N. Returns a dict keyed by
        `_RAW_FIELDS` so the caller can build a `LidarGaussianModel`
        directly from it."""
        model = self.model
        visible = torch.nonzero(leaf_mask, as_tuple=True)[0]
        if visible.numel() == 0:
            return {name: getattr(model, name).new_zeros((0,) + getattr(model, name).shape[1:])
                    for name in _RAW_FIELDS}
        starts = self._node_offsets_gpu[visible]
        ends = self._node_offsets_gpu[visible + 1]
        lengths = ends - starts
        total = int(lengths.sum().item())
        idx = torch.repeat_interleave(starts, lengths) + (
            torch.arange(total, device=starts.device)
            - torch.repeat_interleave(torch.cumsum(lengths, 0) - lengths, lengths)
        )
        return {name: getattr(model, name)[idx] for name in _RAW_FIELDS}

    def _append_proxies(self, raw: dict[str, torch.Tensor], leaf_coarse: torch.Tensor) -> dict[str, torch.Tensor]:
        """Concatenates coarse leaves' proxy splats onto the gathered
        fine-detail raw tensors, converting proxy values (already
        activated -- see `lod.py`'s `build_leaf_proxies`) back through the
        inverse of `LidarGaussianModel`'s own activation functions
        (`logit` for opacity, `log` for scaling) so they round-trip
        correctly through `get_opacity`/`get_scaling` exactly like a real
        raw splat would -- this is what lets `render_lidar_panorama` take
        the resulting `LidarGaussianModel` completely unchanged, no
        separate proxy-color code path needed (unlike the camera branch's
        `_colors_with_proxies`, which has to special-case proxy SH
        evaluation because `GaussianRasterizerWrapper` activates outside
        the model). `raw_rotation` needs no inverse (`get_rotation`
        re-normalizes, idempotent on an already-unit quaternion).
        `features_rest` for proxy rows is zero-padded to the real splats'
        own K (SH band count) -- zero higher-order SH coefficients make
        the kernel's SH evaluation correctly reduce to the DC-only term for
        those rows, the same "proxies render at SH degree 0" behavior the
        camera branch's proxies have, achieved here via the shared
        `sh_degree` the kernel call uses for every splat in one invocation
        rather than a separate color path.
        `raw_t=0`/`velocity=0` for every proxy row -- see this class's
        docstring for why (no merged velocity available, static-only)."""
        proxy_idx = torch.nonzero(leaf_coarse, as_tuple=True)[0]
        if proxy_idx.numel() == 0:
            return raw
        n_proxy = proxy_idx.numel()
        eps = 1e-4
        proxy_opacity = self._proxy_opacity_gpu[proxy_idx].clamp(eps, 1.0 - eps)
        raw_opacity_proxy = torch.logit(proxy_opacity)
        # self._proxy_scale_gpu is already 3 columns here -- the octree
        # this rasterizer was built with must have been built via
        # load_or_build_octree(..., keep_normal_axis=True) (lidar_debug_
        # node.py's job), matching LidarGaussianModel.raw_scaling's genuine
        # 3D extent (this kernel is 3D-GS-family, not the camera branch's
        # 2D surfel one -- see lod.py's build_leaf_proxies docstring for
        # why the column count differs by kernel). A 2-column proxy_scale
        # here (an index built for the camera branch instead) would raise
        # a real shape-mismatch error at the torch.cat below (concatenating
        # a 2-column proxy against the 3-column gathered-fine-splat
        # tensor), not silently render wrong -- that's deliberate, see
        # culling.py's load_or_build_octree docstring.
        raw_scaling_proxy = torch.log(self._proxy_scale_gpu[proxy_idx].clamp_min(eps))
        raw_rotation_proxy = self._proxy_rotation_gpu[proxy_idx]
        features_dc_proxy = self._proxy_features_dc_gpu[proxy_idx]  # [n_proxy, 1, 4], no activation to invert
        k_rest = raw["features_rest"].shape[1]
        features_rest_proxy = features_dc_proxy.new_zeros((n_proxy, k_rest, 4))
        xyz_proxy = self._proxy_xyz_gpu[proxy_idx]
        zeros_1 = features_dc_proxy.new_zeros((n_proxy, 1))
        zeros_3 = features_dc_proxy.new_zeros((n_proxy, 3))

        proxy_raw = {
            "xyz": xyz_proxy, "raw_opacity": raw_opacity_proxy, "raw_scaling": raw_scaling_proxy,
            "raw_rotation": raw_rotation_proxy, "features_dc": features_dc_proxy,
            "features_rest": features_rest_proxy, "raw_t": zeros_1, "raw_scaling_t": zeros_1,
            "velocity": zeros_3,
        }
        return {name: torch.cat([raw[name], proxy_raw[name]], dim=0) for name in _RAW_FIELDS}

    def _cull_and_gather(self, pose_gs: Pose) -> LidarGaussianModel:
        """Vertical-FOV-band broad phase -> angular-size LOD split ->
        gather fine leaves + append coarse leaves' proxies -> a new
        `LidarGaussianModel` built from the (possibly much smaller)
        result. See `render/lidar/culling.py`'s module docstring for why
        this replaces camera-style frustum culling for a panoramic sensor."""
        model = self.model
        r_w2c_fwd = torch.tensor(
            quat_to_rotmat(pose_gs.orientation).T, dtype=torch.float32, device=self.device)
        sensor_position = torch.tensor(pose_gs.position, dtype=torch.float32, device=self.device)

        leaf_vis = visible_leaf_mask_lidar_torch(
            self._node_aabbs_gpu, sensor_position, r_w2c_fwd,
            self.profile.vfov[0], self.profile.vfov[1], margin_deg=self.culling_margin_deg)

        leaf_fine, leaf_coarse = leaf_vis, None
        if self.octree_lod and self._proxy_xyz_gpu is not None:
            leaf_full_detail = angular_size_mask_torch(
                self._leaf_center_gpu, self._leaf_radius_gpu, sensor_position,
                self._ray_pitch_rad, cutoff=self.lod_ray_pitch_cutoff)
            leaf_coarse = leaf_vis & ~leaf_full_detail
            leaf_fine = leaf_vis & leaf_full_detail

        raw = self._gather_leaf_slices(leaf_fine)
        if leaf_coarse is not None:
            raw = self._append_proxies(raw, leaf_coarse)

        return LidarGaussianModel(
            active_sh_degree=model.active_sh_degree, T=model.T, velocity_decay=model.velocity_decay,
            **raw,
        )

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

        render_model = self.model
        if self._has_octree:
            render_model = self._cull_and_gather(pose_gs)
        lap("cull")

        with torch.no_grad():
            pano = render_lidar_panorama(
                render_model, self.raydrop_prior, cam_forward, cam_backward,
                timestamp=timestamp, dynamic=self.dynamic,
                scale_factor=self.profile.scale_factor,
            )
            lap("rasterize")

            raydrop = pano.raydrop
            if self.refine_unet is not None:
                raydrop = refine_raydrop(self.refine_unet, pano.raydrop, pano.intensity, pano.depth)
                lap("refine")

            depth = pano.depth * (raydrop <= self.raydrop_threshold).float()
            if self.range_noise_stddev_m > 0.0:
                valid = depth > 0.0
                noise_gs = torch.randn_like(depth) * (self.range_noise_stddev_m * self.gs_scale)
                depth = torch.where(valid, (depth + noise_gs).clamp_min(0.0), depth)
                lap("range_noise")

            points_xyz, points_intensity = pano_to_points(depth, pano.intensity, vfov=self.profile.vfov)
            lap("unproject")

            if self.intensity_noise_stddev > 0.0:
                points_intensity = points_intensity + torch.randn_like(points_intensity) * self.intensity_noise_stddev

            points_xyz = (points_xyz / self.gs_scale).cpu().numpy().astype(np.float32)
            points_intensity = points_intensity.clamp(0.0, 1.0).cpu().numpy().astype(np.float32)
            lap("copy_to_cpu")

        return LidarRenderResult(
            points_xyz=points_xyz, intensity=points_intensity,
            num_returned=points_xyz.shape[0], num_rendered=pano.num_rendered,
            timings=timings,
        )
