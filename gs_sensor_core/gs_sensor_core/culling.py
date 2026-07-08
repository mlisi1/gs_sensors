"""Octree spatial index (build/save/load) + GPU-native frustum culling.

Standard, generic graphics techniques -- an axis-aligned octree over splat
centers, and Gribb & Hartmann's plane-extraction-from-clip-matrix method for
the frustum test -- not tied to any particular renderer. Independent
implementation; the matrix convention matches `render/camera.py`
(`clip_row = point_row @ full_proj_transform`, so planes are extracted from
*columns* of `full_proj_transform`, not rows).

Only 5 planes are tested (left/right/top/bottom/near); the far plane is
deliberately omitted because the CUDA rasterizer itself doesn't hard-clip at
zfar, so culling against it would incorrectly drop splats it would have
still rendered.

Frustum tests here are all torch (GPU-native), not numpy: `render/
rasterizer.py`'s per-frame gather works from leaf-contiguous index ranges
(GaussianModel.reorder_ + node_offsets), not a per-point boolean mask, so
there's never a reason to materialize an N-length mask on CPU -- an earlier
numpy CPU-mask implementation existed here and was removed once the gather
rewrite made it unreachable; see TODO.md's rendering-performance section for
that history if you're looking for it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Octree:
    node_aabbs: np.ndarray     # [L, 6] float32 (xmin,ymin,zmin,xmax,ymax,zmax) per leaf
    node_offsets: np.ndarray   # [L+1] int64, into flat_indices
    flat_indices: np.ndarray   # [N] int64, permutation of point indices, leaf-ordered

    # Two-level LOD (lod.py's build_leaf_proxies): one merged "proxy"
    # Gaussian per leaf, precomputed at index-build time, swapped in for a
    # whole leaf's individual splats when the leaf's projected screen size
    # is small (see render/rasterizer.py's leaf_fine/leaf_coarse split).
    # None unless the index was built with octree_lod enabled -- an older
    # cached index without these fields degrades gracefully to "LOD
    # unavailable", not an error.
    proxy_xyz: np.ndarray | None = None             # [L, 3], world position
    proxy_scale: np.ndarray | None = None            # [L, 2], already activated (world units) -- 2D, see lod.py
    proxy_rotation: np.ndarray | None = None         # [L, 4], normalized quaternion (w, x, y, z)
    proxy_opacity: np.ndarray | None = None          # [L, 1], already activated ([0, 1])
    proxy_features_dc: np.ndarray | None = None      # [L, 1, 3], raw SH-DC space (pre C0/+0.5)

    @property
    def has_lod(self) -> bool:
        return self.proxy_xyz is not None


def build_octree(xyz: np.ndarray, leaf_max: int = 5000, max_depth: int = 8) -> Octree:
    n = xyz.shape[0]
    leaves_indices: list[np.ndarray] = []
    leaves_aabb: list[np.ndarray] = []

    stack: list[tuple[np.ndarray, int]] = [(np.arange(n, dtype=np.int64), 0)]
    while stack:
        indices, depth = stack.pop()
        if indices.size == 0:
            continue
        pts = xyz[indices]
        aabb_min = pts.min(axis=0)
        aabb_max = pts.max(axis=0)
        if indices.size <= leaf_max or depth >= max_depth:
            leaves_indices.append(indices)
            leaves_aabb.append(np.concatenate([aabb_min, aabb_max]))
            continue

        center = (aabb_min + aabb_max) * 0.5
        octant = (
            (pts[:, 0] >= center[0]).astype(np.int64)
            + (pts[:, 1] >= center[1]).astype(np.int64) * 2
            + (pts[:, 2] >= center[2]).astype(np.int64) * 4
        )
        for o in range(8):
            child = indices[octant == o]
            if child.size:
                stack.append((child, depth + 1))

    if leaves_indices:
        node_offsets = np.zeros(len(leaves_indices) + 1, dtype=np.int64)
        for i, idx in enumerate(leaves_indices):
            node_offsets[i + 1] = node_offsets[i] + idx.size
        flat_indices = np.concatenate(leaves_indices).astype(np.int64)
        node_aabbs = np.stack(leaves_aabb).astype(np.float32)
    else:
        node_offsets = np.zeros(1, dtype=np.int64)
        flat_indices = np.zeros(0, dtype=np.int64)
        node_aabbs = np.zeros((0, 6), dtype=np.float32)

    return Octree(node_aabbs=node_aabbs, node_offsets=node_offsets, flat_indices=flat_indices)


def save_octree(path: str | Path, octree: Octree) -> None:
    kwargs = dict(
        node_aabbs=octree.node_aabbs,
        node_offsets=octree.node_offsets,
        flat_indices=octree.flat_indices,
    )
    if octree.has_lod:
        kwargs.update(
            proxy_xyz=octree.proxy_xyz,
            proxy_scale=octree.proxy_scale,
            proxy_rotation=octree.proxy_rotation,
            proxy_opacity=octree.proxy_opacity,
            proxy_features_dc=octree.proxy_features_dc,
        )
    np.savez_compressed(str(path), **kwargs)


def load_octree(path: str | Path) -> Octree:
    data = np.load(str(path))
    has_lod = "proxy_xyz" in data.files
    return Octree(
        node_aabbs=data["node_aabbs"],
        node_offsets=data["node_offsets"],
        flat_indices=data["flat_indices"],
        proxy_xyz=data["proxy_xyz"] if has_lod else None,
        proxy_scale=data["proxy_scale"] if has_lod else None,
        proxy_rotation=data["proxy_rotation"] if has_lod else None,
        proxy_opacity=data["proxy_opacity"] if has_lod else None,
        proxy_features_dc=data["proxy_features_dc"] if has_lod else None,
    )


def visible_leaf_mask_torch(node_aabbs, full_proj_transform):
    """Gribb-Hartmann p-vertex test, vectorized over all leaves -> [L] bool,
    entirely in torch on `full_proj_transform`'s own device. `node_aabbs`
    must already be a torch tensor on that device. Local `import torch`
    keeps this module import-time numpy-only. This is the only frustum
    broad-phase test in the package now -- see the module docstring."""
    import torch

    m = full_proj_transform
    planes = torch.stack([
        m[:, 0] + m[:, 3],  # left
        m[:, 3] - m[:, 0],  # right
        m[:, 1] + m[:, 3],  # bottom
        m[:, 3] - m[:, 1],  # top
        m[:, 2],            # near
    ], dim=0)
    normals, d_vals = planes[:, :3], planes[:, 3]

    aabb_min = node_aabbs[:, :3]
    aabb_max = node_aabbs[:, 3:]

    pos_mask = normals.unsqueeze(1) >= 0  # [5, 1, 3]
    p_vertex = torch.where(pos_mask, aabb_max.unsqueeze(0), aabb_min.unsqueeze(0))  # [5, L, 3]
    dots = (p_vertex * normals.unsqueeze(1)).sum(dim=2) + d_vals.unsqueeze(1)
    return (dots >= 0).all(dim=0)


def visible_point_mask_exact_torch(xyz, full_proj_transform, margin: float = 0.0):
    """Exact per-point Gribb-Hartmann test against the same 5 planes as
    `visible_leaf_mask_torch` (no far plane, same rationale), but on point
    centers directly instead of a leaf's AABB -- a narrow-phase refinement
    meant to run only on the (much smaller) candidate set that already
    passed the leaf-level broad phase, not the full point cloud, so it
    stays cheap. See GaussianRasterizerWrapper.render's culling_narrow_phase
    branch in render/rasterizer.py.

    `margin`: this tests splat *centers*, which have zero screen-space
    extent, unlike the splats actually being rendered -- at margin=0 a
    splat can visibly pop out right as its center crosses the frustum edge,
    before its rendered footprint has actually left the screen. Inflates
    the plane test by this amount to compensate; tune from what you
    actually see at the frame edges, this isn't derived from splat scale."""
    import torch

    m = full_proj_transform
    planes = torch.stack([
        m[:, 0] + m[:, 3],  # left
        m[:, 3] - m[:, 0],  # right
        m[:, 1] + m[:, 3],  # bottom
        m[:, 3] - m[:, 1],  # top
        m[:, 2],            # near
    ], dim=0)
    normals, d_vals = planes[:, :3], planes[:, 3]  # [5, 3], [5]

    dots = xyz @ normals.T + d_vals  # [K, 5]
    return (dots >= -margin).all(dim=1)


def visible_point_mask_screen_size_torch(xyz, scales, world_view_transform,
                                          focal_x: float, focal_y: float,
                                          cutoff: float = 3.0, min_pixel_radius: float = 1.0):
    """Screen-space size test: culls candidates whose projected footprint
    is smaller than `min_pixel_radius` pixels -- a splat that small
    contributes negligible unique detail. Meant to run only on points that
    already passed the frustum broad phase, same reasoning as
    `visible_point_mask_exact_torch`. Also reused (with cutoff=1.0) for the
    leaf-level LOD fine/coarse decision in render/rasterizer.py, applied to
    a leaf's own center/radius instead of one splat's.

    Deliberately a coarse, conservative proxy rather than replicating the
    CUDA kernel's own exact anisotropic footprint math (compute_transmat/
    compute_aabb in forward.cu, which projects the splat's local tangent
    frame through the projection matrix) -- reimplementing that exactly in
    Python is real surface area for a subtle mismatch bug. Instead this
    uses the standard real-time-rendering pinhole approximation, projected
    radius ~= focal * world_radius / depth, with `world_radius` taken as
    the *larger* in-plane scale axis (worst case, so this never
    underestimates and over-culls) times `cutoff` to match the ~3-standard-
    deviation effective radius the kernel itself uses by default (its own
    `cutoff = 3.0f` in forward.cu). Points behind the camera (depth <= 0)
    are never culled here -- that's frustum culling's job, not this test's;
    a non-positive depth just means 'not testable, keep it'."""
    import torch

    n = xyz.shape[0]
    ones = xyz.new_ones((n, 1))
    xyz_h = torch.cat([xyz, ones], dim=-1)
    p_view = xyz_h @ world_view_transform
    depth = p_view[:, 2]

    world_radius = scales.amax(dim=-1) * cutoff
    focal = (focal_x + focal_y) * 0.5
    safe_depth = torch.clamp(depth, min=1e-6)
    pixel_radius = focal * world_radius / safe_depth

    return (depth <= 0) | (pixel_radius >= min_pixel_radius)


def index_cache_path(ply_path: str | Path, opacity_threshold: float = 0.0) -> Path:
    """<ply_dir>/.gs_sensors/<ply_stem>[_opacityN].idx.npz

    `opacity_threshold` changes *which* splats exist (see
    compression.py's prune_low_opacity), not just how they're stored --
    an octree built at one threshold is structurally invalid for a model
    loaded at a different one (wrong point count, wrong point identities).
    Suffixing the cache filename only when nonzero keeps every existing
    cache built before opacity pruning existed valid and reachable at its
    original path (opacity_threshold=0.0 is still the default)."""
    ply_path = Path(ply_path)
    suffix = f"_opacity{opacity_threshold:g}" if opacity_threshold > 0.0 else ""
    return ply_path.parent / ".gs_sensors" / f"{ply_path.stem}{suffix}.idx.npz"


def load_or_build_octree(
    ply_path: str | Path,
    xyz: np.ndarray,
    leaf_max: int = 5000,
    max_depth: int = 8,
    build_index: bool = False,
    compute_lod: bool = False,
    opacity: np.ndarray | None = None,
    scale: np.ndarray | None = None,
    rotation: np.ndarray | None = None,
    features_dc: np.ndarray | None = None,
    opacity_threshold: float = 0.0,
) -> Octree | None:
    """Loads a cached index if present; builds (and caches) one if
    `build_index` is set and no cache exists; otherwise returns None
    (culling disabled -- the caller renders every splat every frame).

    `compute_lod`: also builds (and caches) per-leaf LOD proxies via
    lod.py's build_leaf_proxies -- requires opacity/scale/rotation/
    features_dc (already-activated arrays, see that function's docstring)
    when actually building a fresh index. If a *cached* index is loaded
    that predates LOD (no proxy data in the .npz), this degrades
    gracefully to "LOD unavailable this run" with a printed notice rather
    than an error -- rebuild with build_index=True to add it.

    `opacity_threshold` must be the same value the model was actually
    loaded with (see index_cache_path) -- passed through here only to pick
    the right cache file, this function doesn't prune anything itself.
    A loaded cache whose point count doesn't match `xyz` (e.g. a stale
    cache from before this per-threshold cache-path split, or `xyz` itself
    changed on disk) fails loudly here rather than crashing confusingly
    later in GaussianModel.reorder_ with an out-of-range index."""
    cache_path = index_cache_path(ply_path, opacity_threshold)
    if cache_path.is_file():
        octree = load_octree(cache_path)
        print(f"[gs_sensor_core] Loaded octree index from {cache_path}")
        if octree.flat_indices.shape[0] != xyz.shape[0]:
            raise ValueError(
                f"Cached octree at {cache_path} covers {octree.flat_indices.shape[0]:,} points, "
                f"but the loaded model has {xyz.shape[0]:,} -- stale cache (e.g. from a different "
                f"opacity_threshold, or the PLY changed on disk). Delete {cache_path} or pass "
                "build_index=True to regenerate it."
            )
        needs_lod_rebuild = compute_lod and not octree.has_lod
        if not needs_lod_rebuild:
            return octree
        if not build_index:
            print(f"[gs_sensor_core] Cached index at {cache_path} has no LOD proxies -- "
                  "pass build_index=True to regenerate it with octree_lod enabled; "
                  "LOD disabled for this run")
            return octree
        print(f"[gs_sensor_core] Cached index at {cache_path} has no LOD proxies -- "
              "rebuilding it with LOD since build_index=True ...")
        # falls through to the build below, deliberately not returning here
    elif not build_index:
        print(f"[gs_sensor_core] No octree index at {cache_path} and build_index=False -- culling disabled")
        return None

    print(f"[gs_sensor_core] Building octree index (leaf_max={leaf_max:,}) ...")
    octree = build_octree(xyz, leaf_max=leaf_max, max_depth=max_depth)
    if compute_lod:
        if opacity is None or scale is None or rotation is None or features_dc is None:
            raise ValueError("compute_lod=True requires opacity/scale/rotation/features_dc")
        from gs_sensor_core.lod import build_leaf_proxies
        print(f"[gs_sensor_core] Building LOD proxies ({len(octree.node_aabbs):,} leaves) ...")
        (octree.proxy_xyz, octree.proxy_scale, octree.proxy_rotation,
         octree.proxy_opacity, octree.proxy_features_dc) = build_leaf_proxies(
            octree, xyz, opacity, scale, rotation, features_dc)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    save_octree(cache_path, octree)
    print(f"[gs_sensor_core] Built {len(octree.node_aabbs):,} leaf nodes, saved to {cache_path}")
    return octree
