"""Octree spatial index (build/save/load) + frustum culling.

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
    np.savez_compressed(
        str(path),
        node_aabbs=octree.node_aabbs,
        node_offsets=octree.node_offsets,
        flat_indices=octree.flat_indices,
    )


def load_octree(path: str | Path) -> Octree:
    data = np.load(str(path))
    return Octree(
        node_aabbs=data["node_aabbs"],
        node_offsets=data["node_offsets"],
        flat_indices=data["flat_indices"],
    )


def _extract_planes(full_proj_transform: np.ndarray) -> np.ndarray:
    """5 planes (left, right, bottom, top, near) as [5, 4] (a, b, c, d)."""
    m = full_proj_transform
    return np.stack([
        m[:, 0] + m[:, 3],  # left
        m[:, 3] - m[:, 0],  # right
        m[:, 1] + m[:, 3],  # bottom
        m[:, 3] - m[:, 1],  # top
        m[:, 2],            # near (camera-Z >= znear)
    ], axis=0)


def visible_leaf_mask(octree: Octree, full_proj_transform: np.ndarray) -> np.ndarray:
    """Gribb-Hartmann p-vertex test, vectorized over all leaves -> [L] bool."""
    planes = _extract_planes(full_proj_transform)
    normals, d_vals = planes[:, :3], planes[:, 3]

    aabb_min = octree.node_aabbs[:, :3]
    aabb_max = octree.node_aabbs[:, 3:]

    pos_mask = normals[:, np.newaxis, :] >= 0
    p_vertex = np.where(pos_mask, aabb_max[np.newaxis], aabb_min[np.newaxis])  # [5, L, 3]
    dots = (p_vertex * normals[:, np.newaxis, :]).sum(axis=2) + d_vals[:, np.newaxis]
    return (dots >= 0).all(axis=0)


def visible_point_mask(octree: Octree, full_proj_transform: np.ndarray, n_points: int) -> np.ndarray:
    """Per-point boolean mask (in original point order) from the leaf-level
    test. A per-leaf Python loop turned out to be faster than a "vectorized"
    np.repeat + fancy-index scatter when measured against a real 7.19M-point
    model at realistic (~50%) visibility ratios (8ms vs 15ms) -- the loop
    only touches visible leaves' contiguous slices, while the repeat/scatter
    approach always materializes full N-length arrays regardless of how much
    is actually visible. Benchmark before "optimizing" this again."""
    leaf_vis = visible_leaf_mask(octree, full_proj_transform)
    mask = np.zeros(n_points, dtype=bool)
    visible_leaves = np.where(leaf_vis)[0]
    for leaf in visible_leaves:
        s, e = octree.node_offsets[leaf], octree.node_offsets[leaf + 1]
        mask[octree.flat_indices[s:e]] = True
    return mask


def point_leaf_ids(octree: Octree) -> np.ndarray:
    """Inverse of the leaf-order permutation: leaf id owning each point, in
    original point order -- [N] int64. Precomputed once so GPU culling can
    turn a leaf-visibility mask into a point mask with a single gather
    instead of the per-frame Python loop `visible_point_mask` uses (that
    loop wins on CPU, per the benchmark note above, but a gather is the
    equivalent GPU-native op)."""
    n_points = octree.flat_indices.shape[0]
    ids = np.empty(n_points, dtype=np.int64)
    for leaf in range(len(octree.node_offsets) - 1):
        s, e = octree.node_offsets[leaf], octree.node_offsets[leaf + 1]
        ids[octree.flat_indices[s:e]] = leaf
    return ids


def visible_leaf_mask_torch(node_aabbs, full_proj_transform):
    """Same Gribb-Hartmann p-vertex test as `visible_leaf_mask`, but done in
    torch on `full_proj_transform`'s own device -- no `.cpu()`/`.numpy()`
    round trip. `node_aabbs` must already be a torch tensor on that device.
    Local `import torch` keeps this module numpy-only unless this path is
    actually used (see `culling_backend="gpu"` in rasterizer.py)."""
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
    `visible_leaf_mask`/`visible_leaf_mask_torch` (no far plane, same
    rationale), but on point centers directly instead of a leaf's AABB --
    a narrow-phase refinement meant to run only on the (much smaller)
    candidate set that already passed the leaf-level broad phase, not the
    full point cloud, so it stays cheap. See
    GaussianRasterizerWrapper._narrow_phase_mask.

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


def index_cache_path(ply_path: str | Path) -> Path:
    """<ply_dir>/.gs_sensors/<ply_stem>.idx.npz"""
    ply_path = Path(ply_path)
    return ply_path.parent / ".gs_sensors" / f"{ply_path.stem}.idx.npz"


def load_or_build_octree(
    ply_path: str | Path,
    xyz: np.ndarray,
    leaf_max: int = 5000,
    max_depth: int = 8,
    build_index: bool = False,
) -> Octree | None:
    """Loads a cached index if present; builds (and caches) one if
    `build_index` is set and no cache exists; otherwise returns None
    (culling disabled -- the caller renders every splat every frame)."""
    cache_path = index_cache_path(ply_path)
    if cache_path.is_file():
        print(f"[gs_sensor_core] Loaded octree index from {cache_path}")
        return load_octree(cache_path)
    if not build_index:
        print(f"[gs_sensor_core] No octree index at {cache_path} and build_index=False -- culling disabled")
        return None

    print(f"[gs_sensor_core] Building octree index (leaf_max={leaf_max:,}) ...")
    octree = build_octree(xyz, leaf_max=leaf_max, max_depth=max_depth)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    save_octree(cache_path, octree)
    print(f"[gs_sensor_core] Built {len(octree.node_aabbs):,} leaf nodes, saved to {cache_path}")
    return octree
