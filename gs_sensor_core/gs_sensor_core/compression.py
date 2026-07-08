"""In-memory splat compression, applied right after a PLY is read.

Three levels, same idea and same real effect as offline PLY-compression
tools in the Gaussian-Splatting ecosystem, reimplemented independently and
applied to already-loaded arrays rather than writing a separate cached
`.ply` file to disk -- for a node that loads its model once at startup
(not repeatedly, as an interactive viewer would), an on-disk cache buys
nothing a config flag doesn't already give you, so it's left out here (see
TODO.md if repeated-load time ever becomes a real problem).

Level 0: no change.
Level 1: fp16 round-trip on every field (precision preview / safety net --
         does NOT reduce tensor size, matches upstream tooling's own
         "same disk size" behavior for this level).
Level 2: fp16 round-trip + SH `f_rest` truncated to `target_sh_degree`
         (default 1) -- real reduction in tensor size and SH-eval cost.
Level 3: fp16 round-trip + SH dropped entirely (degree 0, view-independent
         color only) -- maximal reduction. int8-packing rotation/normals
         for extra disk savings (as some viewers do) is intentionally not
         implemented -- see TODO.md.
"""
from __future__ import annotations

import numpy as np

FP16_MAX = 65504.0


def to_fp16_safe(arr: np.ndarray, label: str = "") -> np.ndarray:
    """Round-trip through float16 (NaN/Inf zeroed, overflow clipped at the
    99.9th percentile first) -- returned as float32."""
    arr = arr.astype(np.float32)
    n_bad = int(np.sum(~np.isfinite(arr)))
    if n_bad > 0:
        print(f"[gs_sensor_core] {label}: zeroing {n_bad:,} NaN/Inf values before fp16 round-trip")
        arr = np.where(np.isfinite(arr), arr, 0.0)
    max_abs = float(np.abs(arr).max()) if arr.size else 0.0
    if max_abs > FP16_MAX:
        threshold = float(min(np.percentile(np.abs(arr), 99.9), FP16_MAX))
        n_clip = int(np.sum(np.abs(arr) > threshold))
        print(f"[gs_sensor_core] {label}: clipping {n_clip:,} fp16-overflow values to ±{threshold:.1f}")
        arr = np.clip(arr, -threshold, threshold)
    return arr.astype(np.float16).astype(np.float32)


def truncate_sh(features_rest: np.ndarray, current_degree: int, target_degree: int) -> tuple[np.ndarray, int]:
    """features_rest: [N, K, 3] with K = (current_degree+1)**2 - 1.
    Returns (truncated_features_rest, new_degree)."""
    new_degree = min(current_degree, target_degree)
    if new_degree >= current_degree:
        return features_rest, current_degree
    keep = (new_degree + 1) ** 2 - 1
    return features_rest[:, :keep, :], new_degree


def apply_compression(
    arrays: dict[str, np.ndarray],
    degree: int,
    level: int,
    target_sh_degree: int = 1,
) -> tuple[dict[str, np.ndarray], int]:
    """arrays: xyz / opacity / scaling / rotation / features_dc / features_rest,
    as produced by ply_loader before the torch conversion. Returns the
    (possibly modified) arrays dict and the (possibly reduced) SH degree."""
    if level <= 0:
        return arrays, degree

    fp16_fields = ("xyz", "opacity", "scaling", "rotation", "features_dc", "features_rest")
    arrays = dict(arrays)
    for name in fp16_fields:
        if arrays[name].size:
            arrays[name] = to_fp16_safe(arrays[name], name)

    if level == 2:
        arrays["features_rest"], degree = truncate_sh(arrays["features_rest"], degree, target_sh_degree)
    elif level >= 3:
        arrays["features_rest"], degree = truncate_sh(arrays["features_rest"], degree, 0)

    return arrays, degree
