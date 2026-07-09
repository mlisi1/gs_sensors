"""Thin loader for GS-LiDAR's vendored panoramic CUDA kernel.

This kernel (`gs_sensor_core/third_party/GS-LiDAR/diff-gaussian-rasterization-2d`,
Python package name `diff_gaussian_rasterization`) is a fork of the Inria 3D-GS
rasterizer with `vfov`/`hfov`/`scale_factor` baked into the CUDA forward pass
itself for equirectangular (panoramic) ray generation -- a different kernel
from `diff-surfel-rasterization` (the perspective 2D-GS kernel the camera
branch vendors), not an alternate name for the same thing. See CLAUDE.md's
"Rendering core is standalone" section for the license (same family as
`diff-surfel-rasterization`: Inria/MPII non-commercial,
`LICENSE_gaussian_splatting.md` inside the vendored submodule).

GS-LiDAR's own training code never `pip install`s this kernel -- its only
working entry point is `gaussian_renderer/diff_gaussian_rasterization_2d.py`,
which JIT-compiles the CUDA sources via `torch.utils.cpp_extension.load()`
at import time (the `setup.py` alongside the CUDA sources references a
`diff_gaussian_rasterization` package directory that doesn't actually exist
in the repo -- confirmed by inspection, not assumed -- so `pip install -e`
would not work here the way it does for `diff-surfel-rasterization`). This
module ports that same JIT-load call verbatim (same source file list, same
`-I .../third_party/glm/` include), pointed at our vendored submodule path
instead -- it does not modify the submodule's own files, so the vendored
tree stays a clean, diffable copy of upstream.

JIT compilation happens on first call to `load_kernel()`, not at import
time -- mirrors `render/rasterizer.py`'s `GaussianRasterizerWrapper.__init__`
importing `diff_surfel_rasterization` lazily, so importing gs_sensor_core
doesn't force a slow (and CUDA-toolchain-dependent) compile for callers who
only need the camera branch. Compiles once per environment; PyTorch's
extension loader caches the built `.so` under `~/.cache/torch_extensions/`
across runs.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

_KERNEL_ROOT = (
    Path(__file__).resolve().parents[3] / "third_party" / "GS-LiDAR" / "diff-gaussian-rasterization-2d"
)

_cached_module = None


def load_kernel():
    """Returns the JIT-compiled `diff_gaussian_rasterization` extension
    module (has `.rasterize_gaussians`, `.rasterize_gaussians_backward`,
    `.mark_visible`). Cached after the first call."""
    global _cached_module
    if _cached_module is not None:
        return _cached_module

    if not _KERNEL_ROOT.is_dir():
        raise RuntimeError(
            f"GS-LiDAR submodule not found at {_KERNEL_ROOT} -- run "
            "'git submodule update --init --recursive' from the repo root."
        )

    from torch.utils.cpp_extension import load

    glm_include = _KERNEL_ROOT / "third_party" / "glm"
    _cached_module = load(
        name="gs_sensors_diff_gaussian_rasterization_2d",
        # `-include cstdint`: cuda_rasterizer/rasterizer_impl.h uses
        # uint32_t/uint64_t/std::uintptr_t without including <cstdint>
        # itself -- silently worked against whatever older host libstdc++
        # this kernel was originally built against (some standard header
        # transitively pulled it in), but fails outright on a newer one
        # (confirmed: GCC 13/Ubuntu 24.04 in test_env/docker, "namespace
        # std has no member uintptr_t" / "identifier uint32_t is
        # undefined"). Force-including it via a compiler flag here --
        # rather than patching the vendored submodule's source directly --
        # keeps the vendored tree an untouched, diffable copy of upstream
        # (see CLAUDE.md's "never modified in place" rule for vendored
        # deps) and fixes every .cu file in one place instead of hunting
        # down which specific files need the include added.
        extra_cuda_cflags=["-I", str(glm_include), "-include", "cstdint"],
        sources=[
            str(_KERNEL_ROOT / "cuda_rasterizer" / "rasterizer_impl.cu"),
            str(_KERNEL_ROOT / "cuda_rasterizer" / "forward.cu"),
            str(_KERNEL_ROOT / "cuda_rasterizer" / "backward.cu"),
            str(_KERNEL_ROOT / "rasterize_points.cu"),
            str(_KERNEL_ROOT / "ext.cpp"),
        ],
        verbose=True,
    )
    return _cached_module


class LidarRasterizationSettings(NamedTuple):
    """Mirrors GS-LiDAR's own `GaussianRasterizationSettings`
    (`gaussian_renderer/diff_gaussian_rasterization_2d.py`) field-for-field
    -- this is the exact positional contract the kernel's `forward()`
    expects, not a redesigned API."""
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: "torch.Tensor"
    scale_modifier: float
    viewmatrix: "torch.Tensor"
    projmatrix: "torch.Tensor"
    sh_degree: int
    campos: "torch.Tensor"
    prefiltered: bool
    debug: bool
    vfov: tuple
    hfov: tuple
    scale_factor: float


def rasterize(means3D, opacities, shs, scales, rotations, settings: LidarRasterizationSettings,
              mask=None):
    """Inference-only forward call (no autograd) -- gs_sensors never trains,
    so this skips GS-LiDAR's `torch.autograd.Function` wrapper entirely and
    calls the compiled op directly. Returns `(contrib, color, feature,
    depth, alpha, radii)`, matching
    `_RasterizeGaussians.forward`'s return `(contrib, color, feature, depth,
    1 - T, radii)` in the ported reference -- `color` is the SH-evaluated
    per-pixel output (4 channels here: see `rasterizer.py`), `depth` is
    4-channel (mean, median, distortion, depth_square, in that order, per
    GS-LiDAR's own `render()`), `alpha` is accumulated opacity.

    `mask`: per-splat bool prefilter, kernel-side (not a Python gather) --
    `None` defaults to all-visible. `rasterizer.py` builds the real
    opacity/marginal_t-based mask GS-LiDAR's own `render()` computes before
    calling its rasterizer; ported here rather than left as a bare
    all-ones default so a caller that doesn't build one explicitly doesn't
    silently diverge from upstream's prefiltering."""
    import torch

    _C = load_kernel()
    means2D = torch.zeros_like(means3D)
    empty = torch.empty(0, device=means3D.device)
    features = torch.empty_like(means3D[..., :0])
    if mask is None:
        mask = torch.ones_like(means3D[:, :1], dtype=torch.bool)

    args = (
        settings.bg,
        means3D,
        empty,          # colors_precomp -- unused, we always drive color via shs
        features,       # 'other'/normal auxiliary channels -- unused for LiDAR inference
        opacities,
        scales,
        rotations,
        settings.scale_modifier,
        empty,          # cov3Ds_precomp -- unused, we always pass scales/rotations
        mask,
        settings.viewmatrix,
        settings.projmatrix,
        settings.tanfovx,
        settings.tanfovy,
        settings.image_height,
        settings.image_width,
        shs,
        settings.sh_degree,
        settings.campos,
        settings.prefiltered,
        settings.debug,
        settings.vfov[0],
        settings.vfov[1],
        settings.hfov[0],
        settings.hfov[1],
        settings.scale_factor,
    )
    (num_rendered, contrib, color, feature, depth, transmittance,
     radii, _geom, _binning, _img) = _C.rasterize_gaussians(*args)
    return contrib, color, feature, depth, 1.0 - transmittance, radii
