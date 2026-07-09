"""LiDAR sensor profile schema: panorama geometry + render resolution +
frame_id + rate. Parallels `camera_profiles/schema.py` -- profiles are data
(YAML), never Python, so adding a sensor is a new file, no code change.

`vfov`/`hfov`/`hw` must match the values the checkpoint was actually trained
with (`setting.txt`'s `vfov`/`hfov`/`hw`, verbatim) -- these aren't
independent render knobs the way a camera's `width`/`height` loosely are;
they're baked into what the model learned to reconstruct at. Both `hfov`
and `hw` are **per-half** values: GS-LiDAR always renders a LiDAR pose as
two opposing `hfov`-wide passes (`render/lidar/camera.py`) and stitches
them into one `[hw[0], hw[1] * 2]` full panorama -- confirmed against
`~/GS-LiDAR/gaussian_renderer/__init__.py`'s `render_range_map` (`depth_pano
= zeros([3, h, w * 2])` where `h, w = hw`) and against the loaded
`Crosslab_lidar` raydrop prior's actual shape (`[1, 32, 1024]` for
`hw=[32, 512]`). A training config's `hfov: [-90, 90]` is this per-half
value already, not a full-circle range -- there is no separate "full 360"
config anywhere upstream, the full circle comes only from stitching.

`scale_factor` is GS-LiDAR's own `pipe.scale_factor` training arg -- **must
equal the checkpoint's `GSFrameTransform.scale` (the PCA transform's scale,
`transform_poses_pca.npz`'s `scale_factor` field / `scale_factor.txt`), not
an independently chosen value.** Traced end-to-end:
`~/GS-LiDAR/scene/kitti360_loader.py:265` sets `args.scale_factor =
float(scale_factor)` (the PCA scale) immediately after
`transform_poses_pca` runs, overwriting whatever CLI default `setting.txt`
logged (`setting.txt` is dumped from argparse *before* this overwrite, so
its `scale_factor: 1.0` is stale, not the value training actually used --
`scale_factor.txt`, written post-overwrite at `train.py:43-44`, is the real
one). This value then flows straight into the CUDA kernel's
`in_frustum_panorama` near/far clip (`auxiliary.h`: `near_n = 2.0`, `far_n =
90.0`, clip planes are `near_n * scale_factor` / `far_n * scale_factor`) --
since the Gaussian model lives in PCA-scaled space, not real-world meters,
those clip planes only land at a sane real-world distance (e.g. `2.0 * 0.1
= 0.2` GS-units = 2m near-clip, a plausible real LiDAR minimum range) when
`scale_factor` matches the PCA scale. Passing the wrong value (e.g. this
field's own too-innocent-looking `1.0` default) silently near-clips the
*entire* model at every frame for an indoor-scale scene -- confirmed by
`scripts/validate_lidar.py` initially using the default and rendering
near-total garbage despite verified-correct pose/geometry.

`beam_count`/`frame_id`/`update_rate` are the real-driver-facing fields --
**not derivable from the checkpoint**, need the real sensor's own values,
same as `realsense_d435.yaml` needed real D435 intrinsics rather than
whatever the camera branch's model happened to train at.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_REQUIRED_KEYS = ("vfov", "hfov", "hw", "frame_id", "update_rate")


@dataclass(frozen=True)
class LidarProfile:
    vfov: tuple[float, float]
    hfov: tuple[float, float]    # per-half (each of the two 180-degree passes), see module docstring
    hw: tuple[int, int]          # (height, width), per-half -- stitched panorama is (hw[0], hw[1] * 2)
    frame_id: str
    update_rate: float
    scale_factor: float = 1.0    # see module docstring -- GS-LiDAR's pipe.scale_factor
    beam_count: int | None = None  # informational only so far, not used by the render path

    @staticmethod
    def from_yaml(path: str | Path) -> "LidarProfile":
        with open(path) as f:
            data = yaml.safe_load(f)
        missing = [k for k in _REQUIRED_KEYS if k not in data]
        if missing:
            raise ValueError(f"LiDAR profile {path} missing required keys: {missing}")
        return LidarProfile(
            vfov=(float(data["vfov"][0]), float(data["vfov"][1])),
            hfov=(float(data["hfov"][0]), float(data["hfov"][1])),
            hw=(int(data["hw"][0]), int(data["hw"][1])),
            frame_id=str(data["frame_id"]),
            update_rate=float(data["update_rate"]),
            scale_factor=float(data.get("scale_factor", 1.0)),
            beam_count=int(data["beam_count"]) if "beam_count" in data else None,
        )
