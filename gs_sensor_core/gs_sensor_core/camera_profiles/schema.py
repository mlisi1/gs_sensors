"""Camera profile schema: intrinsics + resolution + frame_id + rate.

Profiles are data (YAML), never Python -- see config/camera_profiles/*.yaml.
Adding a camera should never require a code change.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_REQUIRED_KEYS = ("width", "height", "fx", "fy", "cx", "cy", "frame_id", "update_rate")


@dataclass(frozen=True)
class CameraProfile:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    frame_id: str
    update_rate: float

    @staticmethod
    def from_yaml(path: str | Path) -> "CameraProfile":
        with open(path) as f:
            data = yaml.safe_load(f)
        missing = [k for k in _REQUIRED_KEYS if k not in data]
        if missing:
            raise ValueError(f"Camera profile {path} missing required keys: {missing}")
        return CameraProfile(
            width=int(data["width"]),
            height=int(data["height"]),
            fx=float(data["fx"]),
            fy=float(data["fy"]),
            cx=float(data["cx"]),
            cy=float(data["cy"]),
            frame_id=str(data["frame_id"]),
            update_rate=float(data["update_rate"]),
        )
