"""tensor/numpy render outputs -> sensor_msgs/Image + CameraInfo + PointCloud2.

No rendering or frame-transform decisions belong here -- only message
construction from already-rendered arrays and an already-loaded profile.
"""
from __future__ import annotations

import numpy as np
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from gs_sensor_core.camera_profiles.schema import CameraProfile

_POINTCLOUD_FIELDS = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
]


def points_to_pointcloud2_msg(xyz: np.ndarray, intensity: np.ndarray, header: Header) -> PointCloud2:
    """`xyz`: `[K, 3]` float32, metric meters, in the LiDAR's own frame (see
    `render/lidar/pointcloud.py` -- points are NOT re-transformed by pose
    here, same as a real spinning-LiDAR driver: `header.frame_id` names the
    sensor frame, and TF is what places these in world/map for a consumer
    that needs it). `intensity`: `[K]` float32."""
    structured = np.zeros(xyz.shape[0], dtype=[
        ("x", np.float32), ("y", np.float32), ("z", np.float32), ("intensity", np.float32),
    ])
    structured["x"], structured["y"], structured["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    structured["intensity"] = intensity
    return point_cloud2.create_cloud(header, _POINTCLOUD_FIELDS, structured)


def rgb_to_image_msg(rgb: np.ndarray, header: Header) -> Image:
    msg = Image()
    msg.header = header
    msg.height, msg.width = rgb.shape[:2]
    msg.encoding = "rgb8"
    msg.is_bigendian = 0
    msg.step = msg.width * 3
    msg.data = np.ascontiguousarray(rgb).tobytes()
    return msg


def depth_to_image_msg(depth: np.ndarray, header: Header) -> Image:
    msg = Image()
    msg.header = header
    msg.height, msg.width = depth.shape[:2]
    msg.encoding = "32FC1"
    msg.is_bigendian = 0
    msg.step = msg.width * 4
    msg.data = np.ascontiguousarray(depth, dtype=np.float32).tobytes()
    return msg


def profile_to_camera_info(profile: CameraProfile, header: Header) -> CameraInfo:
    msg = CameraInfo()
    msg.header = header
    msg.width = profile.width
    msg.height = profile.height
    msg.distortion_model = "plumb_bob"
    msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    msg.k = [profile.fx, 0.0, profile.cx,
             0.0, profile.fy, profile.cy,
             0.0, 0.0, 1.0]
    msg.r = [1.0, 0.0, 0.0,
             0.0, 1.0, 0.0,
             0.0, 0.0, 1.0]
    msg.p = [profile.fx, 0.0, profile.cx, 0.0,
             0.0, profile.fy, profile.cy, 0.0,
             0.0, 0.0, 1.0, 0.0]
    return msg
