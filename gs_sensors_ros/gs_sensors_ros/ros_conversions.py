"""tensor/numpy render outputs -> sensor_msgs/Image + CameraInfo.

No rendering or frame-transform decisions belong here -- only message
construction from already-rendered arrays and an already-loaded profile.
"""
from __future__ import annotations

import numpy as np
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header

from gs_sensor_core.camera_profiles.schema import CameraProfile


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
