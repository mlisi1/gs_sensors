"""Pose source abstraction: ground-truth (Gazebo bridge) vs TF lookup.

Switchable by config -- see CLAUDE.md "Pose handling". Both paths are wired
in from the start rather than retrofitted later, since ground-truth is for
validating the renderer in isolation and TF is for closed-loop tests once a
localization stack sits in front of this node.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

from gs_sensor_core.frames import Pose


class PoseSource(ABC):
    @abstractmethod
    def get_pose(self, stamp: Time) -> Pose | None:
        """World-frame camera pose at `stamp`, or None if not available yet."""


class GroundTruthPoseSource(PoseSource):
    """Subscribes to a geometry_msgs/PoseStamped bridged straight from Gazebo
    (gz-sim-pose-publisher-system -> ros_gz_bridge)."""

    def __init__(self, node: Node, topic: str):
        self._latest: Pose | None = None
        node.create_subscription(PoseStamped, topic, self._on_pose, 10)

    def _on_pose(self, msg: PoseStamped) -> None:
        o = msg.pose.orientation
        p = msg.pose.position
        self._latest = Pose(
            position=np.array([p.x, p.y, p.z]),
            orientation=np.array([o.x, o.y, o.z, o.w]),
        )

    def get_pose(self, stamp: Time) -> Pose | None:
        return self._latest


class TFPoseSource(PoseSource):
    """Looks up `camera_frame` in `world_frame` via tf2 -- must be the
    camera's *optical* frame (REP 103: x-right, y-down, z-forward), not the
    robot's base_link-style mounting frame."""

    def __init__(self, node: Node, world_frame: str, camera_frame: str):
        self._world_frame = world_frame
        self._camera_frame = camera_frame
        self._buffer = Buffer()
        self._listener = TransformListener(self._buffer, node)

    def get_pose(self, stamp: Time) -> Pose | None:
        try:
            tf = self._buffer.lookup_transform(self._world_frame, self._camera_frame, stamp)
        except TransformException:
            return None
        t = tf.transform.translation
        r = tf.transform.rotation
        return Pose(
            position=np.array([t.x, t.y, t.z]),
            orientation=np.array([r.x, r.y, r.z, r.w]),
        )


def make_pose_source(node: Node, *, kind: str, ground_truth_topic: str,
                      world_frame: str, camera_frame: str) -> PoseSource:
    if kind == "ground_truth":
        return GroundTruthPoseSource(node, ground_truth_topic)
    if kind == "tf":
        return TFPoseSource(node, world_frame, camera_frame)
    raise ValueError(f"Unknown pose_source kind: {kind!r} (expected 'ground_truth' or 'tf')")
