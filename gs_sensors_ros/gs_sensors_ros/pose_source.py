"""Pose source abstraction: ground-truth (Gazebo bridge) vs TF lookup.

Switchable by config -- see CLAUDE.md "Pose handling". Both paths are wired
in from the start rather than retrofitted later, since ground-truth is for
validating the renderer in isolation and TF is for closed-loop tests once a
localization stack sits in front of this node.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import CallbackGroup
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

from gs_sensor_core.frames import Pose


class PoseSource(ABC):
    @abstractmethod
    def get_pose(self, stamp: Time) -> Pose | None:
        """World-frame camera pose at `stamp`, or None if not available yet."""

    def pose_stamp(self) -> Time | None:
        """The ROS timestamp the last pose returned by `get_pose` is
        actually valid at -- publish THIS as the render output's own
        `header.stamp`, not the render loop's own `now()`, so a downstream
        TF-based transform (e.g. RViz's Fixed Frame view) resolves the
        exact same pose sample the render was computed from. Default
        (`None`) means "trust the `stamp` passed into `get_pose`", true for
        `TFPoseSource` (tf2 already interpolates to give the pose *at* that
        exact stamp) but NOT for `GroundTruthPoseSource`, which ignores
        `stamp` and returns whatever was last received -- see its override.
        Callers should fall back to their own current time when this
        returns `None` (no pose received yet, or a source that doesn't
        override it)."""
        return None


class GroundTruthPoseSource(PoseSource):
    """Subscribes to a geometry_msgs/PoseStamped bridged straight from Gazebo
    (gz-sim-pose-publisher-system -> ros_gz_bridge).

    `callback_group`: pass a group distinct from the render timer's so a
    MultiThreadedExecutor can process an incoming pose while a render is
    still in flight on the single-threaded-executor-blocking numpy/CUDA
    work in the timer callback -- otherwise pose updates queue up behind
    whatever render is currently running and every frame renders at a pose
    that's stale by up to one render duration. See camera_debug_node.py."""

    def __init__(self, node: Node, topic: str, callback_group: CallbackGroup | None = None):
        self._latest: Pose | None = None
        self._latest_stamp: Time | None = None
        self._latest_received_at: float | None = None
        node.create_subscription(PoseStamped, topic, self._on_pose, 10, callback_group=callback_group)

    def _on_pose(self, msg: PoseStamped) -> None:
        o = msg.pose.orientation
        p = msg.pose.position
        self._latest = Pose(
            position=np.array([p.x, p.y, p.z]),
            orientation=np.array([o.x, o.y, o.z, o.w]),
        )
        self._latest_stamp = Time.from_msg(msg.header.stamp)
        self._latest_received_at = time.monotonic()

    def get_pose(self, stamp: Time) -> Pose | None:
        return self._latest

    def pose_stamp(self) -> Time | None:
        return self._latest_stamp

    def pose_age_s(self) -> float | None:
        """Wall-clock seconds since the cached pose was received -- a
        debug-only diagnostic for measuring staleness at render time, not
        part of the PoseSource interface (a TF lookup has no single
        'latest' pose to measure staleness against the way a cached
        subscription value does)."""
        if self._latest_received_at is None:
            return None
        return time.monotonic() - self._latest_received_at


class TFPoseSource(PoseSource):
    """Looks up `camera_frame` in `world_frame` via tf2 -- must be the
    camera's *optical* frame (REP 103: x-right, y-down, z-forward), not the
    robot's base_link-style mounting frame.

    `spin_thread=True`: TransformListener doesn't take a callback_group in
    this tf2_ros version, so it gets its own dedicated thread/executor for
    /tf and /tf_static instead -- same decoupling motivation as
    GroundTruthPoseSource's callback_group (see its docstring): tf buffer
    updates shouldn't stall behind an in-flight render."""

    def __init__(self, node: Node, world_frame: str, camera_frame: str):
        self._world_frame = world_frame
        self._camera_frame = camera_frame
        self._buffer = Buffer()
        self._listener = TransformListener(self._buffer, node, spin_thread=True)

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
                      world_frame: str, camera_frame: str,
                      callback_group: CallbackGroup | None = None) -> PoseSource:
    if kind == "ground_truth":
        return GroundTruthPoseSource(node, ground_truth_topic, callback_group=callback_group)
    if kind == "tf":
        return TFPoseSource(node, world_frame, camera_frame)
    raise ValueError(f"Unknown pose_source kind: {kind!r} (expected 'ground_truth' or 'tf')")
