"""Phase 1 standalone ROS 2 debug node for the GS-LiDAR branch: renders
simulated LiDAR scans from a trained GS-LiDAR checkpoint at the pose of a
robot moving in Gazebo, and publishes them like a real spinning-LiDAR
driver would (`sensor_msgs/PointCloud2`). See CLAUDE.md "Phase 1 -- debug
node" -- same shape as `camera_debug_node.py`, adapted for the LiDAR
checkpoint format and panoramic render pipeline (`gs_sensor_core.render.lidar`).

This file only does parameter declarations, the timer callback, and
delegation into gs_sensor_core + ros_conversions -- no rendering math or
frame algebra here, that lives in gs_sensor_core.
"""
from __future__ import annotations

import time

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header

from gs_sensor_core.frames import GSFrameTransform, load_gs_frame_transform
from gs_sensor_core.lidar_profiles.schema import LidarProfile
from gs_sensor_core.models.lidar_checkpoint_loader import load_lidar_gaussian_model, load_raydrop_prior
from gs_sensor_core.render.lidar import LidarRasterizer
from gs_sensor_core.render.lidar.refine import load_refine_unet

from gs_sensors_ros.pose_source import make_pose_source
from gs_sensors_ros.ros_conversions import points_to_pointcloud2_msg


class LidarDebugNode(Node):

    def __init__(self):
        super().__init__("lidar_debug_node")

        # Model loading
        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("raydrop_prior_path", "")
        self.declare_parameter("refine_unet_path", "")  # empty = raw raydrop, no refinement
        self.declare_parameter("dynamic", False)
        self.declare_parameter("raydrop_threshold", 0.5)

        # Sensor / pose
        self.declare_parameter("lidar_profile", "")
        self.declare_parameter("gs_frame_transform", "")
        self.declare_parameter("pose_source", "ground_truth")
        self.declare_parameter("ground_truth_topic", "pose")
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("lidar_frame", "")

        # Debug / diagnostics
        self.declare_parameter("debug", False)
        self.declare_parameter("enable_profiling", False)

        checkpoint_path = self.get_parameter("checkpoint_path").value
        raydrop_prior_path = self.get_parameter("raydrop_prior_path").value
        profile_path = self.get_parameter("lidar_profile").value
        if not checkpoint_path or not raydrop_prior_path or not profile_path:
            raise RuntimeError(
                "'checkpoint_path', 'raydrop_prior_path', and 'lidar_profile' parameters are required")

        self.profile = LidarProfile.from_yaml(profile_path)

        transform_path = self.get_parameter("gs_frame_transform").value
        self.gs_transform = (
            load_gs_frame_transform(transform_path) if transform_path
            else GSFrameTransform.identity()
        )

        self.get_logger().info(f"Loading GS-LiDAR checkpoint from {checkpoint_path} ...")
        model = load_lidar_gaussian_model(checkpoint_path)
        raydrop_prior = load_raydrop_prior(raydrop_prior_path)

        refine_path = self.get_parameter("refine_unet_path").value
        refine_unet = load_refine_unet(refine_path) if refine_path else None
        if refine_unet is None:
            self.get_logger().warn(
                "No refine_unet_path given -- publishing raw (unrefined) raydrop mask. "
                "GS-LiDAR's own eval always applies the refine UNet; expect a noisier/"
                "denser point cloud than the validated pipeline.")

        self._total_splats = model.num_points
        self._debug = bool(self.get_parameter("debug").value)
        self._enable_profiling = bool(self.get_parameter("enable_profiling").value)

        self.rasterizer = LidarRasterizer(
            model, raydrop_prior, self.profile, refine_unet=refine_unet,
            gs_scale=self.gs_transform.scale,
            dynamic=bool(self.get_parameter("dynamic").value),
            raydrop_threshold=float(self.get_parameter("raydrop_threshold").value),
        )

        # Same rationale as camera_debug_node.py: a dedicated callback group
        # so pose updates aren't queued behind an in-flight render under a
        # MultiThreadedExecutor.
        pose_callback_group = MutuallyExclusiveCallbackGroup()
        lidar_frame = self.get_parameter("lidar_frame").value or self.profile.frame_id
        self.pose_source = make_pose_source(
            self,
            kind=self.get_parameter("pose_source").value,
            ground_truth_topic=self.get_parameter("ground_truth_topic").value,
            world_frame=self.get_parameter("world_frame").value,
            camera_frame=lidar_frame,
            callback_group=pose_callback_group,
        )

        self._points_pub = self.create_publisher(PointCloud2, "points", 10)

        self._period_s = 1.0 / self.profile.update_rate
        self._first_pose_seen = False
        self._timer = self.create_timer(self._period_s, self._on_timer)

    def _on_timer(self) -> None:
        stamp = self.get_clock().now()
        pose_world = self.pose_source.get_pose(stamp)
        if pose_world is None:
            if not self._first_pose_seen:
                self.get_logger().info("Waiting for first pose ...", throttle_duration_sec=5.0)
            return
        self._first_pose_seen = True

        pose_age_s = getattr(self.pose_source, "pose_age_s", lambda: None)()
        pose_gs = self.gs_transform.apply(pose_world)

        t0 = time.perf_counter()
        result = self.rasterizer.render(pose_gs, profile=self._enable_profiling)
        elapsed_s = time.perf_counter() - t0

        if self._debug:
            msg = (
                f"Rendered {result.num_rendered:,} / {self._total_splats:,} splats, "
                f"{result.num_returned:,} points returned, in {elapsed_s * 1000:.1f} ms"
            )
            if result.timings:
                breakdown = " ".join(f"{k}={v:.1f}ms" for k, v in result.timings.items())
                msg += f" [{breakdown}]"
            if pose_age_s is not None:
                msg += f" (pose age {pose_age_s * 1000:.1f} ms)"
            self.get_logger().info(msg, throttle_duration_sec=1.0)

        if elapsed_s > self._period_s:
            self.get_logger().warn(
                f"Render took {elapsed_s * 1000:.1f} ms, over the "
                f"{self._period_s * 1000:.1f} ms frame budget at "
                f"{self.profile.update_rate:.1f} Hz -- dropping frame timing",
                throttle_duration_sec=5.0,
            )
        if result.num_returned == 0:
            self.get_logger().warn(
                "0 points returned this frame -- the LiDAR pose is outside the model's "
                "content, entirely raydropped, or the CUDA kernel produced no valid range "
                "readings. Check gs_frame_transform and the pose source before the renderer.",
                throttle_duration_sec=5.0,
            )

        header = Header()
        header.stamp = stamp.to_msg()
        header.frame_id = self.profile.frame_id
        self._points_pub.publish(points_to_pointcloud2_msg(result.points_xyz, result.intensity, header))


def main(args=None):
    rclpy.init(args=args)
    node = LidarDebugNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
