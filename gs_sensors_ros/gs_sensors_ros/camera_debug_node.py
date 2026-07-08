"""Phase 1 standalone ROS 2 debug node: renders simulated camera frames from a
trained 2DGS model at the pose of a robot moving in Gazebo, and publishes them
like a real camera driver would. See CLAUDE.md "Phase 1 -- debug node".

This file only does parameter declarations, the timer callback, and
delegation into gs_sensor_core + ros_conversions -- no rendering math or
frame algebra here, that lives in gs_sensor_core.
"""
from __future__ import annotations

import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header

from gs_sensor_core.camera_profiles.schema import CameraProfile
from gs_sensor_core.culling import load_or_build_octree
from gs_sensor_core.frames import GSFrameTransform, load_gs_frame_transform
from gs_sensor_core.models import load_gaussian_model, resolve_ply_path
from gs_sensor_core.render import CameraRasterizer

from gs_sensors_ros.pose_source import make_pose_source
from gs_sensors_ros.ros_conversions import depth_to_image_msg, profile_to_camera_info, rgb_to_image_msg


class CameraDebugNode(Node):

    def __init__(self):
        super().__init__("camera_debug_node")

        self.declare_parameter("ply_path", "")
        self.declare_parameter("iterations", 30000)
        self.declare_parameter("sh_degree", -1)
        self.declare_parameter("compression_level", 0)
        self.declare_parameter("target_sh_degree", 1)
        self.declare_parameter("culling_enabled", True)
        self.declare_parameter("build_index", False)
        self.declare_parameter("leaf_max", 5000)
        self.declare_parameter("camera_profile", "")
        self.declare_parameter("gs_frame_transform", "")
        self.declare_parameter("pose_source", "ground_truth")
        self.declare_parameter("ground_truth_topic", "pose")
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("camera_frame", "")
        self.declare_parameter("publish_depth", True)
        self.declare_parameter("debug", False)

        ply_path_param = self.get_parameter("ply_path").value
        profile_path = self.get_parameter("camera_profile").value
        if not ply_path_param or not profile_path:
            raise RuntimeError("Both 'ply_path' and 'camera_profile' parameters are required")

        # Accepts either a direct .ply, or a training model directory (resolved
        # via 'iterations') -- see gs_sensor_core.models.paths.resolve_ply_path.
        ply_path = resolve_ply_path(ply_path_param, iterations=self.get_parameter("iterations").value)

        self.profile = CameraProfile.from_yaml(profile_path)

        transform_path = self.get_parameter("gs_frame_transform").value
        self.gs_transform = (
            load_gs_frame_transform(transform_path) if transform_path
            else GSFrameTransform.identity()
        )

        self.get_logger().info(f"Loading Gaussian-splat model from {ply_path} ...")
        model = load_gaussian_model(
            ply_path,
            sh_degree=self.get_parameter("sh_degree").value,
            compression_level=self.get_parameter("compression_level").value,
            target_sh_degree=self.get_parameter("target_sh_degree").value,
        )

        culling_enabled = bool(self.get_parameter("culling_enabled").value)
        octree = None
        if culling_enabled:
            octree = load_or_build_octree(
                ply_path,
                model.get_xyz.detach().cpu().numpy(),
                leaf_max=self.get_parameter("leaf_max").value,
                build_index=bool(self.get_parameter("build_index").value),
            )

        self._total_splats = model.num_points
        self._debug = bool(self.get_parameter("debug").value)

        publish_depth = bool(self.get_parameter("publish_depth").value)
        self.rasterizer = CameraRasterizer(
            model, self.profile,
            gs_scale=self.gs_transform.scale,
            publish_depth=publish_depth,
            octree=octree,
            culling_enabled=culling_enabled,
        )

        camera_frame = self.get_parameter("camera_frame").value or self.profile.frame_id
        self.pose_source = make_pose_source(
            self,
            kind=self.get_parameter("pose_source").value,
            ground_truth_topic=self.get_parameter("ground_truth_topic").value,
            world_frame=self.get_parameter("world_frame").value,
            camera_frame=camera_frame,
        )

        self._image_pub = self.create_publisher(Image, "image_raw", 10)
        self._info_pub = self.create_publisher(CameraInfo, "camera_info", 10)
        self._depth_pub = (
            self.create_publisher(Image, "depth/image_raw", 10) if publish_depth else None
        )

        self._period_s = 1.0 / self.profile.update_rate
        self._first_pose_seen = False
        self._timer = self.create_timer(self._period_s, self._on_timer)

    def _on_timer(self) -> None:
        stamp = self.get_clock().now()
        pose_world = self.pose_source.get_pose(stamp)
        if pose_world is None:
            # Don't publish anything before the first successful pose lookup
            # -- avoids garbage frames while TF/bridge is still coming up.
            if not self._first_pose_seen:
                self.get_logger().info("Waiting for first pose ...", throttle_duration_sec=5.0)
            return
        self._first_pose_seen = True

        pose_gs = self.gs_transform.apply(pose_world)

        t0 = time.perf_counter()
        result = self.rasterizer.render(pose_gs)
        elapsed_s = time.perf_counter() - t0

        if self._debug:
            # A dedicated flag rather than the ROS log-level mechanism --
            # --log-level debug also turns on rcl/rmw's own internal debug
            # noise, which drowns out the one line we actually want.
            self.get_logger().info(
                f"Rendered {result.num_rendered:,} / {self._total_splats:,} splats "
                f"in {elapsed_s * 1000:.1f} ms",
                throttle_duration_sec=1.0,
            )

        if elapsed_s > self._period_s:
            self.get_logger().warn(
                f"Render took {elapsed_s * 1000:.1f} ms, over the "
                f"{self._period_s * 1000:.1f} ms frame budget at "
                f"{self.profile.update_rate:.1f} Hz -- dropping frame timing",
                throttle_duration_sec=5.0,
            )
        if result.num_rendered == 0:
            self.get_logger().warn(
                "0 splats rendered this frame -- the camera pose is outside the model's "
                "content (or entirely culled). Check gs_frame_transform and the pose source, "
                "not the renderer itself.",
                throttle_duration_sec=5.0,
            )

        header = Header()
        header.stamp = stamp.to_msg()
        header.frame_id = self.profile.frame_id

        self._image_pub.publish(rgb_to_image_msg(result.rgb, header))
        self._info_pub.publish(profile_to_camera_info(self.profile, header))
        if self._depth_pub is not None and result.depth is not None:
            self._depth_pub.publish(depth_to_image_msg(result.depth, header))


def main(args=None):
    rclpy.init(args=args)
    node = CameraDebugNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
