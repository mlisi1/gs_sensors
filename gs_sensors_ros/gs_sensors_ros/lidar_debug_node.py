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
import torch
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header

from gs_sensor_core.culling import load_or_build_octree
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
        self.declare_parameter("range_noise_stddev_m", 0.0)     # 0.0 = off, see LidarRasterizer
        self.declare_parameter("intensity_noise_stddev", 0.0)   # 0.0 = off, see LidarRasterizer
        self.declare_parameter("opacity_threshold", 0.0)        # 0.0 = off, see LidarGaussianModel.prune_low_opacity_

        # Culling / LOD (see gs_sensor_core/render/lidar/culling.py, lod.py --
        # LOD, not visibility culling, is the lever that matches this
        # sensor's actual bottleneck; see LidarRasterizer's class docstring).
        # Defaults to OFF: measured directly against Crosslab_lidar
        # (~474K splats, room-scale), the cull+gather machinery's own
        # per-frame overhead exceeds its savings at conservative settings
        # (default culling_margin_deg=5.0/lod_ray_pitch_cutoff=1.0 barely
        # excludes anything at room-scale distances) -- real speedup only
        # shows up at aggressive lod_ray_pitch_cutoff values that also cost
        # real accuracy (range_MAE roughly doubles by cutoff=16), so this
        # isn't a safe "just turn it on" default the way the camera
        # branch's culling is. Opt in and tune deliberately; see TODO.md's
        # "LiDAR branch" section for the actual before/after numbers.
        self.declare_parameter("culling_enabled", False)
        self.declare_parameter("culling_margin_deg", 5.0)
        self.declare_parameter("octree_lod", False)
        self.declare_parameter("lod_ray_pitch_cutoff", 1.0)
        self.declare_parameter("build_index", False)
        self.declare_parameter("leaf_max", 5000)

        # Sensor / pose
        self.declare_parameter("lidar_profile", "")
        self.declare_parameter("gs_frame_transform", "")
        self.declare_parameter("pose_source", "ground_truth")
        self.declare_parameter("ground_truth_topic", "pose")
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("lidar_frame", "")
        # 0.0 = use the profile's own update_rate (the normal case: this
        # value is baked into what the checkpoint was trained/tuned at,
        # not a free knob) -- overriding it doesn't change the render
        # itself, just how often it's triggered, so it's a legitimate way
        # to probe how fast this pose/model/culling combination can
        # actually go without editing the shared profile YAML.
        self.declare_parameter("update_rate_override", 0.0)

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

        opacity_threshold = float(self.get_parameter("opacity_threshold").value)
        self.get_logger().info(f"Loading GS-LiDAR checkpoint from {checkpoint_path} ...")
        model = load_lidar_gaussian_model(checkpoint_path, opacity_threshold=opacity_threshold)
        raydrop_prior = load_raydrop_prior(raydrop_prior_path)

        refine_path = self.get_parameter("refine_unet_path").value
        refine_unet = load_refine_unet(refine_path) if refine_path else None
        if refine_unet is None:
            self.get_logger().warn(
                "No refine_unet_path given -- publishing raw (unrefined) raydrop mask. "
                "GS-LiDAR's own eval always applies the refine UNet; expect a noisier/"
                "denser point cloud than the validated pipeline.")

        culling_enabled = bool(self.get_parameter("culling_enabled").value)
        octree_lod = bool(self.get_parameter("octree_lod").value)
        octree = None
        if culling_enabled:
            # LOD proxies need opacity/scale/rotation/features_dc too, not
            # just xyz -- only pulled from the model (extra GPU->CPU
            # copies) when octree_lod is actually requested.
            # keep_normal_axis=True: this kernel is 3D-GS-family (genuine
            # 3D extent per splat), not the camera branch's 2D-surfel one
            # -- see lod.py's build_leaf_proxies docstring. checkpoint_path
            # stands in for ply_path here purely for cache-file naming
            # (index_cache_path isn't actually PLY-specific).
            octree = load_or_build_octree(
                checkpoint_path,
                model.get_xyz.detach().cpu().numpy(),
                leaf_max=self.get_parameter("leaf_max").value,
                build_index=bool(self.get_parameter("build_index").value),
                compute_lod=octree_lod,
                opacity=model.get_opacity.detach().cpu().numpy() if octree_lod else None,
                scale=model.get_scaling.detach().cpu().numpy() if octree_lod else None,
                rotation=model.get_rotation.detach().cpu().numpy() if octree_lod else None,
                features_dc=model.features_dc.detach().cpu().numpy() if octree_lod else None,
                keep_normal_axis=True,
                opacity_threshold=opacity_threshold,
            )
            # Required precondition for LidarRasterizer's contiguous-slice
            # gather (see its class docstring) -- one-time cost at
            # startup, not per-frame.
            perm = torch.from_numpy(octree.flat_indices).long().to(model.xyz.device)
            model.reorder_(perm)

        self._total_splats = model.num_points
        self._debug = bool(self.get_parameter("debug").value)
        self._enable_profiling = bool(self.get_parameter("enable_profiling").value)

        self.rasterizer = LidarRasterizer(
            model, raydrop_prior, self.profile, refine_unet=refine_unet,
            gs_scale=self.gs_transform.scale,
            dynamic=bool(self.get_parameter("dynamic").value),
            raydrop_threshold=float(self.get_parameter("raydrop_threshold").value),
            range_noise_stddev_m=float(self.get_parameter("range_noise_stddev_m").value),
            intensity_noise_stddev=float(self.get_parameter("intensity_noise_stddev").value),
            octree=octree,
            culling_enabled=culling_enabled,
            culling_margin_deg=float(self.get_parameter("culling_margin_deg").value),
            octree_lod=octree_lod,
            lod_ray_pitch_cutoff=float(self.get_parameter("lod_ray_pitch_cutoff").value),
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

        update_rate_override = float(self.get_parameter("update_rate_override").value)
        self._update_rate = update_rate_override if update_rate_override > 0.0 else self.profile.update_rate
        self._period_s = 1.0 / self._update_rate
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
        # The pose actually used to render may be older than `stamp` (e.g.
        # GroundTruthPoseSource ignores `stamp` and returns whatever was
        # last received) -- publish with the pose's own valid-at timestamp,
        # not the render loop's `now()`, so a downstream TF-based transform
        # (RViz's Fixed Frame view) resolves the exact same pose sample
        # this render used. See pose_source.py's `pose_stamp` docstring --
        # without this, the published cloud visibly lags TF during
        # rotation (angular motion displaces far points more per unit time
        # than linear motion does for the same timestamp gap).
        publish_stamp = self.pose_source.pose_stamp() or stamp
        pose_gs = self.gs_transform.apply(pose_world)

        t0 = time.perf_counter()
        result = self.rasterizer.render(pose_gs, profile=self._enable_profiling)
        elapsed_s = time.perf_counter() - t0

        if self._debug:
            # result.num_rendered sums the prefilter-mask pass-count across
            # BOTH panoramic passes (forward + backward) -- unlike the
            # camera branch's single-pass equivalent, this is NOT bounded
            # by _total_splats when culling/LOD are off (== _total_splats *
            # 2 every frame in that case, every splat passes the
            # opacity-only prefilter on both passes). "X / Y splats" reads
            # as a bug (X > Y) if phrased like the camera branch's log line
            # -- spelled out explicitly instead. With culling/LOD enabled
            # this number reflects the actual (typically much smaller)
            # post-cull/LOD candidate set for this frame's pose.
            culling_note = "no culling/LOD" if self.rasterizer.octree is None else "culling/LOD active"
            msg = (
                f"Rendered {result.num_rendered:,} splat-evaluations across 2 passes "
                f"(model: {self._total_splats:,} splats, {culling_note}), "
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
                f"{self._update_rate:.1f} Hz -- dropping frame timing",
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
        header.stamp = publish_stamp.to_msg()
        header.frame_id = self.profile.frame_id
        self._points_pub.publish(points_to_pointcloud2_msg(result.points_xyz, result.intensity, header))


def main(args=None):
    rclpy.init(args=args)
    node = LidarDebugNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        # ExternalShutdownException: rclpy's own SIGINT handler can
        # invalidate the context while executor.spin() is mid-wait,
        # raising this instead of (or in addition to) KeyboardInterrupt --
        # both are a normal Ctrl+C, not an error.
        pass
    finally:
        # Without this, the executor's worker thread pool is never told to
        # stop/join -- the process can hang around well after Ctrl+C
        # waiting for threads that were never signaled to exit, especially
        # if one was mid-CUDA-call (blocking, uninterruptible from Python)
        # when the signal arrived.
        executor.shutdown()
        node.destroy_node()
        # rclpy's own SIGINT handler already calls rclpy.shutdown() on the
        # context before this finally block runs -- calling it again
        # unconditionally raises RCLError("rcl_shutdown already called"),
        # a harmless but noisy traceback on every Ctrl+C. rclpy.ok() is
        # False once that's already happened.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
