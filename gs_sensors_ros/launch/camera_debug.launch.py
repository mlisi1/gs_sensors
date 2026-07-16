"""Launches one gs_sensors_ros camera_debug_node instance.

One node instance = one simulated camera. For a multi-camera rig, include
this launch file multiple times (e.g. from a parent launch file) with
different `camera_name` / `camera_profile` / `gs_frame_transform` values --
`camera_name` becomes the node's namespace so topics don't collide.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    args = [
        DeclareLaunchArgument("camera_name", default_value="camera",
                              description="Namespace for this camera instance"),
        DeclareLaunchArgument("ply_path", description="Path to a .ply, or a training model directory"),
        DeclareLaunchArgument("iterations", default_value="30000",
                              description="Used only when ply_path is a training model directory"),
        DeclareLaunchArgument("camera_profile", description="Path to a camera_profiles/*.yaml"),
        DeclareLaunchArgument("gs_frame_transform", default_value="",
                              description="Path to a gs_T_world transform JSON (empty = identity)"),
        DeclareLaunchArgument("sh_degree", default_value="-1"),
        DeclareLaunchArgument("compression_level", default_value="0",
                              description="0 (none) - 3 (aggressive); see gsplat2d_rendering.compression"),
        DeclareLaunchArgument("target_sh_degree", default_value="1",
                              description="Only used at compression_level 2"),
        DeclareLaunchArgument("opacity_threshold", default_value="0.0",
                              description="Permanently drops splats at/under this activated "
                                          "opacity when the model loads (0.0 = off, no pruning). "
                                          "Kestrel's own default is 0.05 -- on this project's real "
                                          "test model that removes ~67% of all splats (median "
                                          "opacity across the whole model is ~0.004), so this is "
                                          "not a marginal cut. Off by default pending visual "
                                          "verification -- see gsplat2d_rendering.compression's "
                                          "prune_low_opacity"),
        DeclareLaunchArgument("culling_enabled", default_value="true",
                              description="Octree frustum culling (GPU-native)"),
        DeclareLaunchArgument("culling_narrow_phase", default_value="false",
                              description="Exact per-point frustum test on top of the leaf-level "
                                          "one, restricted to points that already passed it -- "
                                          "tightens the leaf test's over-inclusion at scene edges. "
                                          "Measured close to a wash on real (spatially-coherent) "
                                          "scenes, where octree leaves are already tight -- the "
                                          "synthetic benchmark that motivated this used uniformly "
                                          "random points, an AABB-looseness worst case that isn't "
                                          "representative of trained-scene structure. Off by "
                                          "default; may still help on sparser/scattered models"),
        DeclareLaunchArgument("culling_margin", default_value="0.0",
                              description="Slack added to the narrow-phase frustum test since it "
                                          "checks splat centers, not their rendered footprint -- "
                                          "raise from 0 if splats visibly pop at frame edges"),
        DeclareLaunchArgument("screen_size_culling", default_value="false",
                              description="Culls candidates whose projected screen footprint is "
                                          "below screen_size_min_pixels -- a coarse pinhole-"
                                          "approximation proxy, not the CUDA kernel's exact "
                                          "footprint math (see visible_point_mask_screen_size_torch "
                                          "in culling.py). Off by default; benchmark with "
                                          "debug:=true enable_profiling:=true (screen_size_cull "
                                          "stage) before enabling"),
        DeclareLaunchArgument("screen_size_min_pixels", default_value="1.0",
                              description="Projected-radius threshold in pixels below which a "
                                          "splat is culled when screen_size_culling:=true"),
        DeclareLaunchArgument("octree_lod", default_value="false",
                              description="Two-level LOD: visible octree leaves smaller than "
                                          "lod_leaf_pixel_threshold on screen render as one "
                                          "moment-matched merged proxy Gaussian instead of all "
                                          "their individual splats (see build_leaf_proxies in "
                                          "culling.py). Needs build_index:=true the first time "
                                          "(or after leaf_max/compression changes) to compute and "
                                          "cache the proxies -- a cached index built without this "
                                          "set degrades to 'LOD unavailable' with a printed notice, "
                                          "not an error. Off by default, unbenchmarked on real data yet"),
        DeclareLaunchArgument("lod_leaf_pixel_threshold", default_value="16.0",
                              description="Projected-radius threshold in pixels below which a "
                                          "whole leaf collapses to its proxy when octree_lod:=true"),
        DeclareLaunchArgument("build_index", default_value="false",
                              description="Build the octree index if no cached one exists yet"),
        DeclareLaunchArgument("leaf_max", default_value="5000",
                              description="Max splats per octree leaf node"),
        DeclareLaunchArgument("publish_depth", default_value="true"),
        DeclareLaunchArgument("pose_source", default_value="ground_truth",
                              description="'ground_truth' or 'tf'"),
        DeclareLaunchArgument("ground_truth_topic", default_value="pose"),
        DeclareLaunchArgument("world_frame", default_value="world"),
        DeclareLaunchArgument("camera_frame", default_value="",
                              description="Defaults to the profile's frame_id if empty"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("debug", default_value="false",
                              description="Print rendered/total splat count + timing once per second"),
        DeclareLaunchArgument("enable_profiling", default_value="false",
                              description="Per-stage render timing breakdown in the debug log "
                                          "(cull/gather/sh_eval/rasterize/depth_extract/copy_to_cpu). "
                                          "Costs real ms (forces a torch.cuda.synchronize() per stage) "
                                          "-- separate from 'debug' so it's off by default; only takes "
                                          "effect when debug:=true is also set, since that's what logs it"),
        DeclareLaunchArgument("debug_view", default_value="false",
                              description="Opens an OpenCV window showing exactly what this node "
                                          "rendered (RGB | depth side by side), bypassing ROS "
                                          "transport/RViz/rqt entirely -- use to rule those out as "
                                          "the source of a smoothness problem. See "
                                          "gs_sensors_ros/debug_view.py"),
        DeclareLaunchArgument("debug_view_max_depth_m", default_value="10.0",
                              description="Depth normalization range for debug_view's colorized "
                                          "depth panel only -- doesn't affect published depth"),
    ]

    node = Node(
        package="gs_sensors_ros",
        executable="camera_debug_node",
        name="camera_debug_node",
        namespace=LaunchConfiguration("camera_name"),
        output="screen",
        parameters=[{
            "ply_path": LaunchConfiguration("ply_path"),
            "iterations": ParameterValue(LaunchConfiguration("iterations"), value_type=int),
            "camera_profile": LaunchConfiguration("camera_profile"),
            "gs_frame_transform": LaunchConfiguration("gs_frame_transform"),
            "sh_degree": ParameterValue(LaunchConfiguration("sh_degree"), value_type=int),
            "compression_level": ParameterValue(LaunchConfiguration("compression_level"), value_type=int),
            "target_sh_degree": ParameterValue(LaunchConfiguration("target_sh_degree"), value_type=int),
            "opacity_threshold": ParameterValue(LaunchConfiguration("opacity_threshold"), value_type=float),
            "culling_enabled": ParameterValue(LaunchConfiguration("culling_enabled"), value_type=bool),
            "culling_narrow_phase": ParameterValue(LaunchConfiguration("culling_narrow_phase"), value_type=bool),
            "culling_margin": ParameterValue(LaunchConfiguration("culling_margin"), value_type=float),
            "screen_size_culling": ParameterValue(LaunchConfiguration("screen_size_culling"), value_type=bool),
            "screen_size_min_pixels": ParameterValue(LaunchConfiguration("screen_size_min_pixels"), value_type=float),
            "octree_lod": ParameterValue(LaunchConfiguration("octree_lod"), value_type=bool),
            "lod_leaf_pixel_threshold": ParameterValue(LaunchConfiguration("lod_leaf_pixel_threshold"), value_type=float),
            "build_index": ParameterValue(LaunchConfiguration("build_index"), value_type=bool),
            "leaf_max": ParameterValue(LaunchConfiguration("leaf_max"), value_type=int),
            "publish_depth": ParameterValue(LaunchConfiguration("publish_depth"), value_type=bool),
            "pose_source": LaunchConfiguration("pose_source"),
            "ground_truth_topic": LaunchConfiguration("ground_truth_topic"),
            "world_frame": LaunchConfiguration("world_frame"),
            "camera_frame": LaunchConfiguration("camera_frame"),
            "use_sim_time": ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool),
            "debug": ParameterValue(LaunchConfiguration("debug"), value_type=bool),
            "enable_profiling": ParameterValue(LaunchConfiguration("enable_profiling"), value_type=bool),
            "debug_view": ParameterValue(LaunchConfiguration("debug_view"), value_type=bool),
            "debug_view_max_depth_m": ParameterValue(LaunchConfiguration("debug_view_max_depth_m"), value_type=float),
        }],
    )

    return LaunchDescription(args + [node])
