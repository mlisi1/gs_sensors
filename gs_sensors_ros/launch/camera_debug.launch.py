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
                              description="0 (none) - 3 (aggressive); see gs_sensor_core/compression.py"),
        DeclareLaunchArgument("target_sh_degree", default_value="1",
                              description="Only used at compression_level 2"),
        DeclareLaunchArgument("culling_enabled", default_value="true",
                              description="Octree frustum culling"),
        DeclareLaunchArgument("culling_backend", default_value="cpu",
                              description="'cpu' (numpy, existing) or 'gpu' (torch-native, "
                                          "no GPU->CPU->GPU round trip per frame) -- benchmark "
                                          "with debug:=true before picking one"),
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
            "culling_enabled": ParameterValue(LaunchConfiguration("culling_enabled"), value_type=bool),
            "culling_backend": LaunchConfiguration("culling_backend"),
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
        }],
    )

    return LaunchDescription(args + [node])
