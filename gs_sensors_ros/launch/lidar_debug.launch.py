"""Launches one gs_sensors_ros lidar_debug_node instance.

One node instance = one simulated LiDAR. For a multi-sensor rig, include
this launch file multiple times (e.g. from a parent launch file) with
different `lidar_name` / `lidar_profile` / `gs_frame_transform` values --
`lidar_name` becomes the node's namespace so topics don't collide.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    args = [
        DeclareLaunchArgument("lidar_name", default_value="lidar",
                              description="Namespace for this LiDAR instance"),
        DeclareLaunchArgument("checkpoint_path",
                              description="Path to ckpt/chkpnt<N>.pth"),
        DeclareLaunchArgument("raydrop_prior_path",
                              description="Path to ckpt/lidar_raydrop_prior_chkpnt<N>.pth"),
        DeclareLaunchArgument("refine_unet_path", default_value="",
                              description="Path to ckpt/refine.pth (empty = skip raydrop "
                                          "refinement, publish the raw kernel raydrop mask -- "
                                          "GS-LiDAR's own eval always applies this, expect a "
                                          "noisier/denser point cloud without it)"),
        DeclareLaunchArgument("lidar_profile", description="Path to a lidar_profiles/*.yaml"),
        DeclareLaunchArgument("gs_frame_transform", default_value="",
                              description="Path to a gs_T_world transform JSON (empty = identity)"),
        DeclareLaunchArgument("dynamic", default_value="false",
                              description="Apply the time-varying (marginal_t) opacity/prefilter "
                                          "GS-LiDAR's own render() gates behind pipe.dynamic -- "
                                          "off matches Crosslab_lidar's own training (dynamic: False)"),
        DeclareLaunchArgument("raydrop_threshold", default_value="0.5",
                              description="Raydrop probability above which a range reading is "
                                          "dropped before unprojecting to points"),
        DeclareLaunchArgument("range_noise_stddev_m", default_value="0.0",
                              description="Synthetic per-frame Gaussian noise stddev added to "
                                          "each valid range reading, in meters, before "
                                          "unprojection (radial, matching how real LiDAR "
                                          "accuracy specs are quoted) -- 0.0 (default) disables "
                                          "it. The trained field itself is smooth/deterministic, "
                                          "not a substitute for real per-shot sensor noise"),
        DeclareLaunchArgument("intensity_noise_stddev", default_value="0.0",
                              description="Synthetic per-frame Gaussian noise stddev added to "
                                          "each point's intensity ([0,1] range) -- 0.0 (default) "
                                          "disables it"),
        DeclareLaunchArgument("pose_source", default_value="ground_truth",
                              description="'ground_truth' or 'tf'"),
        DeclareLaunchArgument("ground_truth_topic", default_value="pose"),
        DeclareLaunchArgument("world_frame", default_value="world"),
        DeclareLaunchArgument("lidar_frame", default_value="",
                              description="Defaults to the profile's frame_id if empty"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("debug", default_value="false",
                              description="Print rendered splat/returned point count + timing "
                                          "once per second"),
        DeclareLaunchArgument("enable_profiling", default_value="false",
                              description="Per-stage render timing breakdown in the debug log. "
                                          "Costs real ms (forces a torch.cuda.synchronize() per "
                                          "stage) -- only takes effect when debug:=true is also set"),
    ]

    node = Node(
        package="gs_sensors_ros",
        executable="lidar_debug_node",
        name="lidar_debug_node",
        namespace=LaunchConfiguration("lidar_name"),
        output="screen",
        parameters=[{
            "checkpoint_path": LaunchConfiguration("checkpoint_path"),
            "raydrop_prior_path": LaunchConfiguration("raydrop_prior_path"),
            "refine_unet_path": LaunchConfiguration("refine_unet_path"),
            "lidar_profile": LaunchConfiguration("lidar_profile"),
            "gs_frame_transform": LaunchConfiguration("gs_frame_transform"),
            "dynamic": ParameterValue(LaunchConfiguration("dynamic"), value_type=bool),
            "raydrop_threshold": ParameterValue(LaunchConfiguration("raydrop_threshold"), value_type=float),
            "range_noise_stddev_m": ParameterValue(LaunchConfiguration("range_noise_stddev_m"), value_type=float),
            "intensity_noise_stddev": ParameterValue(LaunchConfiguration("intensity_noise_stddev"), value_type=float),
            "pose_source": LaunchConfiguration("pose_source"),
            "ground_truth_topic": LaunchConfiguration("ground_truth_topic"),
            "world_frame": LaunchConfiguration("world_frame"),
            "lidar_frame": LaunchConfiguration("lidar_frame"),
            "use_sim_time": ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool),
            "debug": ParameterValue(LaunchConfiguration("debug"), value_type=bool),
            "enable_profiling": ParameterValue(LaunchConfiguration("enable_profiling"), value_type=bool),
        }],
    )

    return LaunchDescription(args + [node])
