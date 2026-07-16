# gs_sensors

Renders simulated sensor data directly from trained Gaussian-Splat models at
a robot's pose in Gazebo, and publishes it like a real sensor driver would.
Two branches: a camera branch (2D Gaussian Splatting -> `sensor_msgs/Image` +
`sensor_msgs/CameraInfo`) and a LiDAR branch (GS-LiDAR -> `sensor_msgs/PointCloud2`).
See `CLAUDE.md` for project rationale/scope and `TODO.md` for known
limitations.

## Layout

- `gs_sensor_core/` -- plain pip package, the rendering core (zero ROS/Gazebo imports).
- `gs_sensors_ros/` -- the ROS 2 (ament_python) debug nodes (camera + LiDAR).
- `config/camera_profiles/`, `config/lidar_profiles/` -- data-only sensor intrinsics/rate.
- `scripts/validate_lidar.py` -- LiDAR branch vs. real ground truth.
- `test_env/test_package/` -- Gazebo-free ROS 2 test harness (see "Testing without Gazebo" below).
- `test_env/docker/` -- CUDA 12.8+ container for the LiDAR kernel (see "LiDAR CUDA kernel: Docker" below).
- `gs_sensors_gz/` -- phase 2 (Gazebo plugin), not started yet.

## Setup

```bash
# 1. Submodules (pulls the diff-surfel-rasterization CUDA kernel; carries
#    Inria's non-commercial research license, see its LICENSE.md -- and the
#    GS-LiDAR fork, for the LiDAR branch, same license family)
git submodule update --init --recursive

# 2. gs_sensor_core
pip install -e gs_sensor_core

# 3. Camera branch's CUDA rasterizer -- needs an nvcc matching your installed
#    torch's CUDA version EXACTLY, or the build fails. Check if it's already
#    built/importable first (e.g. if you also have a Kestrel checkout, it's
#    the same extension):
python3 -c "import diff_surfel_rasterization" || \
    pip install -e gs_sensor_core/third_party/diff-surfel-rasterization

# 4. LiDAR branch's CUDA rasterizer -- a different kernel (panoramic, forked
#    from GS-LiDAR), same nvcc/torch-CUDA-version-matching caveat as above,
#    PLUS it needs an nvcc new enough for your actual GPU's compute
#    capability (e.g. a Blackwell/sm_120 GPU needs CUDA 12.8+ -- this isn't
#    just "matching torch," an older-but-matching nvcc can still fail here,
#    see TODO.md). Unlike the camera branch's kernel, this one is NOT
#    `pip install`-able (its own setup.py packaging doesn't work -- confirmed
#    by inspection) -- it JIT-compiles on first use instead
#    (`gs_sensor_core/render/lidar/_kernel.py`), so there's no separate
#    install step here; just be aware the first LiDAR render pays a real
#    compile-time cost (PyTorch caches the result under
#    `~/.cache/torch_extensions/` after that).

# 5. ROS 2 packages, from a workspace root
colcon build --packages-select gs_sensors_ros test_package
source install/setup.bash
```

`gs_sensors_ros` also depends on `python3-opencv` (for the `debug_view:=true`
diagnostic window, see below) -- typically already present in a desktop ROS 2
install; `rosdep install --from-paths . --ignore-src -y` from the workspace
root pulls it otherwise.

**Don't use `colcon build --symlink-install`** on Python 3.12 -- it installs
via a legacy `setup.py develop` mechanism (`.egg-link`) that Python 3.12's
`importlib.metadata` can't resolve, so every node's console-script entry
point fails with `PackageNotFoundError` at launch. Plain `colcon build`
works; it just means re-running it after editing node `.py` files (launch
files are copied as-is and pick up edits immediately either way).

## Running

```bash
ros2 launch gs_sensors_ros camera_debug.launch.py \
    ply_path:=/path/to/point_cloud/iteration_30000/point_cloud.ply \
    camera_profile:=config/camera_profiles/realsense_d435.yaml
```

`ply_path` accepts either a direct `.ply` or a training output directory
(resolved as `<dir>/point_cloud/iteration_<iterations>/point_cloud.ply`).
Nothing is published before the first successful pose lookup. For a
multi-camera rig, include this launch file multiple times with different
`camera_name` (becomes the node's ROS namespace).

Publishes, under the node's namespace: `image_raw` (`rgb8`), `camera_info`,
and `depth/image_raw` (`32FC1`, metric meters, if `publish_depth:=true`).

### Launch arguments

**Model loading / compression**

| Argument | Default | Meaning |
|---|---|---|
| `ply_path` | *(required)* | `.ply` file or training model directory |
| `iterations` | `30000` | Used only when `ply_path` is a directory |
| `sh_degree` | `-1` (auto-detect) | Force a specific SH degree instead of detecting from the PLY |
| `compression_level` | `0` | `0`-`3`, see `gs_sensor_core/compression.py` |
| `target_sh_degree` | `1` | Only used at `compression_level=2` |
| `opacity_threshold` | `0.0` (off) | Permanently drops splats at/under this activated opacity when the model loads. Kestrel's own default (`0.05`) removes ~67% of a real trained model's splats (median opacity across a whole model is typically well under 0.05) -- a real lever, not a marginal one. Off by default pending visual verification; see `prune_low_opacity` in `gs_sensor_core/compression.py` |

**Culling / LOD** (see `gs_sensor_core/culling.py`, `lod.py`, `render/rasterizer.py`)

| Argument | Default | Meaning |
|---|---|---|
| `culling_enabled` | `true` | Octree frustum culling (GPU-native) |
| `build_index` | `false` | Build (and cache) the octree index if none exists yet next to the PLY |
| `leaf_max` | `5000` | Max splats per octree leaf |
| `culling_narrow_phase` | `false` | Exact per-point frustum refinement on top of the leaf-level test. Measured close to a wash on the real test model (its own trained-scene octree leaves are already tight) -- may still help on sparser/scattered models |
| `culling_margin` | `0.0` | Slack for `culling_narrow_phase`'s test, since it checks splat centers, not rendered footprint -- raise if splats visibly pop at frame edges |
| `screen_size_culling` | `false` | Culls splats whose projected footprint is below `screen_size_min_pixels`. Measured close to a wash on the real test model -- its own pass cost roughly offset what it saved |
| `screen_size_min_pixels` | `1.0` | Projected-radius threshold in pixels for `screen_size_culling` |
| `octree_lod` | `false` | Two-level LOD: distant/small octree leaves render as one merged proxy splat instead of all their individual ones. Needs `build_index:=true` once to compute proxies (slower than a normal index build). Measured close to a wash (or a net loss, at aggressive thresholds) on the real test model at typical viewing distances -- see `TODO.md` before enabling |
| `lod_leaf_pixel_threshold` | `16.0` | Projected-radius threshold in pixels below which a whole leaf collapses to its proxy when `octree_lod:=true` |

**Camera / pose**

| Argument | Default | Meaning |
|---|---|---|
| `camera_profile` | *(required)* | Path to a `camera_profiles/*.yaml` |
| `gs_frame_transform` | `""` (identity) | Path to a `gs_T_world` JSON (see `docs/coordinate_frames.md`) |
| `publish_depth` | `true` | Publish `depth/image_raw` |
| `pose_source` | `ground_truth` | `ground_truth` (Gazebo bridge `PoseStamped`) or `tf` |
| `ground_truth_topic` | `pose` | Topic for the `ground_truth` pose source |
| `world_frame` | `world` | TF world frame, used by the `tf` pose source |
| `camera_frame` | `""` (defaults to the profile's `frame_id`) | TF camera frame -- **must be the optical frame** (see `docs/coordinate_frames.md`) |
| `use_sim_time` | `true` | Key everything off `/clock`, not wall time |

**Debug / diagnostics**

| Argument | Default | Meaning |
|---|---|---|
| `debug` | `false` | Print rendered/total splat count, render time, and pose age, once per second |
| `enable_profiling` | `false` | Per-stage render timing breakdown (cull/gather/sh_eval/rasterize/...) added to the `debug` log line. Costs real ms (forces a GPU sync per stage) -- separate flag so it's never paid just because `debug` is on; only takes effect when `debug:=true` is also set |
| `debug_view` | `false` | Opens a local OpenCV window showing exactly what this node rendered (RGB \| depth side by side), bypassing ROS transport/RViz/rqt entirely -- use to rule those out when diagnosing a smoothness problem |
| `debug_view_max_depth_m` | `10.0` | Depth normalization range for `debug_view`'s colorized depth panel only -- doesn't affect published depth |

The octree index is cached at `<ply_dir>/.gs_sensors/<ply_stem>.idx.npz`
(or `<ply_stem>_opacityN.idx.npz` when `opacity_threshold` is nonzero,
since pruning changes *which* splats exist, not just how they're stored) --
build it once with `build_index:=true`, then omit that flag on later runs.
Changing `leaf_max`, `opacity_threshold`, or turning `octree_lod` on for
the first time all need a fresh `build_index:=true` pass to regenerate it.

Camera profiles are plain YAML (`width`, `height`, `fx`, `fy`, `cx`, `cy`,
`frame_id`, `update_rate`) -- adding a camera is a new file, no code change.
`realsense_d435.yaml` uses real D435 `fx`/`fy` but a centered `cx`/`cy` (see
`docs/coordinate_frames.md` for why).

## Running the LiDAR debug node

```bash
ros2 launch gs_sensors_ros lidar_debug.launch.py \
    checkpoint_path:=/path/to/ckpt/chkpnt30000.pth \
    raydrop_prior_path:=/path/to/ckpt/lidar_raydrop_prior_chkpnt30000.pth \
    refine_unet_path:=/path/to/ckpt/refine.pth \
    lidar_profile:=config/lidar_profiles/crosslab_vlp16.yaml \
    gs_frame_transform:=/path/to/gs_frame_transform.json
```

`checkpoint_path` and `raydrop_prior_path` are always required and always paired to the same
training iteration (a `chkpnt<N>.pth` from one run doesn't match a `lidar_raydrop_prior_chkpnt<M>.pth`
from another). `refine_unet_path` is optional but strongly recommended -- see `CLAUDE.md`'s
"LiDAR branch" section for why skipping it changes the published point cloud's density/noise
characteristics, not just a minor quality knob.

**`gs_frame_transform` is effectively required for any checkpoint whose training pipeline
recentered/rescaled poses (GS-LiDAR's own `transform_poses_pca` always does, unless a training
run explicitly disabled it)** -- leaving it unset silently falls back to identity (scale 1.0),
which is almost never the checkpoint's real scale (`Crosslab_lidar`'s is `0.1`) and produces a
point cloud that's the wrong size by that exact factor, not an error. For `Crosslab_lidar`, the
correct file already exists at `test_data/Crosslab_lidar/gs_frame_transform.json` (built by
composing `axis_fix.json` + `transform_poses_pca.npz`, the same way `scripts/validate_lidar.py`'s
`load_combined_transform()` does -- regenerate the same way for a different checkpoint that
doesn't already have one saved).

Publishes, under the node's namespace: `points` (`sensor_msgs/PointCloud2`, xyz + intensity,
in the LiDAR's own frame -- see `CLAUDE.md` for why no world-frame reprojection happens here).

### Launch arguments

| Argument | Default | Meaning |
|---|---|---|
| `checkpoint_path` | *(required)* | Path to `ckpt/chkpnt<N>.pth` |
| `raydrop_prior_path` | *(required)* | Path to the matching `ckpt/lidar_raydrop_prior_chkpnt<N>.pth` |
| `refine_unet_path` | `""` (off) | Path to `ckpt/refine.pth`. Off publishes the raw kernel raydrop mask -- GS-LiDAR's own eval never runs without this stage |
| `lidar_profile` | *(required)* | Path to a `lidar_profiles/*.yaml` -- `vfov`/`hfov`/`hw` must match what the checkpoint actually trained at, see `gs_sensor_core/lidar_profiles/schema.py` |
| `gs_frame_transform` | `""` (identity) | Path to a `gs_T_world` JSON, same format as the camera branch's |
| `opacity_threshold` | `0.0` (off) | Permanently drops splats at/under this activated opacity at load time -- a real, unconditional win unlike the culling/LOD knobs below: `0.01` removes ~2% of `Crosslab_lidar`'s splats (already contributing nothing) for a real ~38-60% render-time reduction with bit-identical QA metrics. **Don't reuse the camera branch's own `0.05` default without checking** -- for this checkpoint it costs a real ~20-30% `range_MAE` increase, see `TODO.md` |
| `dynamic` | `false` | Apply GS-LiDAR's time-varying opacity/prefilter gate (`pipe.dynamic`). Off matches every trained model available so far |
| `raydrop_threshold` | `0.5` | Raydrop probability above which a range reading is dropped before publishing |
| `range_noise_stddev_m` | `0.0` (off) | Synthetic per-frame Gaussian noise stddev added to each valid range reading, in meters, before unprojection -- radial (matches how real accuracy specs like a VLP-16's "+/-3cm" are quoted), not isotropic 3D jitter. The trained field is otherwise smooth/deterministic; see `gs_sensor_core/render/lidar/pipeline.py`'s `LidarRasterizer` docstring |
| `intensity_noise_stddev` | `0.0` (off) | Synthetic per-frame Gaussian noise stddev added to each point's intensity ([0,1]) |
| `culling_enabled` | `false` | Octree-based vertical-FOV-band broad phase + LOD -- **not** camera-style frustum culling (a panoramic pose already covers the full 360° azimuth). Defaults off: measured as a net *loss* at conservative settings on the real test model (cull+gather overhead isn't paid for by how little a room-scale capture excludes) -- real speedup only shows up at aggressive `lod_ray_pitch_cutoff` values that cost real accuracy. See `TODO.md`'s "LiDAR branch" section before enabling. Has no effect until `build_index:=true` is run at least once regardless |
| `culling_margin_deg` | `5.0` | Conservative buffer added to the vertical-FOV-band broad-phase test, degrees |
| `octree_lod` | `false` | Merge distant/angularly-small octree leaves into precomputed proxies -- the lever that actually matches this sensor's bottleneck, see `gs_sensor_core/render/lidar/pipeline.py`'s `LidarRasterizer` docstring |
| `lod_ray_pitch_cutoff` | `1.0` | A leaf uses its LOD proxy when its angular size (as seen from the sensor) is below this many ray-widths |
| `build_index` | `false` | Build (and cache under `<checkpoint_dir>/.gs_sensors/`) the octree index if none exists yet -- one-time cost |
| `leaf_max` | `5000` | Max points per octree leaf |
| `pose_source` | `ground_truth` | `ground_truth` (Gazebo bridge `PoseStamped`) or `tf` |
| `ground_truth_topic` | `pose` | Topic for the `ground_truth` pose source |
| `world_frame` | `world` | TF world frame, used by the `tf` pose source |
| `lidar_frame` | `""` (defaults to the profile's `frame_id`) | TF LiDAR frame |
| `debug` | `false` | Print rendered splat / returned point count, render time, and pose age, once per second |
| `enable_profiling` | `false` | Per-stage render timing breakdown added to the `debug` log line |

LiDAR profiles are plain YAML (`vfov`, `hfov`, `hw`, `frame_id`, `update_rate`, `scale_factor`,
`beam_count`) -- `vfov`/`hfov`/`hw` come straight from the checkpoint's own training config
(not free choices), `frame_id`/`update_rate`/`beam_count` are real-sensor-facing values that
need your input, same category of gap the camera branch's `realsense_d435.yaml` had for
`cx`/`cy`. `crosslab_vlp16.yaml`'s values for those three are best-effort defaults for a real
VLP-16 (10 Hz, 16 channels) -- confirm against your actual driver before trusting them downstream.

### Validating against real ground truth

`test_data/gs_lidar_source/qa/` has real ground-truth range/intensity panoramas (3 known
frames) and an accumulated point cloud from the actual trajectory `test_data/Crosslab_lidar`
trained from:

```bash
python3 scripts/validate_lidar.py
python3 scripts/validate_lidar.py --dump-accumulated-cloud /tmp/accumulated.ply  # slower, all 456 frames
```

Needs the vendored `diff-gaussian-rasterization-2d` kernel to actually build -- see the CUDA
toolchain note in Setup and `TODO.md`.

## Diagnosing performance / smoothness problems

- `debug:=true` -- rendered/total splat count, render time, and pose age, once per second.
- `debug:=true enable_profiling:=true` -- adds a per-stage timing breakdown to that log
  line. Costs real time itself (forces a GPU sync per stage), so leave it off except when
  actively diagnosing.
- `debug_view:=true` -- opens a local window showing exactly what this node rendered,
  independent of RViz/rqt/ROS transport. If that window is smooth while RViz isn't, the
  problem is downstream of this node, not in it.

The culling/LOD knobs above (`culling_narrow_phase`, `screen_size_culling`, `octree_lod`)
are all real, tested mechanisms, but none of them measured as a clear win on this project's
own real test model -- see `TODO.md`'s "Rendering performance" section for the actual
numbers and why. The two changes that *did* measure as clear, unconditional wins aren't
flags at all: masking the model's raw tensors before activating them instead of after
(`GaussianModel._activate`), and genuine (not round-tripped) float16 storage at
`compression_level >= 1`.

## Testing without Gazebo: `test_package`

Lives at `test_env/test_package/` (a colcon package name, `test_package`, is what
`colcon build`/`ros2 launch` actually reference -- its source directory location doesn't
matter to either). Two ways to drive the camera without Gazebo running, both usable together
with `camera_debug_node` in one launch command (`launch_camera:=true`,
the default) or standalone (`launch_camera:=false run_clock:=false`
against a `camera_debug_node`/Gazebo you're already running).

**Interactive** (`teleop_camera_test.launch.py`), driven by `teleop_twist_keyboard`:

```bash
ros2 launch test_package teleop_camera_test.launch.py \
    ply_path:=/path/to/point_cloud/iteration_30000/point_cloud.ply \
    camera_profile:=config/camera_profiles/realsense_d435.yaml
# in a second terminal:
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

**Scripted** (`moving_camera_test.launch.py`), an unattended oscillating dolly, same arguments.

Both start at an arbitrary point in the model's own coordinate frame
(`base_x/y/z`, default `0,0,3`) -- if the render comes out all black
(`camera_debug_node` also logs "0 splats rendered" when this happens),
pass a real in-scene position from that model's own `cameras.json`:

```bash
... base_x:=10.24 base_y:=0.76 base_z:=-2.48
```

`look_direction` (`+x`/`-x`/`+y`/`-y`/`+z`/`-z`, default `+x`) sets which way
the camera faces (and, since motion is always relative to view direction,
which way it moves); `world_up` (default `+z`) must match whichever axis is
actually "up" for the model at hand -- this varies per reconstruction. See
`test_env/test_package/directions.py` and `test_env/test_package/*_pose_publisher.py` module
docstrings for the full convention, and `moving_camera_pose_publisher` also
accepts an exact `qx/qy/qz/qw` (via `look_direction:=""`) for a real
training-view orientation instead.

The LiDAR branch has the same two harnesses (`teleop_lidar_test.launch.py` /
`moving_lidar_test.launch.py`), paired with `lidar_debug_node` instead
(`launch_lidar:=true`, the default):

```bash
ros2 launch test_package teleop_lidar_test.launch.py \
    config_file:=test_env/test_package/config/crosslab_lidar.yaml
# in a second terminal:
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

`config_file` loads a YAML file of launch-argument defaults (`test_env/test_package/config/
crosslab_lidar.yaml` has `checkpoint_path`/`raydrop_prior_path`/`refine_unet_path`/
`lidar_profile`/`gs_frame_transform`/`debug` filled in for `Crosslab_lidar`, plus
`range_noise_stddev_m`/`intensity_noise_stddev` both defaulted to `0.02` -- a real VLP-16's
quoted "+/-3cm typical" range accuracy, read as a 1-sigma stddev) -- an explicit `key:=value` on
the command line still overrides whatever the config file set, same precedence as any other
launch-argument default (e.g. `... range_noise_stddev_m:=0.0` to compare against the noise-free
render). Same mechanism on `moving_lidar_test.launch.py`. Without a config file, pass the same
paths individually (see either launch file's module docstring) -- **don't skip
`gs_frame_transform`** either way: without it the published cloud is silently 10x too small
(identity-scale fallback vs. this checkpoint's real `0.1`), not obviously broken, just the wrong
size relative to the room.

Both LiDAR harnesses also publish a `pose_marker` (`visualization_msgs/MarkerArray`, in the
LiDAR's own frame) -- a yellow sphere at the sensor's position and a red arrow along its
forward-pass boresight direction. Add a MarkerArray display on `/lidar/pose_marker` in RViz
(needs `publish_tf:=true`, the default) to see where the simulated sensor actually is and which
way it's facing.

Needs the vendored `diff-gaussian-rasterization-2d` kernel built (see the CUDA toolchain
note in Setup and `test_env/docker/`) -- run it inside `test_env/docker`'s container if the
host `nvcc` can't target your GPU. Default base pose is `0,0,0` (a real in-scene position for
`Crosslab_lidar`, unlike the camera harness's arbitrary default -- see
`teleop_lidar_pose_publisher.py`'s docstring if it needs to change for a different checkpoint).
Same `look_direction`/`world_up` convention as the camera harness; the published pose still
needs a defined "forward" even though the sensor is omnidirectional, since GS-LiDAR's
panoramic render always splits into a forward-centered pass and its opposing backward pass.
Watch results with `ros2 topic hz /lidar/points`, `ros2 topic echo /lidar/points --field header`,
or an RViz `PointCloud2` display on `/lidar/points` (`test_env/docker`'s container has X11-
passthrough `rviz2` -- see its README).

