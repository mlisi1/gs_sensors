#!/usr/bin/env python3
"""Validates the GS-LiDAR render pipeline against real ground truth, closing
the gap flagged in TODO.md for the camera branch's `GSFrameTransform`
(tested only at identity, never against a real non-trivial pose/scale) --
see CLAUDE.md's plan for the LiDAR branch.

For QA frames 0, 227, 455 (the ones `gs_lidar_source/qa/panorama/` has
ground truth for): reads that frame's real `lidar2world` pose from
`transforms_0000_all.json`, maps it into GS-training space (composing
`axis_fix.json`'s rigid remap with `transform_poses_pca.npz`'s Sim(3),
exactly the two pieces of prior art `gs_sensor_core/frames.py`'s
`GSFrameTransform` generalizes -- see its module docstring), renders at
*native* resolution (66x515 per-half, matching `w_lidar`/`h_lidar` in
`transforms_0000_all.json` and `qa/panorama/*.npz`'s own shape -- NOT the
`[32, 512]` training/publish resolution), and reports range/intensity error
against the ground-truth panorama.

Requires a built `diff_gaussian_rasterization_2d` kernel (see
`gs_sensor_core/render/lidar/_kernel.py`) -- this script has NOT been run to
completion in this environment (nvcc 12.0 here can't target this machine's
sm_120 GPU; see TODO.md). Run it wherever that kernel actually builds
before trusting the render pipeline against real data.

Usage:
    python3 scripts/validate_lidar.py [--dump-accumulated-cloud OUT.ply]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from gs_sensor_core.frames import GSFrameTransform, Pose
from gs_sensor_core.lidar_profiles.schema import LidarProfile
from gs_sensor_core.models.lidar_checkpoint_loader import load_lidar_gaussian_model, load_raydrop_prior
from gs_sensor_core.render.lidar.camera import build_lidar_cameras
from gs_sensor_core.render.lidar.pointcloud import pano_to_points
from gs_sensor_core.render.lidar.rasterizer import render_lidar_panorama
from gs_sensor_core.render.lidar.refine import load_refine_unet, refine_raydrop
from gs_sensor_core.rotations import quat_to_rotmat, rotmat_to_quat

REPO_ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = REPO_ROOT / "test_data" / "Crosslab_lidar"
SOURCE_DIR = REPO_ROOT / "test_data" / "gs_lidar_source"
QA_FRAMES = (0, 227, 455)


def load_combined_transform() -> GSFrameTransform:
    """axis_fix.json (rigid remap) then transform_poses_pca.npz (Sim(3)) --
    same two pieces of prior art CLAUDE.md cites for frames.py, composed
    here instead of pre-merged into one JSON since this is a one-off
    validation, not the production frame-transform artifact."""
    axis_fix = json.loads((SOURCE_DIR / "0000" / "axis_fix.json").read_text())
    m = np.asarray(axis_fix["axis_fix_world_T_world"], dtype=np.float64)
    axis_fix_transform = GSFrameTransform(rotation=m[:3, :3], translation=m[:3, 3], scale=1.0)

    npz = np.load(CKPT_DIR / "transform_poses_pca.npz")
    transform, scale = npz["transform"], float(npz["scale_factor"])
    # transform_poses_pca's saved 4x4 has scale folded into the rotation
    # block (transform[:3,:3] == scale * rot, a genuine similarity, not a
    # pure rotation) -- GSFrameTransform needs rotation and scale kept
    # separate (see its docstring: "orientation is rotated ... never
    # scaled"), so divide the scale back out to recover the pure rotation
    # and the pre-scale translation. See this script's module docstring
    # derivation; not an assumption -- matches
    # ~/GS-LiDAR/scene/kitti360_loader.py:transform_poses_pca's own
    # construction (`transform = diag([scale]*3+[1]) @ transform`, applied
    # AFTER the recenter/rotate transform was already built).
    rot = transform[:3, :3] / scale
    translation = transform[:3, 3] / scale
    pca_transform = GSFrameTransform(rotation=rot, translation=translation, scale=scale)

    # Compose: apply axis_fix first, then pca, matching the pipeline order
    # (GSLidarPreprocess's axis_fix runs before GS-LiDAR's own
    # transform_poses_pca during training-data prep).
    combined_rotation = pca_transform.rotation @ axis_fix_transform.rotation
    combined_translation = pca_transform.rotation @ axis_fix_transform.translation + pca_transform.translation
    return GSFrameTransform(rotation=combined_rotation, translation=combined_translation,
                             scale=pca_transform.scale)


# KITTI-360's raw `lidar2world` encodes the LiDAR's pose in its own raw
# (velodyne) axis convention (x-forward, y-left, z-up) -- NOT the optical
# convention (x-right, y-down, z-forward) `build_lidar_cameras`/the
# panoramic direction math assumes (see render/lidar/camera.py's
# docstring). GS-LiDAR's own loader
# (~/GS-LiDAR/scene/kitti360_loader.py:204-207) applies this exact fixed
# local-frame remap before building each Camera -- ported here as a pure
# rotation applied to lidar2world's rotation block only (translation is
# unaffected; derivation: that loader's `w2l = M4 @ inv(lidar2globals)`
# inverts to `world_T_optical_lidar = lidar2globals @ M4.T`, and M4's
# translation is zero).
_RAW_LIDAR_TO_OPTICAL = np.array([
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
])


def lidar2world_to_pose(lidar2world: list) -> Pose:
    m = np.asarray(lidar2world, dtype=np.float64)
    r_optical = m[:3, :3] @ _RAW_LIDAR_TO_OPTICAL
    return Pose(position=m[:3, 3], orientation=rotmat_to_quat(r_optical))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-accumulated-cloud", type=str, default=None,
                         help="Also render all 456 frames, unproject, and write a .ply for "
                              "visual comparison against qa/accumulated_cloud.pcd (slow).")
    args = parser.parse_args()

    device = "cuda"
    model = load_lidar_gaussian_model(CKPT_DIR / "ckpt" / "chkpnt30000.pth", device=device)
    raydrop_prior = load_raydrop_prior(CKPT_DIR / "ckpt" / "lidar_raydrop_prior_chkpnt30000.pth", device=device)
    refine_unet = load_refine_unet(CKPT_DIR / "ckpt" / "refine.pth", device=device)
    gs_transform = load_combined_transform()

    frames_data = json.loads((SOURCE_DIR / "0000" / "transforms_0000_all.json").read_text())
    frames_by_idx = {f["idx"]: f for f in frames_data["frames"]}
    w_native, h_native = frames_data["w_lidar"], frames_data["h_lidar"]
    native_profile = LidarProfile(
        vfov=(-15.0, 15.0), hfov=(-90.0, 90.0), hw=(h_native, w_native // 2),
        frame_id="lidar_optical", update_rate=10.0,
        # Must equal gs_transform.scale, not the LidarProfile default -- see
        # schema.py's docstring: this is a CUDA-kernel near/far-clip
        # parameter tied to the PCA transform's scale, not an independent
        # knob. Using the default (1.0) here silently near-clipped the
        # entire model at every frame.
        scale_factor=gs_transform.scale,
    )

    print(f"Native render resolution (per-half): {native_profile.hw}, "
          f"stitched: ({h_native}, {w_native})")

    # The checkpoint's raydrop envmap is saved at training resolution
    # ([32, 512] per half, per setting.txt) -- resize it to match the native
    # QA resolution we're rendering at here (see RayDropPrior.resize's
    # docstring). Everything below this point (including the
    # --dump-accumulated-cloud pass, which also renders at native_profile)
    # uses this resized prior.
    raydrop_prior = raydrop_prior.resize(h_native, w_native // 2)

    xyz_min = model.xyz.detach().min(dim=0).values.cpu().numpy()
    xyz_max = model.xyz.detach().max(dim=0).values.cpu().numpy()
    print(f"[debug] model xyz bbox (gs-training space): min={xyz_min} max={xyz_max}")

    for idx in QA_FRAMES:
        pose_world = lidar2world_to_pose(frames_by_idx[idx]["lidar2world"])
        pose_gs = gs_transform.apply(pose_world)
        cam_fwd, cam_bwd = build_lidar_cameras(pose_gs, native_profile, device=device)
        print(f"[debug] frame {idx:06d}: world pos={pose_world.position} "
              f"gs pos={pose_gs.position} (scale={gs_transform.scale})")

        with torch.no_grad():
            pano = render_lidar_panorama(model, raydrop_prior, cam_fwd, cam_bwd,
                                          scale_factor=native_profile.scale_factor)
            raydrop = refine_raydrop(refine_unet, pano.raydrop, pano.intensity, pano.depth)

        print(f"[debug] frame {idx:06d}: num_rendered_splats={pano.num_rendered} "
              f"pre-refine raydrop min/mean/max="
              f"{pano.raydrop.min().item():.3f}/{pano.raydrop.mean().item():.3f}/{pano.raydrop.max().item():.3f} "
              f"post-refine raydrop min/mean/max="
              f"{raydrop.min().item():.3f}/{raydrop.mean().item():.3f}/{raydrop.max().item():.3f} "
              f"depth(gs-space) min/mean/max="
              f"{pano.depth.min().item():.3f}/{pano.depth.mean().item():.3f}/{pano.depth.max().item():.3f}")

        depth_m = (pano.depth / gs_transform.scale).squeeze(0).cpu().numpy()
        intensity = pano.intensity.squeeze(0).cpu().numpy()
        pred_valid = (raydrop.squeeze(0) <= 0.5).cpu().numpy()

        qa = np.load(SOURCE_DIR / "qa" / "panorama" / f"{idx:06d}.npz")
        gt_range, gt_intensity, gt_valid = qa["range"], qa["intensity"], qa["valid_mask"]

        both_valid = pred_valid & gt_valid
        n_both = int(both_valid.sum())
        n_gt = int(gt_valid.sum())
        range_mae = float(np.abs(depth_m[both_valid] - gt_range[both_valid]).mean()) if n_both else float("nan")
        intensity_mae = float(np.abs(intensity[both_valid] - gt_intensity[both_valid]).mean()) if n_both else float("nan")
        iou = n_both / max(1, int((pred_valid | gt_valid).sum()))

        print(f"frame {idx:06d}: gt_valid={n_gt} pred&gt_valid={n_both} "
              f"valid_mask_iou={iou:.3f} range_MAE={range_mae:.4f}m intensity_MAE={intensity_mae:.4f}")

    if args.dump_accumulated_cloud:
        print(f"Accumulating all {len(frames_by_idx)} frames -> {args.dump_accumulated_cloud} ...")
        all_points = []
        for idx in sorted(frames_by_idx):
            pose_world = lidar2world_to_pose(frames_by_idx[idx]["lidar2world"])
            pose_gs = gs_transform.apply(pose_world)
            cam_fwd, cam_bwd = build_lidar_cameras(pose_gs, native_profile, device=device)
            with torch.no_grad():
                pano = render_lidar_panorama(model, raydrop_prior, cam_fwd, cam_bwd)
                raydrop = refine_raydrop(refine_unet, pano.raydrop, pano.intensity, pano.depth)
                depth = pano.depth * (raydrop <= 0.5).float()
                xyz_local, _ = pano_to_points(depth, pano.intensity, vfov=native_profile.vfov)
            # xyz_local is in the LiDAR's own *optical* local frame (see
            # pointcloud.py) -- rotate/translate by this frame's real
            # optical-frame world pose (BEFORE gs_transform, i.e.
            # pose_world, which already carries the raw-to-optical remap
            # from lidar2world_to_pose) to accumulate into one common
            # world-frame map comparable to qa/accumulated_cloud.pcd. Using
            # the raw (un-remapped) lidar2world rotation here would put
            # points back in the wrong frame, same bug as
            # lidar2world_to_pose had.
            r_c2w = quat_to_rotmat(pose_world.orientation)
            t_c2w = pose_world.position
            xyz_np = (xyz_local / gs_transform.scale).cpu().numpy()
            xyz_world = xyz_np @ r_c2w.T + t_c2w
            all_points.append(xyz_world)
        cloud = np.concatenate(all_points, axis=0).astype(np.float32)
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)
        o3d.io.write_point_cloud(args.dump_accumulated_cloud, pcd)
        print(f"Wrote {cloud.shape[0]:,} points. Compare visually against "
              f"{SOURCE_DIR / 'qa' / 'accumulated_cloud.pcd'} (Open3D/CloudCompare).")


if __name__ == "__main__":
    main()
