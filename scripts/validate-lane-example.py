#!/usr/bin/env python3
"""Check real supplied-map pose/projection conventions; never invent 2D truth.

Run import_4d_clip.py first. This reads source and platform data without changing
either. Its pixel residual measures implementation equivalence, NOT accuracy
against independently annotated image lanes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from import_4d_clip import CAMERAS, quaternion_transform
from omnilabel.lane_calibration import OptimizeRequest, resolve_geometry
from omnilabel.lane_optimizer import resample
from omnilabel.projection_calibration import _image_path, project_camera_points
from omnilabel.scene import get_scene
from omnilabel.storage import Store


def validate(source: Path, data: Path, clip_id: str, output: Path):
    if output.exists(): raise FileExistsError("Choose a new output directory; existing reports are preserved.")
    store = Store(data); clip = store.source_clip(clip_id); scene = get_scene(store, clip)
    pose_paths = list((source / "global_pose").glob("*.json"))
    if len(pose_paths) != 1: raise ValueError("Expected one source global_pose JSON.")
    source_poses = json.loads(pose_paths[0].read_text())["frame_items"]
    if len(source_poses) != len(clip.frames): raise ValueError("Source/imported frame count differs.")
    lanes = [v for v in scene.vectors if not v.closed]
    points = np.concatenate([resample(v.points, 160) for v in lanes])
    request = OptimizeRequest(pairing_revision=0)
    max_matrix, max_pixel, checked, visible = 0., 0., 0, 0
    output.mkdir(parents=True)
    overlays = output / "projections"; overlays.mkdir()
    for camera_index, (camera_id, _, _) in enumerate(CAMERAS):
        camera = next(c for c in clip.cameras if c.id == camera_id)
        poses, initial, mount = resolve_geometry(store, clip, camera, clip.frames, request)
        for index, (item, frame) in enumerate(zip(source_poses, clip.frames)):
            config = json.loads((source / "camera_config" / f'{item["target_frame_stem"]}.json').read_text())[camera_index]
            k = config["camera_internal"]; radial, tangential = config["distortionK"], config["distortionP"]
            np.testing.assert_allclose(camera.intrinsics, [[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]])
            np.testing.assert_allclose(camera.distortion, [radial[0], radial[1], tangential[0], tangential[1], radial[2]])
            expected = np.asarray(config["camera_external"]).reshape(4, 4)
            source_ego = quaternion_transform(item["ego_pose_local"]["rotation"], item["ego_pose_local"]["translation"])
            source_mount = quaternion_transform(item["lidar_calib"]["rotation"], item["lidar_calib"]["translation"])
            np.testing.assert_allclose(mount, source_mount, atol=1e-9)
            np.testing.assert_allclose(poses[frame.id], source_ego @ source_mount, atol=1e-8)
            derived = initial @ np.linalg.inv(poses[frame.id])
            max_matrix = max(max_matrix, float(np.max(np.abs(derived-expected))))
            a, z = project_camera_points(points, expected, np.asarray(camera.intrinsics), np.asarray(camera.distortion))
            local = (points - poses[frame.id][:3, 3]) @ poses[frame.id][:3, :3]
            b, _ = project_camera_points(local, initial, np.asarray(camera.intrinsics), np.asarray(camera.distortion))
            valid = (z > .05) & np.isfinite(a).all(1) & (a[:, 0] >= 0) & (a[:, 0] < camera.width) & (a[:, 1] >= 0) & (a[:, 1] < camera.height)
            if valid.any(): max_pixel = max(max_pixel, float(np.linalg.norm(a[valid]-b[valid], axis=1).max()))
            visible += int(valid.sum()); checked += 1
            with Image.open(_image_path(store, frame, camera_id)) as raw:
                if raw.size != (camera.width, camera.height): raise ValueError("Image resolution differs from intrinsics.")
                if index not in {0, len(clip.frames)//2, len(clip.frames)-1}: continue
                image = raw.convert("RGB")
            draw = ImageDraw.Draw(image)
            for lane in lanes:
                xyz = resample(lane.points, 160)
                local = (xyz - poses[frame.id][:3, 3]) @ poses[frame.id][:3, :3]
                uv, depth = project_camera_points(local, initial, np.asarray(camera.intrinsics), np.asarray(camera.distortion))
                good = (depth > .05) & np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < camera.width) & (uv[:, 1] >= 0) & (uv[:, 1] < camera.height)
                for j in range(len(uv)-1):
                    if good[j] and good[j+1]: draw.line([tuple(uv[j]), tuple(uv[j+1])], fill="#fb923c", width=3)
                indices = np.flatnonzero(good)
                if len(indices): draw.text(tuple(uv[indices[len(indices)//2]]), lane.id, fill="#38bdf8", stroke_width=1)
            draw.rectangle((0, 0, image.width, 28), fill="black")
            draw.text((8, 8), f"{frame.id} / {camera_id}: raw initial projection; NO independent 2D truth", fill="white")
            image.save(overlays / f"{frame.id}-{camera_id}.jpg", quality=90)
    if max_matrix > 2e-5 or max_pixel > .01: raise ValueError("Source and physical pose/extrinsic projections disagree.")
    report = {"clip_id": clip_id, "frames": len(clip.frames), "cameras": len(clip.cameras),
        "checked_camera_frames": checked, "open_world_lanes": len(lanes), "visible_projection_samples": visible,
        "max_source_vs_optimizer_matrix_difference": max_matrix,
        "max_source_vs_optimizer_pixel_difference": max_pixel,
        "interpretation": "Implementation equivalence only; this is not calibration accuracy.",
        "independent_2d_lane_annotations": "Not supplied in this example. Add ordered manual pairs in the platform.",
        "source_projection_corrections": "Not used as image ground truth or as a shared initial extrinsic.",
        "time_alignment": "Source frame association checked; independent camera exposure timestamps are absent.",
        "extrinsic_updated": False}
    (output / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "example/4D_clip_DATA_01_20260808160513v12")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--clip-id", default="4d-clip-data-01-version2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.source, args.data_dir, args.clip_id, args.output), ensure_ascii=False, indent=2))
