"""CPU optimizer checks against known geometric truth, without mocked solvers."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import pytest
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation

from omnilabel.automatic_projection import automatic_refine
from omnilabel.models import Camera, Clip, Frame
from omnilabel.projection_calibration import project_camera_points


@pytest.fixture
def synthetic_scan(tmp_path):
    """One textured, nonplanar static surface, with exactly known pixel matches."""
    width, height, focal = 640, 480, 450
    texture = gaussian_filter(np.random.default_rng(62071).normal(size=(height, width)), 3)
    texture = np.rint(255 * (texture - texture.min()) / np.ptp(texture)).astype(np.uint8)
    image_path = tmp_path / "textured.png"
    Image.fromarray(texture).save(image_path)
    u, v = np.meshgrid(np.arange(25, width - 25, 3), np.arange(25, height - 25, 3))
    u, v = u.ravel(), v.ravel()
    z = 10 + 12 * u / width + 2 * np.sin(v / 80)
    points = np.column_stack(((u - width / 2) * z / focal,
                              (v - height / 2) * z / focal, z, texture[v, u]))
    camera = Camera(id="front", name="Front", width=width, height=height,
                    intrinsics=[[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]],
                    camera_from_ego=np.eye(4).tolist())
    initial = np.eye(4)
    initial[:3, :3] = Rotation.from_rotvec([.002, -.004, .001]).as_matrix()
    initial[:3, 3] = [.12, -.05, .03]
    return camera, image_path, points, initial, np.column_stack((u, v))


def test_real_cpu_refinement_reduces_known_projection_error(synthetic_scan):
    camera, image_path, points, initial, true_pixels = synthetic_scan
    result = automatic_refine(points, image_path, camera, initial)
    assert result["usable"], result["rejection_reasons"]
    k = np.asarray(camera.intrinsics)
    before, _ = project_camera_points(points[:, :3], initial, k, np.array([]))
    after, _ = project_camera_points(points[:, :3], np.asarray(result["matrix"]), k, np.array([]))
    before_error = np.median(np.linalg.norm(before - true_pixels, axis=1))
    after_error = np.median(np.linalg.norm(after - true_pixels, axis=1))
    assert before_error > 1.0
    assert after_error < before_error * .65, (before_error, after_error)
    assert result["metrics"]["holdout_improvement"] > .02


def test_real_batch_cpu_result_survives_save_reload_and_export(synthetic_scan, tmp_path):
    from omnilabel.automatic_calibration import AutomaticCalibrationManager, AutomaticCalibrationRequest
    from omnilabel.storage import Store
    camera, image_path, points, initial, _ = synthetic_scan
    store = Store(tmp_path / "store")
    asset = store.root / "assets" / "frame.png"
    asset.write_bytes(image_path.read_bytes())
    scan = store.root / "pointclouds" / "frame.npy"
    np.save(scan, points)
    identity = np.eye(4).tolist()
    frame = Frame(id="f0", timestamp=0, ego_to_world=identity, lidar_to_ego=identity,
                  pointcloud_file="frame.npy", point_count=len(points), images={camera.id: "/api/assets/frame.png"})
    camera = camera.model_copy(update={"camera_from_ego": initial.tolist()})
    store.add_clip(Clip(id="real", name="Real CPU integration", fps=10, cameras=[camera], frames=[frame]))
    source = store.root / "clips" / "real.json"
    before = source.read_bytes()
    with ThreadPoolExecutor(max_workers=1) as executor:
        manager = AutomaticCalibrationManager(store, executor)
        job = manager.start("real", AutomaticCalibrationRequest(engine="direct_nid"))
        # A single executor guarantees this marker runs after the batch.
        executor.submit(lambda: None).result(timeout=20)
        candidate = manager.get("real", job["id"])
        assert candidate["accepted"] == 1, candidate
        assert not store.clip("real").frames[0].projection_camera_from_lidar
        applied = manager.apply("real", job["id"])
        assert applied["saved"] == 1, applied
        saved = applied["items"][0]["matrix"]
        assert Store(store.root).clip("real").frames[0].projection_camera_from_lidar[camera.id] == saved
        assert manager.export("real")["frames"][0]["cameras"][camera.id]["camera_from_lidar"] == saved
    assert source.read_bytes() == before


@pytest.mark.parametrize("missing", ["image_texture", "lidar_intensity"])
def test_real_cpu_refinement_rejects_missing_information(synthetic_scan, missing):
    camera, image_path, points, initial, _ = synthetic_scan
    if missing == "image_texture":
        Image.fromarray(np.full((camera.height, camera.width), 128, dtype=np.uint8)).save(image_path)
        expected_reason = "图像纹理不足"
    else:
        points[:, 3] = 0
        expected_reason = "intensity"
    result = automatic_refine(points, image_path, camera, initial)
    assert not result["usable"]
    assert any(expected_reason in reason for reason in result["rejection_reasons"])
    assert np.allclose(result["matrix"], initial)
