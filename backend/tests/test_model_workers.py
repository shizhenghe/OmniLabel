"""Real output conversion/protocol tests; no GPU inference is claimed here."""
import base64
import io
from types import SimpleNamespace

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

from model_workers.app import create_app
from model_workers.adapters import (SAM2Adapter, OpenOccupancyAdapter,
                                    normalize_detection, occupancy_from_logits)
from model_workers.common import (InvalidInput, image_from_payload,
                                  mask_to_polygons, points_array, validate_options)


def payload():
    image = io.BytesIO()
    Image.new("RGB", (20, 10), (20, 30, 40)).save(image, "PNG")
    return {"schema_version": "1.0", "model_id": "sam2", "clip": {"id": "test", "cameras": [
        {"id": "front", "width": 20, "height": 10}]}, "frame": {"id": "f0", "points": [[1, 2, 3, .5]],
        "images": {"front": "data:image/png;base64," + base64.b64encode(image.getvalue()).decode()}},
        "annotations": {"boxes": [], "vectors": [], "image_polygons": [], "point_labels": {}, "revision": 2}, "options": {}}


def test_embedded_image_validation_and_original_dimensions():
    request = payload()
    image, camera = image_from_payload(request, "front")
    assert image.size == (20, 10) and camera["id"] == "front"
    request["clip"]["cameras"][0]["width"] = 21
    with pytest.raises(InvalidInput, match="matching"):
        image_from_payload(request, "front")
    request["frame"]["images"]["front"] = "https://private-service/secret"
    with pytest.raises(InvalidInput): image_from_payload(request, "front")


def test_mask_exteriors_keep_disconnected_components_and_report_holes():
    mask = np.zeros((30, 40), dtype=bool)
    mask[2:14, 3:15] = True
    mask[5:9, 6:10] = False
    mask[20:28, 23:35] = True
    polygons, metadata = mask_to_polygons(mask)
    assert len(polygons) == 2
    assert metadata["hole_count"] == 1
    assert metadata["approximate_polygon_conversion"]
    assert sorted([len(p) for p in polygons]) == [4, 4]
    assert all(0 <= x <= 40 and 0 <= y <= 30 for p in polygons for x, y in p)


def test_masks_touching_at_one_corner_remain_separate():
    mask = np.zeros((12, 12), dtype=bool)
    mask[1:5, 1:5] = True
    mask[5:9, 5:9] = True
    polygons, metadata = mask_to_polygons(mask, minimum_area=1)
    assert len(polygons) == 2 and metadata["hole_count"] == 0


def test_masks_resize_back_to_original_pixel_coordinates():
    mask = np.ones((1200, 1800), dtype=bool)
    polygons, metadata = mask_to_polygons(mask)
    assert metadata["working_scale"] < 1
    assert max(x for p in polygons for x, y in p) == 1800
    assert max(y for p in polygons for x, y in p) == 1200


def test_detector_uses_gravity_center_not_bottom_center_and_preserves_human_labels():
    class LiDARInstance3DBoxes:
        gravity_center = np.array([[1, 2, 3], [4, 5, 6]])
        dims = np.array([[4, 2, 2], [1, 1, 1]])
        yaw = np.array([.2, -.4])
        tensor = np.array([[1, 2, 2, 4, 2, 2, .2], [4, 5, 5.5, 1, 1, 1, -.4]])
    sample = SimpleNamespace(pred_instances_3d=SimpleNamespace(bboxes_3d=LiDARInstance3DBoxes(),
        scores_3d=np.array([.95, .1]), labels_3d=np.array([1, 0])))
    annotations = payload()["annotations"]
    annotations["point_labels"] = {"0": "manual"}
    confidence = normalize_detection((sample, {}), ["person", "car"], .3, annotations)
    assert len(annotations["boxes"]) == 1
    assert annotations["boxes"][0]["center"] == [1, 2, 3]
    assert annotations["boxes"][0]["label"] == "car"
    assert list(confidence.values()) == [.95]
    assert annotations["point_labels"] == {"0": "manual"}


def test_detector_rejects_camera_coordinate_boxes():
    class CameraInstance3DBoxes:
        gravity_center = dims = np.array([[1, 1, 1]])
        yaw = np.array([0])
    sample = SimpleNamespace(pred_instances_3d=SimpleNamespace(bboxes_3d=CameraInstance3DBoxes(),
        scores_3d=np.array([1]), labels_3d=np.array([0])))
    with pytest.raises(InvalidInput, match="LiDAR-coordinate"):
        normalize_detection(sample, ["car"], .3, payload()["annotations"])


def test_dense_model_occupancy_axis_order_origin_confidence_and_free_semantics():
    logits = np.full((2, 2, 2, 1), -5, dtype=float)
    logits[0, :, :, :] = 5  # confidently free
    logits[1, 1, 0, 0] = 20
    result = occupancy_from_logits(logits, [-2, -4, -1, 2, 4, 1], .9, ["free", "car"])
    cell = next(v for v in result["voxels"] if v["label"] == "car")
    assert cell["index"] == [1, 0, 0] and cell["center"] == [1, -2, 0]
    assert result["voxel_dimensions"] == [2, 4, 2]
    assert result["prediction_only"] and result["occupied_count"] == 1 and result["free_count"] == 3
    assert "unknown" in result["unknown_policy"]


def test_sparse_fine_occupancy_and_explicit_truncation_prioritize_occupied():
    logits = np.array([[10, -10], [-10, 10], [10, -10]])
    coordinates = np.array([[0, 3, 1], [0, 2, 1], [0, 1, 0]])
    result = occupancy_from_logits(logits, [0, 0, 0, 4, 4, 2], .9, ["free", "car"], coordinates, [4, 4, 2], maximum=1)
    assert result["truncated"] and result["candidate_count"] == 3
    assert result["voxels"][0]["index"] == [3, 2, 1]
    assert result["voxels"][0]["label"] == "car"


def test_worker_with_missing_dependencies_reports_unavailable_and_never_returns_labels(monkeypatch):
    monkeypatch.setattr("model_workers.adapters.module_present", lambda name: False)
    with TestClient(create_app("sam2", eager_load=False)) as client:
        record = client.get("/models").json()["models"][0]
        assert not record["available"] and record["status"] == "missing_dependencies"
        assert client.post("/predict", json=payload()).status_code == 503


def test_worker_protocol_dispatch_auth_and_response(monkeypatch):
    class ProtocolTestRuntime:
        def status(self): return {"id": "sam2", "available": True, "status": "ready"}
        def predict(self, data): return {"annotations": data["annotations"], "model_version": "protocol-test-only"}
    monkeypatch.setenv("MODEL_WORKER_TOKEN", "test-fixture-token")
    with TestClient(create_app("sam2", ProtocolTestRuntime(), eager_load=False)) as client:
        assert client.get("/models").status_code == 401
        headers = {"Authorization": "Bearer test-fixture-token"}
        assert client.get("/models", headers=headers).json()["models"][0]["available"]
        response = client.post("/predict", json=payload(), headers=headers)
        assert response.status_code == 200 and response.json()["annotations"]["revision"] == 2
        wrong_model = payload(); wrong_model["model_id"] = "openoccupancy"
        assert client.post("/predict", json=wrong_model, headers=headers).status_code == 422


@pytest.mark.parametrize("threshold", [True, -1, 2, float("nan")])
def test_invalid_model_thresholds_rejected(threshold):
    with pytest.raises(InvalidInput): validate_options({"score_threshold": threshold})


def test_points_are_original_xyzi_with_no_invented_measurements():
    points = points_array({"points": [[1, 2, 3]]})
    assert points.tolist() == [[1, 2, 3, 0]]
    with pytest.raises(InvalidInput): points_array({"points": [[float("nan"), 2, 3]]})
