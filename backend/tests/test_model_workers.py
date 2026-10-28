"""Real output conversion/protocol tests; no GPU inference is claimed here."""
import base64
import io
from contextlib import contextmanager, nullcontext
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

from model_workers.app import create_app
from model_workers.adapters import (SAM2Adapter, LATRAdapter, OpenOccupancyAdapter,
                                    decode_latr_predictions, latr_coordinate_matrices,
                                    normalize_detection, occupancy_from_logits, resolve_sam2_precision)
from model_workers.common import (InvalidInput, image_from_payload,
                                  mask_to_polygons, ModelUnavailable, points_array, validate_options)


def test_sam2_turing_uses_fp16_even_if_torch_can_emulate_bf16():
    torch = SimpleNamespace(cuda=SimpleNamespace(device=lambda value: nullcontext(),
        get_device_capability=lambda value: (7, 5), is_bf16_supported=lambda: True))
    assert resolve_sam2_precision(torch, "cuda:0") == "float16"
    assert resolve_sam2_precision(torch, "cuda:0", "float32") == "float32"
    with pytest.raises(ModelUnavailable, match="Turing"):
        resolve_sam2_precision(torch, "cuda:0", "bfloat16")


def test_sam2_native_bf16_requires_capability_and_actual_support_on_selected_device():
    selected = []
    torch = SimpleNamespace(cuda=SimpleNamespace(device=lambda value: nullcontext(),
        get_device_capability=lambda value: selected.append(value) or (8, 6), is_bf16_supported=lambda: True))
    assert resolve_sam2_precision(torch, "cuda:3") == "bfloat16"
    assert selected == ["cuda:3"]
    torch.cuda.is_bf16_supported = lambda: False
    assert resolve_sam2_precision(torch, "cuda:3") == "float16"


def test_sam2_cpu_keeps_float32_and_rejects_invalid_precision():
    # CPU preflight must not probe CUDA devices or require a GPU runtime.
    torch = SimpleNamespace()
    assert resolve_sam2_precision(torch, "cpu") == "float32"
    with pytest.raises(ModelUnavailable, match="CUDA"):
        resolve_sam2_precision(torch, "cpu", "float16")
    with pytest.raises(ModelUnavailable, match="SAM2_PRECISION"):
        resolve_sam2_precision(torch, "cpu", "invalid")


def payload():
    image = io.BytesIO()
    Image.new("RGB", (20, 10), (20, 30, 40)).save(image, "PNG")
    return {"schema_version": "1.0", "model_id": "sam2", "clip": {"id": "test", "cameras": [
        {"id": "front", "width": 20, "height": 10}]}, "frame": {"id": "f0", "points": [[1, 2, 3, .5]],
        "images": {"front": "data:image/png;base64," + base64.b64encode(image.getvalue()).decode()}},
        "annotations": {"boxes": [], "vectors": [], "image_polygons": [], "point_labels": {}, "revision": 2}, "options": {}}


@pytest.mark.parametrize("prompted", [True, False])
def test_sam2_adapter_encodes_and_predicts_under_fp16_and_forwards_automatic_batch(monkeypatch, prompted):
    """Exercise official API call wiring with explicit fakes, without GPU claims."""
    state = {"inference": False, "autocast": False, "called": False}

    @contextmanager
    def inference_mode():
        state["inference"] = True
        yield
        state["inference"] = False

    @contextmanager
    def autocast(*, device_type, dtype):
        assert device_type == "cuda" and dtype == "fake-float16"
        assert state["inference"]
        state["autocast"] = True
        yield
        state["autocast"] = False

    torch = SimpleNamespace(inference_mode=inference_mode, autocast=autocast, float16="fake-float16")
    monkeypatch.setitem(sys.modules, "torch", torch)
    mask = np.zeros((10, 20), dtype=bool)
    mask[1:9, 2:18] = True

    class PromptPredictor:
        def __init__(self, model):
            assert model is adapter.model

        def set_image(self, image):
            assert image.shape == (10, 20, 3) and state["autocast"] and state["inference"]

        def predict(self, **options):
            assert state["autocast"] and state["inference"]
            np.testing.assert_array_equal(options["point_coords"], [[10, 5]])
            np.testing.assert_array_equal(options["point_labels"], [1])
            assert options["multimask_output"] is False
            state["called"] = True
            return np.asarray([mask]), np.array([.95]), None

    class AutomaticGenerator:
        def __init__(self, model, **options):
            assert model is adapter.model
            assert options["points_per_batch"] == 4

        def generate(self, image):
            assert image.shape == (10, 20, 3) and state["autocast"] and state["inference"]
            state["called"] = True
            return [{"segmentation": mask, "predicted_iou": .95}]

    monkeypatch.setitem(sys.modules, "sam2.sam2_image_predictor", SimpleNamespace(SAM2ImagePredictor=PromptPredictor))
    monkeypatch.setitem(sys.modules, "sam2.automatic_mask_generator", SimpleNamespace(SAM2AutomaticMaskGenerator=AutomaticGenerator))
    adapter = SAM2Adapter()
    adapter.model, adapter.precision, adapter.points_per_batch = object(), "float16", 4
    request = payload()
    request["options"] = {"camera_id": "front", "label": "road"}
    if prompted:
        request["options"].update(points=[[10, 5]], point_labels=[1])
    result = adapter.predict(request)
    assert state == {"inference": False, "autocast": False, "called": True}
    assert result["inference_precision"] == "float16"
    assert len(result["annotations"]["image_polygons"]) == 1
    assert result["annotations"]["image_polygons"][0]["label"] == "road"


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


def test_worker_reports_actionable_precision_prerequisite_without_claiming_ready(tmp_path, monkeypatch):
    checkpoint = tmp_path / "trusted-test-fixture.pt"
    checkpoint.write_bytes(b"fixture, not a real checkpoint")
    monkeypatch.setenv("SAM2_CHECKPOINT", str(checkpoint))
    monkeypatch.setattr("model_workers.adapters.module_present", lambda name: True)
    adapter = SAM2Adapter()

    def fail_load():
        raise ModelUnavailable("SAM2 bfloat16 requires native BF16; use auto or float16 on RTX 2080/Turing")

    monkeypatch.setattr(adapter, "load", fail_load)
    with TestClient(create_app("sam2", adapter)) as client:
        record = client.get("/models").json()["models"][0]
        assert not record["available"] and record["status"] == "error"
        assert "RTX 2080/Turing" in record["reason"]


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


def test_latr_calibration_converts_right_forward_up_to_lidar_coordinates():
    identity = np.eye(4).tolist()
    # Ego [forward,left,up] -> optical camera [right,down,forward].
    camera_from_ego = [[0, -1, 0, 0], [0, 0, -1, 1.5],
                       [1, 0, 0, 0], [0, 0, 0, 1]]
    request = {"clip": {"cameras": [{"id": "front", "intrinsics": [[1000, 0, 960], [0, 1000, 640], [0, 0, 1]],
                                        "camera_from_ego": camera_from_ego}]},
               "frame": {"lidar_to_ego": identity}}
    _, intrinsic, latr_to_lidar, camera_from_latr = latr_coordinate_matrices(request, "front")
    np.testing.assert_array_equal(intrinsic, [[1000, 0, 960], [0, 1000, 640], [0, 0, 1]])
    np.testing.assert_allclose(latr_to_lidar @ [2, 10, .5, 1], [10, -2, .5, 1])
    # LATR forward points have positive optical depth for the chosen camera.
    assert camera_from_latr[2, 1] == 1


def test_latr_decoder_adds_only_visible_confident_3d_lane_vectors():
    anchors = np.array([1, 2, 3], dtype=float)
    output = {
        "all_line_preds": np.array([[[
            [1, 2, 3, .1, .2, .3, 1, 1, -1],
            [-1, -2, -3, 0, 0, 0, 1, 1, 1],
        ]]], dtype=float),
        "all_cls_scores": np.array([[[[0, 4], [4, 0]]]], dtype=float),
    }
    annotations = {"boxes": [], "vectors": [{"id": "manual", "label": "lane", "points": [[0, 0, 0], [1, 0, 0]], "closed": False}],
                   "image_polygons": [], "point_labels": {}, "revision": 3}
    latr_to_lidar = np.array([[0, 1, 0, 0], [-1, 0, 0, 0],
                              [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    confidence = decode_latr_predictions(output, anchors, .8, 0, "lane_solid", annotations, latr_to_lidar)
    assert len(annotations["vectors"]) == 2
    predicted = annotations["vectors"][1]
    assert predicted["label"] == "lane_solid" and not predicted["closed"]
    np.testing.assert_allclose(predicted["points"], [[1, -1, .1], [2, -2, .2]])
    assert list(confidence) == [predicted["id"]]
    assert confidence[predicted["id"]] > .98


def test_latr_rejects_side_or_rear_camera_calibration():
    identity = np.eye(4).tolist()
    request = {"clip": {"cameras": [{"id": "rear", "intrinsics": np.eye(3).tolist(),
                                        "camera_from_ego": [[0, 1, 0, 0], [0, 0, -1, 1],
                                                            [-1, 0, 0, 0], [0, 0, 0, 1]]}]},
               "frame": {"lidar_to_ego": identity}}
    with pytest.raises(InvalidInput, match="forward-facing"):
        latr_coordinate_matrices(request, "rear")


def test_latr_weight_variants_resolve_matching_checkpoint_and_config(tmp_path, monkeypatch):
    repo = tmp_path / "latr"
    for relative in ("models/latr.py", "ckpt/once.pth", "ckpt/openlane.pth",
                     "config/release_iccv/once.py", "config/release_iccv/latr_1000_baseline.py"):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    monkeypatch.setenv("LATR_REPO", str(repo))
    adapter = LATRAdapter()
    once = adapter.variant_paths("once")
    openlane = adapter.variant_paths("openlane")
    assert once == (repo / "ckpt/once.pth", repo / "config/release_iccv/once.py")
    assert openlane == (repo / "ckpt/openlane.pth", repo / "config/release_iccv/latr_1000_baseline.py")
    with pytest.raises(InvalidInput, match="once or openlane"):
        adapter.variant_paths("unknown")
