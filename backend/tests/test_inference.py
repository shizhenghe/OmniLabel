import base64
import io
import json

import httpx
import numpy as np
import pytest
from PIL import Image

import omnilabel.inference as inference

REAL_HTTPX_CLIENT = httpx.Client


@pytest.fixture
def inputs(monkeypatch):
    monkeypatch.setenv("OMNILABEL_INFERENCE_URL", "https://worker.example.test/v1/predict")
    monkeypatch.delenv("OMNILABEL_INFERENCE_TOKEN", raising=False)
    camera = {"id": "front", "name": "Front", "width": 100, "height": 80,
              "intrinsics": [[50, 0, 50], [0, 50, 40], [0, 0, 1]], "camera_from_ego": np.eye(4).tolist()}
    frame = {"id": "f0", "timestamp": 0, "points": [[1, 2, 3, 0.5]],
             "ego_to_world": np.eye(4).tolist(), "lidar_to_ego": np.eye(4).tolist(), "images": {}}
    annotations = {"boxes": [], "vectors": [], "image_polygons": [], "point_labels": {}, "revision": 7}
    return {"id": "clip", "cameras": [camera]}, frame, annotations


def transport(monkeypatch, handler):
    def factory(**kwargs):
        assert kwargs["follow_redirects"] is False
        assert kwargs["verify"] is True
        return REAL_HTTPX_CLIENT(transport=httpx.MockTransport(handler), **kwargs)
    monkeypatch.setattr(inference.httpx, "Client", factory)


def test_actual_worker_request_validated_proposal_revision_preserved_and_no_mutation(inputs, monkeypatch):
    monkeypatch.setenv("OMNILABEL_INFERENCE_TOKEN", "test-secret-token")
    def worker(request):
        assert str(request.url) == "https://worker.example.test/v1/predict"
        assert request.headers["authorization"] == "Bearer test-secret-token"
        payload = json.loads(request.content)
        assert payload["model_id"] == "sam2"
        assert payload["prompt"] == "road"
        assert payload["frame"]["points"] == [[1, 2, 3, 0.5]]
        assert payload["clip"]["cameras"][0]["id"] == "front"
        return httpx.Response(200, json={"annotations": {"point_labels": {"0": "road"}, "revision": 99}, "model_version": "checkpoint-sha256:abc"})
    transport(monkeypatch, worker)
    result = inference.infer_proposal(*inputs, "sam2", "road")
    assert result["proposal"]["revision"] == 7
    assert result["proposal"]["point_labels"] == {"0": "road"}
    assert result["review_required"] is True
    assert result["source"] == "external-worker"
    assert inputs[2]["point_labels"] == {}


def test_local_camera_pixels_are_embedded_and_external_urls_are_not_fetched(inputs, monkeypatch, tmp_path):
    assets = tmp_path / "assets"
    (assets / "clip").mkdir(parents=True)
    image = io.BytesIO()
    Image.new("RGB", (100, 80), color=(30, 40, 50)).save(image, format="PNG")
    pixels = image.getvalue()
    (assets / "clip" / "front.png").write_bytes(pixels)
    inputs[1]["images"] = {"front": "/api/assets/clip/front.png"}
    def worker(request):
        payload = json.loads(request.content)
        encoded = payload["frame"]["images"]["front"]
        assert encoded == "data:image/png;base64," + base64.b64encode(pixels).decode()
        return httpx.Response(200, json={"annotations": {}, "model_version": "v1"})
    transport(monkeypatch, worker)
    inference.infer_proposal(*inputs, "sam2", "road", assets_root=assets)
    inputs[1]["images"]["front"] = "https://arbitrary.invalid/picture.png"
    with pytest.raises(inference.InferenceError, match="imported local assets"):
        inference.infer_proposal(*inputs, "sam2", "road", assets_root=assets)
    inputs[1]["images"]["front"] = "/api/assets/../../secret.png"
    with pytest.raises(inference.InferenceError, match="unavailable"):
        inference.infer_proposal(*inputs, "sam2", "road", assets_root=assets)


@pytest.mark.parametrize("proposed", [
    {"point_labels": {"1": "road"}},  # out-of-range point
    {"point_labels": {"0": ""}},
    {"image_polygons": [{"id": "p", "camera_id": "unknown", "label": "road", "points": [[0, 0], [1, 0], [0, 1]]}]},
    {"boxes": [{"id": "b", "track_id": "t", "label": "car", "center": [0, 0, 0], "size": [-1, 2, 2], "yaw": 0}]},
])
def test_worker_invalid_annotations_rejected(inputs, monkeypatch, proposed):
    transport(monkeypatch, lambda request: httpx.Response(200, json={"annotations": proposed, "model_version": "v1"}))
    with pytest.raises(inference.InferenceError, match="invalid annotations"):
        inference.infer_proposal(*inputs, "sam2", "road")


@pytest.mark.parametrize("url", [
    "http://worker.example.test/predict", "https://user:secret@worker.example.test/predict",
    "file:///tmp/worker", "http://localhost.evil.test/predict", "https://worker.example.test/#secret",
    " https://worker.example.test/", "https://worker.example.test:99999/",
])
def test_bad_configuration_rejected_without_disclosing_url(inputs, monkeypatch, url):
    monkeypatch.setenv("OMNILABEL_INFERENCE_URL", url)
    assert not inference.worker_configured()
    with pytest.raises(inference.InferenceError) as captured:
        inference.infer_proposal(*inputs, "sam2", "road")
    assert url not in str(captured.value)
    assert "secret@" not in str(captured.value)


@pytest.mark.parametrize("url", ["https://gpu.example.test/predict", "http://127.0.0.1:9000/predict", "http://[::1]:9000/predict", "http://localhost:9000/predict"])
def test_valid_configuration_does_not_claim_health(monkeypatch, url):
    monkeypatch.setenv("OMNILABEL_INFERENCE_URL", url)
    assert inference.worker_configured()


def test_missing_configuration(inputs, monkeypatch):
    monkeypatch.delenv("OMNILABEL_INFERENCE_URL")
    assert not inference.worker_configured()
    with pytest.raises(inference.InferenceError, match="not configured"):
        inference.infer_proposal(*inputs, "sam2", "road")


@pytest.mark.parametrize("status", [302, 401, 500])
def test_http_failure_no_response_body_or_secrets_disclosed(inputs, monkeypatch, status):
    transport(monkeypatch, lambda request: httpx.Response(status, text="private details: token=super-secret", headers={"location": "https://untrusted.invalid/"}))
    with pytest.raises(inference.InferenceError) as captured:
        inference.infer_proposal(*inputs, "sam2", "road")
    assert f"HTTP {status}" in str(captured.value)
    assert "super-secret" not in str(captured.value)


def test_timeout_is_sanitized(inputs, monkeypatch):
    def worker(request):
        raise httpx.ReadTimeout("url https://user:secret@worker.invalid/", request=request)
    transport(monkeypatch, worker)
    with pytest.raises(inference.InferenceError) as captured:
        inference.infer_proposal(*inputs, "sam2", "road")
    assert "timed out" in str(captured.value)
    assert "secret" not in str(captured.value)


def test_response_limits_and_invalid_json(inputs, monkeypatch):
    monkeypatch.setattr(inference, "MAX_RESPONSE_BYTES", 5)
    transport(monkeypatch, lambda request: httpx.Response(200, content=b"123456"))
    with pytest.raises(inference.InferenceError, match="exceeds"):
        inference.infer_proposal(*inputs, "sam2", "road")
    monkeypatch.setattr(inference, "MAX_RESPONSE_BYTES", 100)
    transport(monkeypatch, lambda request: httpx.Response(200, content=b"not json"))
    with pytest.raises(inference.InferenceError, match="invalid annotations"):
        inference.infer_proposal(*inputs, "sam2", "road")
