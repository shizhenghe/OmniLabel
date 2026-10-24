import base64
import io

import numpy as np
import pytest
from fastapi import HTTPException
from PIL import Image

from model_worker import app as worker
from model_worker.common import decode_data_image, input_points, merged_annotations, parse_labels


def test_common_input_and_merge_helpers():
    assert parse_labels(" vehicle, road，vehicle\npedestrian ") == ["vehicle", "road", "pedestrian"]
    points = input_points({"points": [[1, 2, 3], [4, 5, 6]]}, target_dim=5)
    assert points.shape == (2, 5)
    assert np.all(points[:, 3:] == 0)
    request = {"annotations": {
        "boxes": [{"id": "mmd3d-old"}, {"id": "manual"}],
        "vectors": [], "image_polygons": [], "point_labels": {}, "revision": 4,
    }}
    merged = merged_annotations(request, replace_prefix="mmd3d-")
    assert merged["boxes"] == [{"id": "manual"}]
    assert "revision" not in merged


def test_embedded_image_validation():
    stream = io.BytesIO()
    Image.new("RGB", (12, 8), (10, 20, 30)).save(stream, format="PNG")
    reference = "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()
    assert decode_data_image(reference, (12, 8)).size == (12, 8)


def test_worker_routes_dispatch_and_auth(monkeypatch):
    class FakeAdapter:
        model_id = "fake"

        def status(self):
            return {"id": "fake", "name": "Fake", "tasks": ["test"], "available": True, "loaded": True, "detail": "ok"}

        def predict(self, request):
            assert request["prompt"] == "vehicle"
            return {"boxes": [], "vectors": [], "image_polygons": [], "point_labels": {}}, "fake-v1"

    monkeypatch.setattr(worker, "ADAPTERS", {"fake": FakeAdapter()})
    monkeypatch.setenv("OMNILABEL_WORKER_TOKEN", "secret")
    payload = {
        "schema_version": "1.0", "model_id": "fake", "prompt": "vehicle",
        "clip": {"id": "clip", "cameras": []},
        "frame": {"id": "frame", "points": [], "images": {}},
        "annotations": {},
    }
    with pytest.raises(HTTPException) as captured:
        worker.authorize(None)
    assert captured.value.status_code == 401
    worker.authorize("Bearer secret")
    assert worker.models(None)["models"][0]["id"] == "fake"
    response = worker.predict(worker.PredictRequest.model_validate(payload), None)
    assert response["model_version"] == "fake-v1"

