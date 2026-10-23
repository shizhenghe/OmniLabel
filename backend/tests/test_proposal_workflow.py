"""A model result must stay a proposal until a person saves it explicitly."""
import json
import time
import base64
import io

import httpx
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from omnilabel.main import create_app


def test_inference_job_preserves_saved_annotations_until_review(tmp_path, monkeypatch):
    from omnilabel import inference

    identity = np.eye(4).tolist()
    clip = {"id": "inference-test", "name": "Worker test", "fps": 10,
            "cameras": [{"id": "front", "name": "Front", "width": 100, "height": 100,
                         "intrinsics": [[10, 0, 50], [0, 10, 50], [0, 0, 1]], "camera_from_ego": identity}],
            "frames": [{"id": "f0", "timestamp": 0, "ego_to_world": identity,
                        "lidar_to_ego": identity, "points": [[0, 0, 5]], "images": {}}]}
    monkeypatch.setenv("OMNILABEL_INFERENCE_URL", "http://127.0.0.1:9901/predict")

    pixels = io.BytesIO()
    Image.new("RGB", (100, 100), (12, 34, 56)).save(pixels, "PNG")
    clip["frames"][0]["images"]["front"] = "data:image/png;base64," + base64.b64encode(pixels.getvalue()).decode()
    received = []

    def worker(request):
        payload = json.loads(request.content)
        received.append(payload)
        assert payload["frame"]["images"]["front"].startswith("data:image/png;base64,")
        return httpx.Response(200, json={"annotations": {**payload["annotations"], "point_labels": {"0": "road"}},
                                          "model_version": "test-checkpoint-v1"})

    with TestClient(create_app(tmp_path, seed_demo=False)) as client:
        assert client.post("/api/clips/import", files={"file": ("clip.json", json.dumps(clip))}).status_code == 201
        original_http_client = httpx.Client
        monkeypatch.setattr(inference.httpx, "Client", lambda **kwargs: original_http_client(transport=httpx.MockTransport(worker), **kwargs))
        route = "/api/clips/inference-test/frames/f0/annotations"
        original = client.get(route).json()
        submitted = client.post("/api/clips/inference-test/jobs", json={
            "kind": "prelabel", "frame_id": "f0", "model_id": "test-worker", "prompt": "road"})
        assert submitted.status_code == 202
        for _ in range(100):
            job = client.get(f"/api/jobs/{submitted.json()['id']}").json()
            if job["status"] in ("completed", "failed"):
                break
            time.sleep(.01)
        assert job["status"] == "completed", job
        assert job["result"]["review_required"]
        assert job["result"]["frame_id"] == "f0"
        assert received[0]["model_id"] == "test-worker"
        assert client.get(route).json() == original
        reviewed = client.put(route, json=job["result"]["proposal"])
        assert reviewed.status_code == 200
        assert reviewed.json()["point_labels"] == {"0": "road"}
