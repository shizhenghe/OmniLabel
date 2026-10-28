import json

import httpx
import pytest
from fastapi.testclient import TestClient

from omnilabel.main import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    for suffix in ("INFERENCE", "SAM2", "LATR", "MMDET3D", "OPENOCCUPANCY"):
        monkeypatch.delenv(f"OMNILABEL_{suffix}_URL", raising=False)
    with TestClient(create_app(tmp_path, seed_demo=False)) as client:
        yield client


def test_model_endpoints_persist_independently_and_disable_environment_fallback(client, tmp_path, monkeypatch):
    monkeypatch.setenv("OMNILABEL_SAM2_URL", "https://initial.example/predict")
    updated = client.put("/api/model-connections", json={"connections": [
        {"model_id": "sam2", "endpoint": "http://127.0.0.1:9101/predict"},
        {"model_id": "mmdetection3d", "endpoint": "https://lidar.example/v1/predict"},
    ]})
    assert updated.status_code == 200
    with TestClient(create_app(tmp_path, seed_demo=False)) as reopened:
        values = {c["model_id"]: c for c in reopened.get("/api/model-connections").json()["connections"]}
        assert values["sam2"]["endpoint"] == "http://127.0.0.1:9101/predict"
        assert values["mmdetection3d"]["endpoint"] == "https://lidar.example/v1/predict"
        assert not values["latr"]["configured"]
        assert not values["openoccupancy"]["configured"]
    result = client.put("/api/model-connections", json={"connections": [{"model_id": "sam2", "endpoint": ""}]})
    assert not result.json()["connections"][0]["configured"]


@pytest.mark.parametrize("endpoint", ["file:///tmp/config", "https://user:password@example.com/predict", "https://example.com/predict?token=private", "http://public.example/predict"])
def test_connection_rejects_unsafe_or_credential_bearing_endpoints(client, endpoint):
    result = client.put("/api/model-connections", json={"connections": [{"model_id": "sam2", "endpoint": endpoint}]})
    assert result.status_code == 422
    assert "password" not in result.text and "private" not in result.text


def test_worker_status_requires_the_requested_model_and_real_ready_flag(client, monkeypatch):
    client.put("/api/model-connections", json={"connections": [
        {"model_id": "sam2", "endpoint": "https://models.example/predict"},
        {"model_id": "openoccupancy", "endpoint": "https://models.example/predict"},
    ]})
    monkeypatch.setenv("OMNILABEL_SAM2_TOKEN", "test-token-only")
    original = httpx.Client

    def remote(request):
        assert request.url.path == "/models"
        return httpx.Response(200, json={"models": [
            {"id": "sam2", "available": True, "tasks": ["image-segmentation"]},
            {"id": "openoccupancy", "available": False, "tasks": ["occupancy"]},
        ]})

    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(remote), **kwargs))
    sam = client.post("/api/model-connections/sam2/check").json()
    occ = client.post("/api/model-connections/openoccupancy/check").json()
    assert sam["available"] and sam["status"] == "ready"
    assert occ["configured"] and not occ["available"] and occ["status"] == "not_ready"
    assert "endpoint" not in sam


def test_model_status_does_not_echo_worker_errors_or_credentials(client, monkeypatch):
    client.put("/api/model-connections", json={"connections": [{"model_id": "sam2", "endpoint": "https://models.example/predict"}]})
    original = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(lambda request: httpx.Response(401, text="private token=test-secret")), **kwargs))
    result = client.post("/api/model-connections/sam2/check")
    assert result.json()["status"] == "http_error" and not result.json()["available"]
    assert "test-secret" not in result.text and "private" not in result.text
