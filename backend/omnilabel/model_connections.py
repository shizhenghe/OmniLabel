"""Persistent per-model endpoints and actual remote worker capability checks."""
from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit, urlunsplit

import httpx
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .storage import atomic_json

MODEL_ENV = {"sam2": "SAM2", "latr": "LATR", "mmdetection3d": "MMDET3D", "openoccupancy": "OPENOCCUPANCY"}


class Connection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str
    endpoint: str = Field(default="", max_length=2048)
    enabled: bool = True


class ConnectionUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    connections: list[Connection] = Field(max_length=4)


class ModelConnections:
    def __init__(self, store):
        self.store = store
        self.path = store.root / "model-connections.json"
        self.cache = {}
        self.lock = threading.RLock()

    def saved(self):
        with self.lock:
            return json.loads(self.path.read_text()) if self.path.exists() else {}

    def endpoint(self, model_id):
        if model_id not in MODEL_ENV:
            return os.environ.get("OMNILABEL_INFERENCE_URL", ""), "environment"
        configured = self.saved().get(model_id)
        if configured is not None:
            return (configured["endpoint"] if configured["enabled"] else ""), "settings"
        return os.environ.get(f"OMNILABEL_{MODEL_ENV[model_id]}_URL", os.environ.get("OMNILABEL_INFERENCE_URL", "")), "environment"

    def token(self, model_id):
        suffix = MODEL_ENV.get(model_id)
        return os.environ.get(f"OMNILABEL_{suffix}_TOKEN", os.environ.get("OMNILABEL_INFERENCE_TOKEN", "")) if suffix else os.environ.get("OMNILABEL_INFERENCE_TOKEN", "")

    def list(self):
        result = []
        for model_id in MODEL_ENV:
            endpoint, source = self.endpoint(model_id)
            result.append({"model_id": model_id, "endpoint": endpoint, "configured": bool(endpoint), "source": source})
        return {"connections": result}

    def update(self, connections):
        from .inference import InferenceError, validate_worker_url
        if len({c.model_id for c in connections}) != len(connections):
            raise ValueError("Duplicate model connections")
        values = self.saved()
        for connection in connections:
            if connection.model_id not in MODEL_ENV:
                raise ValueError("Unknown model; choose sam2, latr, mmdetection3d or openoccupancy")
            if connection.endpoint:
                try:
                    validate_worker_url(connection.endpoint)
                except InferenceError as exc:
                    raise ValueError(str(exc)) from None
                if urlsplit(connection.endpoint).query:
                    raise ValueError("Put credentials in the server token environment binding, not the endpoint URL")
            values[connection.model_id] = connection.model_dump(exclude={"model_id"})
        with self.lock:
            atomic_json(self.path, values)
            self.cache.clear()
        return self.list()

    def probe(self, model_id, force=False):
        from .inference import InferenceError, validate_worker_url
        endpoint, _ = self.endpoint(model_id)
        result = {"model_id": model_id, "configured": bool(endpoint), "available": False,
                  "status": "not_configured", "status_detail": "尚未配置远程推理地址"}
        if not endpoint:
            return result
        key = (model_id, endpoint)
        with self.lock:
            entry = self.cache.get(key)
            if not force and entry and time.monotonic() - entry[0] < 10:
                return dict(entry[1])
        try:
            validate_worker_url(endpoint)
            parsed = urlsplit(endpoint)
            path = parsed.path.rsplit("/", 1)[0] + "/models"
            status_url = urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
            token = self.token(model_id)
            if "\r" in token or "\n" in token:
                raise ValueError("Invalid token binding")
            headers = {"Authorization": "Bearer " + token} if token else {}
            with httpx.Client(timeout=httpx.Timeout(3, connect=2), verify=True, follow_redirects=False) as client:
                with client.stream("GET", status_url, headers=headers) as response:
                    if response.status_code != 200:
                        result.update(status="http_error", status_detail=f"模型状态接口返回 HTTP {response.status_code}")
                    else:
                        body = bytearray()
                        for chunk in response.iter_bytes():
                            body.extend(chunk)
                            if len(body) > 256 * 1024:
                                raise ValueError("Worker model catalogue too large")
                        payload = json.loads(body)
                        if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
                            raise ValueError("Invalid model catalogue")
                        models = payload.get("models", [])
                        entry = next((m for m in models if isinstance(m, dict) and m.get("id", m.get("model_id")) == model_id), None)
                        if entry is None:
                            result.update(status="unsupported_model", status_detail="服务未提供所选模型，请检查部署的适配器")
                        else:
                            ready = entry.get("available") is True
                            result.update(available=ready, status="ready" if ready else "not_ready",
                                          status_detail="模型服务已报告就绪" if ready else "服务可达，模型依赖、配置或权重尚未就绪")
                            if not ready:
                                reasons = {"missing_dependencies": "服务缺少模型运行依赖", "missing_checkpoint": "服务缺少模型权重",
                                           "missing_config": "服务缺少模型配置或训练相机定义", "configured": "服务已配置，模型尚未加载",
                                           "missing_repository": "服务缺少 LATR 源码目录",
                                           "error": "模型加载失败，请检查运行环境与权重版本"}
                                result["status_detail"] = reasons.get(entry.get("status"), result["status_detail"])
                            tasks = entry.get("tasks", [])
                            result["capabilities"] = [t for t in tasks[:20] if isinstance(t, str) and len(t) <= 80] if isinstance(tasks, list) else []
        except (httpx.HTTPError, ValueError, TypeError, KeyError, InferenceError):
            result.update(status="unreachable", status_detail="无法验证模型服务；请检查地址、网络和鉴权配置")
        with self.lock:
            self.cache[key] = (time.monotonic(), dict(result))
        return result

    def statuses(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            values = list(pool.map(self.probe, MODEL_ENV))
        return {item["model_id"]: item for item in values}


def mount_model_routes(app, connections):
    @app.get("/api/model-connections")
    def read_connections():
        return connections.list()

    @app.put("/api/model-connections")
    def update_connections(update: ConnectionUpdate):
        try:
            return connections.update(update.connections)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    @app.post("/api/model-connections/{model_id}/check")
    def check_connection(model_id: str):
        if model_id not in MODEL_ENV:
            raise HTTPException(404, "Unknown model")
        return connections.probe(model_id, force=True)
