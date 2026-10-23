"""Launch one actual model per isolated Python environment/GPU process.

Example: OMNILABEL_WORKER_MODEL=sam2 python -m uvicorn model_workers.app:app
The worker never fabricates labels when dependencies or checkpoints are absent.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
import hmac
import json
import os
import threading

from fastapi import FastAPI, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from .adapters import ADAPTERS
from .common import InvalidInput, ModelUnavailable

MAX_REQUEST_BYTES = 64 * 1024 * 1024


def create_app(model_id=None, adapter=None, eager_load=True):
    selected = model_id or os.environ.get("OMNILABEL_WORKER_MODEL", "sam2")
    if selected not in ADAPTERS:
        raise ValueError("OMNILABEL_WORKER_MODEL must be sam2, mmdetection3d or openoccupancy")
    runtime = adapter or ADAPTERS[selected]()
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(application):
        if eager_load and runtime.status()["status"] == "configured":
            try:
                await run_in_threadpool(runtime.load)
            except Exception as exc:
                # Show an actionable class without URLs, checkpoint data or tokens.
                runtime.failure = "Model initialization failed (" + type(exc).__name__ + "). Check the matching dependency/config/checkpoint versions on the worker."
        yield

    application = FastAPI(title="OmniLabel " + selected + " worker", lifespan=lifespan)
    application.state.adapter = runtime

    def authenticate(request):
        token = os.environ.get("MODEL_WORKER_TOKEN", "")
        if token and not hmac.compare_digest(request.headers.get("authorization", ""), "Bearer " + token):
            raise HTTPException(401, "Worker authentication required")

    @application.get("/health")
    def health(request: Request):
        authenticate(request)
        return {"status": "ok", "model": runtime.status()}

    @application.get("/models")
    def models(request: Request):
        authenticate(request)
        return {"models": [runtime.status()]}

    @application.post("/predict")
    async def predict(request: Request):
        authenticate(request)
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > MAX_REQUEST_BYTES:
                raise HTTPException(413, "Model request exceeds 64 MiB")
        try:
            payload = json.loads(data)
            if not isinstance(payload, dict) or payload.get("schema_version") != "1.0" or payload.get("model_id") != selected:
                raise ValueError
            if not isinstance(payload.get("frame"), dict) or not isinstance(payload.get("clip"), dict) or not isinstance(payload["clip"].get("cameras"), list):
                raise ValueError
            if not isinstance(payload.get("options", {}), dict): raise ValueError
        except (ValueError, TypeError, KeyError):
            raise HTTPException(422, "Expected the OmniLabel 1.0 frame/clip request for this model ID") from None
        status = runtime.status()
        if not status["available"]:
            raise HTTPException(503, status["reason"])

        def run():
            with lock:
                return runtime.predict(payload)
        try:
            return await run_in_threadpool(run)
        except InvalidInput as exc:
            raise HTTPException(422, str(exc)) from None
        except ModelUnavailable as exc:
            raise HTTPException(503, str(exc)) from None
        except Exception as exc:
            # The model framework may include local paths or URLs in exceptions.
            raise HTTPException(500, "Model inference failed (" + type(exc).__name__ + "); inspect the worker runtime and compatible checkpoint") from None

    return application


app = create_app()
