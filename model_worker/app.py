from __future__ import annotations

import logging
import os
import re
import secrets
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .common import WorkerError
from .mmdetection3d_adapter import MMDetection3DAdapter
from .sam3_adapter import Sam3Adapter

LOGGER = logging.getLogger(__name__)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"]
    model_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}$")
    prompt: str = Field(max_length=2000)
    clip: dict[str, Any]
    frame: dict[str, Any]
    annotations: dict[str, Any]


ADAPTERS = {adapter.model_id: adapter for adapter in (Sam3Adapter(), MMDetection3DAdapter())}


def authorize(authorization: str | None = Header(default=None)) -> None:
    expected = os.environ.get("OMNILABEL_WORKER_TOKEN", "")
    if not expected:
        return
    supplied = authorization.removeprefix("Bearer ") if authorization and authorization.startswith("Bearer ") else ""
    if not supplied or not secrets.compare_digest(supplied, expected):
        raise HTTPException(401, "Invalid worker authorization")


app = FastAPI(title="OmniLabel Model Worker", version="0.1.0")


@app.get("/health")
def health(_: None = Depends(authorize)):
    return {"status": "ok", "models": [adapter.status() for adapter in ADAPTERS.values()]}


@app.get("/v1/models")
def models(_: None = Depends(authorize)):
    return {"models": [adapter.status() for adapter in ADAPTERS.values()]}


@app.post("/v1/predict")
def predict(request: PredictRequest, _: None = Depends(authorize)):
    adapter = ADAPTERS.get(request.model_id)
    if adapter is None:
        raise HTTPException(404, "Unknown model_id")
    if not adapter.status()["available"]:
        raise HTTPException(503, "Requested model is not configured in this worker")
    try:
        annotations, model_version = adapter.predict(request.model_dump())
        if not isinstance(model_version, str) or not model_version.strip() or len(model_version) > 200:
            raise WorkerError("adapter returned invalid model version metadata")
        return {"model_version": model_version, "annotations": annotations}
    except WorkerError as exc:
        raise HTTPException(422, str(exc)) from None
    except Exception:
        LOGGER.exception("Unhandled model worker failure for %s", re.sub(r"[^a-zA-Z0-9_.-]", "", request.model_id))
        raise HTTPException(500, "Model inference failed; inspect worker logs") from None

