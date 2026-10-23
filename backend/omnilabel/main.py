"""Local single-user annotation API. Run behind authentication before deployment."""
from __future__ import annotations

import logging
import io
import json
import os
import shutil
import time
import tempfile
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import ValidationError
from PIL import Image, ImageDraw
from starlette.background import BackgroundTask

from .importer import MAX_EXPANDED, MAX_FILES, MAX_UPLOAD, parse_import
from .models import Annotations, Clip, JobRequest, ProjectRequest
from .storage import ConflictError, Store
from .model_connections import ModelConnections, mount_model_routes

LOGGER = logging.getLogger(__name__)
REPOSITORY = Path(__file__).resolve().parents[2]

MODELS = [
    {"id": "geometric-ground", "name": "Geometric ground baseline", "tasks": ["point-segmentation"], "available": True,
     "description": "CPU geometric ground candidate labels for human review. This is not a trained model or verified ground truth.",
     "url": "https://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html"},
    {"id": "sam2", "name": "SAM 2", "tasks": ["image-segmentation", "video-tracking"], "available": False,
     "description": "Integration reference only. Install a licensed checkpoint and GPU inference worker before use; this service does not run SAM 2.",
     "url": "https://github.com/facebookresearch/sam2"},
    {"id": "grounding-dino", "name": "Grounding DINO", "tasks": ["image-detection"], "available": False,
     "description": "Open-vocabulary detector integration reference. Requires weights and inference service; no automatic pseudo-labels are fabricated.",
     "url": "https://github.com/IDEA-Research/GroundingDINO"},
    {"id": "mmdetection3d", "name": "MMDetection3D", "tasks": ["3d-detection", "point-segmentation"], "available": False,
     "description": "3D model toolkit integration reference; requires a dataset-compatible model, checkpoint and GPU worker.",
     "url": "https://github.com/open-mmlab/mmdetection3d"},
    {"id": "openoccupancy", "name": "OpenOccupancy", "tasks": ["occupancy"], "available": False,
     "description": "Occupancy research reference. Current built-in conversion uses observed LiDAR rays and cannot infer occluded ground truth.",
     "url": "https://github.com/JeffWang987/OpenOccupancy"},
]


def create_app(data_dir: str | Path | None = None, seed_demo: bool = True) -> FastAPI:
    root = Path(data_dir or os.environ.get("OMNILABEL_DATA_DIR", REPOSITORY / "data"))
    store = Store(root)
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="omnilabel-job")

    @asynccontextmanager
    async def lifespan(application):
        # This local store supports one service process (uvicorn --workers 1).
        store.recover_interrupted_jobs()
        if seed_demo and not (store.root / "clips" / "demo.json").exists():
            from .demo import build_demo
            demo_assets = store.root / "assets" / "demo"
            demo_assets.mkdir(parents=True, exist_ok=True)
            clip_raw, initial_annotations = build_demo(demo_assets)
            clip = Clip.model_validate(clip_raw)
            annotations = {f.id: Annotations.model_validate(initial_annotations.get(f.id, {})).validate_for_frame(f, clip.cameras)
                           for f in clip.frames}
            store.add_clip(clip, annotations)
        yield
        executor.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(title="OmniLabel API", version="0.1.0", lifespan=lifespan)
    app.state.store = store
    model_connections = ModelConnections(store)
    mount_model_routes(app, model_connections)
    from .scene_routes import mount_scene_routes
    mount_scene_routes(app, store, executor)

    @app.exception_handler(ConflictError)
    async def handle_conflict(request, exc):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    def get_clip(clip_id: str) -> Clip:
        try:
            return store.clip(clip_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(404, "Clip not found") from exc

    def get_frame(clip: Clip, frame_id: str):
        frame = next((f for f in clip.frames if f.id == frame_id), None)
        if frame is None:
            raise HTTPException(404, "Frame not found")
        return frame

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/clips")
    def list_clips():
        return store.summaries()

    @app.post("/api/clips/import", response_model=Clip, status_code=201)
    async def import_clip(file: UploadFile = File(...)):
        payload = await file.read(MAX_UPLOAD + 1)
        await file.close()
        if len(payload) > MAX_UPLOAD:
            raise HTTPException(413, "Upload exceeds 128 MiB")
        staging = None
        destination = None
        published = False
        try:
            clip, labels, staging, namespace, scene_bundle = parse_import(payload, file.filename or "", store.root / "assets", include_scene=True)
            with store.lock:
                if (store.root / "clips" / f"{clip.id}.json").exists():
                    raise ConflictError("A clip with this ID already exists. Change clip.id before importing.")
                destination = store.root / "assets" / namespace
                staging.rename(destination)
                staging = None
                store.add_clip(clip, labels)
                published = True
                if scene_bundle:
                    from .scene import restore_scene
                    restore_scene(store, clip, scene_bundle.get("scene_annotations"),
                                  scene_bundle.get("scene_map"), scene_bundle.get("scene_registration"))
            return clip
        except ConflictError:
            raise
        except (ValueError, TypeError, KeyError, UnicodeError, OSError) as exc:
            detail = str(exc)
            if isinstance(exc, ValidationError):
                detail = "; ".join(f"{'.'.join(map(str, item['loc']))}: {item['msg']}" for item in exc.errors()[:8])
            raise HTTPException(422, detail[:1500]) from exc
        finally:
            if staging:
                shutil.rmtree(staging, ignore_errors=True)
            if destination and not published:
                shutil.rmtree(destination, ignore_errors=True)

    @app.get("/api/clips/{clip_id}", response_model=Clip)
    def read_clip(clip_id: str):
        return get_clip(clip_id)

    @app.get("/api/clips/{clip_id}/frames/{frame_id}/annotations", response_model=Annotations)
    def read_annotations(clip_id: str, frame_id: str):
        get_frame(get_clip(clip_id), frame_id)
        return store.annotations(clip_id, frame_id)

    @app.put("/api/clips/{clip_id}/frames/{frame_id}/annotations", response_model=Annotations)
    def write_annotations(clip_id: str, frame_id: str, annotations: Annotations):
        clip = get_clip(clip_id)
        frame = get_frame(clip, frame_id)
        try:
            annotations.validate_for_frame(frame, clip.cameras)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return store.save_annotations(clip_id, frame_id, annotations)

    @app.post("/api/clips/{clip_id}/project")
    def project(clip_id: str, request: ProjectRequest):
        from .geometry import project_points
        clip = get_clip(clip_id)
        frame = get_frame(clip, request.frame_id)
        camera = next((c for c in clip.cameras if c.id == request.camera_id), None)
        if camera is None:
            raise HTTPException(404, "Camera not found")
        annotations = store.annotations(clip_id, frame.id)
        return project_points(frame.model_dump(), camera.model_dump(), annotations.model_dump(), request.stride)

    def mask_data(clip_id: str, frame_id: str, camera_id: str):
        clip = get_clip(clip_id)
        frame = get_frame(clip, frame_id)
        camera = next((c for c in clip.cameras if c.id == camera_id), None)
        if camera is None:
            raise HTTPException(404, "Camera not found")
        if camera.width * camera.height > 40_000_000:
            raise HTTPException(422, "Semantic mask dimensions exceed the 40-million-pixel interactive limit")
        all_annotations = store.all_annotations(clip)
        taxonomy = ["road", "sidewalk", "vehicle", "pedestrian", "vegetation", "building", "lane", "sky", "terrain", "pole", "traffic-sign", "cyclist"]
        custom = sorted({p["label"] for a in all_annotations.values() for p in a["image_polygons"]} - set(taxonomy) - {"unlabeled"})
        labels = {0: "unlabeled", **{i + 1: label for i, label in enumerate(taxonomy + custom)}}
        if len(labels) > 65535:
            raise HTTPException(422, "Too many semantic classes for a 16-bit mask")
        return camera, all_annotations[frame.id], labels

    def mask_metadata_value(camera, annotations, labels):
        return {"labels": labels, "width": camera.width, "height": camera.height,
                "revision": annotations["revision"], "background": 0, "overlap_rule": "last polygon wins",
                "note": "Built-in class IDs are fixed; custom labels are sorted across this clip. Save this mapping with each mask."}

    def rasterize_mask(camera, annotations, labels):
        label_ids = {label: index for index, label in labels.items()}
        mask = Image.new("I", (camera.width, camera.height), 0)
        draw = ImageDraw.Draw(mask)
        for polygon in annotations["image_polygons"]:
            if polygon["camera_id"] == camera.id:
                draw.polygon([tuple(point) for point in polygon["points"]], fill=label_ids[polygon["label"]])
        output = io.BytesIO()
        mask.convert("I;16").save(output, format="PNG")
        return output.getvalue()

    @app.get("/api/clips/{clip_id}/frames/{frame_id}/masks/{camera_id}.json")
    def mask_metadata(clip_id: str, frame_id: str, camera_id: str):
        camera, annotations, labels = mask_data(clip_id, frame_id, camera_id)
        return mask_metadata_value(camera, annotations, labels)

    @app.get("/api/clips/{clip_id}/frames/{frame_id}/masks/{camera_id}.png")
    def semantic_mask(clip_id: str, frame_id: str, camera_id: str):
        camera, annotations, labels = mask_data(clip_id, frame_id, camera_id)
        return Response(rasterize_mask(camera, annotations, labels), media_type="image/png", headers={
            "Content-Disposition": f'attachment; filename="{clip_id}-{frame_id}-{camera_id}-mask.png"',
            "X-Annotation-Revision": str(annotations["revision"]), "X-Content-Type-Options": "nosniff"})

    @app.get("/api/clips/{clip_id}/frames/{frame_id}/masks/{camera_id}.zip")
    def semantic_mask_archive(clip_id: str, frame_id: str, camera_id: str):
        # One snapshot supplies both pixels and class IDs, even during concurrent edits.
        camera, annotations, labels = mask_data(clip_id, frame_id, camera_id)
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("mask.png", rasterize_mask(camera, annotations, labels))
            archive.writestr("labels.json", json.dumps(mask_metadata_value(camera, annotations, labels), ensure_ascii=False))
        return Response(output.getvalue(), media_type="application/zip", headers={
            "Content-Disposition": f'attachment; filename="{clip_id}-{frame_id}-{camera_id}-semantic-mask.zip"',
            "X-Annotation-Revision": str(annotations["revision"]), "X-Content-Type-Options": "nosniff"})

    def run_job(job: dict, clip: Clip, request: JobRequest):
        from .geometry import build_occupancy, fuse_map, propagate_annotations, segment_ground
        try:
            job.update(status="running", started_at=time.time())
            store.save_job(job)
            clip_data = clip.model_dump()
            annotations = store.all_annotations(clip)
            source = get_frame(clip, request.frame_id or clip.frames[0].id)
            if request.kind == "map":
                result = fuse_map(clip_data, annotations, request.voxel_size)
            elif request.kind == "occupancy":
                result = build_occupancy(source.model_dump(), annotations[source.id], request.voxel_size)
            elif request.kind == "prelabel":
                from .inference import infer_proposal
                current_annotations = annotations[source.id]
                scene_revision = None
                if request.scene_mode:
                    from .scene import load_scene, evaluate_scene
                    with store.lock:
                        scene = load_scene(store, clip)
                        current_annotations = evaluate_scene(store, clip, source.id, scene)
                        scene_revision = scene.revision
                endpoint, _ = model_connections.endpoint(request.model_id)
                result = infer_proposal(clip_data, source.model_dump(), current_annotations,
                                        request.model_id, request.prompt, assets_root=store.root / "assets",
                                        options=request.options.model_dump(exclude_none=True) if request.options else None,
                                        endpoint=endpoint, token=model_connections.token(request.model_id))
                result.update(frame_id=source.id, clip_id=clip.id)
                if scene_revision is not None:
                    result["scene_revision"] = scene_revision
            else:
                changes = {}
                if request.kind == "ground-segmentation":
                    changes[source.id] = segment_ground(source.model_dump(), annotations[source.id])
                else:
                    targets = [get_frame(clip, request.target_frame_id)] if request.target_frame_id else [f for f in clip.frames if f.id != source.id]
                    for target in targets:
                        if target.id == source.id:
                            raise ValueError("Propagation target must differ from source frame")
                        changes[target.id] = propagate_annotations(source.model_dump(), target.model_dump(), annotations[source.id], annotations[target.id])
                validated = {frame_id: Annotations.model_validate(value).validate_for_frame(get_frame(clip, frame_id), clip.cameras)
                             for frame_id, value in changes.items()}
                with store.lock:
                    for frame_id in validated:
                        if store.annotations(clip.id, frame_id).revision != annotations[frame_id]["revision"]:
                            raise ConflictError("Annotations changed while this job was running. Reload and retry.")
                    for frame_id, value in validated.items():
                        value.revision = annotations[frame_id]["revision"]
                        store.save_annotations(clip.id, frame_id, value)
                result = {"updated_frames": len(validated)}
            job.update(status="completed", result=result, finished_at=time.time())
        except Exception as exc:
            LOGGER.exception("Annotation job %s failed", job["id"])
            job.update(status="failed", error=str(exc)[:1500], finished_at=time.time())
        store.save_job(job)

    @app.post("/api/clips/{clip_id}/jobs", status_code=202)
    def submit_job(clip_id: str, request: JobRequest):
        clip = get_clip(clip_id)
        if request.kind == "prelabel":
            from .inference import worker_configured
            endpoint, _ = model_connections.endpoint(request.model_id)
            if not worker_configured(request.model_id, endpoint):
                raise HTTPException(503, "所选模型未配置有效的推理地址，请在模型面板设置远程 GPU 服务。")
            source = get_frame(clip, request.frame_id or clip.frames[0].id)
            if request.options and request.options.camera_id:
                camera = next((c for c in clip.cameras if c.id == request.options.camera_id), None)
                if camera is None:
                    raise HTTPException(422, "Prompt camera does not exist")
                vertices = list(request.options.points)
                if request.options.box:
                    vertices += [(request.options.box[0], request.options.box[1]), (request.options.box[2], request.options.box[3])]
                if any(not 0 <= x <= camera.width or not 0 <= y <= camera.height for x, y in vertices):
                    raise HTTPException(422, "SAM2 prompt lies outside the selected image")
                if request.options.camera_id not in source.images and request.model_id == "sam2":
                    raise HTTPException(422, "Selected camera has no image in this frame")
        if request.frame_id:
            get_frame(clip, request.frame_id)
        if request.target_frame_id:
            get_frame(clip, request.target_frame_id)
        job = {"id": uuid.uuid4().hex, "clip_id": clip_id, "kind": request.kind, "status": "queued", "created_at": time.time(),
               "parameters": request.model_dump()}
        store.save_job(job)
        response = dict(job)
        executor.submit(run_job, job, clip, request)
        return response

    @app.get("/api/jobs/{job_id}")
    def read_job(job_id: str):
        try:
            return store.job(job_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(404, "Job not found") from exc

    @app.get("/api/clips/{clip_id}/export")
    def export_clip(clip_id: str):
        clip = get_clip(clip_id)
        from .scene import scene_export
        return JSONResponse({"schema_version": "1.1", "clip": clip.model_dump(), "annotations": store.all_annotations(clip), **scene_export(store, clip)},
                            headers={"Content-Disposition": f'attachment; filename="{clip.id}-annotations.json"'})

    @app.get("/api/clips/{clip_id}/export.zip")
    def export_archive(clip_id: str):
        clip = get_clip(clip_id)
        clip_data = clip.model_dump()
        annotations = store.all_annotations(clip)
        asset_files = {}
        assets_root = (store.root / "assets").resolve()
        expanded = 0
        for frame in clip_data["frames"]:
            for camera_id, url in list(frame["images"].items()):
                if not url.startswith("/api/assets/"):
                    raise HTTPException(422, "Clip contains an image that is not stored locally")
                source = (assets_root / url.removeprefix("/api/assets/")).resolve()
                if not source.is_relative_to(assets_root) or not source.is_file():
                    raise HTTPException(422, "A referenced image asset is missing")
                if source not in asset_files:
                    asset_files[source] = f"assets/image-{len(asset_files):06d}{source.suffix.lower()}"
                    expanded += source.stat().st_size
                frame["images"][camera_id] = asset_files[source]
        if len(asset_files) + 1 > MAX_FILES:
            raise HTTPException(422, "Portable archive exceeds the 5000-member import limit; split this clip before export")
        temporary = tempfile.NamedTemporaryFile(prefix=".export-", suffix=".zip", dir=store.root, delete=False)
        destination = Path(temporary.name)
        temporary.close()
        try:
            with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
                for source, name in asset_files.items():
                    archive.write(source, name)
                from .scene import scene_export
                envelope = {"schema_version": "1.1", "clip": clip_data, "annotations": annotations, **scene_export(store, clip)}
                encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                with archive.open("manifest.json", "w") as manifest:
                    for chunk in encoder.iterencode(envelope):
                        encoded = chunk.encode("utf-8")
                        expanded += len(encoded)
                        if expanded > MAX_EXPANDED:
                            raise HTTPException(422, "Portable archive exceeds 256 MiB expanded; split this clip before export")
                        manifest.write(encoded)
            if destination.stat().st_size > MAX_UPLOAD:
                raise HTTPException(422, "Portable archive exceeds 128 MiB compressed; split this clip before export")
            return FileResponse(destination, media_type="application/zip", filename=f"{clip.id}-omnilabel.zip",
                                background=BackgroundTask(destination.unlink, missing_ok=True))
        except Exception:
            destination.unlink(missing_ok=True)
            raise

    @app.get("/api/models")
    def models():
        from .inference import worker_configured
        statuses = model_connections.statuses()
        entries = []
        for model in MODELS:
            if model["id"] in statuses:
                status = statuses[model["id"]]
                model = {**model, **status}
                model["description"] = {
                    "sam2": "SAM2 提示分割：在左侧相机上添加正负点或框选区域，输出可审核的分割候选。",
                    "mmdetection3d": "MMDetection3D 点云目标检测：按已部署配置和权重预测三维框；使用原始 LiDAR 坐标。",
                    "openoccupancy": "OpenOccupancy 学习型占用预测：调用已部署的 occupancy 模型，输出语义体素和候选结果。"
                }[model["id"]]
            entries.append(model)
        worker = {"id": "external-worker", "name": "外部模型推理服务", "tasks": ["prelabel-proposals"],
                  "available": worker_configured(), "url": "https://github.com/shizhenghe/OmniLabel",
                  "description": "支持统一协议的外部 GPU / 模型服务。已配置仅表示服务地址有效，实际推理须连接验证；候选结果必须人工审核。部署协议见 docs/model-worker.md。"}
        return {"models": [*entries, worker]}

    @app.get("/api/assets/{asset_path:path}")
    def asset(asset_path: str):
        root = store.root / "assets"
        path = (root / asset_path).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or any(p.startswith(".") for p in Path(asset_path).parts):
            raise HTTPException(404, "Asset not found")
        return FileResponse(path, headers={"X-Content-Type-Options": "nosniff"})

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str):
        if path == "api" or path.startswith("api/"):
            raise HTTPException(404, "API route not found")
        dist = REPOSITORY / "dist"
        asset_file = (dist / path).resolve()
        if path and asset_file.is_relative_to(dist.resolve()) and asset_file.is_file():
            return FileResponse(asset_file)
        if (dist / "index.html").is_file():
            return FileResponse(dist / "index.html")
        return JSONResponse({"service": "OmniLabel API", "docs": "/docs", "frontend": "Run npm run dev, or npm run build for the production UI."})

    return app


app = create_app()
