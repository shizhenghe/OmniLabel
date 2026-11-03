"""Ordered manual 3D/2D lane pairing and durable multi-frame optimization jobs."""
from __future__ import annotations

from datetime import datetime, timezone
from importlib.util import find_spec
import json
from pathlib import Path
import threading
from typing import Literal
import uuid
import zipfile

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import numpy as np
from pydantic import Field, field_validator, model_validator
from scipy.spatial.transform import Rotation, Slerp

from .calibration import CalibrationUpdate, _revision, save_calibration
from .lane_optimizer import OptimizationCancelled, optimize, prepare, resample, statistics, write_outputs
from .models import Camera, Identifier, Model, rigid_transform
from .projection_calibration import _base_matrix, _fingerprint, _image_path, project_camera_points
from .scene import get_scene, registered_poses
from .storage import ConflictError, Store, atomic_json, safe_id


class LanePairFields(Model):
    id: Identifier
    frame_id: Identifier
    lane_id: Identifier
    world_range: tuple[float, float] = (0, 1)
    image_world_range: tuple[float, float] = (0, 1)
    split: Literal["train", "validation"] = "train"
    weight: float = Field(default=1, gt=0, le=100)

    @field_validator("world_range", "image_world_range")
    @classmethod
    def interval(cls, value):
        if not 0 <= value[0] < value[1] <= 1: raise ValueError("三维取样范围必须在 0–1 内且起点小于终点。")
        return value


class ImageLaneReference(Model):
    frame_id: Identifier
    camera: Camera
    camera_from_primary: list[list[float]]
    world_from_primary: list[list[float]]
    source_line_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    world_range: tuple[float, float] = (0, 1)
    _matrices = field_validator("camera_from_primary", "world_from_primary")(rigid_transform)


class LanePair(LanePairFields):
    image_points: list[tuple[float, float]] = Field(min_length=2, max_length=512)
    image_size: tuple[int, int]
    image_lane_id: Identifier | None = None
    image_reference: ImageLaneReference | None = None

    @model_validator(mode="after")
    def projected_source(self):
        if self.image_lane_id is not None:
            if self.image_lane_id == self.lane_id: raise ValueError("真值线与图像投影线不能选择同一条线。")
            if self.image_reference is None: raise ValueError("请使用 /pairs/from-lanes 保存两条点云线，生成固定图像约束。")
            if self.image_reference.frame_id != self.frame_id: raise ValueError("图像投影线的参考帧与配对不一致。")
            if self.image_reference.world_range != self.image_world_range: raise ValueError("固定图像线的取样范围已变化，请重新选择配对。")
        elif self.image_reference is not None: raise ValueError("图像线参考记录缺少 image_lane_id。")
        return self


class ProjectedLaneSelection(LanePairFields):
    revision: int = Field(ge=0)
    image_lane_id: Identifier


class PairUpdate(Model):
    revision: int = Field(ge=0)
    pairs: list[LanePair] = Field(max_length=128)

    @model_validator(mode="after")
    def identities(self):
        if len({p.id for p in self.pairs}) != len(self.pairs): raise ValueError("配对 ID 重复。")
        image_ids = [(p.frame_id, p.image_lane_id) for p in self.pairs if p.image_lane_id is not None]
        if len(set(image_ids)) != len(image_ids): raise ValueError("同帧同一条图像投影线不能重复配给不同真值线。")
        owners = {}
        for pair in self.pairs:
            if pair.image_lane_id is None: continue
            if pair.image_lane_id in owners and owners[pair.image_lane_id] != pair.lane_id:
                raise ValueError("同一条图像投影线只能归属一条真值线。")
            owners[pair.image_lane_id] = pair.lane_id
        roles = {}
        for pair in self.pairs:
            if pair.frame_id in roles and roles[pair.frame_id] != pair.split:
                raise ValueError("同一帧不能同时出现在训练集和独立验证集。")
            roles[pair.frame_id] = pair.split
        return self


class OptimizerConfig(Model):
    samples_per_lane: int = Field(default=48, ge=10, le=128)
    adam_iterations: int = Field(default=180, ge=1, le=1000)
    lbfgs_iterations: int = Field(default=25, ge=0, le=100)
    learning_rate: float = Field(default=.04, gt=0, le=.5)
    huber_delta_px: float = Field(default=4, gt=0, le=100)
    chamfer_weight: float = Field(default=.5, ge=0, le=10)
    prior_weight: float = Field(default=.01, ge=0, le=100)
    max_translation_m: float = Field(default=.5, gt=0, le=2)
    max_rotation_degrees: float = Field(default=3, gt=0, le=10)
    timeout_seconds: float = Field(default=120, ge=5, le=600)


class PoseSample(Model):
    timestamp: float
    matrix: list[list[float]]
    _matrix = field_validator("matrix")(rigid_transform)


class OptimizeRequest(Model):
    pairing_revision: int = Field(ge=0)
    mode: Literal["multi", "single"] = "multi"
    frame_ids: list[Identifier] | None = Field(default=None, max_length=128)
    conventions_confirmed: bool = False
    time_alignment_confirmed: bool = False
    pose_mode: Literal["auto", "ego", "provided"] = "auto"
    pose_direction: Literal["world_from_lidar", "lidar_from_world"] = "world_from_lidar"
    pose_samples: list[PoseSample] = Field(default_factory=list, max_length=4000)
    image_timestamps: dict[Identifier, float] = Field(default_factory=dict)
    max_pose_gap_seconds: float = Field(default=.2, gt=0, le=10)
    sensor_lidar_to_ego: list[list[float]] | None = None
    initial_matrix: list[list[float]] | None = None
    initial_direction: Literal["camera_from_lidar", "lidar_from_camera"] = "camera_from_lidar"
    config: OptimizerConfig = Field(default_factory=OptimizerConfig)

    @field_validator("sensor_lidar_to_ego", "initial_matrix")
    @classmethod
    def transforms(cls, value):
        return rigid_transform(value) if value is not None else value


class ApplyRequest(Model):
    confirm_shared_update: bool = False


def runtime_status():
    available = all(find_spec(name) is not None for name in ("torch", "yaml", "matplotlib"))
    return {"available": available, "detail": "CPU PyTorch 多帧 SE(3) 优化已安装。" if available
            else "请运行 bash scripts/setup-lane-calibration.sh 安装优化依赖。"}


def _interpolate(request: OptimizeRequest, frames):
    if len(request.pose_samples) < 2: raise ValueError("外部 pose 至少需要两个带时间戳的样本。")
    times = np.asarray([p.timestamp for p in request.pose_samples])
    if np.any(np.diff(times) <= 0): raise ValueError("Pose 样本时间戳必须严格递增，单位秒。")
    matrices = np.asarray([p.matrix for p in request.pose_samples])
    if request.pose_direction == "lidar_from_world": matrices = np.linalg.inv(matrices)
    rotations = Slerp(times, Rotation.from_matrix(matrices[:, :3, :3])); output = {}
    for frame in frames:
        if frame.id not in request.image_timestamps: raise ValueError(f"缺少 {frame.id} 的相机图像时间戳。")
        stamp = request.image_timestamps[frame.id]
        if stamp < times[0] or stamp > times[-1]: raise ValueError("图像时间戳超出 pose 覆盖范围，不允许外推。")
        index = min(max(int(np.searchsorted(times, stamp, side="right")), 1), len(times)-1)
        exact = np.any(np.isclose(times, stamp, rtol=0, atol=1e-9))
        if not exact and times[index]-times[index-1] > request.max_pose_gap_seconds:
            raise ValueError("Pose 插值间隔过大，请提供更密的同步位姿。")
        pose = np.eye(4); pose[:3, :3] = rotations(stamp).as_matrix()
        pose[:3, 3] = [np.interp(stamp, times, matrices[:, axis, 3]) for axis in range(3)]
        output[frame.id] = pose
    return output


def resolve_geometry(store, clip, camera, frames, request: OptimizeRequest):
    """Resolve physical sensor poses without requiring or fabricating 2D truth."""
    mounts = []
    fused = clip.source.startswith("4d-") or all(np.allclose(p, np.eye(4)) for p in registered_poses(store, clip).values()) and any(not np.allclose(f.ego_to_world, np.eye(4)) for f in frames)
    for frame in frames:
        if request.pose_mode == "ego": mount = np.eye(4)
        elif request.sensor_lidar_to_ego is not None: mount = np.asarray(request.sensor_lidar_to_ego)
        elif frame.calibration_lidar_to_ego is not None: mount = np.asarray(frame.calibration_lidar_to_ego)
        elif frame.sensor_lidar_to_ego is not None: mount = np.asarray(frame.sensor_lidar_to_ego)
        elif not fused: mount = np.asarray(frame.lidar_to_ego)
        else: mount = None
        mounts.append(mount)
    known_mount = all(mount is not None for mount in mounts)
    if known_mount and not all(np.allclose(mount, mounts[0], atol=1e-6) for mount in mounts):
        raise ValueError("所选帧的真实 LiDAR 安装变换不一致，不适合共享同一个外参。")
    if request.pose_mode == "provided":
        poses = _interpolate(request, frames)
        if request.initial_matrix is None: raise ValueError("外部 pose 模式需要显式提供初始相机–LiDAR 外参及方向。")
    else:
        if not known_mount:
            raise ValueError("全局地图的单位注册矩阵不是真实 LiDAR pose。请提供 sensor_lidar_to_ego、外部 pose，或明确选择 LiDAR 与 ego 重合模式。")
        primary_poses = registered_poses(store, clip)
        # The primary cloud can be a world map or another LiDAR. Recover the
        # registered ego pose first, then compose the actual calibration sensor.
        poses = {f.id: primary_poses[f.id] @ np.linalg.inv(np.asarray(f.lidar_to_ego)) @ mount
                 for f, mount in zip(frames, mounts)}
    if request.initial_matrix is not None:
        initial = np.asarray(request.initial_matrix)
        if request.initial_direction == "lidar_from_camera": initial = np.linalg.inv(initial)
    else: initial = np.asarray(camera.camera_from_ego) @ mounts[0]
    return poses, initial, mounts[0] if known_mount else None


def _line_fingerprint(lane):
    return _fingerprint({"id": lane.id, "points": lane.points, "closed": lane.closed})


def _validate_image_lane(pair, lanes, camera):
    if pair.image_lane_id is None: return  # Previously saved measured image polylines remain usable.
    lane = lanes.get(pair.image_lane_id)
    if lane is None: raise ValueError("图像投影线已删除或不再是开放折线，请重新选择配对。")
    if pair.image_reference.camera.id != camera.id: raise ValueError("图像投影线参考相机与当前相机不一致。")
    if pair.image_reference.source_line_fingerprint != _line_fingerprint(lane):
        raise ConflictError("图像投影线的三维几何已修改，请删除旧配对后重新选择；不会自动移动固定图像约束。")


def build_problem(store: Store, clip_id: str, camera_id: str, pairing: dict, request: OptimizeRequest, *, validate_split=True):
    from PIL import Image
    if pairing["revision"] != request.pairing_revision: raise ConflictError("配对已变化，请重新读取。")
    if not request.conventions_confirmed or not request.time_alignment_confirmed:
        raise ValueError("必须先确认坐标系方向、原图像素坐标和图像/pose 时间对齐。")
    clip = store.source_clip(clip_id); scene = get_scene(store, clip)
    camera = next((c for c in clip.cameras if c.id == camera_id), None)
    if camera is None: raise KeyError("未找到相机。")
    pairs = PairUpdate(revision=pairing["revision"], pairs=pairing["pairs"]).pairs
    if request.frame_ids is not None:
        if not request.frame_ids or len(set(request.frame_ids)) != len(request.frame_ids): raise ValueError("帧选择为空或重复。")
        if set(request.frame_ids) - {p.frame_id for p in pairs}: raise ValueError("所选帧没有已保存的车道线配对。")
        pairs = [p for p in pairs if p.frame_id in request.frame_ids]
    if request.mode == "single":
        if len({p.frame_id for p in pairs}) != 1: raise ValueError("单帧调试必须显式选择一个已配对帧。")
        pairs = [p.model_copy(update={"split": "train"}) for p in pairs]
    if not pairs: raise ValueError("请先按顺序选择三维真值线和二维图像线，保存至少一组配对。")
    frames = [f for f in clip.frames if f.id in {p.frame_id for p in pairs}]
    if len(frames) != len({p.frame_id for p in pairs}): raise ValueError("配对引用了不存在的帧。")
    lanes = {v.id: v for v in scene.vectors if not v.closed}
    train = {p.frame_id for p in pairs if p.split == "train"}; validation = {p.frame_id for p in pairs if p.split == "validation"}
    if validate_split and request.mode == "multi" and (len(train) < 2 or not validation):
        raise ValueError("多帧正式标定至少需要两个训练帧和一个独立验证帧。")
    if validate_split and request.mode == "single" and len(train) != 1: raise ValueError("单帧调试必须只选择一个训练帧。")
    poses, initial, mount = resolve_geometry(store, clip, camera, frames, request)
    observations, image_tokens = [], []
    for pair in pairs:
        if pair.lane_id not in lanes: raise ValueError(f"未找到世界坐标车道线 {pair.lane_id}；请先保存三维标注。")
        _validate_image_lane(pair, lanes, camera)
        if pair.image_size != (camera.width, camera.height): raise ValueError("二维线分辨率与原始相机不一致，禁止猜测缩放或裁剪。")
        pixels = np.asarray(pair.image_points)
        if (pixels < 0).any() or (pixels[:, 0] >= camera.width).any() or (pixels[:, 1] >= camera.height).any():
            raise ValueError("二维线顶点必须位于原图像素范围内。")
        frame = next(f for f in frames if f.id == pair.frame_id)
        image = _image_path(store, frame, camera.id)
        with Image.open(image) as source:
            if source.size != (camera.width, camera.height): raise ValueError("实际图像分辨率与内参声明不一致。")
        stat = image.stat(); image_tokens.append([str(image), stat.st_size, stat.st_mtime_ns])
        observations.append({**pair.model_dump(mode="json"), "world_points": lanes[pair.lane_id].model_dump(mode="json")["points"],
            "world_from_lidar": poses[frame.id].tolist(), "image_path": str(image.resolve()),
            "image_timestamp": request.image_timestamps.get(frame.id, frame.timestamp)})
    problem = {"clip_id": clip_id, "camera": camera.model_dump(mode="json"), "initial_matrix": initial.tolist(),
        "initial_input_matrix": request.initial_matrix if request.initial_matrix is not None else initial.tolist(),
        "request": request.model_dump(mode="json"),
        "mode": request.mode, "config": request.config.model_dump(), "observations": observations,
        "conventions": {"points": "world", "pose": "world_from_lidar", "extrinsic": "camera_from_lidar",
            "update": "T0 @ Exp(xi^)", "pixels": "original raw image; Brown-Conrady distortion",
            "pose_mode": request.pose_mode, "pose_source": "registered world_from_primary @ inverse(primary_to_ego) @ sensor_to_ego" if request.pose_mode != "provided" else "explicit timestamped pose samples",
            "initial_input_direction": request.initial_direction if request.initial_matrix is not None else "derived camera_from_lidar",
            "pose_interpolation": "SLERP + linear translation" if request.pose_mode == "provided" else "frame association confirmed by user"}}
    token = _fingerprint({"pairing": pairing, "request": request.model_dump(mode="json"), "camera": camera.model_dump(),
                          "scene": scene.model_dump(), "frames": [f.model_dump(exclude={"points"}) for f in frames],
                          "poses": {k: v.tolist() for k, v in poses.items()}, "images": image_tokens})
    return problem, token, mount


class LaneCalibrationManager:
    def __init__(self, store: Store, executor):
        self.store = store; self.executor = executor; self.root = store.root / "lane_calibration"
        self.root.mkdir(parents=True, exist_ok=True); self.active = {}; self.events = {}

    def directory(self, clip_id, camera_id):
        return self.root / safe_id(clip_id) / safe_id(camera_id)

    def pairing(self, clip_id, camera_id):
        path = self.directory(clip_id, camera_id) / "pairs.json"
        return json.loads(path.read_text()) if path.exists() else {"revision": 0, "pairs": []}

    def read(self, clip_id, camera_id):
        with self.store.lock:
            clip = self.store.source_clip(clip_id); scene = get_scene(self.store, clip)
            if camera_id not in {c.id for c in clip.cameras}: raise KeyError("未找到相机。")
            jobs = sorted((self.directory(clip_id, camera_id) / "jobs").glob("*/job.json"), key=lambda p: p.stat().st_mtime_ns, reverse=True)
            latest_job = next((job for path in jobs if not (job := json.loads(path.read_text())).get("discarded")), None)
            return {"runtime": runtime_status(), "pairing": self.pairing(clip_id, camera_id),
                    "lanes": [v.model_dump(mode="json") for v in scene.vectors if not v.closed],
                    "latest_job": self.application_status(latest_job) if latest_job else None,
                    "coordinate_info": {"world_lanes": "scene.vectors", "ego_pose": "ego_to_world: ego → world",
                        "camera_mount": "camera_from_ego: ego → camera", "primary_lidar_mount": "lidar_to_ego: primary points → ego",
                        "fused_map": clip.source.startswith("4d-"), "sensor_mount_available": all(
                            f.calibration_lidar_to_ego is not None or f.sensor_lidar_to_ego is not None for f in clip.frames)}}

    def save_pairs(self, clip_id, camera_id, value: PairUpdate):
        with self.store.lock:
            state = self.read(clip_id, camera_id)
            if self.active: raise ConflictError("优化运行中，请先取消或等待完成再修改配对。")
            if state["pairing"]["revision"] != value.revision: raise ConflictError("配对版本冲突，请重新读取。")
            clip = self.store.source_clip(clip_id); camera = next(c for c in clip.cameras if c.id == camera_id)
            lanes = {v.id: v for v in get_scene(self.store, clip).vectors if not v.closed}; frames = {f.id for f in clip.frames}
            previous = {p["id"]: LanePair.model_validate(p) for p in state["pairing"]["pairs"]}
            for pair in value.pairs:
                if pair.frame_id not in frames or pair.lane_id not in lanes: raise ValueError("配对必须引用已保存的帧与世界坐标三维线。")
                _validate_image_lane(pair, lanes, camera)
                old = previous.get(pair.id)
                if old and old.image_lane_id is not None and any(getattr(old, name) != getattr(pair, name)
                        for name in ("image_points", "image_lane_id", "image_reference", "image_world_range")):
                    raise ValueError("已保存的图像约束保持固定；如需更换参考，请删除旧配对并重新选择两条线。")
                if pair.image_size != (camera.width, camera.height): raise ValueError("配对必须使用原图分辨率。")
                pixels = np.asarray(pair.image_points)
                if (pixels < 0).any() or (pixels[:, 0] >= camera.width).any() or (pixels[:, 1] >= camera.height).any(): raise ValueError("图像线超出原图。")
                resample(pair.image_points, 10)
            document = {"revision": value.revision + 1, "pairs": [p.model_dump(mode="json") for p in value.pairs]}
            atomic_json(self.directory(clip_id, camera_id) / "pairs.json", document); return document

    def pair_from_lanes(self, clip_id, camera_id, value: ProjectedLaneSelection):
        """Freeze a selected image-derived world lane in its reference image.

        Both IDs are explicit user choices. Geometry has no camera provenance,
        so the current frame/camera is the declared source. Its effective saved
        projection (including frame overrides) is captured once, never optimized.
        """
        from PIL import Image
        with self.store.lock:
            current = self.pairing(clip_id, camera_id)
            if current["revision"] != value.revision: raise ConflictError("配对版本冲突，请重新读取。")
            if self.active: raise ConflictError("优化运行中，请先取消或等待完成再修改配对。")
            clip = self.store.clip(clip_id)
            frame = next((f for f in clip.frames if f.id == value.frame_id), None)
            camera = next((c for c in clip.cameras if c.id == camera_id), None)
            if frame is None or camera is None: raise ValueError("请选择存在的帧和相机。")
            lanes = {v.id: v for v in get_scene(self.store, clip).vectors if not v.closed}
            if value.lane_id not in lanes or value.image_lane_id not in lanes:
                raise ValueError("两次选择都必须引用已保存的世界坐标开放车道线。")
            if value.lane_id == value.image_lane_id: raise ValueError("真值线与图像投影线不能选择同一条线。")
            with Image.open(_image_path(self.store, frame, camera.id)) as image:
                if image.size != (camera.width, camera.height): raise ValueError("实际图像分辨率与内参声明不一致。")
            lane = lanes[value.image_lane_id]
            pose = registered_poses(self.store, clip)[frame.id]
            matrix = np.asarray(_base_matrix(frame, camera))
            world = resample(lane.points, 512, *value.image_world_range)
            primary = (world - pose[:3, 3]) @ pose[:3, :3]
            pixels, depth = project_camera_points(primary, matrix, np.asarray(camera.intrinsics), np.asarray(camera.distortion))
            visible = (np.isfinite(pixels).all(1) & (depth > .05) & (pixels[:, 0] >= 0) & (pixels[:, 0] < camera.width)
                       & (pixels[:, 1] >= 0) & (pixels[:, 1] < camera.height))
            indices = np.flatnonzero(visible)
            if len(indices) < 2: raise ValueError("图像投影线在当前帧/相机中没有足够的可见线段，请检查来源和取样范围。")
            if np.any(np.diff(indices) != 1): raise ValueError("图像投影线有不连续的可见区间，请缩小其取样范围，避免跨过不可见区域连线。")
            reference = ImageLaneReference(frame_id=frame.id, camera=camera,
                world_from_primary=pose.tolist(), camera_from_primary=matrix.tolist(),
                world_range=value.image_world_range, source_line_fingerprint=_line_fingerprint(lane))
            pair = LanePair(**value.model_dump(exclude={"revision"}), image_points=pixels[visible].tolist(),
                            image_size=(camera.width, camera.height), image_reference=reference)
            pairs = [LanePair.model_validate(p) for p in current["pairs"]]
            pairs = [p.model_copy(update={"split": value.split}) if p.frame_id == value.frame_id else p for p in pairs]
            return self.save_pairs(clip_id, camera_id, PairUpdate(revision=value.revision, pairs=[*pairs, pair]))

    def discard(self, clip_id, camera_id, value: PairUpdate):
        with self.store.lock:
            if value.pairs: raise ValueError("重新选择必须清空所有配对。")
            state = self.read(clip_id, camera_id)
            if state["pairing"]["revision"] != value.revision: raise ConflictError("配对版本冲突，请重新读取。")
            if self.active: raise ConflictError("请先取消正在运行的优化，等待结束后再丢弃。")
            document = {"revision": value.revision + 1, "pairs": []}
            # Never touch calibration or scene annotations. Invalidate every old
            # candidate so reopening the panel cannot resurrect an earlier job.
            for path in (self.directory(clip_id, camera_id) / "jobs").glob("*/job.json"):
                job = json.loads(path.read_text())
                job.update(discarded=True, result=None, can_apply=False)
                atomic_json(path, job)
            atomic_json(self.directory(clip_id, camera_id) / "pairs.json", document)
            return self.read(clip_id, camera_id)

    def preview(self, clip_id, camera_id, request: OptimizeRequest):
        with self.store.lock:
            problem, token, _ = build_problem(self.store, clip_id, camera_id, self.pairing(clip_id, camera_id), request, validate_split=False)
        groups = prepare(problem)
        return {"initial_matrix": problem["initial_matrix"], "conventions": problem["conventions"],
                "statistics": statistics(problem, groups, np.asarray(problem["initial_matrix"])), "source_fingerprint": token}

    def job_path(self, clip_id, camera_id, job_id):
        return self.directory(clip_id, camera_id) / "jobs" / safe_id(job_id)

    def get_job(self, clip_id, camera_id, job_id):
        path = self.job_path(clip_id, camera_id, job_id) / "job.json"
        if not path.is_file(): raise KeyError("未找到优化任务。")
        return self.application_status(json.loads(path.read_text()))

    @staticmethod
    def application_status(job):
        # Quality metrics inform the user's review, including for older jobs
        # saved when validation improvement was a mandatory acceptance gate.
        job["can_apply"] = (job["status"] == "completed" and not job.get("discarded")
            and job.get("result") is not None and job.get("sensor_lidar_to_ego") is not None
            and job["request"].get("pose_mode") != "provided" and job["request"].get("mode") != "single")
        return job

    def start(self, clip_id, camera_id, request: OptimizeRequest):
        with self.store.lock:
            if not runtime_status()["available"]: raise ValueError(runtime_status()["detail"])
            if self.active: raise ConflictError("已有车道线优化任务正在运行。")
            problem, token, mount = build_problem(self.store, clip_id, camera_id, self.pairing(clip_id, camera_id), request)
            prepare(problem)
            job_id = uuid.uuid4().hex; directory = self.job_path(clip_id, camera_id, job_id)
            job = {"id": job_id, "clip_id": clip_id, "camera_id": camera_id, "status": "queued", "iteration": 0,
                "created_at": datetime.now(timezone.utc).isoformat(), "source_fingerprint": token,
                "request": request.model_dump(mode="json"), "result": None, "applied": False}
            atomic_json(directory / "job.json", job)
            self.active[job_id] = True; self.events[job_id] = threading.Event()
            try: self.executor.submit(self.run, job, problem, mount)
            except RuntimeError:
                self.active.pop(job_id); self.events.pop(job_id); job.update(status="failed", error="服务正在关闭，请重新启动任务。")
                atomic_json(directory / "job.json", job); raise
            return job

    def run(self, job, problem, mount):
        directory = self.job_path(job["clip_id"], job["camera_id"], job["id"])
        def save(): atomic_json(directory / "job.json", job)
        try:
            job["status"] = "running"; save()
            def progress(row):
                if row["iteration"] % 10 == 0:
                    with self.store.lock: job.update(iteration=row["iteration"], loss=row["loss"]); save()
            result = optimize(problem, progress, self.events[job["id"]].is_set)
            if self.events[job["id"]].is_set(): raise OptimizationCancelled()
            write_outputs(problem, result, directory)
            with self.store.lock:
                if self.events[job["id"]].is_set(): raise OptimizationCancelled()
                _, current, _ = build_problem(self.store, job["clip_id"], job["camera_id"],
                    self.pairing(job["clip_id"], job["camera_id"]), OptimizeRequest.model_validate(job["request"]))
                if current != job["source_fingerprint"]: raise ConflictError("求解期间标注、pose、内参或图像已变化；请重新运行。")
                job.update(status="completed", result={k: v for k, v in result.items() if k != "history"},
                    can_apply=problem["mode"] != "single" and mount is not None and job["request"]["pose_mode"] != "provided",
                    apply_block_reason=("外部同步 Pose 尚未写入平台，结果仅供导出，避免应用时使用不同位姿。" if job["request"]["pose_mode"] == "provided"
                        else "缺少真实 LiDAR 安装变换，结果仅供导出。" if mount is None else ""),
                    sensor_lidar_to_ego=mount.tolist() if mount is not None else None)
                save()
        except OptimizationCancelled:
            with self.store.lock: job.update(status="cancelled", error="用户取消；原始外参保持不变。"); save()
        except Exception as exc:
            with self.store.lock: job.update(status="failed", error=str(exc)[:1800]); save()
        finally:
            with self.store.lock: self.active.pop(job["id"], None); self.events.pop(job["id"], None)

    def cancel(self, clip_id, camera_id, job_id):
        with self.store.lock:
            job = self.get_job(clip_id, camera_id, job_id)
            if job_id in self.events: self.events[job_id].set(); job["cancel_requested"] = True
            atomic_json(self.job_path(clip_id, camera_id, job_id) / "job.json", job)
            return job

    def apply(self, clip_id, camera_id, job_id, value: ApplyRequest):
        with self.store.lock:
            if not value.confirm_shared_update: raise ValueError("请明确确认将共享外参应用到整个 clip 的当前相机。")
            job = self.get_job(clip_id, camera_id, job_id)
            if job.get("discarded"): raise ConflictError("候选已丢弃，请重新选择配对并优化。")
            if job.get("applied"): return job
            if job["status"] != "completed" or not job.get("can_apply"): raise ValueError("候选尚未完成、仅供调试或缺少可应用的坐标变换；可导出检查。")
            _, current, mount = build_problem(self.store, clip_id, camera_id, self.pairing(clip_id, camera_id), OptimizeRequest.model_validate(job["request"]))
            if current != job["source_fingerprint"]: raise ConflictError("数据或配对已变化，禁止应用旧结果。")
            clip = self.store.source_clip(clip_id)
            cameras = {c.id: c.camera_from_ego for c in clip.cameras}
            cameras[camera_id] = (np.asarray(job["result"]["optimized_matrix"]) @ np.linalg.inv(mount)).tolist()
            save_calibration(self.store, clip_id, CalibrationUpdate(revision=_revision(clip), cameras=cameras,
                lidar_to_ego={f.id: f.lidar_to_ego for f in clip.frames}), reset_projection_cameras={camera_id})
            job["applied"] = True; atomic_json(self.job_path(clip_id, camera_id, job_id) / "job.json", job); return job

    def recover(self):
        for path in self.root.glob("*/*/jobs/*/job.json"):
            job = json.loads(path.read_text())
            if job["status"] in {"queued", "running"}:
                job.update(status="failed", error="服务重启中断优化，请重新运行；原始外参未改变。")
                atomic_json(path, job)

    def shutdown(self):
        for event in self.events.values(): event.set()


def mount_lane_calibration_routes(app: FastAPI, manager: LaneCalibrationManager):
    route = "/api/clips/{clip_id}/lane-calibration/{camera_id}"
    def respond(function, *args):
        try: return function(*args)
        except KeyError as exc: raise HTTPException(404, str(exc)) from exc
        except ConflictError as exc: raise HTTPException(409, str(exc)) from exc
        except (ValueError, OSError) as exc: raise HTTPException(422, str(exc)) from exc

    @app.get(route)
    def state(clip_id: str, camera_id: str): return respond(manager.read, clip_id, camera_id)

    @app.put(route + "/pairs")
    def pairs(clip_id: str, camera_id: str, value: PairUpdate): return respond(manager.save_pairs, clip_id, camera_id, value)

    @app.post(route + "/pairs/from-lanes")
    def pair_from_lanes(clip_id: str, camera_id: str, value: ProjectedLaneSelection):
        return respond(manager.pair_from_lanes, clip_id, camera_id, value)

    @app.post(route + "/preview")
    def preview(clip_id: str, camera_id: str, value: OptimizeRequest): return respond(manager.preview, clip_id, camera_id, value)

    @app.post(route + "/discard")
    def discard(clip_id: str, camera_id: str, value: PairUpdate): return respond(manager.discard, clip_id, camera_id, value)

    @app.post(route + "/jobs", status_code=202)
    def start(clip_id: str, camera_id: str, value: OptimizeRequest): return respond(manager.start, clip_id, camera_id, value)

    @app.get(route + "/jobs/{job_id}")
    def job(clip_id: str, camera_id: str, job_id: str): return respond(manager.get_job, clip_id, camera_id, job_id)

    @app.post(route + "/jobs/{job_id}/cancel")
    def cancel(clip_id: str, camera_id: str, job_id: str): return respond(manager.cancel, clip_id, camera_id, job_id)

    @app.post(route + "/jobs/{job_id}/apply")
    def apply(clip_id: str, camera_id: str, job_id: str, value: ApplyRequest): return respond(manager.apply, clip_id, camera_id, job_id, value)

    @app.get(route + "/jobs/{job_id}/download")
    def download(clip_id: str, camera_id: str, job_id: str):
        job = respond(manager.get_job, clip_id, camera_id, job_id)
        if job.get("discarded"): raise HTTPException(409, "候选已丢弃，请重新选择配对并优化。")
        if job["status"] != "completed": raise HTTPException(409, "任务尚未完成，没有完整结果包。")
        directory = manager.job_path(clip_id, camera_id, job_id); archive = directory / "results.zip"
        with manager.store.lock:
            if not archive.exists():
                temporary = directory / "results.tmp"
                with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as output:
                    for path in directory.rglob("*"):
                        if path.is_file() and path.name != "job.json" and not any(part.startswith('.') for part in path.relative_to(directory).parts) and path.suffix in {".npy", ".txt", ".csv", ".json", ".yaml", ".png"} and path.resolve().is_relative_to(directory.resolve()):
                            output.write(path, str(path.relative_to(directory)))
                temporary.replace(archive)
        return FileResponse(archive, media_type="application/zip", filename=f"{clip_id}-{camera_id}-lane-calibration.zip")
