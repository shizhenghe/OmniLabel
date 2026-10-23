"""Validated interchange models. Coordinate conventions are in docs/contract.md."""
from __future__ import annotations

import re
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Identifier = Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}$")]
Vec3 = tuple[float, float, float]


class Model(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")


def rigid_transform(value: list[list[float]]) -> list[list[float]]:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("transform must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError("transform last row must be [0,0,0,1]")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-3) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-3):
        raise ValueError("transform rotation must be right-handed and orthonormal")
    return value


class Camera(Model):
    id: Identifier
    name: str = Field(min_length=1, max_length=120)
    width: int = Field(gt=0, le=16384)
    height: int = Field(gt=0, le=16384)
    intrinsics: list[list[float]]
    camera_from_ego: list[list[float]]
    distortion: list[float] = Field(default_factory=list, max_length=5)

    _transform = field_validator("camera_from_ego")(rigid_transform)

    @field_validator("intrinsics")
    @classmethod
    def validate_intrinsics(cls, value):
        k = np.asarray(value, dtype=float)
        if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
            raise ValueError("intrinsics must be a finite 3x3 camera matrix with positive focal lengths")
        return value

    @field_validator("distortion")
    @classmethod
    def validate_distortion(cls, value):
        if len(value) not in (0, 5):
            raise ValueError("distortion must be empty or [k1,k2,p1,p2,k3]")
        return value


class Frame(Model):
    id: Identifier
    timestamp: float
    ego_to_world: list[list[float]]
    lidar_to_ego: list[list[float]]
    points: list[list[float]] = Field(max_length=500_000)
    images: dict[str, str] = Field(default_factory=dict)

    _transforms = field_validator("ego_to_world", "lidar_to_ego")(rigid_transform)

    @field_validator("points")
    @classmethod
    def validate_points(cls, value):
        if any(len(p) not in (3, 4) for p in value):
            raise ValueError("each point must contain x,y,z or x,y,z,intensity")
        return [p if len(p) == 4 else [*p, 0.0] for p in value]


class Clip(Model):
    id: Identifier
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=10000)
    fps: float = Field(gt=0, le=1000)
    cameras: list[Camera] = Field(min_length=1, max_length=32)
    frames: list[Frame] = Field(min_length=1, max_length=2000)
    source: str = Field(default="import", max_length=200)

    @model_validator(mode="after")
    def validate_clip(self):
        camera_ids = [c.id for c in self.cameras]
        frame_ids = [f.id for f in self.frames]
        if len(set(camera_ids)) != len(camera_ids) or len(set(frame_ids)) != len(frame_ids):
            raise ValueError("camera IDs and frame IDs must be unique")
        if any(set(f.images) - set(camera_ids) for f in self.frames):
            raise ValueError("frame images reference an unknown camera")
        if any(self.frames[i].timestamp >= self.frames[i + 1].timestamp for i in range(len(self.frames) - 1)):
            raise ValueError("frame timestamps must be strictly increasing")
        if sum(len(f.points) for f in self.frames) > 2_000_000:
            raise ValueError("interactive clip limit is 2 million points; split larger clips")
        return self


class Box(Model):
    id: Identifier
    track_id: Identifier
    label: str = Field(min_length=1, max_length=80)
    center: Vec3
    size: Vec3
    yaw: float

    @field_validator("size")
    @classmethod
    def positive_size(cls, value):
        if any(x <= 0 or x > 1000 for x in value):
            raise ValueError("box dimensions must be positive and at most 1000 metres")
        return value


class Vector(Model):
    id: Identifier
    label: str = Field(min_length=1, max_length=80)
    points: list[Vec3] = Field(min_length=2, max_length=20000)
    closed: bool = False

    @model_validator(mode="after")
    def valid_polygon(self):
        if self.closed and len(self.points) < 3:
            raise ValueError("closed vectors need at least three vertices")
        return self


class ImagePolygon(Model):
    id: Identifier
    camera_id: Identifier
    label: str = Field(min_length=1, max_length=80)
    points: list[tuple[float, float]] = Field(min_length=3, max_length=20000)


class Annotations(Model):
    boxes: list[Box] = Field(default_factory=list, max_length=5000)
    vectors: list[Vector] = Field(default_factory=list, max_length=5000)
    image_polygons: list[ImagePolygon] = Field(default_factory=list, max_length=5000)
    point_labels: dict[str, str] = Field(default_factory=dict)
    revision: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_identifiers(self):
        ids = [o.id for objects in (self.boxes, self.vectors, self.image_polygons) for o in objects]
        if len(set(ids)) != len(ids):
            raise ValueError("annotation IDs must be unique within a frame")
        if len({b.track_id for b in self.boxes}) != len(self.boxes):
            raise ValueError("each track can have only one box per frame")
        if any(not re.fullmatch(r"0|[1-9][0-9]*", k) for k in self.point_labels):
            raise ValueError("point labels must use non-negative integer point indices")
        if any(not label or len(label) > 80 for label in self.point_labels.values()):
            raise ValueError("point labels must be nonempty and at most 80 characters")
        return self

    def validate_for_frame(self, frame: Frame, cameras: list[Camera]):
        camera_map = {c.id: c for c in cameras}
        if any(int(index) >= len(frame.points) for index in self.point_labels):
            raise ValueError("point label index is outside this frame")
        for polygon in self.image_polygons:
            camera = camera_map.get(polygon.camera_id)
            if camera is None:
                raise ValueError("image polygon references an unknown camera")
            if any(not (0 <= x <= camera.width and 0 <= y <= camera.height) for x, y in polygon.points):
                raise ValueError("image polygon vertex is outside image bounds")
        return self


class ProjectRequest(Model):
    frame_id: Identifier
    camera_id: Identifier
    stride: int = Field(default=1, ge=1, le=1000)


class InferenceOptions(Model):
    camera_id: Identifier | None = None
    label: str = Field(default="object", min_length=1, max_length=80)
    points: list[tuple[float, float]] = Field(default_factory=list, max_length=256)
    point_labels: list[Literal[0, 1]] = Field(default_factory=list, max_length=256)
    box: tuple[float, float, float, float] | None = None
    score_threshold: float = Field(default=0.3, ge=0, le=1)
    max_masks: int = Field(default=64, ge=1, le=256)

    @model_validator(mode="after")
    def validate_prompts(self):
        if self.point_labels and len(self.points) != len(self.point_labels):
            raise ValueError("SAM2 points and point_labels must have matching lengths")
        if self.box and (self.box[2] <= self.box[0] or self.box[3] <= self.box[1]):
            raise ValueError("SAM2 box must be [left,top,right,bottom] with positive area")
        return self


class JobRequest(Model):
    kind: Literal["map", "occupancy", "propagate", "ground-segmentation", "prelabel"]
    frame_id: Identifier | None = None
    voxel_size: float = Field(default=0.5, ge=0.1, le=5)
    target_frame_id: Identifier | None = None
    model_id: str = Field(default="", max_length=96)
    prompt: str = Field(default="", max_length=2000)
    options: InferenceOptions | None = None
    scene_mode: bool = False

    @model_validator(mode="after")
    def validate_model(self):
        if self.kind == "prelabel" and not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}", self.model_id):
            raise ValueError("prelabel requires a short alphanumeric model_id supported by your configured inference worker")
        return self
