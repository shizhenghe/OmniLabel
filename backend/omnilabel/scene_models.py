"""World-space 4D scene annotations, separate from legacy per-frame labels."""
from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from .models import Clip, Identifier, Model, Vec3, Vector


class SceneKeyframe(Model):
    frame_id: Identifier
    center: Vec3
    size: Vec3
    yaw: float

    @field_validator("size")
    @classmethod
    def positive_size(cls, value):
        if any(d <= 0 or d > 1000 for d in value):
            raise ValueError("scene box dimensions must be positive and at most 1000 metres")
        return value


class SceneTrack(Model):
    id: Identifier
    label: str = Field(min_length=1, max_length=80)
    mode: Literal["static", "dynamic"] = "dynamic"
    keyframes: list[SceneKeyframe] = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def unique_frames(self):
        if len({k.frame_id for k in self.keyframes}) != len(self.keyframes):
            raise ValueError("a track may have only one keyframe per frame")
        return self


class SceneAnnotations(Model):
    revision: int = Field(default=0, ge=0)
    coordinate_frame: Literal["world"] = "world"
    vectors: list[Vector] = Field(default_factory=list, max_length=5000)
    tracks: list[SceneTrack] = Field(default_factory=list, max_length=5000)

    @model_validator(mode="after")
    def unique_identifiers(self):
        ids = [t.id for t in self.tracks] + [v.id for v in self.vectors]
        if len(set(ids)) != len(ids):
            raise ValueError("track and vector IDs must be unique within a scene")
        if sum(len(t.keyframes) for t in self.tracks) > 50_000:
            raise ValueError("a scene may contain at most 50000 box keyframes")
        return self

    def validate_for_clip(self, clip: Clip):
        frames = {f.id for f in clip.frames}
        if any(k.frame_id not in frames for t in self.tracks for k in t.keyframes):
            raise ValueError("scene keyframe references a frame not present in this clip")
        return self


class ReconstructionRequest(Model):
    voxel_size: float = Field(default=0.3, ge=0.1, le=5)
    registration: Literal["poses", "icp"] = "poses"
    exclude_dynamic: bool = True
    preserve_detail: bool = True
    detail_voxel_size: float = Field(default=0.05, ge=0.05, le=0.2)
