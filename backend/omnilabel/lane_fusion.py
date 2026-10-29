"""Conservative, read-only fusion of overlapping 3D lane polylines.

This deliberately requires measured overlap: nearby endpoints, crossings and
diverging branches are not enough to identify one road marking. The first
vector's geometry wins in the shared interval, preserving reviewed vertices;
later observations can extend either end. All calculations use float64 in a
local origin so world/UTM coordinates retain their precision.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians

import numpy as np
from pydantic import Field, model_validator

from .models import Identifier, Model, Vector


LANE_LABELS = frozenset({"lane", "lane_line", "lane_marking", "lane_solid", "lane_dashed"})


class LaneMergeRequest(Model):
    vectors: list[Vector] = Field(max_length=5000)
    max_distance: float = Field(default=0.25, ge=0.01, le=1.0)
    max_angle_degrees: float = Field(default=15.0, ge=1.0, le=30.0)
    min_overlap: float = Field(default=1.0, ge=0.2, le=20.0)
    scene_revision: int | None = Field(default=None, ge=0)
    candidate_ids: list[Identifier] | None = Field(default=None, max_length=5000)

    @model_validator(mode="after")
    def unique_vectors_and_candidates(self):
        identifiers = {v.id for v in self.vectors}
        if len(identifiers) != len(self.vectors):
            raise ValueError("lane merge vector IDs must be unique")
        if self.candidate_ids is not None and not set(self.candidate_ids) <= identifiers:
            raise ValueError("candidate_ids must reference vectors in the preview")
        return self


class LaneMergeGroup(Model):
    kept_id: Identifier
    source_ids: list[Identifier]


class LaneMergeReport(Model):
    input_count: int
    output_count: int
    merged_count: int
    groups: list[LaneMergeGroup]


class LaneMergePreview(Model):
    vectors: list[Vector]
    report: LaneMergeReport


@dataclass
class _Path:
    points: np.ndarray
    steps: np.ndarray
    lengths: np.ndarray
    stations: np.ndarray

    @property
    def length(self):
        return self.stations[-1]


def _path(points: np.ndarray) -> _Path | None:
    # Consecutive repeated vertices must not create NaN tangents.
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-8]
    points = points[keep]
    if len(points) < 2:
        return None
    steps = np.diff(points, axis=0)
    lengths = np.linalg.norm(steps, axis=1)
    return _Path(points, steps, lengths, np.r_[0., np.cumsum(lengths)])


def _sample(path: _Path, step: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stations = np.linspace(0., path.length, max(2, int(np.ceil(path.length / step)) + 1))
    indices = np.minimum(np.searchsorted(path.stations, stations, side="right") - 1, len(path.steps) - 1)
    fraction = (stations - path.stations[indices]) / path.lengths[indices]
    return (path.points[indices] + fraction[:, None] * path.steps[indices],
            path.steps[indices] / path.lengths[indices, None], stations)


def _project(points: np.ndarray, target: _Path):
    """Exact nearest segment projection, bounded intermediate array sizes."""
    count = len(points)
    distance = np.full(count, np.inf)
    station = np.zeros(count)
    direction = np.zeros((count, 3))
    for start in range(0, count, 256):
        sample = points[start:start + 256]
        best = np.full(len(sample), np.inf)
        best_segment = np.zeros(len(sample), dtype=int)
        best_fraction = np.zeros(len(sample))
        for offset in range(0, len(target.steps), 1024):
            steps = target.steps[offset:offset + 1024]
            lengths = target.lengths[offset:offset + 1024]
            delta = sample[:, None, :] - target.points[None, offset:offset + len(steps), :]
            fraction = np.clip(np.einsum("ijk,jk->ij", delta, steps) / lengths[None, :] ** 2, 0., 1.)
            residual = delta - fraction[:, :, None] * steps[None, :, :]
            squared = np.einsum("ijk,ijk->ij", residual, residual)
            nearest = squared.argmin(axis=1)
            selected = squared[np.arange(len(sample)), nearest]
            improve = selected < best
            best[improve] = selected[improve]
            best_segment[improve] = nearest[improve] + offset
            best_fraction[improve] = fraction[np.arange(len(sample)), nearest][improve]
        chosen = slice(start, start + len(sample))
        distance[chosen] = np.sqrt(best)
        station[chosen] = target.stations[best_segment] + best_fraction * target.lengths[best_segment]
        direction[chosen] = target.steps[best_segment] / target.lengths[best_segment, None]
    return distance, station, direction


def _overlap(source: _Path, target: _Path, request: LaneMergeRequest):
    sample, tangent, stations = _sample(source, min(0.25, request.min_overlap / 4))
    distance, projected, target_tangent = _project(sample, target)
    aligned = np.einsum("ij,ij->i", tangent, target_tangent) >= cos(radians(request.max_angle_degrees))
    near = distance <= request.max_distance + 1e-8
    good = near & aligned
    # A branch can share a substantial prefix then diverge. Unmatched vertices
    # are allowed only past the other polyline's endpoints, never alongside its
    # interior. Test both directions before accepting a pair.
    interior = (projected > request.max_distance) & (projected < target.length - request.max_distance)
    if np.any(interior & ~good):
        return None
    indices = np.flatnonzero(good)
    if len(indices) < 2:
        return None
    # A single continuous, order-preserving correspondence prevents a crossing
    # or a looping path from stitching unrelated stretches together.
    if np.any(np.diff(indices) != 1) or np.any(np.diff(projected[indices]) < -1e-6):
        return None
    length = stations[indices[-1]] - stations[indices[0]]
    if length + 1e-6 < request.min_overlap:
        return None
    return indices


def _merge_pair(first: Vector, second: Vector, request: LaneMergeRequest) -> Vector | None:
    origin = np.asarray(first.points[0], dtype=np.float64)
    primary = _path(np.asarray(first.points, dtype=np.float64) - origin)
    other = _path(np.asarray(second.points, dtype=np.float64) - origin)
    if primary is None or other is None:
        return None
    # Determine direction from the closest portion, not from global endpoint
    # headings (a valid curved lane may bend by more than 90 degrees).
    sample, tangent, _ = _sample(other, min(0.25, request.min_overlap / 4))
    distance, _, projected_tangent = _project(sample, primary)
    close = distance <= request.max_distance + 1e-8
    if not np.any(close):
        return None
    alignment = np.einsum("ij,ij->i", tangent[close], projected_tangent[close])
    if np.median(alignment) < 0:
        other = _path(other.points[::-1])
    if _overlap(other, primary, request) is None or _overlap(primary, other, request) is None:
        return None
    endpoint_distance, endpoint_station, _ = _project(primary.points[[0, -1]], other)
    if np.any(endpoint_station[1:] < endpoint_station[:-1] - 1e-6):
        return None
    # Retain every reviewed primary vertex. Only add the portions of the new
    # observation extending past its endpoints, with a single joining segment.
    before = other.points[:0]
    after = other.points[:0]
    if endpoint_distance[0] <= request.max_distance + 1e-8 and endpoint_station[0] > 1e-6:
        before = other.points[other.stations < endpoint_station[0] - 1e-6]
    if endpoint_distance[1] <= request.max_distance + 1e-8 and endpoint_station[1] < other.length - 1e-6:
        after = other.points[other.stations > endpoint_station[1] + 1e-6]
    points = np.concatenate([before, primary.points, after], axis=0) + origin
    if len(points) > 20000:
        # Do not silently decimate authored vertices to fit the interchange
        # schema; leave these two very large annotations independently editable.
        return None
    return first.model_copy(update={"points": [tuple(point) for point in points.tolist()]})


def merge_lanes(request: LaneMergeRequest) -> LaneMergePreview:
    vectors = list(request.vectors)
    sources = [[v.id] for v in vectors]
    candidates = set(request.candidate_ids) if request.candidate_ids is not None else None
    # Expanded axis-aligned boxes discard spatially unrelated pairs before the
    # more expensive exact segment checks. No point cloud is loaded or sampled.
    lows = np.asarray([np.min(v.points, axis=0) for v in vectors], dtype=float).reshape(-1, 3)
    highs = np.asarray([np.max(v.points, axis=0) for v in vectors], dtype=float).reshape(-1, 3)
    changed = True
    while changed:
        changed = False
        for i, first in enumerate(vectors):
            if first.closed or first.label.lower() not in LANE_LABELS:
                continue
            while True:
                # Vectorize the broad phase: thousands of unrelated vectors
                # should not incur a Python/NumPy call for every possible pair.
                spatial = (np.all(lows[i + 1:] <= highs[i] + request.max_distance, axis=1)
                           & np.all(highs[i + 1:] >= lows[i] - request.max_distance, axis=1))
                merged_any = False
                for j in np.flatnonzero(spatial) + i + 1:
                    second = vectors[j]
                    eligible = (not second.closed and second.label == first.label
                                and (candidates is None or candidates.intersection(sources[i] + sources[j])))
                    merged = _merge_pair(first, second, request) if eligible else None
                    if merged is None:
                        continue
                    vectors[i] = first = merged
                    sources[i].extend(sources.pop(j))
                    vectors.pop(j)
                    lows[i], highs[i] = np.min(merged.points, axis=0), np.max(merged.points, axis=0)
                    lows, highs = np.delete(lows, j, axis=0), np.delete(highs, j, axis=0)
                    changed = merged_any = True
                    break  # Re-query after extension and index removal.
                if not merged_any:
                    break
    groups = [LaneMergeGroup(kept_id=v.id, source_ids=ids) for v, ids in zip(vectors, sources) if len(ids) > 1]
    return LaneMergePreview(vectors=vectors, report=LaneMergeReport(
        input_count=len(request.vectors), output_count=len(vectors),
        merged_count=len(request.vectors) - len(vectors), groups=groups,
    ))
