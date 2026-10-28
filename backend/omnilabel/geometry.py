"""Deterministic CPU geometry; no learned predictions are fabricated here.

All annotations and point coordinates are in the frame's LiDAR coordinate
system. Sparse occupancy omits unknown cells and reports free-ray truncation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
from math import atan2
import tempfile
from uuid import uuid4

import numpy as np


MAX_POINTS = None  # Backward-compatible name; no source point-count ceiling.
MAX_OCCUPIED_VOXELS = 150_000
MAX_FREE_VOXELS = 150_000
MAX_RAYS = 12_000
MAX_RAY_VISITS = 750_000
MAX_BOX_COMPARISONS = 50_000_000
DYNAMIC_LABELS = {
    "car", "truck", "bus", "vehicle", "pedestrian", "person", "bicycle",
    "cyclist", "motorcycle", "motorcyclist", "trailer", "construction_vehicle",
}


def _points(frame: dict, point_loader=None) -> np.ndarray:
    raw = point_loader(frame) if point_loader is not None else frame.get("points", [])
    # The interchange accepts both XYZ and XYZI rows in one frame.
    if isinstance(raw, np.ndarray):
        points = np.asarray(raw, dtype=np.float64)
    else:
        points = np.asarray([p if len(p) == 4 else [*p, 0] for p in raw], dtype=np.float64)
    if not points.size:
        return np.empty((0, 4), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] not in (3, 4):
        raise ValueError("Points must be an Nx3 or Nx4 array")
    if not np.isfinite(points).all():
        raise ValueError("Points must be finite")
    if points.shape[1] == 3:
        points = np.column_stack((points, np.zeros(len(points))))
    return points


def _matrix(value: list, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError(f"{name} must be a homogeneous rigid transform")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-3) or not np.isclose(
        np.linalg.det(rotation), 1, atol=1e-3
    ):
        raise ValueError(f"{name} rotation must be orthonormal and right-handed")
    return matrix


def _world_from_lidar(frame: dict) -> np.ndarray:
    return _matrix(frame["ego_to_world"], "ego_to_world") @ _matrix(
        frame["lidar_to_ego"], "lidar_to_ego"
    )


def _transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _voxel_size(value: float) -> float:
    size = float(value)
    if not np.isfinite(size) or not 0.05 <= size <= 5:
        raise ValueError("voxel_size must be between 0.05 and 5 metres")
    return size


def _voxel_indices(xyz: np.ndarray, size: float) -> np.ndarray:
    # Prevent NumPy's float->int overflow from silently merging unrelated cells.
    # Beyond 2**52, half-cell centers are not representable in float64 either.
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = np.floor(xyz / size)
    if not np.isfinite(scaled).all() or np.any(np.abs(scaled) >= 2**52):
        raise ValueError("Point coordinates exceed the safe voxel-index range")
    return scaled.astype(np.int64)


def _inside_box(points: np.ndarray, box: dict) -> np.ndarray:
    center = np.asarray(box["center"], dtype=float)
    size = np.asarray(box["size"], dtype=float)
    yaw = float(box["yaw"])
    delta = points[:, :3] - center
    cosine, sine = np.cos(yaw), np.sin(yaw)
    local_x = cosine * delta[:, 0] + sine * delta[:, 1]
    local_y = -sine * delta[:, 0] + cosine * delta[:, 1]
    return (
        (np.abs(local_x) <= size[0] / 2)
        & (np.abs(local_y) <= size[1] / 2)
        & (np.abs(delta[:, 2]) <= size[2] / 2)
    )


def project_points(
    frame: dict, camera: dict, annotations: dict | None = None, stride: int = 1
) -> dict:
    """Brown-Conrady projection with positive-depth/frustum clipping.

    This is a geometric projection, not an occlusion z-buffer. `index` always
    refers to the original, unsampled LiDAR array, allowing semantic edits.
    """
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be a positive integer")
    points = _points(frame)
    indices = np.arange(0, len(points), stride)
    direct = (frame.get("projection_camera_from_lidar", {}).get(camera["id"])
              or frame.get("annotation_camera_from_lidar", {}).get(camera["id"]))
    camera_from_lidar = _matrix(direct, "camera_from_lidar") if direct is not None else (
        _matrix(camera["camera_from_ego"], "camera_from_ego") @ _matrix(frame["lidar_to_ego"], "lidar_to_ego")
    )
    xyz = _transform(points[indices, :3], camera_from_lidar)
    forward = xyz[:, 2] > 1e-6
    xyz, indices = xyz[forward], indices[forward]
    width, height = int(camera["width"]), int(camera["height"])
    intrinsic = np.asarray(camera["intrinsics"], dtype=float)
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError("intrinsics must be a finite 3x3 matrix")
    distortion = list(camera.get("distortion") or [])
    if len(distortion) not in (0, 4, 5):
        raise ValueError("distortion must have 0, 4 or 5 Brown-Conrady coefficients")
    distortion += [0.0] * (5 - len(distortion))
    k1, k2, p1, p2, k3 = distortion
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        x, y = xyz[:, 0] / xyz[:, 2], xyz[:, 1] / xyz[:, 2]
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        distorted_x = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        distorted_y = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        pixels = np.column_stack((distorted_x, distorted_y, np.ones(len(x)))) @ intrinsic.T
        uv = pixels[:, :2] / pixels[:, 2:3]
    visible = (
        np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= 0) & (uv[:, 0] < width)
        & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    )
    labels = (annotations or {}).get("point_labels", {})
    projected = []
    for index, pixel, depth in zip(indices[visible], uv[visible], xyz[visible, 2]):
        point = {"index": int(index), "u": float(pixel[0]), "v": float(pixel[1]), "depth": float(depth)}
        if str(index) in labels:
            point["label"] = labels[str(index)]
        projected.append(point)
    return {"points": projected, "width": width, "height": height, "occlusion_filtering": False}


def _point_chunks(frame: dict, point_loader=None, chunk_size: int = 100_000):
    """Read all source points, with bounded temporary arrays and stable indices.

    A memmapped external scan stays on disk until its current chunk is needed.
    Chunk size controls working memory, never how many input points are accepted.
    """
    raw = point_loader(frame) if point_loader is not None else frame.get("points", [])
    for start in range(0, len(raw), chunk_size):
        values = raw[start:start + chunk_size]
        if isinstance(values, np.ndarray):
            points = np.asarray(values, dtype=np.float64)
        else:
            points = np.asarray([p if len(p) == 4 else [*p, 0] for p in values], dtype=np.float64)
        if points.ndim != 2 or points.shape[1] not in (3, 4) or not np.isfinite(points).all():
            raise ValueError("Points must be finite numeric Nx3 or Nx4 arrays")
        if points.shape[1] == 3:
            points = np.column_stack((points, np.zeros(len(points))))
        yield start, points


def _statistical_inliers(points: np.ndarray, neighbors: int, std_ratio: float) -> np.ndarray:
    from scipy.spatial import cKDTree
    xyz = np.asarray(points[:, :3] if isinstance(points, np.ndarray) else [p[:3] for p in points], dtype=np.float64)
    if not np.isfinite(xyz).all():
        raise ValueError("Points must be finite")
    if len(xyz) <= neighbors:
        return np.ones(len(xyz), dtype=bool)
    tree = cKDTree(xyz)
    distances = np.empty(len(xyz))
    for start in range(0, len(xyz), 25_000):
        values, _ = tree.query(xyz[start:start + 25_000], k=neighbors + 1, workers=1)
        distances[start:start + len(values)] = values[:, 1:].mean(axis=1)
    threshold = float(distances.mean() + std_ratio * distances.std())
    return distances <= threshold


def _static_chunks(clip: dict, annotations_by_frame: dict, point_loader=None, *,
                   outlier_filter=False, outlier_neighbors=16, outlier_std_ratio=3.0):
    for frame in clip["frames"]:
        annotations = annotations_by_frame.get(frame["id"], {})
        labels = annotations.get("point_labels", {})
        raw = point_loader(frame) if point_loader is not None else frame.get("points", [])
        inliers = _statistical_inliers(raw, outlier_neighbors, outlier_std_ratio) if outlier_filter and len(raw) else None
        dynamic_indices = np.asarray([int(index) for index, label in labels.items()
                                      if str(index).isdigit() and label.lower() in DYNAMIC_LABELS], dtype=np.int64)
        transform = _world_from_lidar(frame)
        for start, points in _point_chunks(frame, lambda _: raw):
            remove = np.zeros(len(points), dtype=bool)
            for box in annotations.get("boxes", []):
                if box["label"].lower() in DYNAMIC_LABELS:
                    remove |= _inside_box(points, box)
            selected_dynamic = dynamic_indices[(dynamic_indices >= start) & (dynamic_indices < start + len(points))]
            remove[selected_dynamic - start] = True
            removed_dynamic = int(remove.sum())
            removed_outlier = 0
            if inliers is not None:
                rejected = ~inliers[start:start + len(points)]
                removed_outlier = int((rejected & ~remove).sum())
                remove |= rejected
            original_indices = np.flatnonzero(~remove) + start
            static = points[~remove].copy()
            static[:, :3] = _transform(static[:, :3], transform)
            if not np.isfinite(static).all():
                raise ValueError("Transformed point coordinates must remain finite")
            point_labels = [labels.get(str(index), "unknown") for index in original_indices]
            yield static, point_labels, len(points), removed_dynamic, removed_outlier


def fuse_map(clip: dict, annotations_by_frame: dict, voxel_size: float = 0.15, *,
             preserve_detail: bool = False, detail_voxel_size: float = 0.05,
             point_loader=None, fusion_mode: str | None = None,
             return_array: bool = False, outlier_filter: bool = False,
             outlier_neighbors: int = 16, outlier_std_ratio: float = 3.0) -> dict:
    """Fuse every scan in chunks; no input/output point-count ceiling is used.

    full keeps every measured static point. detail retains the brightest measured
    sample in each fine voxel, surface picks the measured sample nearest its voxel
    centroid, and centroid averages measurements. Surface does not invent a bright
    centroid or move a measured intensity onto a different surface point.
    """
    size = _voxel_size(voxel_size)
    detail_size = float(detail_voxel_size)
    if not np.isfinite(detail_size) or not 0.02 <= detail_size <= 5:
        raise ValueError("detail_voxel_size must be between 0.02 and 5 metres")
    mode = fusion_mode or ("detail" if preserve_detail else "centroid")
    if mode not in ("full", "detail", "surface", "centroid"):
        raise ValueError("fusion_mode must be full, detail, surface or centroid")
    effective_size = 0.0 if mode == "full" else min(size, detail_size) if mode in ("detail", "surface") else size
    if outlier_neighbors < 2 or not np.isfinite(outlier_std_ratio) or outlier_std_ratio <= 0:
        raise ValueError("Outlier neighbor count must be at least 2 and standard-deviation ratio positive")
    filtering = dict(outlier_filter=outlier_filter, outlier_neighbors=outlier_neighbors,
                     outlier_std_ratio=outlier_std_ratio)
    cells = {}
    semantic_labels = {}
    input_count = excluded_count = output_count = outlier_count = 0
    minimum, maximum = float("inf"), -float("inf")
    # Spooling full-resolution output avoids retaining a copy of every frame and
    # then concatenating the whole clip. Only the final result occupies RAM.
    with tempfile.TemporaryFile() as spool:
        for static, labels, count, excluded, outliers in _static_chunks(clip, annotations_by_frame, point_loader, **filtering):
            input_count += count
            excluded_count += excluded
            outlier_count += outliers
            if not len(static):
                continue
            minimum = min(minimum, float(static[:, 3].min()))
            maximum = max(maximum, float(static[:, 3].max()))
            if mode == "full":
                static.tofile(spool)
                semantic_labels.update({str(output_count + i): label for i, label in enumerate(labels)
                                        if label not in ("unknown", "unlabeled")})
                output_count += len(static)
                continue
            keys, inverse, counts = np.unique(_voxel_indices(static[:, :3], effective_size), axis=0,
                                               return_inverse=True, return_counts=True)
            if mode == "detail":
                order = np.lexsort((np.arange(len(static)), -static[:, 3], inverse))
                selected = order[np.r_[True, np.diff(inverse[order]) != 0]]
                for key, index in zip(keys, selected):
                    cell = tuple(int(v) for v in key)
                    previous = cells.get(cell)
                    if previous is None or static[index, 3] > previous[0][3]:
                        cells[cell] = (static[index].copy(), labels[index])
            else:
                sums = np.column_stack([np.bincount(inverse, weights=static[:, axis]) for axis in range(4)])
                for key, summed, count in zip(keys, sums, counts):
                    cell = tuple(int(v) for v in key)
                    previous = cells.get(cell)
                    if previous is None:
                        cells[cell] = [summed.copy(), int(count), Counter()]
                    else:
                        previous[0] += summed
                        previous[1] += int(count)
                for key, label in zip(_voxel_indices(static[:, :3], effective_size), labels):
                    if label not in ("unknown", "unlabeled"):
                        cells[tuple(int(v) for v in key)][2][label] += 1
        if mode == "full":
            spool.seek(0)
            downsampled = np.fromfile(spool, dtype=np.float64, count=output_count * 4).reshape((-1, 4))
        else:
            ordered = sorted(cells)
            downsampled = np.empty((len(ordered), 4), dtype=np.float64)
            if mode == "detail":
                for index, key in enumerate(ordered):
                    point, label = cells[key]
                    downsampled[index] = point
                    if label not in ("unknown", "unlabeled"):
                        semantic_labels[str(index)] = label
            else:
                for index, key in enumerate(ordered):
                    summed, count, votes = cells[key]
                    downsampled[index] = summed / count
                    if votes:
                        semantic_labels[str(index)] = sorted(votes, key=lambda label: (-votes[label], label))[0]
                if mode == "surface":
                    # A second streamed pass selects an actual sample nearest
                    # each final centroid. No high-reflectance outlier preference.
                    positions = {key: index for index, key in enumerate(ordered)}
                    distances = np.full(len(ordered), np.inf)
                    centroids = downsampled[:, :3].copy()
                    for static, labels, _, _, _ in _static_chunks(clip, annotations_by_frame, point_loader, **filtering):
                        if not len(static):
                            continue
                        keys = _voxel_indices(static[:, :3], effective_size)
                        for point, key, label in zip(static, keys, labels):
                            index = positions[tuple(int(v) for v in key)]
                            distance = float(np.sum((point[:3] - centroids[index]) ** 2))
                            if distance < distances[index]:
                                distances[index] = distance
                                downsampled[index] = point
                                if label not in ("unknown", "unlabeled"):
                                    semantic_labels[str(index)] = label
                                else:
                                    semantic_labels.pop(str(index), None)
    reductions = {"full": "all-measured-points", "detail": "brightest-measured-point",
                  "surface": "nearest-measured-surface-point", "centroid": "centroid-mean"}
    has_points = bool(len(downsampled))
    return {
        "points": downsampled if return_array else downsampled.tolist(), "frame_count": len(clip["frames"]),
        "point_count": len(downsampled), "input_point_count": input_count,
        "excluded_dynamic_points": excluded_count, "excluded_outlier_points": outlier_count,
        "outlier_filter": outlier_filter, "voxel_size": size,
        "fusion_mode": mode, "preserve_detail": mode in ("detail", "surface"), "detail_voxel_size": detail_size,
        "effective_voxel_size": effective_size, "point_reduction": reductions[mode],
        "intensity_available": bool(has_points and (minimum != 0 or maximum != 0)),
        "intensity_has_contrast": bool(has_points and maximum - minimum > 1e-8),
        "intensity_range": [minimum, maximum] if has_points else None,
        "point_labels": semantic_labels, "coordinate_frame": "world",
        "input_truncated": False, "output_truncated": False,
        "method": "pose-aligned streamed static fusion",
    }


def _ray_cells(endpoint: np.ndarray, voxel_size: float):
    """Amanatides-Woo traversal; endpoint cell excluded, tied axes step together."""
    target = tuple(int(value) for value in np.floor(endpoint / voxel_size))
    cell = [0, 0, 0]
    direction = endpoint.tolist()
    step = [1 if value > 0 else -1 if value < 0 else 0 for value in direction]
    t_delta = [voxel_size / abs(value) if value else float("inf") for value in direction]
    t_max = [(voxel_size if value > 0 else 0) / value if value else float("inf") for value in direction]
    # Radius <= 150 m, min voxel 0.05 m means < 9003 crossed cells.
    for _ in range(10_000):
        if tuple(cell) == target:
            return
        yield tuple(cell)
        next_t = min(t_max)
        if next_t > 1:
            return
        for axis in range(3):
            if abs(t_max[axis] - next_t) <= 1e-12:
                cell[axis] += step[axis]
                t_max[axis] += t_delta[axis]


def build_occupancy(frame: dict, annotations: dict, voxel_size: float = 0.5) -> dict:
    """Observed sparse occupancy with semantic endpoint votes and ray free space.

    Unobserved cells are absent, never silently treated as free. All in-range
    endpoints are used. Rays are deterministically sampled/budgeted for CPU
    responsiveness and any truncation is explicit in result metadata.
    """
    size = _voxel_size(voxel_size)
    points = _points(frame)
    with np.errstate(over="ignore", invalid="ignore"):
        distances = np.linalg.norm(points[:, :3], axis=1)
    in_range = distances <= 150
    source_indices = np.flatnonzero(in_range)
    points = points[in_range]
    if len(points) * len(annotations.get("boxes", [])) > MAX_BOX_COMPARISONS:
        raise ValueError("Too many box/point comparisons; reduce annotations or split the frame")
    indices = np.floor(points[:, :3] / size).astype(np.int64)
    labels = annotations.get("point_labels", {})
    point_classes = [labels.get(str(index), "unknown") for index in source_indices]
    for box in annotations.get("boxes", []):
        for index in np.flatnonzero(_inside_box(points, box)):
            if point_classes[index] == "unknown":
                point_classes[index] = box["label"]
    votes = defaultdict(Counter)
    for index, label in zip(indices, point_classes):
        key = tuple(int(value) for value in index)
        votes[key][label] += 1
        if len(votes) > MAX_OCCUPIED_VOXELS:
            raise ValueError("Too many occupied voxels; increase voxel_size or split the clip")
    occupied = {}
    for key, counts in votes.items():
        # Explicit semantic votes take precedence over unlabelled measurements.
        semantic = {label: count for label, count in counts.items() if label not in ("unknown", "unlabeled")}
        candidates = semantic or counts
        occupied[key] = sorted(candidates, key=lambda label: (-candidates[label], label))[0]
    ray_indices = np.linspace(0, len(points) - 1, min(len(points), MAX_RAYS), dtype=int) if len(points) else []
    free = set()
    visits = traced_rays = 0
    budget_exhausted = False
    for index in ray_indices:
        completed = True
        for key in _ray_cells(points[index, :3], size):
            visits += 1
            if key not in occupied:
                free.add(key)
            if visits >= MAX_RAY_VISITS or len(free) >= MAX_FREE_VOXELS:
                budget_exhausted = True
                completed = False
                break
        traced_rays += int(completed)
        if budget_exhausted:
            break
    voxels = [
        {"index": list(index), "center": [(value + 0.5) * size for value in index],
         "state": "occupied", "label": label}
        for index, label in sorted(occupied.items())
    ]
    voxels.extend(
        {"index": list(index), "center": [(value + 0.5) * size for value in index],
         "state": "free", "label": "free"}
        for index in sorted(free)
    )
    return {
        "voxels": voxels, "voxel_size": size, "coordinate_frame": "lidar",
        "occupied_count": len(occupied), "free_count": len(free),
        "unknown_policy": "Unobserved cells are omitted, not free",
        "max_range_metres": 150, "out_of_range_points": int((~in_range).sum()),
        "total_rays": len(points), "traced_rays": traced_rays,
        "ray_sampling": len(ray_indices) < len(points),
        "truncated": budget_exhausted or len(ray_indices) < len(points),
    }


def segment_ground(frame: dict, annotations: dict) -> dict:
    """Suggest ground/obstacle labels using a robust low-seed ground plane.

    This simple geometric baseline is useful for editing, not learned semantic
    segmentation. Existing per-point labels are preserved as human annotations.
    """
    points = _points(frame)
    output = deepcopy(annotations)
    labels = output.setdefault("point_labels", {})
    if not len(points):
        return output
    xyz = points[:, :3]
    coarse = _voxel_indices(xyz[:, :2], 2)
    _, inverse = np.unique(coarse, axis=0, return_inverse=True)
    # Lowest point per horizontal tile avoids vehicle roofs dominating seeds.
    order = np.lexsort((xyz[:, 2], inverse))
    first = np.r_[True, inverse[order][1:] != inverse[order][:-1]]
    seeds = xyz[order[first]]
    floor_height = float(np.percentile(xyz[:, 2], 15))
    seeds = seeds[np.abs(seeds[:, 2] - floor_height) < 0.8]
    plane = np.array([0.0, 0.0, floor_height])
    if len(seeds) >= 3:
        for _ in range(3):
            design = np.column_stack((seeds[:, :2], np.ones(len(seeds))))
            candidate, _, rank, _ = np.linalg.lstsq(design, seeds[:, 2], rcond=None)
            if rank < 3 or np.linalg.norm(candidate[:2]) > np.tan(np.deg2rad(20)):
                break
            plane = candidate
            residual = np.abs(design @ plane - seeds[:, 2])
            seeds = seeds[residual < 0.25]
            if len(seeds) < 3:
                break
    ground_z = xyz[:, 0] * plane[0] + xyz[:, 1] * plane[1] + plane[2]
    is_ground = np.abs(xyz[:, 2] - ground_z) <= 0.2
    for index, ground in enumerate(is_ground):
        labels.setdefault(str(index), "ground" if ground else "obstacle")
    return output


def propagate_annotations(
    source_frame: dict, target_frame: dict, annotations: dict,
    target_annotations: dict | None = None,
) -> dict:
    """Copy tracks/vectors using ego-motion compensation, preserving target edits.

    This assumes static world position between frames. It is not object tracking
    or motion prediction. Upright boxes retain size and use projected yaw.
    Camera masks and LiDAR point-index labels cannot safely propagate this way.
    """
    transform = np.linalg.inv(_world_from_lidar(target_frame)) @ _world_from_lidar(source_frame)
    output = deepcopy(target_annotations) if target_annotations is not None else {
        "boxes": [], "vectors": [], "image_polygons": [], "point_labels": {}, "revision": 0,
    }
    tracks = {box["track_id"] for box in output.setdefault("boxes", [])}
    for box in annotations.get("boxes", []):
        if box["track_id"] in tracks:
            continue
        copied = deepcopy(box)
        copied["id"] = uuid4().hex
        copied["center"] = _transform(np.asarray([box["center"]]), transform)[0].tolist()
        heading = transform[:3, :3] @ np.array([np.cos(box["yaw"]), np.sin(box["yaw"]), 0])
        copied["yaw"] = atan2(heading[1], heading[0])
        output["boxes"].append(copied)
        tracks.add(copied["track_id"])
    # Matching IDs identify already copied vectors. Avoid duplicate propagation.
    vector_ids = {vector["id"] for vector in output.setdefault("vectors", [])}
    all_ids = {
        obj["id"] for key in ("boxes", "vectors", "image_polygons")
        for obj in output.get(key, [])
    }
    for vector in annotations.get("vectors", []):
        propagated_id = "prop-" + sha256(
            f"{source_frame['id']}:{target_frame['id']}:{vector['id']}".encode()
        ).hexdigest()[:32]
        if vector["id"] in vector_ids or propagated_id in vector_ids:
            continue
        copied = deepcopy(vector)
        if copied["id"] in all_ids:
            copied["id"] = propagated_id
        copied["points"] = _transform(np.asarray(vector["points"]), transform).tolist()
        output["vectors"].append(copied)
        vector_ids.add(copied["id"])
        all_ids.add(copied["id"])
    return output
