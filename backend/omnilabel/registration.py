"""Robust CPU registration; sampling here never reduces the exported source map.

This is local registration around supplied poses, not global SLAM. A balanced
voxel sample avoids favouring the ordering of LiDAR rings or packet files.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _xyz(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 3 or not np.isfinite(values[:, :3]).all():
        raise ValueError("ICP point arrays must contain finite XYZ coordinates")
    return values[:, :3]


def spatial_sample(points: np.ndarray, voxel_size: float, max_points: int = 15000) -> np.ndarray:
    """Voxel centroids followed by deterministic spatial-hash sampling.

    Only the registration solver uses this sample. It is independent of scan
    packet order and does not truncate the original point data or fused map.
    A zero point budget means all occupied registration cells.
    """
    xyz = _xyz(points)
    size = float(voxel_size)
    if not np.isfinite(size) or size <= 0 or max_points < 0:
        raise ValueError("Registration voxel size must be positive and its sample budget nonnegative")
    if not len(xyz):
        return np.empty((0, 3), dtype=np.float64)
    scaled = np.floor(xyz / size)
    if not np.isfinite(scaled).all() or np.any(np.abs(scaled) >= 2**52):
        raise ValueError("Registration coordinates exceed the safe voxel-index range")
    cells, inverse, counts = np.unique(scaled.astype(np.int64), axis=0, return_inverse=True, return_counts=True)
    centroids = np.column_stack([np.bincount(inverse, weights=xyz[:, axis]) / counts for axis in range(3)])
    if max_points and len(centroids) > max_points:
        # Hash whole occupied cells, rather than taking a prefix/linear sample
        # of a ring-ordered input. Fixed uint64 arithmetic makes this repeatable.
        unsigned = cells.astype(np.uint64)
        hashed = ((unsigned[:, 0] * np.uint64(0x9E3779B185EBCA87)) ^
                  (unsigned[:, 1] * np.uint64(0xC2B2AE3D27D4EB4F)) ^
                  (unsigned[:, 2] * np.uint64(0x165667B19E3779F9)))
        hashed ^= hashed >> np.uint64(29)
        selected = np.lexsort((cells[:, 2], cells[:, 1], cells[:, 0], hashed))[:max_points]
        # Keep a canonical spatial order after choosing a balanced subset.
        centroids = centroids[np.sort(selected)]
    return centroids


def _supported(points: np.ndarray) -> bool:
    if len(points) < 30:
        return False
    singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    return len(singular) >= 2 and singular[1] > max(1e-6, singular[0] * 1e-4)


def _transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _rigid_delta(source: np.ndarray, target: np.ndarray, distances: np.ndarray,
                 scale_floor: float) -> np.ndarray:
    # Huber weights bound the influence of residual movers/outliers that survive
    # the correspondence cutoff and trimming, while fitting actual measurements.
    huber = max(scale_floor, float(np.median(distances)) * 1.5)
    weights = np.minimum(1.0, huber / np.maximum(distances, 1e-12))
    weights /= weights.sum()
    center_source = np.sum(source * weights[:, None], axis=0)
    center_target = np.sum(target * weights[:, None], axis=0)
    covariance = (source - center_source).T @ ((target - center_target) * weights[:, None])
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    delta = np.eye(4)
    delta[:3, :3] = rotation
    delta[:3, 3] = center_target - rotation @ center_source
    return delta


def register_multiscale(sourceXYZ: np.ndarray, targetXYZ: np.ndarray, initial: np.ndarray,
                        base_voxel: float = .1, max_distance: float = 3,
                        max_points: int = 15000, iterations: int = 45) -> tuple[np.ndarray, dict]:
    """Align one scan to a world submap with three trimmed/Huber ICP stages.

    Coarse registration increases capture range; the final smaller voxel and
    correspondence window improve surface alignment. Explicit overlap/RMSE gates
    reject unrelated or degenerate geometry instead of claiming a valid map.
    """
    source, target = _xyz(sourceXYZ), _xyz(targetXYZ)
    matrix = np.array(initial, dtype=np.float64, copy=True)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all() or
            not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7) or
            not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5) or
            not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-5)):
        raise ValueError("ICP initial transform must be a finite right-handed rigid matrix")
    if (not np.isfinite(base_voxel) or base_voxel <= 0 or not np.isfinite(max_distance) or
            max_distance <= 0 or (0 < max_points < 30) or max_points < 0 or iterations < 1):
        raise ValueError("ICP resolution, correspondence range and iterations must be positive; sample budget at least 30")
    if not _supported(source) or not _supported(target):
        raise ValueError("ICP needs at least 30 non-collinear static points in source and submap")
    final_distance = min(max_distance, max(.075, base_voxel * 2))
    scales = [(base_voxel * 4, max_distance),
              (base_voxel * 2, max(final_distance, max_distance * .5)),
              (base_voxel * .5, final_distance)]
    stages = []
    total_iterations = 0
    final_source = final_target = None
    for voxel, distance_limit in scales:
        sampled_source = spatial_sample(source, voxel, max_points)
        sampled_target = spatial_sample(target, voxel, max_points)
        if not _supported(sampled_source) or not _supported(sampled_target):
            # Sparse scans can have fewer than 30 coarse cells. Keep all fine
            # registration detail rather than pretending the scan is invalid.
            sampled_source = spatial_sample(source, min(voxel, base_voxel * .5), max_points)
            sampled_target = spatial_sample(target, min(voxel, base_voxel * .5), max_points)
        if not _supported(sampled_source) or not _supported(sampled_target):
            raise ValueError("ICP registration sample has fewer than 30 non-collinear cells")
        tree = cKDTree(sampled_target)
        converged = False
        for step in range(iterations):
            transformed = _transform(sampled_source, matrix)
            distances, indices = tree.query(transformed, k=1, workers=1)
            nearby = np.flatnonzero(distances <= distance_limit)
            if len(nearby) < 30:
                raise ValueError("ICP scan/submap overlap is insufficient (fewer than 30 nearby correspondences)")
            order = nearby[np.argsort(distances[nearby], kind="stable")]
            keep = order[:max(30, int(len(order) * .85))]
            a, b = transformed[keep], sampled_target[indices[keep]]
            if not _supported(a) or not _supported(b):
                raise ValueError("ICP correspondence geometry is degenerate")
            delta = _rigid_delta(a, b, distances[keep], voxel * .05)
            matrix = delta @ matrix
            total_iterations += 1
            angle = float(np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1)))
            if np.linalg.norm(delta[:3, 3]) < 1e-5 and angle < 1e-5:
                converged = True
                break
        distances, _ = tree.query(_transform(sampled_source, matrix), k=1, workers=1)
        valid = distances <= distance_limit
        fitness = float(valid.mean())
        rmse = float(np.sqrt(np.mean(distances[valid] ** 2))) if valid.any() else float("inf")
        stages.append({"voxel_size": voxel, "correspondence_distance": distance_limit,
                       "source_points": len(sampled_source), "target_points": len(sampled_target),
                       "fitness": fitness, "rmse": rmse, "iterations": step + 1, "converged": converged})
        final_source, final_target = sampled_source, sampled_target
    final = stages[-1]
    if final["fitness"] < .25 or not np.isfinite(final["rmse"]) or final["rmse"] > final_distance * .6:
        raise ValueError(f"ICP quality rejected: overlap={final['fitness']:.3f}, inlier RMSE={final['rmse']:.3f}m; improve input poses and scan overlap")
    correction = matrix @ np.linalg.inv(np.asarray(initial))
    return matrix, {"fitness": final["fitness"], "rmse": final["rmse"],
                    "iterations": total_iterations, "converged": final["converged"],
                    "correspondence_distance": final_distance,
                    "source_points": len(final_source), "target_points": len(final_target),
                    "stages": stages, "estimator": "multiscale trimmed Huber point-to-point ICP",
                    "correction_translation": float(np.linalg.norm(correction[:3, 3])),
                    "correction_rotation_degrees": float(np.degrees(np.arccos(np.clip((np.trace(correction[:3, :3]) - 1) / 2, -1, 1))))}
