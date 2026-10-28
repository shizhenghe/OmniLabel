"""Per-frame camera registration with robust 3D/2D correspondences.

The persisted transform maps the frame's *original* point coordinates into
the optical camera.  It is a projection override, never a mutation of scans,
poses or annotations.  One atomic document is the commit for each frame and
camera; Store.clip overlays these documents when serving a clip.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from pydantic import Field, field_validator, model_validator
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation

from .calibration_input import calibration_input
from .models import Camera, Clip, Frame, Identifier, Model, rigid_transform
from .storage import ConflictError, Store, atomic_json, safe_id


MIN_CORRESPONDENCES = 6
INLIER_THRESHOLD_PX = 6.0
MAX_CORRESPONDENCES = 256


class Correspondence(Model):
    id: Identifier
    point: tuple[float, float, float]
    pixel: tuple[float, float]
    enabled: bool = True


class ProjectionSolveRequest(Model):
    revision: str = Field(min_length=1, max_length=128)
    correspondences: list[Correspondence] = Field(max_length=MAX_CORRESPONDENCES)
    initial_matrix: list[list[float]] | None = None

    @field_validator("initial_matrix")
    @classmethod
    def valid_matrix(cls, value):
        return rigid_transform(value) if value is not None else None


class ProjectionAutoRequest(Model):
    revision: str = Field(min_length=1, max_length=128)
    initial_matrix: list[list[float]] | None = None

    @field_validator("initial_matrix")
    @classmethod
    def valid_matrix(cls, value):
        return rigid_transform(value) if value is not None else None


class ProjectionUpdate(Model):
    revision: str = Field(min_length=1, max_length=128)
    matrix: list[list[float]] | None
    correspondences: list[Correspondence] = Field(default_factory=list, max_length=MAX_CORRESPONDENCES)
    method: str = Field(default="correspondences", min_length=1, max_length=80)
    # Accepted for API compatibility. Quality is recomputed by the server.
    metrics: dict[str, Any] | None = None

    @field_validator("matrix")
    @classmethod
    def valid_matrix(cls, value):
        return rigid_transform(value) if value is not None else None

    @model_validator(mode="after")
    def unique_correspondences(self):
        _unique_ids(self.correspondences)
        return self


def _camera_dict(camera: Camera | dict) -> dict:
    return camera.model_dump() if isinstance(camera, Camera) else camera


def project_camera_points(points: np.ndarray, matrix: np.ndarray, intrinsics: np.ndarray,
                          distortion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Brown-Conrady projection, including skew; returns raw pixels and depth.

    No image clipping or occlusion filtering is applied: an outside-image
    correspondence must still contribute its actual reprojection residual.
    """
    points = np.asarray(points, dtype=np.float64)
    camera = points @ np.asarray(matrix)[:3, :3].T + np.asarray(matrix)[:3, 3]
    depths = camera[:, 2]
    z = np.where(np.abs(depths) < 1e-8, np.where(depths < 0, -1e-8, 1e-8), depths)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        x, y = camera[:, 0] / z, camera[:, 1] / z
        if len(distortion):
            k1, k2, p1, p2, k3 = distortion
            r2 = x*x + y*y
            radial = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
            xd = x*radial + 2*p1*x*y + p2*(r2 + 2*x*x)
            yd = y*radial + p1*(r2 + 2*y*y) + 2*p2*x*y
            x, y = xd, yd
        normalized = np.column_stack((x, y, np.ones(len(x))))
        pixels = (normalized @ np.asarray(intrinsics).T)[:, :2]
    return pixels, depths


def _as_correspondences(values: list[dict | Correspondence]) -> list[Correspondence]:
    if len(values) > MAX_CORRESPONDENCES:
        raise ValueError(f"最多支持 {MAX_CORRESPONDENCES} 组对应点。")
    pairs = [value if isinstance(value, Correspondence) else Correspondence.model_validate(value) for value in values]
    _unique_ids(pairs)
    return pairs


def _unique_ids(pairs: list[Correspondence]):
    if len({pair.id for pair in pairs}) != len(pairs):
        raise ValueError("对应点编号重复，请删除重复编号后重新求解。")


def _enabled_arrays(camera: Camera | dict, values: list[dict | Correspondence]):
    camera = _camera_dict(camera)
    pairs = _as_correspondences(values)
    active = [pair for pair in pairs if pair.enabled]
    if len(active) < MIN_CORRESPONDENCES:
        raise ValueError("至少需要 6 组启用的三维点和图像对应点，建议选择 8～12 组。")
    xyz = np.asarray([pair.point for pair in active], dtype=float)
    uv = np.asarray([pair.pixel for pair in active], dtype=float)
    if np.max(np.abs(xyz)) > 1e12:
        raise ValueError("三维坐标超出可安全求解的范围，请确认坐标单位为米。")
    if np.any(uv[:, 0] < 0) or np.any(uv[:, 0] > camera["width"]) or np.any(uv[:, 1] < 0) or np.any(uv[:, 1] > camera["height"]):
        raise ValueError("对应点的图像位置超出原始图像范围，请使用原始像素坐标。")
    distances3 = np.linalg.norm(xyz[:, None] - xyz[None, :], axis=2)
    distances2 = np.linalg.norm(uv[:, None] - uv[None, :], axis=2)
    np.fill_diagonal(distances3, np.inf)
    np.fill_diagonal(distances2, np.inf)
    if np.any(distances3 < 1e-5) or np.any(distances2 < 0.25):
        raise ValueError("存在重复或过近的对应点，请选择不同位置的明确特征。")
    singular3 = np.linalg.svd(xyz - xyz.mean(axis=0), compute_uv=False)
    if singular3[0] < 1e-5 or singular3[1] / singular3[0] < 1e-4:
        raise ValueError("三维对应点几乎共线，无法可靠求解。请加入其它车道、路缘或杆体上的点。")
    singular2 = np.linalg.svd(uv - uv.mean(axis=0), compute_uv=False)
    try:
        area = ConvexHull(uv).volume
    except QhullError:
        area = 0
    span = np.ptp(uv, axis=0)
    if (singular2[1] / max(singular2[0], 1e-12) < .005
            or span[0] < max(4., camera["width"] * .015)
            or span[1] < max(4., camera["height"] * .015)
            or area < max(16., camera["width"] * camera["height"] * .0002)):
        raise ValueError("图像对应点过于集中或接近一条直线，请在图像左右、远近位置分散选点。")
    return pairs, active, xyz, uv


def _matrix_from_delta(delta: np.ndarray, initial: np.ndarray, translation_scale: float) -> np.ndarray:
    adjustment = np.eye(4)
    adjustment[:3, :3] = Rotation.from_rotvec(delta[:3]).as_matrix()
    adjustment[:3, 3] = delta[3:] * translation_scale
    return adjustment @ initial


def _fit(xyz: np.ndarray, uv: np.ndarray, camera: dict, initial: np.ndarray,
         *, robust: bool = True, max_nfev: int = 180) -> np.ndarray:
    k, distortion = np.asarray(camera["intrinsics"]), np.asarray(camera.get("distortion", []))
    camera_xyz = xyz @ initial[:3, :3].T + initial[:3, 3]
    scale = max(1., float(np.median(np.linalg.norm(camera_xyz, axis=1))))

    def residual(delta):
        matrix = _matrix_from_delta(delta, initial, scale)
        pixels, depth = project_camera_points(xyz, matrix, k, distortion)
        raw = np.nan_to_num(pixels - uv, nan=1e8, posinf=1e8, neginf=-1e8)
        # The penalty avoids fitting the mathematically valid mirror behind
        # the camera without hiding the true pixel residual in the metrics.
        behind = depth <= 1e-4
        raw[behind] += (1e3 + np.minimum(np.abs(depth[behind]), 1e4))[:, None]
        return np.clip(raw, -1e8, 1e8).ravel()

    result = least_squares(residual, np.zeros(6), loss="soft_l1" if robust else "linear",
                           f_scale=2., x_scale="jac", max_nfev=max_nfev,
                           ftol=1e-9, xtol=1e-9, gtol=1e-8)
    return _matrix_from_delta(result.x, initial, scale)


def _undistort_pixels(uv: np.ndarray, camera: dict) -> np.ndarray:
    inverse = np.linalg.inv(np.asarray(camera["intrinsics"]))
    target = np.column_stack((uv, np.ones(len(uv)))) @ inverse.T
    estimate = target[:, :2].copy()
    distortion = camera.get("distortion", [])
    if not distortion:
        return estimate
    k1, k2, p1, p2, k3 = distortion
    # Fixed-point inverse is only an initializer; the final solve evaluates
    # the original distorted pixel observations.
    for _ in range(15):
        x, y = estimate[:, 0], estimate[:, 1]
        r2 = x*x + y*y
        radial = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
        if np.any(np.abs(radial) < 1e-6):
            break
        tangential = np.column_stack((2*p1*x*y + p2*(r2+2*x*x), p1*(r2+2*y*y) + 2*p2*x*y))
        estimate = np.clip((target[:, :2] - tangential) / radial[:, None], -100, 100)
    return estimate


def _linear_initializers(xyz: np.ndarray, uv: np.ndarray, camera: dict) -> list[np.ndarray]:
    """DLT/homography initializers, including a planar-road case.

    These are hypotheses only: cheirality, inlier support and nonlinear
    observability checks decide whether a hypothesis is usable.
    """
    center = xyz.mean(axis=0)
    scale = max(1e-6, float(np.sqrt(np.mean(np.sum((xyz-center)**2, axis=1)))))
    normalized = (xyz-center) / scale
    _, singular, basis = np.linalg.svd(normalized, full_matrices=False)
    uvn = _undistort_pixels(uv, camera)
    out = []
    if singular[2] / singular[0] > 1e-3:
        x = np.column_stack((normalized, np.ones(len(xyz))))
        rows = []
        for point, pixel in zip(x, uvn):
            rows.extend([np.r_[point, np.zeros(4), -pixel[0]*point],
                         np.r_[np.zeros(4), point, -pixel[1]*point]])
        _, _, vt = np.linalg.svd(rows, full_matrices=False)
        projection = vt[-1].reshape(3, 4)
        for sign in (1., -1.):
            m = projection * sign
            u, s, vt_rotation = np.linalg.svd(m[:, :3])
            rotation = u @ vt_rotation
            if np.linalg.det(rotation) < 0:
                continue
            translation = m[:, 3] / np.mean(s) * scale - rotation @ center
            matrix = np.eye(4); matrix[:3, :3] = rotation; matrix[:3, 3] = translation
            out.append(matrix)
    else:
        planar = normalized @ basis[:2].T
        rows = []
        for point, pixel in zip(np.column_stack((planar, np.ones(len(xyz)))), uvn):
            rows.extend([np.r_[point, np.zeros(3), -pixel[0]*point],
                         np.r_[np.zeros(3), point, -pixel[1]*point]])
        _, _, vt = np.linalg.svd(rows, full_matrices=False)
        homography = vt[-1].reshape(3, 3)
        plane_basis = np.vstack((basis[:2], np.cross(basis[0], basis[1])))
        for sign in (1., -1.):
            h = homography * sign
            norm = (np.linalg.norm(h[:, 0]) + np.linalg.norm(h[:, 1])) / 2
            r1, r2 = h[:, 0]/norm, h[:, 1]/norm
            u, _, vt_rotation = np.linalg.svd(np.column_stack((r1, r2, np.cross(r1, r2))))
            plane_rotation = u @ vt_rotation
            if np.linalg.det(plane_rotation) < 0:
                u[:, -1] *= -1
                plane_rotation = u @ vt_rotation
            rotation = plane_rotation @ plane_basis
            matrix = np.eye(4); matrix[:3, :3] = rotation
            matrix[:3, 3] = h[:, 2]/norm * scale - rotation @ center
            out.append(matrix)
    return out


def _observability(xyz: np.ndarray, camera: dict, matrix: np.ndarray) -> tuple[int, float]:
    """Scale-normalized reprojection Jacobian, independent of world origin."""
    k, distortion = np.asarray(camera["intrinsics"]), np.asarray(camera.get("distortion", []))
    camera_points = xyz @ matrix[:3, :3].T + matrix[:3, 3]
    scale = max(1., float(np.median(np.linalg.norm(camera_points, axis=1))))
    columns = []
    epsilon = 1e-6
    for axis in range(6):
        step = np.zeros(6); step[axis] = epsilon
        plus, _ = project_camera_points(xyz, _matrix_from_delta(step, matrix, scale), k, distortion)
        minus, _ = project_camera_points(xyz, _matrix_from_delta(-step, matrix, scale), k, distortion)
        columns.append(((plus-minus)/(2*epsilon)).ravel())
    jacobian = np.column_stack(columns)
    norm = np.linalg.norm(jacobian, axis=0)
    jacobian /= np.maximum(norm, 1e-12)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    condition = float(singular[0] / max(singular[-1], 1e-15))
    rank = int(np.count_nonzero(singular > singular[0] * 1e-6))
    return rank, condition


def evaluate_correspondences(camera: Camera | dict, correspondences: list[dict | Correspondence],
                             matrix: list[list[float]] | np.ndarray,
                             initial_matrix: list[list[float]] | np.ndarray) -> dict:
    """Recompute independent, raw-pixel metrics for a proposed matrix."""
    camera = _camera_dict(camera)
    pairs, active, xyz, uv = _enabled_arrays(camera, correspondences)
    matrix = np.asarray(rigid_transform(np.asarray(matrix).tolist()))
    initial = np.asarray(rigid_transform(np.asarray(initial_matrix).tolist()))
    k, distortion = np.asarray(camera["intrinsics"]), np.asarray(camera.get("distortion", []))
    projected, depth = project_camera_points(xyz, matrix, k, distortion)
    baseline, _ = project_camera_points(xyz, initial, k, distortion)
    errors = np.linalg.norm(projected-uv, axis=1)
    baseline_errors = np.linalg.norm(baseline-uv, axis=1)
    finite = np.isfinite(errors) & (depth > 1e-4)
    inliers = finite & (errors <= INLIER_THRESHOLD_PX)
    reasons = []
    required = max(MIN_CORRESPONDENCES, int(np.ceil(len(active) * .6)))
    if np.count_nonzero(inliers) < required:
        reasons.append(f"仅 {np.count_nonzero(inliers)}/{len(active)} 组点误差在 {INLIER_THRESHOLD_PX:g} 像素内，需要至少 {required} 组。")
    if np.count_nonzero(depth > 1e-4) < required:
        reasons.append("过多对应点位于相机后方，可能选错坐标系或对应位置。")
    rank, condition = (0, 1e15)
    if np.count_nonzero(inliers) >= MIN_CORRESPONDENCES:
        try:
            # An outlier must not make an otherwise collinear inlier set
            # appear observable. Repeat spread validation on the consensus.
            inlier_pairs = [pair for pair, keep in zip(active, inliers) if keep]
            _enabled_arrays(camera, inlier_pairs)
            rank, condition = _observability(xyz[inliers], camera, matrix)
            if rank < 6 or condition > 1e5:
                reasons.append("对应点对部分位姿参数约束不足，请增加不同距离、高度和方向的特征。")
        except ValueError as exc:
            reasons.append(str(exc))
    if np.any(inliers) and np.sqrt(np.mean(errors[inliers]**2)) > 4.:
        reasons.append("有效对应点仍有较大投影误差，请检查误配点并补充对应点。")
    point_errors = []
    active_by_id = {pair.id: i for i, pair in enumerate(active)}
    for pair in pairs:
        index = active_by_id.get(pair.id)
        if index is None:
            point_errors.append({"id": pair.id, "enabled": False, "error_px": None,
                                 "baseline_error_px": None, "inlier": False, "projected": None})
        else:
            point_errors.append({"id": pair.id, "enabled": True,
                                 "error_px": float(errors[index]) if np.isfinite(errors[index]) else None,
                                 "baseline_error_px": float(baseline_errors[index]) if np.isfinite(baseline_errors[index]) else None,
                                 "inlier": bool(inliers[index]),
                                 "projected": projected[index].tolist() if np.isfinite(projected[index]).all() else None})
    finite_errors = errors[np.isfinite(errors)]
    finite_baseline = baseline_errors[np.isfinite(baseline_errors)]
    metrics = {"rmse_px": float(np.sqrt(np.mean(finite_errors**2))) if len(finite_errors) else None,
               "baseline_rmse_px": float(np.sqrt(np.mean(finite_baseline**2))) if len(finite_baseline) else None,
               "median_px": float(np.median(finite_errors)) if len(finite_errors) else None,
               "p90_px": float(np.percentile(finite_errors, 90)) if len(finite_errors) else None,
               "inlier_rmse_px": float(np.sqrt(np.mean(errors[inliers]**2))) if np.any(inliers) else None,
               "inlier_count": int(np.count_nonzero(inliers)), "total_count": len(active),
               "inlier_threshold_px": INLIER_THRESHOLD_PX, "positive_depth_count": int(np.count_nonzero(depth > 1e-4)),
               "jacobian_rank": rank, "jacobian_condition": condition, "point_errors": point_errors}
    return {"matrix": matrix.tolist(), "candidate_matrix": matrix.tolist(), "metrics": metrics,
            "correspondences": [pair.model_dump() for pair in pairs], "method": "correspondences",
            "usable": not reasons, "rejection_reasons": reasons}


def solve_correspondences(camera: Camera | dict, correspondences: list[dict | Correspondence],
                          initial_matrix: list[list[float]] | np.ndarray) -> dict:
    """Robust PnP without OpenCV; never persists a candidate.

    Road-plane homography and full-3D DLT provide alternative hypotheses;
    deterministic RANSAC and a robust nonlinear fit tolerate misclicks.
    """
    camera = _camera_dict(camera)
    pairs, _, xyz, uv = _enabled_arrays(camera, correspondences)
    initial = np.asarray(rigid_transform(np.asarray(initial_matrix).tolist()))
    k, distortion = np.asarray(camera["intrinsics"]), np.asarray(camera.get("distortion", []))

    def score(matrix):
        pixels, depths = project_camera_points(xyz, matrix, k, distortion)
        errors = np.linalg.norm(pixels-uv, axis=1)
        valid = (depths > 1e-4) & np.isfinite(errors)
        inliers = valid & (errors <= INLIER_THRESHOLD_PX)
        # Support first, then a bounded robust cost. An extreme outlier cannot
        # dominate the selection amongst otherwise good camera solutions.
        cost = float(np.mean(np.minimum(np.where(valid, errors, 1e8), INLIER_THRESHOLD_PX)**2))
        return (int(np.count_nonzero(inliers)), -cost), inliers

    hypotheses = [initial]
    try:
        hypotheses.extend(_linear_initializers(xyz, uv, camera))
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        pass
    best_matrix, best_score = initial, (-1, -np.inf)
    for hypothesis in hypotheses:
        try:
            candidate = _fit(xyz, uv, camera, hypothesis)
            candidate_score, _ = score(candidate)
            if candidate_score > best_score:
                best_matrix, best_score = candidate, candidate_score
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            continue
    # Six clean points are sufficient for each hypothesis; up to 64 trials
    # cover common 10-30% correspondence contamination without random UI runs.
    if best_score[0] < len(xyz) and len(xyz) > MIN_CORRESPONDENCES:
        rng = np.random.default_rng(20261027)
        trials = min(64, max(24, len(xyz) * 3))
        for _ in range(trials):
            subset = rng.choice(len(xyz), MIN_CORRESPONDENCES, replace=False)
            try:
                candidate = _fit(xyz[subset], uv[subset], camera, initial, robust=False, max_nfev=80)
                candidate_score, _ = score(candidate)
                if candidate_score > best_score:
                    best_matrix, best_score = candidate, candidate_score
                    if best_score[0] == len(xyz):
                        break
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                continue
    for _ in range(3):
        _, inliers = score(best_matrix)
        if np.count_nonzero(inliers) < MIN_CORRESPONDENCES:
            break
        refined = _fit(xyz[inliers], uv[inliers], camera, best_matrix, robust=True)
        refined_score, _ = score(refined)
        if refined_score < best_score:
            break
        best_matrix, best_score = refined, refined_score
    return evaluate_correspondences(camera, pairs, best_matrix, initial)


def _fingerprint(value: Any) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                             separators=(",", ":")).encode()).hexdigest()


def _path(store: Store, clip_id: str, frame_id: str, camera_id: str) -> Path:
    return store.root / "projection_calibrations" / safe_id(clip_id) / safe_id(frame_id) / f"{safe_id(camera_id)}.json"


def _base_matrix(frame: Frame, camera: Camera) -> list[list[float]]:
    source_override = getattr(frame, "projection_camera_from_lidar", {}).get(camera.id)
    if source_override is not None:
        return source_override
    annotation_override = getattr(frame, "annotation_camera_from_lidar", {}).get(camera.id)
    if annotation_override is not None:
        return annotation_override
    return (np.asarray(camera.camera_from_ego) @ np.asarray(frame.lidar_to_ego)).tolist()


def _base_fingerprint(frame: Frame, camera: Camera) -> str:
    return _fingerprint({"base_matrix": _base_matrix(frame, camera), "intrinsics": camera.intrinsics,
                         "distortion": camera.distortion, "width": camera.width, "height": camera.height})


def _source_fingerprint(store: Store, frame: Frame, camera: Camera) -> str:
    def file_identity(path: Path):
        try:
            stat = path.stat()
            return {"inode": stat.st_ino, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        except OSError:
            return {"missing": True}
    reference = frame.images.get(camera.id)
    image = {"reference": reference}
    if reference and reference.startswith("/api/assets/"):
        root = (store.root / "assets").resolve()
        path = (root / reference.removeprefix("/api/assets/")).resolve()
        image.update(file_identity(path) if path.is_relative_to(root) else {"invalid": True})
    if frame.pointcloud_file:
        try:
            points = {"reference": frame.pointcloud_file, **file_identity(store.pointcloud_path(frame))}
        except ValueError:
            points = {"reference": frame.pointcloud_file, "missing": True}
    else:
        points = _fingerprint(frame.points)
    calibration = None
    if frame.calibration_pointcloud_file:
        try:
            calibration = {"reference": frame.calibration_pointcloud_file,
                           **file_identity(store.calibration_pointcloud_path(frame)),
                           "count": frame.calibration_point_count,
                           "lidar_to_ego": frame.calibration_lidar_to_ego,
                           "timestamp": frame.calibration_timestamp}
        except ValueError:
            calibration = {"reference": frame.calibration_pointcloud_file, "missing": True}
    return _fingerprint({"image": image, "points": points, "calibration_points": calibration})


def _record_matches_source(store: Store, record: dict, frame: Frame, camera: Camera) -> bool:
    # Existing documents predate source tokens; retain them until explicitly
    # recalibrated. Newly saved records stop applying if their input changes.
    return (record.get("source_fingerprint") is None
            or record["source_fingerprint"] == _source_fingerprint(store, frame, camera))


def _read_record(store: Store, clip_id: str, frame_id: str, camera_id: str) -> dict | None:
    path = _path(store, clip_id, frame_id, camera_id)
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("matrix") is not None:
        rigid_transform(value["matrix"])
    return value


def overlay_projection_calibrations(store: Store, clip: Clip) -> Clip:
    """Called by Store.clip; source manifests remain byte-for-byte untouched."""
    cameras = {camera.id: camera for camera in clip.cameras}
    frames = []
    changed = False
    for frame in clip.frames:
        matrices = dict(getattr(frame, "projection_camera_from_lidar", {}))
        directory = store.root / "projection_calibrations" / safe_id(clip.id) / safe_id(frame.id)
        if directory.is_dir():
            for path in directory.glob("*.json"):
                camera = cameras.get(path.stem)
                if camera is None:
                    continue
                record = _read_record(store, clip.id, frame.id, camera.id)
                if (record and record.get("matrix") is not None
                        and record.get("base_fingerprint") == _base_fingerprint(frame, camera)
                        and _record_matches_source(store, record, frame, camera)):
                    matrices[camera.id] = record["matrix"]
        if matrices != getattr(frame, "projection_camera_from_lidar", {}):
            frame = frame.model_copy(update={"projection_camera_from_lidar": matrices})
            changed = True
        frames.append(frame)
    return clip.model_copy(update={"frames": frames}) if changed else clip


def _source_context(store: Store, clip_id: str, frame_id: str, camera_id: str):
    path = store.root / "clips" / f"{safe_id(clip_id)}.json"
    if not path.is_file():
        raise KeyError("未找到该片段。")
    clip = Clip.model_validate_json(path.read_text(encoding="utf-8"))
    frame = next((item for item in clip.frames if item.id == frame_id), None)
    camera = next((item for item in clip.cameras if item.id == camera_id), None)
    if frame is None:
        raise KeyError("未找到该帧。")
    if camera is None:
        raise KeyError("未找到该相机。")
    return clip, frame, camera


def document_for(store: Store, clip_id: str, frame_id: str, camera_id: str, *, context=None) -> dict:
    _, frame, camera = context or _source_context(store, clip_id, frame_id, camera_id)
    record = _read_record(store, clip_id, frame_id, camera_id)
    base = _base_matrix(frame, camera)
    fingerprint = _base_fingerprint(frame, camera)
    active = bool(record and record.get("base_fingerprint") == fingerprint
                  and _record_matches_source(store, record, frame, camera))
    saved = record.get("matrix") if active else None
    try:
        from .automatic_projection import automatic_status
        status = automatic_status()
    except ImportError:
        status = {"available": False, "detail": "当前服务未安装自动图像配准模块，可使用对应点求解。"}
    has_image = camera_id in frame.images
    has_points = bool(frame.calibration_point_count or frame.point_count or frame.points)
    return {"schema_version": "1.0", "clip_id": clip_id, "frame_id": frame_id, "camera_id": camera_id,
            "revision": _fingerprint({"base_fingerprint": fingerprint, "record": record}),
            "base_matrix": base, "effective_matrix": saved if saved is not None else base,
            "saved_matrix": saved,
            "projection_matrix": saved if saved is not None else getattr(frame, "projection_camera_from_lidar", {}).get(camera_id),
            "correspondences": record.get("correspondences", []) if record else [],
            "method": record.get("method") if active else None, "metrics": record.get("metrics") if active else None,
            "updated_at": record.get("updated_at") if record else None,
            "stale": bool(record and record.get("matrix") is not None and not active),
            "stale_detail": "基础标定或源图像、点云已变化，旧修正已停用，请重新求解。" if record and not active else None,
            "status_reason": "基础标定或源图像、点云已变化，旧修正已停用，请重新求解。" if record and not active else None,
            "automatic_available": bool(status.get("available") and has_image and has_points),
            "automatic_detail": status.get("detail", "") if has_image and has_points else "当前帧需要原始点云和相机图像才能自动配准。",
            "units": {"translation": "metres", "pixel": "original image pixels"},
            "convention": "column vectors; optical camera_from_original_frame_points"}


def _check_revision(document: dict, revision: str):
    if revision != document["revision"]:
        raise ConflictError("该帧相机的投影标定已变化，请重新载入后求解或保存。")


def _image_path(store: Store, frame: Frame, camera_id: str) -> Path:
    reference = frame.images.get(camera_id, "")
    if not reference.startswith("/api/assets/"):
        raise ValueError("自动配准需要导入平台的本地图像，请先导入包含图像的片段。")
    root = (store.root / "assets").resolve()
    path = (root / reference.removeprefix("/api/assets/")).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("当前帧的原始相机图像不存在，无法自动配准。")
    return path


def _automatic_options(store: Store, clip: Clip, frame: Frame) -> dict:
    # Direct image/intensity registration consumes source scans, not labels.
    return {}


def save_projection_calibration(store: Store, clip_id: str, frame_id: str, camera_id: str,
                                value: ProjectionUpdate, *, context=None,
                                source_job_id: str | None = None,
                                source_input_fingerprint: str | None = None) -> dict:
    # Keep the comparison, validation and one-file publication under a single
    # lock. Heavy automatic solve itself runs outside this lock in its route.
    with store.lock:
        current = document_for(store, clip_id, frame_id, camera_id, context=context)
        _check_revision(current, value.revision)
        clip, frame, camera = context or _source_context(store, clip_id, frame_id, camera_id)
        matrix = value.matrix
        metrics = None
        if matrix is not None:
            if value.method in {"direct_nid", "koide"}:
                from .automatic_projection import evaluate_automatic_candidate
                registration = calibration_input(store, frame, np.asarray(current["effective_matrix"]))
                assessment = evaluate_automatic_candidate(
                    registration.points, _image_path(store, frame, camera_id), camera,
                    registration.initial_camera_from_input,
                    registration.camera_from_input(np.asarray(matrix)),
                    options=_automatic_options(store, clip, frame))
                assessment["metrics"].update(input_source=registration.source,
                                             input_point_count=len(registration.points))
            elif value.method == "correspondences":
                assessment = evaluate_correspondences(camera, value.correspondences, matrix, current["effective_matrix"])
            else:
                raise ValueError("未知的投影修正方式，请重新求解后保存。")
            if not assessment["usable"]:
                raise ValueError("当前候选未通过质量检查，不能保存：" + "；".join(assessment["rejection_reasons"]))
            metrics = assessment["metrics"]
        record = {"schema_version": "1.0", "clip_id": clip_id, "frame_id": frame_id, "camera_id": camera_id,
                  "base_fingerprint": _base_fingerprint(frame, camera),
                  "source_fingerprint": _source_fingerprint(store, frame, camera), "matrix": matrix,
                  "correspondences": [pair.model_dump() for pair in value.correspondences],
                  "method": value.method if matrix is not None else "reset", "metrics": metrics,
                  "sequence": int((_read_record(store, clip_id, frame_id, camera_id) or {}).get("sequence", 0)) + 1,
                  "updated_at": datetime.now(timezone.utc).isoformat()}
        if source_job_id is not None:
            record.update(source_job_id=source_job_id, source_input_fingerprint=source_input_fingerprint)
        previous = _read_record(store, clip_id, frame_id, camera_id)
        if previous:
            history = store.root / "projection_calibration_history" / safe_id(clip_id) / safe_id(frame_id) / safe_id(camera_id)
            atomic_json(history / f"{previous.get('sequence', 0)}-{_fingerprint(previous)[:16]}.json", previous)
        atomic_json(_path(store, clip_id, frame_id, camera_id), record)
        return document_for(store, clip_id, frame_id, camera_id, context=(clip, frame, camera))


def mount_projection_calibration_routes(app: FastAPI, store: Store):
    route = "/api/clips/{clip_id}/frames/{frame_id}/projection-calibration/{camera_id}"

    def translate_error(exc):
        if isinstance(exc, ConflictError):
            raise HTTPException(409, str(exc)) from exc
        if isinstance(exc, KeyError):
            raise HTTPException(404, str(exc).strip("'")) from exc
        raise HTTPException(422, str(exc)) from exc

    @app.get(route)
    def read_projection_calibration(clip_id: str, frame_id: str, camera_id: str):
        try:
            with store.lock:
                return document_for(store, clip_id, frame_id, camera_id)
        except (KeyError, ValueError, OSError) as exc:
            translate_error(exc)

    @app.post(route + "/solve")
    def solve_projection_calibration(clip_id: str, frame_id: str, camera_id: str, value: ProjectionSolveRequest):
        try:
            with store.lock:
                document = document_for(store, clip_id, frame_id, camera_id)
                _check_revision(document, value.revision)
                _, _, camera = _source_context(store, clip_id, frame_id, camera_id)
            result = solve_correspondences(camera, value.correspondences, value.initial_matrix or document["effective_matrix"])
            with store.lock:
                _check_revision(document_for(store, clip_id, frame_id, camera_id), value.revision)
            return {**document, **result}
        except (KeyError, ValueError, OSError, ConflictError) as exc:
            translate_error(exc)

    @app.post(route + "/auto")
    def automatic_projection_calibration(clip_id: str, frame_id: str, camera_id: str, value: ProjectionAutoRequest):
        try:
            from .automatic_projection import automatic_refine
            with store.lock:
                document = document_for(store, clip_id, frame_id, camera_id)
                _check_revision(document, value.revision)
                clip, frame, camera = _source_context(store, clip_id, frame_id, camera_id)
                source_image = _image_path(store, frame, camera_id)
                registration = calibration_input(store, frame, np.asarray(
                    value.initial_matrix or document["effective_matrix"]))
                options = _automatic_options(store, clip, frame)
                options_fingerprint = _fingerprint(options)
            result = automatic_refine(registration.points, source_image, camera,
                                      registration.initial_camera_from_input, options=options)
            result["matrix"] = registration.camera_from_primary(np.asarray(result["matrix"])).tolist()
            result.setdefault("metrics", {}).update(input_source=registration.source,
                                                     input_point_count=len(registration.points))
            with store.lock:
                _check_revision(document_for(store, clip_id, frame_id, camera_id), value.revision)
                updated_clip, updated_frame, _ = _source_context(store, clip_id, frame_id, camera_id)
                if _fingerprint(_automatic_options(store, updated_clip, updated_frame)) != options_fingerprint:
                    raise ConflictError("求解期间三维车道线或场景位姿已变化，请重新自动配准。")
            return {**document, **result}
        except ImportError as exc:
            raise HTTPException(422, "当前服务未安装自动配准模块，请使用对应点求解。") from exc
        except (KeyError, ValueError, OSError, ConflictError) as exc:
            translate_error(exc)

    @app.put(route)
    def write_projection_calibration(clip_id: str, frame_id: str, camera_id: str, value: ProjectionUpdate):
        try:
            return save_projection_calibration(store, clip_id, frame_id, camera_id, value)
        except (KeyError, ValueError, OSError, ConflictError) as exc:
            translate_error(exc)

    @app.get(route + "/export")
    def export_projection_calibration(clip_id: str, frame_id: str, camera_id: str):
        try:
            with store.lock:
                document = document_for(store, clip_id, frame_id, camera_id)
            name = f"{safe_id(clip_id)}-{safe_id(frame_id)}-{safe_id(camera_id)}-projection.json"
            return JSONResponse(document, headers={"Content-Disposition": f'attachment; filename="{name}"'})
        except (KeyError, ValueError, OSError) as exc:
            translate_error(exc)
