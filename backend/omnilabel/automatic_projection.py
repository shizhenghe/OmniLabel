"""Bounded, correspondence-free camera/map refinement using intensity NID.

This is a small CPU implementation of the normalized information distance
objective used by direct visual/LiDAR registration, not a wrapper around Koide's
ROS package. It proposes one *local* rigid correction, never edits source data,
and rejects low-information or non-generalising fits. Its NID score is an
image/intensity consistency check, not an independent reprojection error in px.

All matrices map frame point coordinates into the optical camera (column
vectors). Translation is metres. The caller remains responsible for reviewing
an accepted candidate before persisting it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from scipy.stats import rankdata

from .models import Camera, rigid_transform


@dataclass(frozen=True)
class _Options:
    max_seconds: float = 12.0
    max_evaluations: int = 240
    max_points: int = 16000
    max_range: float = 100.0
    voxel_size: float = 0.10
    max_translation: float = 0.75
    max_rotation_degrees: float = 3.0
    seed: int = 62071


def _options(value: dict | None) -> _Options:
    options = _Options(**(value or {}))
    # Also bound programmatic callers. Routes do not accept arbitrary options.
    for name, lower, upper in (("max_seconds", 0.5, 30), ("max_evaluations", 20, 800),
                               ("max_points", 1000, 40000), ("max_range", 10, 150),
                               ("voxel_size", 0.03, 1), ("max_translation", 0.05, 1.5),
                               ("max_rotation_degrees", 0.1, 5)):
        number = getattr(options, name)
        if not np.isfinite(number) or not lower <= number <= upper:
            raise ValueError(f"automatic registration {name} must be between {lower} and {upper}")
    return options


def automatic_status() -> dict:
    return {"available": True, "method": "direct_nid",
            "detail": "内置 CPU 灰度与点云强度直接配准；有界局部优化，独立留出集检查，候选需人工预览。"}


def _transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    # Avoid spawning large BLAS thread pools for repeated very small matrices.
    return np.column_stack([sum(points[:, j] * matrix[i, j] for j in range(3)) + matrix[i, 3]
                            for i in range(3)])


def _pixels(points: np.ndarray, k: np.ndarray, distortion: np.ndarray) -> np.ndarray:
    z = np.maximum(points[:, 2], 1e-8)
    x, y = points[:, 0] / z, points[:, 1] / z
    if len(distortion) and np.any(distortion):
        k1, k2, p1, p2, k3 = distortion
        r2 = x * x + y * y
        radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
        x, y = (x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
                y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)
    return np.column_stack([k[0, 0] * x + k[0, 1] * y + k[0, 2],
                            k[1, 0] * x + k[1, 1] * y + k[1, 2]])


def _delta(value: np.ndarray, options: _Options) -> np.ndarray:
    delta = np.eye(4)
    delta[:3, :3] = Rotation.from_rotvec(value[3:] * np.deg2rad(options.max_rotation_degrees)).as_matrix()
    delta[:3, 3] = value[:3] * options.max_translation
    return delta


def _entropy(probability: np.ndarray) -> float:
    nonzero = probability[probability > 1e-12]
    return float(-np.sum(nonzero * np.log(nonzero)))


def _nid(intensity: np.ndarray, gray: np.ndarray, bins: int = 16) -> tuple[float, float, float]:
    """Linear partial-volume joint histogram, reducing hard-bin discontinuity."""
    coordinates = np.clip(np.column_stack([intensity, gray]), 0, 1) * (bins - 1)
    base = np.floor(coordinates).astype(np.int32)
    fraction = coordinates - base
    hist = np.zeros(bins * bins)
    for di in (0, 1):
        for dg in (0, 1):
            i = np.minimum(base[:, 0] + di, bins - 1)
            g = np.minimum(base[:, 1] + dg, bins - 1)
            weight = (fraction[:, 0] if di else 1 - fraction[:, 0]) * (fraction[:, 1] if dg else 1 - fraction[:, 1])
            hist += np.bincount(i * bins + g, weights=weight, minlength=bins * bins)
    hist = hist.reshape(bins, bins) / max(float(hist.sum()), 1e-12)
    hi, hg, joint = _entropy(hist.sum(axis=1)), _entropy(hist.sum(axis=0)), _entropy(hist)
    return float(np.clip(2 - (hi + hg) / max(joint, 1e-12), 0, 1)), hi, hg


def _prepare(points: np.ndarray, image_path: Path, camera: Camera | dict, initial: np.ndarray,
             options: _Options) -> dict:
    camera = camera if isinstance(camera, Camera) else Camera.model_validate(camera)
    if points.ndim != 2 or points.shape[1] < 4:
        raise ValueError("自动配准需要含 x、y、z 和 intensity 的点云")
    rigid_transform(initial.tolist())
    with Image.open(image_path) as source:
        if source.size != (camera.width, camera.height):
            raise ValueError("图像尺寸与相机内参声明不一致，无法自动配准")
        scale = min(1.0, 960 / source.width)
        width, height = round(source.width * scale), round(source.height * scale)
        image_gray = source.convert("L").resize((width, height), Image.Resampling.BILINEAR)
        raw_gray_std = float(np.std(np.asarray(image_gray, dtype=float) / 255))
        gray = np.asarray(ImageOps.equalize(image_gray), dtype=float) / 255
    k = np.asarray(camera.intrinsics, dtype=float).copy()
    k[:2] *= scale
    distortion = np.asarray(camera.distortion, dtype=float)
    rng = np.random.default_rng(options.seed)
    selected = []
    roi_count = 0
    # Read every chunk so source-order sampling cannot discard a whole local
    # region. Only nearby points survive; candidate memory stays bounded.
    for start in range(0, len(points), 250000):
        chunk = np.asarray(points[start:start + 250000, :4], dtype=float)
        finite = np.isfinite(chunk).all(axis=1)
        chunk = chunk[finite]
        xyz = _transform(chunk[:, :3], initial)
        z = xyz[:, 2]
        near = (z > 2) & (z < options.max_range) & (np.sum(xyz * xyz, axis=1) < options.max_range ** 2)
        xyz, values = xyz[near], chunk[near, 3]
        if not len(xyz):
            continue
        uv = _pixels(xyz, k, distortion)
        valid = (uv[:, 0] > -width * .12) & (uv[:, 0] < width * 1.12) & (uv[:, 1] > -height * .12) & (uv[:, 1] < height * 1.12)
        xyz, values = xyz[valid], values[valid]
        roi_count += len(xyz)
        if len(xyz) > 24000:
            indices = rng.choice(len(xyz), 24000, replace=False)
            xyz, values = xyz[indices], values[indices]
        selected.append(np.column_stack([xyz, values]))
        if sum(len(part) for part in selected) > 240000:
            merged = np.concatenate(selected)
            selected = [merged[rng.choice(len(merged), 120000, replace=False)]]
    data = np.concatenate(selected) if selected else np.empty((0, 4))
    if len(data):
        # One measured return per voxel: do not average away reflectance detail.
        cells = np.floor(data[:, :3] / options.voxel_size).astype(np.int32)
        _, indices = np.unique(cells, axis=0, return_index=True)
        data = data[indices]
        uv = _pixels(data[:, :3], k, distortion)
        inside = (uv[:, 0] > 3) & (uv[:, 0] < width - 4) & (uv[:, 1] > 3) & (uv[:, 1] < height - 4)
        data, uv = data[inside], uv[inside]
        # Conservative near-return filtering at ~6 original-image pixels; never
        # use background returns behind a foreground object as correspondences.
        cell_width = int(np.ceil(width / 3))
        keys = (uv[:, 1] // 3).astype(int) * cell_width + (uv[:, 0] // 3).astype(int)
        nearest = np.full(cell_width * int(np.ceil(height / 3)), np.inf)
        np.minimum.at(nearest, keys, data[:, 2])
        visible = data[:, 2] <= nearest[keys] + np.maximum(.25, .0075 * nearest[keys])
        data, uv = data[visible], uv[visible]
        if len(data) > options.max_points:
            indices = rng.choice(len(data), options.max_points, replace=False)
            data, uv = data[indices], uv[indices]
    else:
        uv = np.empty((0, 2))
    # Spatially held-out image cells prevent the optimiser from seeing the
    # validation samples. The same fixed points are scored before and after.
    groups = (np.floor(uv[:, 0] / max(1, width / 8)).astype(int)
              + 3 * np.floor(uv[:, 1] / max(1, height / 6)).astype(int)) % 4
    holdout = groups == 0
    intensities = data[:, 3].copy()
    span = float(np.ptp(intensities)) if len(data) else 0.0
    if len(data):
        low, high = np.percentile(intensities, [1, 99])
        if high - low > 1e-8:
            # Histogram equalisation makes different sensor response curves
            # less dominant. Midranks preserve equal reflectance values.
            intensities = (rankdata(intensities, method="average") - .5) / len(intensities)
        else:
            intensities = np.zeros(len(data))
    return {"xyz": data[:, :3], "intensity": intensities, "gray": gray,
            "k": k, "distortion": distortion, "uv": uv, "groups": groups,
            "holdout": holdout, "roi_count": roi_count, "intensity_span": span,
            "width": width, "height": height, "camera": camera, "raw_gray_std": raw_gray_std}


def _score(prepared: dict, transform: np.ndarray, mask: np.ndarray, gray: np.ndarray | None = None) -> dict:
    xyz = _transform(prepared["xyz"][mask], transform)
    uv = _pixels(xyz, prepared["k"], prepared["distortion"])
    valid = ((xyz[:, 2] > .5) & np.isfinite(uv).all(axis=1)
             & (uv[:, 0] >= 1) & (uv[:, 0] < prepared["width"] - 2)
             & (uv[:, 1] >= 1) & (uv[:, 1] < prepared["height"] - 2))
    overlap = float(np.mean(valid)) if len(valid) else 0.0
    if valid.sum() < 80:
        return {"nid": 1.0, "intensity_entropy": 0.0, "image_entropy": 0.0,
                "overlap": overlap, "positive_depth_ratio": float(np.mean(xyz[:, 2] > .5)) if len(xyz) else 0.0,
                "samples": int(valid.sum())}
    image = prepared["gray"] if gray is None else gray
    sampled = map_coordinates(image, [uv[valid, 1], uv[valid, 0]], order=1, mode="nearest", prefilter=False)
    nid, hi, hg = _nid(prepared["intensity"][mask][valid], sampled)
    return {"nid": nid, "intensity_entropy": hi, "image_entropy": hg,
            "overlap": overlap, "positive_depth_ratio": float(np.mean(xyz[:, 2] > .5)),
            "samples": int(valid.sum())}


def _quality(prepared: dict, delta: np.ndarray, options: _Options, *, timed_out: bool = False) -> tuple[dict, list[str]]:
    train = ~prepared["holdout"]
    validation = prepared["holdout"]
    # Validation always uses the same fine image for both poses, independently
    # of the smoothed image(s) used to guide the optimiser.
    fine = gaussian_filter(prepared["gray"], .65)
    before = _score(prepared, np.eye(4), validation, fine)
    after = _score(prepared, delta, validation, fine)
    train_before = _score(prepared, np.eye(4), train, fine)
    train_after = _score(prepared, delta, train, fine)
    improvement = before["nid"] - after["nid"]
    train_improvement = train_before["nid"] - train_after["nid"]
    shift = np.linalg.norm(delta[:3, 3])
    rotation = np.rad2deg(Rotation.from_matrix(delta[:3, :3]).magnitude())
    uv_after = _pixels(_transform(prepared["xyz"], delta), prepared["k"], prepared["distortion"])
    pixel_shift = np.linalg.norm(uv_after - prepared["uv"], axis=1)
    # Require information distributed over the image and at different depths.
    uv = prepared["uv"]
    occupied = len(np.unique(np.floor(uv[:, 0] / max(1, prepared["width"] / 8)).astype(int)
                             + 8 * np.floor(uv[:, 1] / max(1, prepared["height"] / 6)).astype(int))) if len(uv) else 0
    metrics = {"holdout_before": before, "holdout_after": after,
               "train_before": train_before, "train_after": train_after,
               "holdout_improvement": float(improvement), "train_improvement": float(train_improvement),
               "translation_delta_m": float(shift), "rotation_delta_degrees": float(rotation),
               "median_projection_shift_px": float(np.median(pixel_shift)) / min(1., 960 / prepared["camera"].width) if len(pixel_shift) else 0.,
               "occupied_image_cells": occupied, "total_image_cells": 48,
               "roi_point_count": int(prepared["roi_count"]), "sample_point_count": int(len(uv)),
               "timed_out": timed_out, "score_name": "normalized_information_distance",
               "score_note": "NID 是跨模态纹理一致性指标，数值越低越好；不是像素重投影误差。"}
    reasons = []
    if len(uv) < 600 or before["samples"] < 150 or train_before["samples"] < 350:
        reasons.append("可见静态点云不足，无法独立验证自动配准。")
    if occupied < 10:
        reasons.append("点云只覆盖图像局部区域，无法可靠约束当前相机位姿。")
    if prepared["intensity_span"] < 1e-6 or before["intensity_entropy"] < .55:
        reasons.append("点云 intensity 缺少纹理变化，无法与图像进行直接配准。")
    if prepared["raw_gray_std"] < .035 or before["image_entropy"] < .7:
        reasons.append("图像纹理不足或曝光异常，无法可靠自动配准。")
    if after["overlap"] < .95 or after["positive_depth_ratio"] < .99:
        reasons.append("候选修正使过多点离开图像或相机前方，已拒绝。")
    if shift >= options.max_translation * .97 or rotation >= options.max_rotation_degrees * .97:
        reasons.append("优化已接近允许的位姿修正范围，需要对应点校准或检查上游标定。")
    # Pure entropy improvement of a nearly independent modality can be noise.
    if after["nid"] > .975:
        reasons.append("图像与点云强度相关性太弱，无法确认同一静态结构。")
    if improvement < max(.004, before["nid"] * .008) or train_improvement < .003:
        reasons.append("独立留出区域的匹配改善不足；初值可能已准确，或该帧不支持可靠自动修正。")
    if improvement < train_improvement * .3:
        reasons.append("候选主要改善训练区域，未通过独立留出集检查。")
    return metrics, reasons


def _result(matrix: np.ndarray, prepared: dict, metrics: dict, reasons: list[str], elapsed: float) -> dict:
    return {"matrix": matrix.tolist(), "candidate_matrix": matrix.tolist(), "method": "direct_nid",
            "usable": not reasons, "rejection_reasons": reasons, "point_count": int(len(prepared["xyz"])),
            "metrics": {**metrics, "elapsed_seconds": float(elapsed)},
            "quality": "candidate_requires_visual_review" if not reasons else "rejected"}


def evaluate_automatic_candidate(points: np.ndarray, image_path: Path, camera: Camera | dict,
                                 initial_matrix: np.ndarray, candidate_matrix: np.ndarray,
                                 options: dict | None = None) -> dict:
    """Reproduce the independent acceptance check when saving a proposal."""
    start = time.monotonic()
    settings = _options(options)
    initial = np.asarray(initial_matrix, dtype=float)
    candidate = np.asarray(candidate_matrix, dtype=float)
    rigid_transform(candidate.tolist())
    prepared = _prepare(points, image_path, camera, initial, settings)
    delta = candidate @ np.linalg.inv(initial)
    metrics, reasons = _quality(prepared, delta, settings)
    return _result(candidate, prepared, metrics, reasons, time.monotonic() - start)


def automatic_refine(points: np.ndarray, image_path: Path, camera: Camera | dict,
                     initial_matrix: np.ndarray, options: dict | None = None) -> dict:
    """Propose a per-frame correction without 2D/3D correspondences.

    A rejected result may contain a diagnostic matrix but must not be saved or
    described as calibrated. The independent holdout is never used to select
    optimisation steps. Runtime/evaluation bounds prevent unbounded searches.
    """
    start = time.monotonic()
    settings = _options(options)
    initial = np.asarray(initial_matrix, dtype=float)
    prepared = _prepare(points, image_path, camera, initial, settings)
    metrics, initial_reasons = _quality(prepared, np.eye(4), settings)
    # The unchanged-pose quality gate always reports no improvement; only other
    # reasons are preflight blockers. No false success on textureless data.
    initial_reasons = [reason for reason in initial_reasons if not reason.startswith("独立留出区域")]
    # Low initial modality correlation may itself be due to the incorrect pose;
    # it is an acceptance test, not a preflight requirement.
    initial_reasons = [reason for reason in initial_reasons if not reason.startswith("图像与点云强度相关性")]
    if initial_reasons:
        return _result(initial, prepared, metrics, initial_reasons, time.monotonic() - start)
    train = ~prepared["holdout"]
    best_value, best_loss = np.zeros(6), float("inf")
    evaluations = 0
    timed_out = False

    class _BudgetExpired(Exception):
        pass

    # A coarse image reduces local aliases; refinement uses a less blurred
    # image. The held-out spatial cells remain untouched until the final check.
    for sigma, fraction in ((2.0, .55), (.65, 1.0)):
        smoothed = gaussian_filter(prepared["gray"], sigma)
        limit = int(settings.max_evaluations * fraction)
        best_loss = float("inf")

        def objective(value):
            nonlocal evaluations, best_value, best_loss, timed_out
            if time.monotonic() - start > settings.max_seconds or evaluations >= limit:
                timed_out = time.monotonic() - start > settings.max_seconds
                raise _BudgetExpired
            evaluations += 1
            delta = _delta(np.asarray(value), settings)
            # Enforce a norm bound, not merely per-axis box bounds.
            norm_excess = max(0., np.linalg.norm(value[:3]) - 1) + max(0., np.linalg.norm(value[3:]) - 1)
            score = _score(prepared, delta, train, smoothed)
            loss = score["nid"] + .4 * (1 - score["overlap"]) + .08 * norm_excess + .0005 * float(np.dot(value, value))
            if loss < best_loss:
                best_loss, best_value = loss, np.asarray(value).copy()
            return loss

        try:
            minimize(objective, best_value.copy(), method="Powell", bounds=[(-1., 1.)] * 6,
                     options={"xtol": .002, "ftol": .00002, "maxfev": max(20, limit - evaluations)})
        except _BudgetExpired:
            if timed_out:
                break
    delta = _delta(best_value, settings)
    metrics, reasons = _quality(prepared, delta, settings, timed_out=timed_out)
    metrics["evaluations"] = evaluations
    return _result(delta @ initial, prepared, metrics, reasons, time.monotonic() - start)
