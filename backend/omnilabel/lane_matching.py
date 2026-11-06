"""Partial-observation image/point-cloud calibration with frozen 2D evidence.

Only normal distances on matched, visible line portions constrain the pose.
Unobserved endpoints never attract a long map line. Associations are refreshed
between robust least-squares steps, while before/after scores use the same
initial support. This module neither changes source geometry nor saves poses.
"""
from __future__ import annotations

from copy import deepcopy
import math
import time

import numpy as np
from scipy.optimize import least_squares

from .lane_optimizer import OptimizationCancelled, error_metrics, project, resample
from .models import rigid_transform


_DEFAULTS = {"matching_samples": 512, "samples_per_lane": 48,
    "max_match_distance_px": 60., "max_match_angle_degrees": 30.,
    "min_match_length_px": 20., "outer_iterations": 6, "dof_mode": "auto",
    "max_translation_m": .5, "max_rotation_degrees": 3., "huber_delta_px": 4.,
    "prior_weight": .01, "adam_iterations": 180, "timeout_seconds": 120.}
_AXES = ["tx", "ty", "tz", "rx", "ry", "rz"]


def _config(problem):
    return {**_DEFAULTS, **problem.get("config", {})}


def _image_line(points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("图像约束必须为有限的二维原图像素。")
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9]
    points = points[keep]
    if len(points) < 2:
        raise ValueError("图像车道线至少需要两个不同的像素点。")
    return points


def _normal_field(points, line):
    """Nearest finite segment selects a normal, not an endpoint residual."""
    starts = line[:-1]
    directions = np.diff(line, axis=0)
    lengths = np.linalg.norm(directions, axis=1)
    alpha = np.einsum("nki,ki->nk", points[:, None] - starts, directions) / lengths[None]**2
    feet = starts + np.clip(alpha, 0., 1.)[..., None] * directions
    squared = np.sum((points[:, None] - feet)**2, axis=2)
    selected = np.argmin(squared, axis=1)
    index = np.arange(len(points))
    tangent = directions[selected] / lengths[selected, None]
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    unclamped = alpha[index, selected]
    foot = starts[selected] + unclamped[:, None] * directions[selected]
    signed = np.sum((points - foot) * normal, axis=1)
    arc = np.r_[0., np.cumsum(lengths)]
    return {"normal": normal, "foot": foot, "alpha": unclamped,
            "distance": np.sqrt(squared[index, selected]), "signed": signed,
            "arc": arc[selected] + unclamped * lengths[selected],
            "tangent": tangent, "length": float(arc[-1]), "segment": selected}


def _association(problem, group, matrix, *, distance=None):
    config = _config(problem)
    uv, valid = project(problem, group, matrix)
    safe = np.where(np.isfinite(uv), uv, 1e12)
    field = _normal_field(safe, group["pixels"])
    tangent = np.gradient(safe, axis=0)
    speed = np.linalg.norm(tangent, axis=1)
    tangent /= np.maximum(speed[:, None], 1e-12)
    agreement = np.sum(tangent * field["tangent"], axis=1)
    direction = np.where(agreement >= 0, 1, -1)
    limit = config["max_match_distance_px"] if distance is None else distance
    good = (valid & (speed > 1e-8) & (field["alpha"] >= -1e-8)
            & (field["alpha"] <= 1 + 1e-8) & (field["distance"] <= limit)
            & (np.abs(agreement) >= math.cos(math.radians(config["max_match_angle_degrees"]))))
    indices = np.flatnonzero(good)
    runs, current = [], []
    maximum_gap = max(3, int(math.ceil(len(uv) * .005)))
    for index in indices:
        if current:
            previous = current[-1]
            arc_delta = direction[index] * (field["arc"][index] - field["arc"][previous])
            if (index - previous > maximum_gap or direction[index] != direction[previous]
                    or arc_delta < -max(1., field["length"] * .005)):
                runs.append(np.asarray(current, dtype=int)); current = []
        current.append(int(index))
    if current:
        runs.append(np.asarray(current, dtype=int))
    eligible = []
    for run in runs:
        covered = float(np.ptp(field["arc"][run]))
        if len(run) >= 6 and covered >= config["min_match_length_px"]:
            eligible.append((covered, float(np.mean(field["distance"][run])), run))
    if not eligible:
        raise ValueError(f'{group["id"]}: 没有足够的连续重合线段，请核对配对、初值、方向和匹配距离。')
    eligible.sort(key=lambda item: (-item[0], item[1]))
    covered, mean, chosen = eligible[0]
    first_arc = field["arc"][chosen]
    for other_covered, other_mean, other in eligible[1:]:
        other_arc = field["arc"][other]
        common = max(0., min(first_arc.max(), other_arc.max()) - max(first_arc.min(), other_arc.min()))
        if (other_covered >= .8 * covered and common >= .6 * min(covered, other_covered)
                and abs(other_mean - mean) < max(1., min(3., limit * .05))):
            raise ValueError(f'{group["id"]}: 存在多个相似重合区间，请明确选择对应三维路段，避免错误关联。')
    mask = np.zeros(len(uv), dtype=bool); mask[chosen] = True
    start, end = group.get("world_range", [0., 1.])
    report = {"id": group["id"], "frame_id": group["frame_id"], "lane_id": group["lane_id"],
              "matched_samples": int(mask.sum()), "total_samples": len(mask),
              "matched_fraction": float(mask.mean()), "image_coverage": covered / field["length"],
              "matched_world_range": [float(start + (end-start) * chosen[0] / (len(mask)-1)),
                                      float(start + (end-start) * chosen[-1] / (len(mask)-1))],
              "matched_length_px": covered, "mean_distance_px": mean,
              "direction": "forward" if direction[chosen[0]] > 0 else "reverse"}
    return {"indices": chosen, "mask": mask, "normal": field["normal"][chosen],
            "foot": field["foot"][chosen], "report": report}


def prepare_matched(problem: dict, matrix=None) -> list[dict]:
    """Dense source geometry, with initial observed support held for evaluation."""
    config = _config(problem)
    matrix = np.asarray(rigid_transform(np.asarray(matrix if matrix is not None else problem["initial_matrix"]).tolist()))
    groups = []
    for observation in problem["observations"]:
        points = resample(observation["world_points"], config["matching_samples"],
                          *observation.get("world_range", [0, 1]))
        pose = np.asarray(rigid_transform(observation["world_from_lidar"]), dtype=np.float64)
        local = (points - pose[:3, 3]) @ pose[:3, :3]
        group = {**deepcopy(observation), "local": local, "pixels": _image_line(observation["image_points"])}
        association = _association(problem, group, matrix)
        group.update(support=association["mask"], association=association)
        groups.append(group)
    if not groups:
        raise ValueError("请先添加至少一组真实图像线与三维车道线配对。")
    return groups


def _matching_report(groups, associations=None):
    pairs = [(associations or {}).get(group["id"], group["association"])["report"] for group in groups]
    return {"pairs": pairs, "pair_count": len(pairs),
            "matched_samples": sum(row["matched_samples"] for row in pairs),
            "metric": "matched_point_to_line_normal_px"}


def matched_statistics(problem, groups, matrix):
    diagonal = math.hypot(problem["camera"]["width"], problem["camera"]["height"])
    rows, by_split, overlays = [], {"train": [], "validation": []}, []
    for group in groups:
        uv, valid = project(problem, group, matrix)
        support = group["support"]
        errors = np.full(int(support.sum()), diagonal)
        visible = valid[support]
        if visible.any():
            field = _normal_field(uv[support][visible], group["pixels"])
            errors[visible] = np.abs(field["signed"])
        metrics = error_metrics(errors)
        metrics.update(id=group["id"], frame_id=group["frame_id"], lane_id=group["lane_id"],
                       split=group["split"], weight=group["weight"], visibility_retained=float(visible.mean()),
                       matched_samples=int(support.sum()), metric="normal_distance_on_fixed_matched_support")
        rows.append(metrics); by_split[group["split"]].append(errors)
        overlays.append({"id": group["id"], "frame_id": group["frame_id"], "lane_id": group["lane_id"],
                         "pixels": [point.tolist() if good and selected else None
                                    for point, good, selected in zip(uv, valid, support)]})
    splits = {}
    for split, errors in by_split.items():
        if not errors:
            splits[split] = None; continue
        selected = [row for row in rows if row["split"] == split]
        frames = sorted({row["frame_id"] for row in selected})
        balanced = [np.average([row["mean_px"] for row in selected if row["frame_id"] == fid],
                              weights=[row["weight"] for row in selected if row["frame_id"] == fid]) for fid in frames]
        splits[split] = {**error_metrics(np.concatenate(errors)), "frame_balanced_mean_px": float(np.mean(balanced)),
                         "frames": len(frames), "pairs": len(selected)}
    return {"splits": splits, "pairs": rows, "overlays": overlays,
            "metric": "normal_distance_on_fixed_matched_support"}


def _exp(xi):
    v, w = np.asarray(xi[:3]), np.asarray(xi[3:])
    theta = float(np.linalg.norm(w))
    skew = np.array([[0., -w[2], w[1]], [w[2], 0., -w[0]], [-w[1], w[0], 0.]])
    squared = skew @ skew
    if theta < 1e-5:
        a, b, c = 1-theta*theta/6, .5-theta*theta/24, 1/6-theta*theta/120
    else:
        a, b, c = math.sin(theta)/theta, (1-math.cos(theta))/theta**2, (theta-math.sin(theta))/theta**3
    result = np.eye(4)
    result[:3, :3] = np.eye(3) + a*skew + b*squared
    result[:3, 3] = (np.eye(3) + b*skew + c*squared) @ v
    return result


def _normal_residual(problem, groups, associations, matrix):
    values = []
    for group in groups:
        association = associations[group["id"]]
        uv, _ = project(problem, group, matrix)
        values.extend(np.sum((uv[association["indices"]] - association["foot"]) * association["normal"], axis=1))
    return np.asarray(values)


def _observability(problem, training, associations, matrix):
    config = _config(problem)
    columns = []
    for axis in range(6):
        step = np.zeros(6); step[axis] = 1e-4 if axis < 3 else 1e-6
        a = _normal_residual(problem, training, associations, _exp(step) @ matrix)
        b = _normal_residual(problem, training, associations, _exp(-step) @ matrix)
        columns.append((a-b)/(2*step[axis]))
    jacobian = np.column_stack(columns)
    scales = np.asarray([config["max_translation_m"]]*3 + [math.radians(config["max_rotation_degrees"])]*3)
    scaled = jacobian * scales
    singular = np.linalg.svd(scaled, compute_uv=False)
    threshold = max(1e-6, float(singular[0]) * 1e-5) if len(singular) else 1e-6
    rank = int(np.count_nonzero(singular > threshold))
    sensitivities = np.linalg.norm(scaled, axis=0) / math.sqrt(max(1, len(scaled)))
    # Do not normalize near-zero columns: doing so manufactures observability
    # from floating-point noise for translation along a straight road line.
    weak = [name for name, value in zip(_AXES, sensitivities) if value < max(1e-6, float(np.max(sensitivities))*1e-7)]
    rotational_singular = np.linalg.svd(scaled[:, 3:], compute_uv=False)
    rotational_threshold = max(1e-6, float(rotational_singular[0]) * 1e-5)
    rotation_rank = int(np.count_nonzero(rotational_singular > rotational_threshold))
    requested = config["dof_mode"]
    effective = "rotation" if requested == "rotation" or requested == "auto" and rank < 6 else "se3"
    return {"rank": rank, "rotation_rank": rotation_rank,
            "condition": min(1e15, float(singular[0] / max(singular[-1], 1e-15))),
            "singular_values": singular.tolist(), "column_sensitivities": sensitivities.tolist(),
            "weak_directions": weak, "requested_dof_mode": requested, "effective_dof_mode": effective,
            "_jacobian": scaled, "_threshold": threshold}


def _public_observability(observation):
    return {key: value for key, value in observation.items() if not key.startswith("_")}


def preview_matched(problem):
    groups = prepare_matched(problem)
    training = [group for group in groups if group["split"] == "train"]
    if not training:
        return {"statistics": matched_statistics(problem, groups, np.asarray(problem["initial_matrix"])),
                "matching": _matching_report(groups), "observability": None,
                "observability_reason": "仅有验证配对，不用验证数据选择优化自由度。"}
    matrix = np.asarray(problem["initial_matrix"])
    associations = {group["id"]: group["association"] for group in training}
    observation = _observability(problem, training, associations, matrix)
    return {"statistics": matched_statistics(problem, groups, matrix), "matching": _matching_report(groups),
            "observability": _public_observability(observation)}


def optimize_matched(problem: dict, progress=None, cancelled=lambda: False):
    started = time.monotonic()
    config = _config(problem)

    def check():
        if cancelled():
            raise OptimizationCancelled()
        if time.monotonic()-started > config["timeout_seconds"]:
            raise ValueError("匹配优化超时，原始标注和外参未改变。请减少配对或迭代次数。")

    check()
    groups = prepare_matched(problem)
    training = [group for group in groups if group["split"] == "train"]
    if not training:
        raise ValueError("至少需要一组训练配对才能优化外参。")
    initial = np.asarray(problem["initial_matrix"], dtype=np.float64)
    before = matched_statistics(problem, groups, initial)
    associations = {group["id"]: group["association"] for group in training}
    observation = _observability(problem, training, associations, initial)
    if observation["effective_dof_mode"] == "rotation":
        _, singular, vt = np.linalg.svd(observation["_jacobian"][:, 3:], full_matrices=False)
        rank = observation["rotation_rank"]
        basis = np.vstack([np.zeros((3, rank)), vt[:rank].T])
    else:
        _, singular, vt = np.linalg.svd(observation["_jacobian"], full_matrices=False)
        rank = observation["rank"]
        basis = vt[:rank].T if rank < 6 else np.eye(6)
    if not rank:
        raise ValueError("当前配对没有可靠的外参约束，请加入不同方向、近远处的图像特征。")
    scales = np.asarray([config["max_translation_m"]]*3 + [math.radians(config["max_rotation_degrees"])]*3)

    def increment(parameters):
        normalized = basis @ parameters
        denominator = math.sqrt(1 + max(float(normalized[:3] @ normalized[:3]), float(normalized[3:] @ normalized[3:])))
        return normalized / denominator * scales

    def matrix_for(parameters):
        return _exp(increment(parameters)) @ initial

    history = []
    parameters = np.zeros(basis.shape[1])
    best_parameters = parameters.copy()
    best_score = before["splits"]["train"]["frame_balanced_mean_px"]
    best_matrix = initial.copy()
    warnings = []
    frame_weights = {}
    for group in training:
        frame_weights[group["frame_id"]] = frame_weights.get(group["frame_id"], 0.) + group["weight"]

    def record(parameters, residuals, phase):
        xi = increment(parameters)
        delta = config["huber_delta_px"]
        absolute = np.abs(residuals)
        cost = np.where(absolute <= delta, .5*residuals**2, delta*(absolute-.5*delta))
        row = {"iteration": len(history), "phase": phase, "loss": float(np.mean(cost)),
               "data_loss": float(np.mean(cost)),
               **dict(zip(["dtx", "dty", "dtz", "drx", "dry", "drz"], xi.tolist()))}
        history.append(row)
        if progress:
            progress(row)

    for outer in range(int(config["outer_iterations"])):
        check()
        current = matrix_for(parameters)
        if best_score < 1e-7:
            break
        try:
            # Refreshed associations remain inside the user-declared lane pair.
            # Initial range and source geometry are never rewritten.
            associations = {group["id"]: _association(problem, group, current) for group in training}
        except ValueError as exc:
            warnings.append(str(exc)); break
        total = sum(len(association["indices"]) for association in associations.values())
        diagonal = math.hypot(problem["camera"]["width"], problem["camera"]["height"])

        def residual(parameters):
            check()
            matrix = matrix_for(parameters)
            chunks = []
            for group in training:
                association = associations[group["id"]]
                selected = association["indices"]
                uv, valid = project(problem, group, matrix)
                uv = uv[selected]; valid = valid[selected]
                normal = np.sum((uv-association["foot"]) * association["normal"], axis=1)
                # A pose cannot improve the fit by moving matched points behind
                # the camera or outside the calibrated image.
                normal = np.where(valid & np.isfinite(normal), normal, diagonal)
                weight = math.sqrt(group["weight"] * total /
                    (len(frame_weights) * frame_weights[group["frame_id"]] * len(selected)))
                chunks.append(normal * weight)
            xi = increment(parameters)
            data = np.concatenate(chunks)
            # A weak physical prior does not enter the observability estimate.
            values = np.r_[data, math.sqrt(config["prior_weight"]) * xi/scales]
            if not np.isfinite(values).all():
                raise ValueError("匹配优化出现非有限投影，请检查初始外参。")
            record(parameters, data, "matched_least_squares")
            return values

        result = least_squares(residual, parameters, bounds=(-20., 20.), loss="huber",
                               f_scale=config["huber_delta_px"], max_nfev=max(12, int(config["adam_iterations"])//int(config["outer_iterations"])),
                               ftol=1e-9, xtol=1e-9, gtol=1e-8, x_scale="jac")
        candidate = matrix_for(result.x)
        candidate_statistics = matched_statistics(problem, groups, candidate)
        score = candidate_statistics["splits"]["train"]["frame_balanced_mean_px"]
        retain = True
        for group in training:
            try:
                refreshed = _association(problem, group, candidate)
                original = group["association"]["report"]
                if (refreshed["report"]["matched_samples"] < .8 * original["matched_samples"]
                        or refreshed["report"]["image_coverage"] < .7 * original["image_coverage"]):
                    retain = False
            except ValueError:
                retain = False
        if any(row["visibility_retained"] < .98 for row in candidate_statistics["pairs"] if row["split"] == "train"):
            retain = False
        if not retain:
            warnings.append("候选降低了原始匹配支持或覆盖率，已保留覆盖完整的最佳结果。")
            break
        change = float(np.linalg.norm(result.x-parameters))
        parameters = result.x.copy()
        if score < best_score - 1e-9:
            best_score, best_parameters, best_matrix = float(score), parameters.copy(), candidate
        if change < 1e-7:
            break

    check()
    xi = increment(best_parameters)
    after = matched_statistics(problem, groups, best_matrix)
    refreshed = {}
    for group in groups:
        try:
            refreshed[group["id"]] = _association(problem, group, best_matrix)
        except ValueError:
            refreshed[group["id"]] = group["association"]
    final_associations = {group["id"]: refreshed[group["id"]] for group in training}
    final_residuals = []
    total = sum(len(association["indices"]) for association in final_associations.values())
    for group in training:
        association = final_associations[group["id"]]
        uv, valid = project(problem, group, best_matrix)
        selected = association["indices"]
        normal = np.sum((uv[selected]-association["foot"])*association["normal"], axis=1)
        normal = np.where(valid[selected] & np.isfinite(normal), normal,
                          math.hypot(problem["camera"]["width"], problem["camera"]["height"]))
        weight = math.sqrt(group["weight"]*total /
            (len(frame_weights)*frame_weights[group["frame_id"]]*len(selected)))
        final_residuals.extend(normal*weight)
    record(best_parameters, np.asarray(final_residuals), "selected")
    final_observation = _observability(problem, training, final_associations, best_matrix)
    # The solve's selected DOF mode remains fixed through the outer iterations.
    final_observation["effective_dof_mode"] = observation["effective_dof_mode"]
    if observation["rank"] < 6:
        warnings.append("车道线不足以可靠约束六自由度；自动模式仅调整可观测旋转，或已固定显式模式中的弱约束方向。")
    if before["splits"]["validation"] is None:
        warnings.append("没有独立验证约束，请使用未参与拟合的本帧特征复核投影。")
    if np.linalg.norm(xi[:3]) > .95*config["max_translation_m"] or np.linalg.norm(xi[3:]) > .95*math.radians(config["max_rotation_degrees"]):
        warnings.append("修正接近设置边界，请检查初值、配对和时间同步。")
    if problem.get("mode") == "single":
        warnings.append("单帧结果仅供调试，只导出、不应用。")
    reduction, balanced_reduction = {}, {}
    for split in ("train", "validation"):
        a, b = before["splits"][split], after["splits"][split]
        reduction[split] = (0. if max(a["mean_px"], b["mean_px"]) < 1e-8
            else float(1-b["mean_px"]/max(a["mean_px"], 1e-9))) if a else None
        balanced_reduction[split] = (0. if max(a["frame_balanced_mean_px"], b["frame_balanced_mean_px"]) < 1e-8
            else float(1-b["frame_balanced_mean_px"]/max(a["frame_balanced_mean_px"], 1e-9))) if a else None
    matching = _matching_report(groups, refreshed)
    matching["initial_pairs"] = _matching_report(groups)["pairs"]
    return {"initial_matrix": initial.tolist(), "optimized_matrix": best_matrix.tolist(), "increment": xi.tolist(),
            "before": before, "after": after, "matching": matching,
            "error_reduction": reduction, "frame_balanced_error_reduction": balanced_reduction,
            "history": history, "observability": _public_observability(final_observation),
            "optimization": {"requested_dof_mode": config["dof_mode"], "effective_dof_mode": observation["effective_dof_mode"],
                             "update_convention": "Exp(camera_xi) @ T0"},
            "usable": not warnings, "rejection_reasons": list(dict.fromkeys(warnings)),
            "elapsed_seconds": time.monotonic()-started}
