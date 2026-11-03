"""Numerical lane calibration, independent of FastAPI and annotation writes.

All points supplied here are world-space ground truth. Each observation has an
explicit world_from_lidar pose; one shared camera_from_lidar is optimized using
T0 @ exp(xi^). Image constraints are frozen raw-image pixels, measured directly
or recovered from explicitly selected image-derived world lines and a saved
reference projection. The reference is never transformed by the candidate.
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from PIL import Image, ImageDraw

from .models import rigid_transform
from .projection_calibration import project_camera_points


class OptimizationCancelled(Exception):
    pass


def resample(points, count: int, start: float = 0, end: float = 1) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    keep = np.r_[True, np.diff(lengths) > 1e-9]
    points, lengths = points[keep], lengths[keep]
    if len(points) < 2 or lengths[-1] < 1e-6:
        raise ValueError("车道线至少需要两个不同的点。")
    samples = np.linspace(start * lengths[-1], end * lengths[-1], count)
    return np.column_stack([np.interp(samples, lengths, points[:, i]) for i in range(points.shape[1])])


def closest_segments(points, line):
    """Exact point-to-segment displacement, including endpoints; no index pairing."""
    points, line = np.asarray(points), np.asarray(line)
    a, direction = line[:-1], np.diff(line, axis=0)
    alpha = np.sum((points[:, None] - a) * direction, axis=2) / np.maximum(np.sum(direction**2, axis=1), 1e-12)
    feet = a + np.clip(alpha, 0, 1)[..., None] * direction
    delta = points[:, None] - feet
    indices = np.argmin(np.sum(delta**2, axis=2), axis=1)
    return delta[np.arange(len(points)), indices]


def prepare(problem: dict) -> list[dict]:
    groups = []
    for observation in problem["observations"]:
        points = resample(observation["world_points"], problem["config"]["samples_per_lane"],
                          *observation.get("world_range", [0, 1]))
        pose = np.asarray(rigid_transform(observation["world_from_lidar"]), dtype=np.float64)
        # Subtract the world origin in float64 before rotation (UTM-safe).
        local = (points - pose[:3, 3]) @ pose[:3, :3]
        group = {**observation, "local": local,
                 "pixels": resample(observation["image_points"], problem["config"]["samples_per_lane"])}
        uv, valid = project(problem, group, np.asarray(problem["initial_matrix"]))
        if np.count_nonzero(valid) < 6 or np.count_nonzero(valid[:-1] & valid[1:]) < 2:
            raise ValueError(f'{observation["id"]}: 初值下可见三维线段不足，请检查方向、pose、匹配和三维取样范围。')
        group["support"] = valid
        groups.append(group)
    return groups


def project(problem, group, matrix):
    camera = problem["camera"]
    uv, depth = project_camera_points(group["local"], matrix, np.asarray(camera["intrinsics"]), np.asarray(camera["distortion"]))
    valid = (np.isfinite(uv).all(axis=1) & (depth > .05) & (uv[:, 0] >= 0) & (uv[:, 0] < camera["width"])
             & (uv[:, 1] >= 0) & (uv[:, 1] < camera["height"]))
    return uv, valid


def statistics(problem, groups, matrix) -> dict:
    camera = problem["camera"]; diagonal = math.hypot(camera["width"], camera["height"])
    rows, errors_by_split, overlays = [], {"train": [], "validation": []}, []
    for group in groups:
        uv, valid = project(problem, group, matrix)
        # Fixed initial support. Missing points receive the image diagonal,
        # preventing a candidate from improving metrics by hiding bad points.
        errors = np.full(int(group["support"].sum()), diagonal)
        visible = valid[group["support"]]
        if visible.any():
            errors[visible] = np.linalg.norm(closest_segments(uv[group["support"]][visible], group["pixels"]), axis=1)
        metrics = error_metrics(errors)
        metrics.update(id=group["id"], frame_id=group["frame_id"], lane_id=group["lane_id"],
                       split=group["split"], weight=group["weight"], visibility_retained=float(visible.mean()))
        rows.append(metrics); errors_by_split[group["split"]].append(errors)
        overlays.append({"id": group["id"], "frame_id": group["frame_id"], "lane_id": group["lane_id"],
                         "pixels": [point.tolist() if good else None for point, good in zip(uv, valid)]})
    splits = {}
    for split, errors in errors_by_split.items():
        if not errors:
            splits[split] = None; continue
        selected = [row for row in rows if row["split"] == split]
        frames = sorted({row["frame_id"] for row in selected})
        balanced = [np.average([r["mean_px"] for r in selected if r["frame_id"] == fid],
                               weights=[r["weight"] for r in selected if r["frame_id"] == fid]) for fid in frames]
        splits[split] = {**error_metrics(np.concatenate(errors)), "frame_balanced_mean_px": float(np.mean(balanced)),
                         "frames": len(frames), "pairs": len(selected)}
    return {"splits": splits, "pairs": rows, "overlays": overlays}


def error_metrics(errors):
    return {"mean_px": float(np.mean(errors)), "rmse_px": float(np.sqrt(np.mean(errors**2))),
            "median_px": float(np.median(errors)), "p95_px": float(np.percentile(errors, 95)), "samples": len(errors)}


def se3_exp(xi):
    import torch
    v, w = xi[:3], xi[3:]
    zero = w[0] * 0
    skew = torch.stack([zero, -w[2], w[1], w[2], zero, -w[0], -w[1], w[0], zero]).reshape(3, 3)
    hat = torch.cat([torch.cat([skew, v[:, None]], dim=1), xi.new_zeros((1, 4))], dim=0)
    return torch.matrix_exp(hat)


def _torch_project(points, matrix, camera):
    import torch
    pc = points @ matrix[:3, :3].T + matrix[:3, 3]
    x, y = (pc[:, :2] / pc[:, 2:].clamp(min=.05)).clamp(-1000, 1000).unbind(1)
    if camera["distortion"]:
        k1, k2, p1, p2, k3 = camera["distortion"]
        r2 = x*x + y*y; radial = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
        x, y = x*radial + 2*p1*x*y + p2*(r2 + 2*x*x), y*radial + p1*(r2 + 2*y*y) + 2*p2*x*y
    uv = torch.stack([x, y, torch.ones_like(x)], dim=1) @ points.new_tensor(camera["intrinsics"]).T
    return uv[:, :2], pc[:, 2]


def _torch_distances(points, line, segment_valid=None):
    import torch
    direction = line[1:] - line[:-1]
    alpha = ((points[:, None] - line[:-1]) * direction).sum(2) / direction.square().sum(1).clamp(min=1e-12)
    delta = points[:, None] - (line[:-1] + alpha.clamp(0, 1)[..., None] * direction)
    squared = delta.square().sum(2)
    if segment_valid is not None:
        squared = torch.where(segment_valid[None], squared, squared.new_full((), 1e12))
    return (squared.min(1).values + 1e-9).sqrt()


def optimize(problem: dict, progress=None, cancelled=lambda: False) -> dict:
    import torch
    from scipy.linalg import expm
    torch.set_num_threads(min(2, torch.get_num_threads()))
    groups = prepare(problem); config = problem["config"]; camera = problem["camera"]
    initial = np.asarray(rigid_transform(problem["initial_matrix"]), dtype=np.float64)
    before = statistics(problem, groups, initial)
    training = [g for g in groups if g["split"] == "train"]
    tensors = [{**g, "xyz": torch.tensor(g["local"], dtype=torch.float64),
                "image": torch.tensor(g["pixels"], dtype=torch.float64),
                "mask": torch.tensor(g["support"])} for g in training]
    parameters = torch.zeros(6, dtype=torch.float64, requires_grad=True)
    t0 = torch.tensor(initial, dtype=torch.float64)
    diagonal = math.hypot(camera["width"], camera["height"])
    started = time.monotonic(); history = []; best = (float("inf"), parameters.detach().clone())

    def check():
        if cancelled(): raise OptimizationCancelled()
        if time.monotonic() - started > config["timeout_seconds"]:
            raise ValueError("优化超时，请减少帧数或迭代次数；原始外参未改变。")

    def increment(raw):
        # Smooth norm bounds, not independent unbounded rotation entries.
        return torch.cat([raw[:3] / (1 + raw[:3].square().sum()).sqrt() * config["max_translation_m"],
                          raw[3:] / (1 + raw[3:].square().sum()).sqrt() * math.radians(config["max_rotation_degrees"])])

    def objective(raw):
        xi = increment(raw); matrix = t0 @ se3_exp(xi); by_frame = {}
        for g in tensors:
            uv, z = _torch_project(g["xyz"], matrix, camera)
            valid = (z > .05) & (uv[:, 0] >= 0) & (uv[:, 0] < camera["width"]) & (uv[:, 1] >= 0) & (uv[:, 1] < camera["height"])
            support = g["mask"]; usable = support & valid
            forward = _torch_distances(uv, g["image"])
            reverse = _torch_distances(g["image"], uv, usable[:-1] & usable[1:])
            huber = lambda d: torch.nn.functional.huber_loss(d, torch.zeros_like(d), delta=config["huber_delta_px"], reduction="none")
            lane = (huber(forward) * usable).sum() / support.sum()
            symmetric = .5 * (lane + huber(reverse).mean())
            lost = support & ~valid
            barrier = ((.05 - z).relu() * camera["intrinsics"][0][0] + (-uv[:, 0]).relu()
                       + (uv[:, 0] - camera["width"]).relu() + (-uv[:, 1]).relu() + (uv[:, 1] - camera["height"]).relu())
            visibility = lost.double().sum() / support.sum() * diagonal**2 + (barrier * support).mean() * 10
            cost = lane + config["chamfer_weight"] * symmetric + visibility
            by_frame.setdefault(g["frame_id"], []).append((g["weight"], cost))
        data_loss = torch.stack([sum(w*c for w, c in values) / sum(w for w, _ in values) for values in by_frame.values()]).mean()
        prior = (xi[:3] / config["max_translation_m"]).square().sum() + (xi[3:] / math.radians(config["max_rotation_degrees"])).square().sum()
        return data_loss + config["prior_weight"] * prior, data_loss, xi

    def record(iteration, phase):
        nonlocal best
        loss, data_loss, xi = objective(parameters)
        if not torch.isfinite(loss): raise ValueError("优化出现非有限损失，请检查输入标定和车道线。")
        value = float(loss.detach())
        if value < best[0]: best = (value, parameters.detach().clone())
        row = {"iteration": iteration, "phase": phase, "loss": value, "data_loss": float(data_loss.detach()),
               **dict(zip(["dtx", "dty", "dtz", "drx", "dry", "drz"], xi.detach().tolist()))}
        history.append(row)
        if progress: progress(row)
        return loss

    optimizer = torch.optim.Adam([parameters], lr=config["learning_rate"])
    for iteration in range(config["adam_iterations"]):
        check(); optimizer.zero_grad(); loss = record(iteration, "adam"); loss.backward()
        torch.nn.utils.clip_grad_norm_([parameters], 1000.); optimizer.step()
    if config["lbfgs_iterations"]:
        check()
        optimizer = torch.optim.LBFGS([parameters], max_iter=config["lbfgs_iterations"], line_search_fn="strong_wolfe")
        def closure():
            check(); optimizer.zero_grad(); loss = record(len(history), "lbfgs_evaluation")
            loss.backward(); return loss
        optimizer.step(closure)
    record(len(history), "final")
    xi = increment(best[1]).detach(); matrix = (t0 @ se3_exp(xi)).detach().numpy()
    history.append({"iteration": len(history), "phase": "selected", "loss": best[0],
        "data_loss": float(objective(best[1])[1].detach()),
        **dict(zip(["dtx", "dty", "dtz", "drx", "dry", "drz"], xi.tolist()))})
    rigid_transform(matrix.tolist())
    after = statistics(problem, groups, matrix)

    # Data-only, scale-normalized finite-difference Jacobian. A prior cannot
    # manufacture geometric observability or serve as held-out validation.
    def residual(delta):
        hat = np.zeros((4, 4)); v, w = delta[:3], delta[3:]
        hat[:3, :3] = [[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]]; hat[:3, 3] = v
        candidate = matrix @ expm(hat); values = []
        for group in training:
            uv, _ = project(problem, group, candidate)
            values.extend(closest_segments(uv[group["support"]], group["pixels"]).ravel())
        return np.asarray(values)
    columns = []
    for axis in range(6):
        step = np.zeros(6); step[axis] = 1e-4 if axis < 3 else 1e-5
        columns.append((residual(step) - residual(-step)) / (2*step[axis]))
    jacobian = np.column_stack(columns); norms = np.linalg.norm(jacobian, axis=0)
    singular = np.linalg.svd(jacobian / np.maximum(norms, 1e-12), compute_uv=False)
    rank = int(np.count_nonzero(singular > singular[0] * 1e-5))
    condition = float(singular[0] / max(singular[-1], 1e-15)); reasons = []
    reduction = {}; balanced_reduction = {}
    for split in ("train", "validation"):
        a, b = before["splits"][split], after["splits"][split]
        reduction[split] = float(1 - b["mean_px"] / max(a["mean_px"], 1e-9)) if a else None
        balanced_reduction[split] = float(1 - b["frame_balanced_mean_px"] / max(a["frame_balanced_mean_px"], 1e-9)) if a else None
        if not a:
            if split == "validation": reasons.append("没有独立验证帧；仅供单帧调试，不可应用。")
        elif min(reduction[split], balanced_reduction[split]) < .005:
            reasons.append(f'{"训练" if split == "train" else "验证"}帧误差没有改善至少 0.5%。')
    if rank < 6 or condition > 1e8: reasons.append("车道线几何不足以可靠约束六自由度，请增加弯道、左右及近远处特征。")
    if any(row["visibility_retained"] < .98 for row in after["pairs"]): reasons.append("候选丢失了原本可见的车道线点。")
    if problem["mode"] == "single": reasons.append("单帧结果仅供调试。")
    return {"initial_matrix": initial.tolist(), "optimized_matrix": matrix.tolist(), "increment": xi.tolist(),
            "before": before, "after": after, "error_reduction": reduction, "frame_balanced_error_reduction": balanced_reduction, "history": history,
            "observability": {"rank": rank, "condition": condition}, "usable": not reasons,
            "rejection_reasons": reasons, "elapsed_seconds": time.monotonic() - started}


def write_outputs(problem: dict, result: dict, directory: Path):
    import yaml
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "omnilabel-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "initial_extrinsic.npy", result["initial_matrix"])
    np.save(directory / "initial_input_extrinsic.npy", problem.get("initial_input_matrix", result["initial_matrix"]))
    np.savetxt(directory / "initial_extrinsic.txt", result["initial_matrix"], fmt="%.12g")
    np.save(directory / "optimized_extrinsic.npy", result["optimized_matrix"])
    np.savetxt(directory / "optimized_extrinsic.txt", result["optimized_matrix"], fmt="%.12g")
    (directory / "input_snapshot.json").write_text(json.dumps(problem, ensure_ascii=False, indent=2))
    (directory / "optimization_config.yaml").write_text(yaml.safe_dump({"camera_id": problem["camera"]["id"],
        "conventions": problem["conventions"], "mode": problem["mode"], "request": problem.get("request"), **problem["config"]}, allow_unicode=True))
    (directory / "error_statistics.json").write_text(json.dumps({k: v for k, v in result.items() if k != "history"}, ensure_ascii=False, indent=2))
    with (directory / "optimization_log.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["history"][0])); writer.writeheader(); writer.writerows(result["history"])
    figure, axis = plt.subplots(figsize=(7, 4)); axis.plot([r["iteration"] for r in result["history"]], [r["loss"] for r in result["history"]])
    axis.set(xlabel="Iteration", ylabel="Training objective", title="Shared camera-LiDAR extrinsic optimization")
    figure.tight_layout(); figure.savefig(directory / "loss_curve.png", dpi=160); plt.close(figure)
    output = directory / "before_after"; output.mkdir(exist_ok=True)
    groups = prepare(problem)
    for frame_id in sorted({g["frame_id"] for g in groups}):
        selected = [g for g in groups if g["frame_id"] == frame_id]
        for name, matrix in (("before", result["initial_matrix"]), ("after", result["optimized_matrix"])):
            with Image.open(selected[0]["image_path"]) as source: image = source.convert("RGB")
            draw = ImageDraw.Draw(image)
            for number, group in enumerate(selected, 1):
                draw.line([tuple(p) for p in group["pixels"]], fill="#4ade80", width=3)
                uv, valid = project(problem, group, np.asarray(matrix))
                for i in range(len(uv)-1):
                    if valid[i] and valid[i+1]: draw.line([tuple(uv[i]), tuple(uv[i+1])], fill="#fb923c" if name == "before" else "#38bdf8", width=3)
            rows = [r for r in result[name]["pairs"] if r["frame_id"] == frame_id]
            draw.rectangle((0, 0, image.width, 26), fill="black")
            draw.text((7, 7), f'{name}: mean {np.mean([r["mean_px"] for r in rows]):.2f}px | image reference green | projection {"orange" if name == "before" else "cyan"}', fill="white")
            image.save(output / f"{frame_id}-{name}.png")
