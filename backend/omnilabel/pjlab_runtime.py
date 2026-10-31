"""Adapter for PJLab SensorsCalibration/lidar2camera/auto_calib (CRLF).

Run the unmodified native executable on the original sensor coordinates.
It requires a road-line/pole feature mask, not an RGB photograph. Nothing
here substitutes an image-edge heuristic for the upstream segmentation model.
"""
from __future__ import annotations

from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import uuid

import numpy as np
from PIL import Image, UnidentifiedImageError
from scipy.spatial.transform import Rotation

from .models import Camera, rigid_transform
from .koide_runtime import _stop_process


def _executable() -> str:
    name = os.environ.get("OMNILABEL_PJLAB_BINARY", "").strip()
    binary = shutil.which(name) if name else None
    if not binary:
        raise ValueError("PJLab 未部署：请编译 SensorsCalibration/lidar2camera/auto_calib，"
                         "设置 OMNILABEL_PJLAB_BINARY，并为每帧相机上传线特征掩码。")
    return str(Path(binary).resolve())


def pjlab_status() -> dict:
    try:
        _executable()
        automatic_masks = mask_command() is not None
        masks = "已配置逐帧自动线特征提取。" if automatic_masks else "每组图像需上传同尺寸线特征掩码。"
        return {"available": True, "detail": f"PJLab 已配置；{masks}实际运行验证原生依赖与输出，结果需审核。"}
    except ValueError as exc:
        return {"available": False, "detail": str(exc)}


def mask_command() -> list[str] | None:
    raw = os.environ.get("OMNILABEL_PJLAB_MASK_COMMAND", "").strip()
    if not raw:
        return None
    try:
        command = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("OMNILABEL_PJLAB_MASK_COMMAND 必须是 JSON 字符串参数数组。") from exc
    if (not isinstance(command, list) or not command or any(not isinstance(v, str) or not v for v in command)
            or not shutil.which(command[0]) or not any("{image}" in v for v in command)
            or not any("{mask}" in v for v in command)):
        raise ValueError("PJLab 掩码命令需为可执行的 JSON 参数数组，并包含 {image} 与 {mask}。")
    return command


def _run(command: list[str], directory: Path, log_file: Path, timeout: float):
    with log_file.open("wb") as log:
        process = subprocess.Popen(command, cwd=directory, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            _stop_process(process)
            raise ValueError(f"PJLab 超过 {timeout:g} 秒，已终止；日志：{log_file}") from exc
    if process.returncode:
        raise ValueError(f"PJLab 执行失败（exit={process.returncode}），请检查日志：{log_file}")


def validate_mask(content: bytes, camera: Camera) -> Image.Image:
    try:
        with Image.open(BytesIO(content)) as image:
            if image.format != "PNG" or image.size != (camera.width, camera.height):
                raise ValueError("线特征掩码必须是与原图同尺寸的 PNG，不可缩放或预先去畸变。")
            image.load()
            result = image.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("无法读取线特征 PNG 掩码。") from exc
    foreground = np.mean(np.asarray(result), axis=2) > 55
    fraction = float(np.mean(foreground))
    if not 0 < fraction < .8:
        raise ValueError("掩码必须包含道路标线/杆体特征与黑色背景，不能是全黑、全白或普通照片。")
    return result


def _parse_result(log: str) -> list[list[float]]:
    # Upstream prints the final 3x3 rotation immediately before this marker.
    # Read it directly; its separately printed Euler angles have ambiguous units.
    parts = log.rsplit("refine extrinsic:", 1)
    if len(parts) != 2:
        raise ValueError("PJLab 未输出 refine extrinsic，不能使用初值代替结果。")
    try:
        rows = [[float(v) for v in line.split()] for line in parts[0].strip().splitlines()[-3:]]
        translation = re.search(r"^translation:\s*(.+)$", parts[1], re.MULTILINE)
        if translation is None:
            raise ValueError("missing translation")
        matrix = np.eye(4)
        matrix[:3, :3] = np.asarray(rows).reshape(3, 3)
        matrix[:3, 3] = [float(v) for v in translation.group(1).split()]
        return rigid_transform(matrix.tolist())
    except (ValueError, TypeError) as exc:
        raise ValueError("PJLab 输出不是有限、右手正交的刚体外参。") from exc


def assess_candidate(initial, matrix, before, after) -> dict:
    """Native CRLF line-feature support, not a pixel-error/holdout claim."""
    try:
        before, after = float(before), float(after)
    except (TypeError, ValueError) as exc:
        raise ValueError("PJLab 未输出有效的线特征匹配评分。") from exc
    if not np.isfinite([before, after]).all() or not (0 <= before <= 1 and 0 <= after <= 1):
        raise ValueError("PJLab 未输出有效的线特征匹配评分。")
    initial = np.asarray(rigid_transform(np.asarray(initial).tolist()))
    matrix = np.asarray(rigid_transform(np.asarray(matrix).tolist()))
    delta = matrix @ np.linalg.inv(initial)
    shift = float(np.linalg.norm(delta[:3, 3]))
    angle = float(np.rad2deg(Rotation.from_matrix(delta[:3, :3]).magnitude()))
    reasons = []
    if after < .1 or after - before < .01:
        reasons.append("线特征匹配改善不足；初值可能已准确，或掩码/扫描不支持可靠修正。")
    if shift > .75 or angle > 3:
        reasons.append("修正超过 0.75 米或 3 度，请先手动校准初值并检查同步情况。")
    return {"usable": not reasons, "rejection_reasons": reasons, "metrics": {
        "score_name": "pjlab_line_feature_overlap", "line_overlap_before": before,
        "line_overlap_after": after, "line_overlap_improvement": after - before,
        "translation_delta_m": shift, "rotation_delta_degrees": angle,
        "engine": "pjlab", "validation": "native_line_feature_fit", "independent_holdout": False,
        "score_note": "上游提取线特征的掩码匹配比例；不是像素误差或独立验证集精度。"}}


def run_pjlab_calibration(*, points: np.ndarray, image_path: Path, camera: Camera,
                          initial: np.ndarray, mask_path: Path | None, work_dir: Path) -> dict:
    binary = _executable()
    initial = np.asarray(rigid_transform(np.asarray(initial).tolist()))
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 4 or not len(points) or not np.isfinite(points).all():
        raise ValueError("PJLab 需要有效的原始 XYZI 点云。")
    if np.ptp(points[:, 3]) <= 0:
        raise ValueError("PJLab 道路标线提取需要实测且有变化的反射强度。")
    with Image.open(image_path) as image:
        if image.size != (camera.width, camera.height):
            raise ValueError("原始图像尺寸与相机内参不一致。")
    try:
        timeout = float(os.environ.get("OMNILABEL_PJLAB_TIMEOUT", "300"))
    except ValueError as exc:
        raise ValueError("OMNILABEL_PJLAB_TIMEOUT 必须是 1–3600 秒。") from exc
    if not np.isfinite(timeout) or not 1 <= timeout <= 3600:
        raise ValueError("OMNILABEL_PJLAB_TIMEOUT 必须是 1–3600 秒。")
    # Fresh directory prevents stale output or concurrent pairs being confused.
    directory = work_dir / f"pjlab-{uuid.uuid4().hex}"
    directory.mkdir(parents=True)
    mask_file = directory / "mask.png"
    if mask_path is not None and mask_path.is_file():
        mask = validate_mask(mask_path.read_bytes(), camera)
    else:
        command = mask_command()
        if command is None:
            raise ValueError("PJLab 缺少当前帧线特征掩码；请上传或配置 OMNILABEL_PJLAB_MASK_COMMAND 自动提取。")
        command = [arg.replace("{image}", str(image_path.resolve())).replace("{mask}", str(mask_file.resolve()))
                   for arg in command]
        _run(command, directory, directory / "mask.log", timeout)
        if not mask_file.is_file():
            raise ValueError("PJLab 掩码命令未生成当前图像的 mask.png。")
        mask = validate_mask(mask_file.read_bytes(), camera)
    mask.save(mask_file)
    cloud_file = directory / "scan.pcd"
    header = ("# .PCD v0.7\nVERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\n"
              "TYPE F F F F\nCOUNT 1 1 1 1\n" + f"WIDTH {len(points)}\nHEIGHT 1\n"
              f"VIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(points)}\nDATA binary\n")
    cloud = points.astype("<f4")
    # CRLF's upstream lane extractor uses an 80/255 reflectivity threshold.
    # Preserve measured contrast when the importer supplies normalized [0,1].
    if cloud[:, 3].min() >= 0 and cloud[:, 3].max() <= 1:
        cloud[:, 3] *= 255
    if not np.isfinite(cloud).all():
        raise ValueError("点云坐标超出 PJLab PCD float32 范围。")
    with cloud_file.open("wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(cloud.tobytes())
    def cv_matrix(data):
        return {"rows": len(data), "cols": len(data[0]), "type": 6, "continuous": True, "data": data}
    intrinsics = directory / "intrinsic.json"
    extrinsics = directory / "extrinsic.json"
    intrinsics.write_text(json.dumps({"camera-intrinsic": {"param": {
        "img_dist_w": camera.width, "img_dist_h": camera.height,
        "cam_K": cv_matrix(camera.intrinsics), "cam_dist": cv_matrix([camera.distortion or [0.] * 5])}}}))
    extrinsics.write_text(json.dumps({"lidar-to-camera-extrinsic": {"param": {
        "time_lag": 0, "sensor_calib": cv_matrix(initial.tolist())}}}))
    log_file = directory / "pjlab.log"
    _run([binary, str(mask_file.resolve()), str(cloud_file.resolve()),
          str(intrinsics.resolve()), str(extrinsics.resolve())], directory, log_file, timeout)
    with log_file.open("rb") as log:
        log.seek(max(0, log_file.stat().st_size - 65536))
        output = log.read().decode("utf-8", errors="replace")
        matrix = _parse_result(output)
    scores = [re.findall(rf"^{stage} loss:\s*(\S+)", output, re.MULTILINE) for stage in ("before", "after")]
    if any(len(values) != 1 for values in scores):
        raise ValueError("PJLab 缺少唯一的前后线特征匹配评分，不能保存。")
    assessment = assess_candidate(initial, matrix, scores[0][0], scores[1][0])
    return {"matrix": matrix, "engine": "pjlab", "method": "pjlab", **assessment, "artifacts": {
        "directory": str(directory), "log": str(log_file), "mask": str(mask_file)}}
