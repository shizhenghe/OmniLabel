"""Interchange adapter for the real Koide direct_visual_lidar_calibration.

The optimizer runs in an external ROS installation or the supplied Docker
worker.  This module deliberately has no substitute optimizer.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Sequence

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from .models import Camera, rigid_transform


class KoideRuntimeError(RuntimeError):
    def __init__(self, message: str, *, code: str = "koide_failed", artifacts: dict | None = None):
        super().__init__(message)
        self.code = code
        self.artifacts = artifacts or {}


class KoideUnavailableError(KoideRuntimeError):
    def __init__(self, message: str):
        super().__init__(message, code="koide_unavailable")


@dataclass(frozen=True)
class KoideFrame:
    """Synchronized image and Nx4 LiDAR scan in a shared sensor coordinate frame."""

    points: np.ndarray
    image_path: Path


def _worker_path() -> Path:
    return Path(__file__).resolve().parents[2] / "scripts" / "koide_worker.py"


def _command_template() -> tuple[list[str], str]:
    raw = os.environ.get("OMNILABEL_KOIDE_COMMAND", "").strip()
    if raw:
        try:
            command = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise KoideUnavailableError("OMNILABEL_KOIDE_COMMAND 必须是 JSON 参数数组；请参考 docs/koide-calibration.md。") from exc
        if not isinstance(command, list) or not command or any(not isinstance(v, str) or not v for v in command):
            raise KoideUnavailableError("OMNILABEL_KOIDE_COMMAND 必须是非空 JSON 字符串数组。")
        return command, "configured_command"
    image = os.environ.get("OMNILABEL_KOIDE_DOCKER_IMAGE", "").strip()
    if image:
        return [sys.executable, str(_worker_path()), "--docker-image", image, "--data-dir", "{data_dir}"], "docker"
    if shutil.which("ros2"):
        return ["ros2", "run", "direct_visual_lidar_calibration", "calibrate", "{data_dir}"], "ros2"
    if shutil.which("rosrun"):
        return ["rosrun", "direct_visual_lidar_calibration", "calibrate", "{data_dir}"], "ros1"
    raise KoideUnavailableError(
        "Koide 运行环境不可用。请部署 docs/koide-calibration.md 中的 Docker worker，"
        "或配置已安装 Koide 的 ROS 环境及 OMNILABEL_KOIDE_COMMAND。"
    )


def _probe(command: list[str], *, timeout: float = 5) -> str | None:
    try:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)
    return None if result.returncode == 0 else result.stdout.strip()[-1000:] or f"exit={result.returncode}"


def koide_status() -> dict:
    """Report runtime availability, without claiming an accuracy guarantee."""
    try:
        command, runtime = _command_template()
        executable = shutil.which(command[0])
        if not executable:
            raise KoideUnavailableError(f"Koide 启动程序不存在或不可执行：{command[0]}")
        error = None
        if runtime == "docker":
            error = _probe([*command[:], "--check"], timeout=8)
        elif runtime == "ros2":
            error = _probe(["ros2", "pkg", "prefix", "direct_visual_lidar_calibration"])
        elif runtime == "ros1":
            error = _probe(["rospack", "find", "direct_visual_lidar_calibration"])
        if error:
            raise KoideUnavailableError(f"Koide 运行环境检查失败：{error}")
        timeout = _configured_timeout()
        detail = ("已配置外部 Koide 命令，实际执行时验证结果。" if runtime == "configured_command"
                  else "Koide 运行环境已就绪。") + f" 单组最长运行 {timeout:g} 秒。"
        return {"engine": "koide", "available": True, "runtime": runtime,
                "verified": runtime != "configured_command", "timeout_seconds": timeout,
                "message": detail, "detail": detail}
    except KoideRuntimeError as exc:
        return {"engine": "koide", "available": False, "runtime": None,
                "verified": False, "message": str(exc), "detail": str(exc)}


def _equalize_gray(source: Image.Image) -> Image.Image:
    """OpenCV equalizeHist's CDF/minimum convention, using existing dependencies."""
    gray = np.asarray(source.convert("L"), dtype=np.uint8)
    hist = np.bincount(gray.ravel(), minlength=256)
    populated = np.flatnonzero(hist)
    if len(populated) <= 1:
        raise ValueError("图像没有足够的灰度变化，无法执行 Koide 自动配准。")
    cdf = hist.cumsum()
    minimum = cdf[populated[0]]
    lut = np.rint(np.clip((cdf - minimum) * (255.0 / (gray.size - minimum)), 0, 255)).astype(np.uint8)
    return Image.fromarray(lut[gray])


def _write_frame(frame: KoideFrame, camera: Camera, directory: Path, stem: str) -> dict:
    points = np.asarray(frame.points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError("Koide 输入点云必须包含 x,y,z,intensity 四列。")
    finite = np.isfinite(points).all(axis=1)
    # Upstream preprocess also excludes invalid returns and points <1 metre.
    points = points[finite & (np.linalg.norm(points[:, :3], axis=1) >= 1.0)]
    if len(points) < 32:
        raise ValueError("有效 LiDAR 点不足 32 个，无法执行 Koide 自动配准。")
    if np.ptp(points[:, 3]) <= 1e-9:
        raise ValueError("点云 intensity 没有变化；Koide 精准配准需要真实反射强度，不能使用全零或伪造强度。")
    # Same quantized rank equalization as Koide preprocess.cpp.  Rejecting
    # constant raw intensity above prevents manufacturing a useful-looking
    # signal from the rank order of entirely missing intensity measurements.
    order = np.argsort(points[:, 3], kind="stable")
    intensities = np.empty(len(points), dtype=np.float32)
    intensities[order] = np.floor(256.0 * np.arange(len(points)) / len(points)) / 256.0
    vertices = np.empty((len(points), 4), dtype="<f4")
    vertices[:, :3] = points[:, :3]
    vertices[:, 3] = intensities
    if not np.isfinite(vertices).all():
        raise ValueError("点云坐标超出 Koide PLY float32 的有效范围。")
    ply_path = directory / f"{stem}.ply"
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(points)}\nproperty float x\nproperty float y\n"
              "property float z\nproperty float intensity\nend_header\n")
    with ply_path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(vertices.tobytes(order="C"))
    with Image.open(frame.image_path) as source:
        if source.size != (camera.width, camera.height):
            raise ValueError("实际图像尺寸与相机内参记录不一致；请先修正数据，避免错误缩放导致配准偏移。")
        _equalize_gray(source).save(directory / f"{stem}.png")
    return {"name": stem, "source_points": len(frame.points), "used_points": len(points),
            "source_image": str(Path(frame.image_path).resolve())}


def prepare_koide_dataset(*, frames: Sequence[KoideFrame], camera: Camera,
                         initial_camera_from_lidar: np.ndarray, work_dir: Path) -> dict:
    """Write the upstream preprocessed format; source scans/images are untouched."""
    if not frames:
        raise ValueError("Koide 至少需要一组同步图像和点云。")
    initial = np.asarray(initial_camera_from_lidar, dtype=float)
    rigid_transform(initial.tolist())
    k = np.asarray(camera.intrinsics)
    if not np.isclose(k[0, 1], 0) or not np.isclose(k[1, 0], 0):
        raise ValueError("当前 Koide 适配器不支持非零 skew 相机内参。")
    directory = Path(work_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    frame_meta = [_write_frame(frame, camera, directory, f"frame_{index:06d}")
                  for index, frame in enumerate(frames)]
    # Upstream stores camera->LiDAR, opposite to the platform's projection.
    lidar_from_camera = np.linalg.inv(initial)
    quat = Rotation.from_matrix(lidar_from_camera[:3, :3]).as_quat()
    config = {
        "camera": {"camera_model": "plumb_bob", "intrinsics": [float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])],
                   "distortion_coeffs": list(camera.distortion) or [0.0] * 5},
        "meta": {"bag_names": [f["name"] for f in frame_meta], "data_path": str(directory),
                 "intensity_channel": "intensity", "source": "OmniLabel synchronized pairs",
                 "preprocessing": "equalized grayscale; quantized intensity rank; finite returns at least 1m", "frames": frame_meta},
        "results": {"init_T_lidar_camera": [*lidar_from_camera[:3, 3].tolist(), *quat.tolist()]},
    }
    payload = json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    (directory / "calib.input.json").write_text(payload, encoding="utf-8")
    (directory / "calib.json").write_text(payload, encoding="utf-8")
    return config


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=8)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _configured_timeout() -> float:
    raw = os.environ.get("OMNILABEL_KOIDE_TIMEOUT_SECONDS", "900")
    try:
        timeout = float(raw)
    except ValueError as exc:
        raise KoideUnavailableError("OMNILABEL_KOIDE_TIMEOUT_SECONDS 必须是正数秒数。") from exc
    if not np.isfinite(timeout) or timeout <= 0:
        raise KoideUnavailableError("OMNILABEL_KOIDE_TIMEOUT_SECONDS 必须是正数秒数。")
    return timeout


def run_koide_calibration(*, points: np.ndarray | None = None, image_path: Path | None = None,
                          camera: Camera, initial_camera_from_lidar: np.ndarray,
                          work_dir: Path, timeout_seconds: float | None = None,
                          frames: Sequence[KoideFrame] | None = None) -> dict:
    """Run genuine Koide refinement and return a LiDAR->camera matrix.

    ``frames`` optionally shares one rigid extrinsic over multiple synchronized
    pairs.  The ordinary points/image_path form calibrates one pair.  Command
    configuration is server-side environment only, never request-provided code.
    """
    timeout_seconds = _configured_timeout() if timeout_seconds is None else timeout_seconds
    if not np.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("Koide timeout 必须是正数。")
    command, runtime = _command_template()
    if not shutil.which(command[0]):
        raise KoideUnavailableError(f"Koide 启动程序不存在或不可执行：{command[0]}")
    if frames is None:
        if points is None or image_path is None:
            raise ValueError("缺少同步点云或图像。")
        frames = [KoideFrame(points=points, image_path=Path(image_path))]
    elif points is not None or image_path is not None:
        raise ValueError("frames 与单帧 points/image_path 不能同时指定。")
    directory = Path(work_dir).resolve()
    prepare_koide_dataset(frames=frames, camera=camera,
                          initial_camera_from_lidar=initial_camera_from_lidar, work_dir=directory)
    has_path = any("{data_dir}" in arg for arg in command)
    command = [arg.replace("{data_dir}", str(directory)) for arg in command]
    if not has_path:
        command.append(str(directory))
    if runtime == "docker" and "--timeout" not in command:
        # Let the wrapper clean up its named container just before the API's
        # outer deadline. This avoids orphaned calibration containers.
        margin = min(15.0, timeout_seconds * 0.1)
        command.extend(("--timeout", f"{max(0.05, timeout_seconds - margin):g}"))
    for flag in ("--background", "--auto_quit"):
        if flag not in command:
            command.append(flag)
    artifacts = {"directory": str(directory), "calib_json": str(directory / "calib.json"),
                 "input_json": str(directory / "calib.input.json"), "log": str(directory / "koide.log")}
    started = time.monotonic()
    with (directory / "koide.log").open("w", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       cwd=directory, start_new_session=True, shell=False)
        except OSError as exc:
            raise KoideRuntimeError(f"无法启动 Koide：{exc}", code="koide_unavailable", artifacts=artifacts) from exc
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            _stop_process(process)
            raise KoideRuntimeError(f"Koide 超过 {timeout_seconds:g} 秒，已终止；详见运行日志。",
                                    code="koide_timeout", artifacts=artifacts) from exc
        except BaseException:
            _stop_process(process)
            raise
    if returncode:
        raise KoideRuntimeError(f"Koide 执行失败（exit={returncode}）；详见 {directory / 'koide.log'}。", artifacts=artifacts)
    try:
        config = json.loads((directory / "calib.json").read_text(encoding="utf-8"))
        values = np.asarray(config["results"]["T_lidar_camera"], dtype=float)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError("T_lidar_camera must contain seven finite values")
        quaternion = values[3:]
        norm = np.linalg.norm(quaternion)
        if not 0.99 <= norm <= 1.01:
            raise ValueError("T_lidar_camera quaternion is not normalized")
        lidar_from_camera = np.eye(4)
        lidar_from_camera[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
        lidar_from_camera[:3, 3] = values[:3]
        matrix = np.linalg.inv(lidar_from_camera)
        rigid_transform(matrix.tolist())
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise KoideRuntimeError(f"Koide 未生成有效标定结果：{exc}。", code="koide_invalid_result", artifacts=artifacts) from exc
    return {"matrix": matrix.tolist(), "engine": "koide", "runtime": runtime,
            "artifacts": artifacts, "elapsed_seconds": round(time.monotonic() - started, 3),
            "frame_count": len(frames), "convention": "camera_from_lidar",
            "intrinsics_optimized": False}
