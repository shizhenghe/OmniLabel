"""Deterministic, explicitly synthetic seven-camera clip.

The RGB render and LiDAR samples share world geometry. This is a calibration
fixture and an editable sample, not a recorded drive or model prediction.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import math

import numpy as np
from PIL import Image, ImageDraw


WIDTH, HEIGHT = 640, 360
GROUND = -1.7
PALETTE = {
    "road": (63, 76, 94), "sidewalk": (129, 140, 151),
    "vehicle": (45, 188, 207), "pedestrian": (249, 179, 76),
    "vegetation": (65, 137, 98), "building": (106, 127, 150),
}
CAMERA_NAMES = [
    ("front", "前视"), ("front_left", "左前"), ("left", "左侧"),
    ("rear_left", "左后"), ("rear_right", "右后"),
    ("right", "右侧"), ("front_right", "右前"),
]
FACES = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
         (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]


def _pose(x: float, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0, x], [s, c, 0, 0],
                     [0, 0, 1, 1.7], [0, 0, 0, 1]], dtype=float)


def _camera(index: int) -> dict[str, Any]:
    yaw = index * 2 * math.pi / 7
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[s, -c, 0], [0, 0, -1], [c, s, 0]])
    position = np.array([0.6 * c, 0.6 * s, -0.05])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = -rotation @ position
    camera_id, name = CAMERA_NAMES[index]
    return {"id": camera_id, "name": name, "width": WIDTH, "height": HEIGHT,
            "intrinsics": [[295, 0, WIDTH / 2], [0, 295, HEIGHT / 2], [0, 0, 1]],
            "camera_from_ego": matrix.tolist(), "distortion": [0, 0, 0, 0, 0]}


def _corners(center: list[float], size: list[float], yaw: float = 0) -> np.ndarray:
    x, y, z = (v / 2 for v in size)
    points = np.array([[-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
                       [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]])
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return points @ rotation.T + np.asarray(center)


def _surface_points(rng: np.random.Generator, center: list[float],
                    size: list[float], count: int, yaw: float = 0) -> np.ndarray:
    # Sample actual cuboid surfaces, keeping camera/LiDAR geometry consistent.
    p = rng.uniform(-0.5, 0.5, (count, 3)) * np.array(size)
    axes = rng.integers(0, 3, count)
    signs = rng.choice([-0.5, 0.5], count)
    p[np.arange(count), axes] = signs * np.array(size)[axes]
    c, s = math.cos(yaw), math.sin(yaw)
    r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return p @ r.T + np.array(center)


def _objects(frame_index: int) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for i, (x, y, height, width, length) in enumerate([
        (-20, 19, 9, 12, 18), (12, 19, 10, 12, 20), (43, 19, 14, 12, 22),
        (-13, -19, 8, 12, 20), (20, -19, 12, 12, 22), (52, -19, 9, 12, 20),
    ]):
        objects.append({"id": f"building-{i}", "label": "building",
                        "center": [x, y, height / 2], "size": [length, width, height],
                        "yaw": 0.0, "samples": 230})
    for side in [-1, 1]:
        for i, x in enumerate([-26, -5, 15, 36, 57]):
            objects.append({"id": f"tree-{side}-{i}", "label": "vegetation",
                            "center": [x, side * 11.3, 3.4], "size": [3.2, 3.2, 4.4],
                            "yaw": 0.1, "samples": 100})
    for i, (center, size, yaw) in enumerate([
        ([8, -5.3, 0.8], [4.4, 1.9, 1.6], 0),
        ([29, 5.1, 0.9], [4.8, 2, 1.8], math.pi),
        ([-12, -5.3, 0.8], [4.2, 1.8, 1.6], 0),
        ([20 + frame_index * 0.38, 1.8, 0.8], [4.5, 1.9, 1.6], 0),
    ]):
        objects.append({"id": f"vehicle-{i + 1:03}", "label": "vehicle", "center": center,
                        "size": size, "yaw": yaw, "samples": 220})
    objects.append({"id": "pedestrian-001", "label": "pedestrian",
                    "center": [13 + frame_index * 0.04, -7.1, 0.9],
                    "size": [0.55, 0.55, 1.8], "yaw": 0.0, "samples": 95})
    return objects


def _clip_near(polygon: np.ndarray, near: float = 0.15) -> np.ndarray:
    result = []
    previous = polygon[-1]
    for current in polygon:
        a, b = previous[2] >= near, current[2] >= near
        if a != b:
            t = (near - previous[2]) / (current[2] - previous[2])
            result.append(previous + t * (current - previous))
        if b:
            result.append(current)
        previous = current
    return np.asarray(result)


def _render(path: Path, camera: dict[str, Any], pose: np.ndarray,
            objects: list[dict[str, Any]], frame_index: int) -> None:
    # Avoid re-rendering retained assets when reseeding a fresh data database.
    if path.exists():
        return
    image = Image.new("RGB", (WIDTH, HEIGHT))
    draw = ImageDraw.Draw(image)
    for y in range(HEIGHT):
        if y < HEIGHT / 2:
            t = y / (HEIGHT / 2)
            color = tuple(int(a + t * (b - a)) for a, b in zip((58, 84, 117), (157, 181, 199)))
        else:
            t = (y - HEIGHT / 2) / (HEIGHT / 2)
            color = tuple(int(a + t * (b - a)) for a, b in zip((83, 96, 107), (37, 48, 64)))
        draw.line((0, y, WIDTH, y), fill=color)
    world_to_camera = np.asarray(camera["camera_from_ego"]) @ np.linalg.inv(pose)
    k = np.asarray(camera["intrinsics"])
    surfaces: list[tuple[float, list[tuple[float, float]], tuple[int, int, int]]] = []

    def add_polygon(world: np.ndarray, color: tuple[int, int, int]) -> None:
        transformed = world @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
        clipped = _clip_near(transformed)
        if len(clipped) < 3:
            return
        projected = clipped @ k.T
        uv = projected[:, :2] / projected[:, 2:3]
        # Clipping after projection is performed by Pillow; discard offscreen polygons.
        if (np.max(uv[:, 0]) < 0 or np.min(uv[:, 0]) > WIDTH or
                np.max(uv[:, 1]) < 0 or np.min(uv[:, 1]) > HEIGHT):
            return
        uv = np.clip(uv, -100000, 100000)
        surfaces.append((float(clipped[:, 2].mean()), [tuple(v) for v in uv], color))

    for y0, y1 in [(-12.8, -8), (8, 12.8)]:
        add_polygon(np.array([[-60, y0, .16], [95, y0, .16], [95, y1, .16], [-60, y1, .16]]),
                    PALETTE["sidewalk"])
    for x in range(-60, 95, 6):
        for y in [-3.6, 0, 3.6]:
            add_polygon(np.array([[x, y - .06, .02], [x + 3, y - .06, .02],
                                  [x + 3, y + .06, .02], [x, y + .06, .02]]), (225, 225, 206))
    for y in [-7.75, 7.75]:
        add_polygon(np.array([[-60, y - .07, .03], [95, y - .07, .03],
                              [95, y + .07, .03], [-60, y + .07, .03]]), (230, 184, 91))
    for obj in objects:
        corners = _corners(obj["center"], obj["size"], obj["yaw"])
        color = PALETTE[obj["label"]]
        for face_index, face in enumerate(FACES):
            shade = [0.72, 1.1, 0.80, 0.94, 0.85, 1.0][face_index]
            add_polygon(corners[list(face)], tuple(min(255, int(v * shade)) for v in color))
        if obj["label"] == "vehicle":
            # Windshield on the roof plane: the LiDAR still follows the enclosing cuboid.
            roof = corners[[4, 5, 6, 7]].copy()
            roof[:, :2] = roof[:, :2] * .6 + np.asarray(obj["center"][:2]) * .4
            roof[:, 2] += .005
            add_polygon(roof, (31, 60, 79))
    for _, polygon, color in sorted(surfaces, key=lambda item: item[0], reverse=True):
        draw.polygon(polygon, fill=color)
    draw.rectangle((0, 0, WIDTH, 25), fill=(13, 24, 38))
    draw.text((10, 7), f"SYNTHETIC DEMO  /  {camera['id'].upper()}  /  {frame_index:03}",
              fill=(153, 224, 235))
    draw.text((10, HEIGHT - 20), "Calibrated geometric render - not recorded driving data", fill=(217, 226, 236))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "JPEG", quality=87, optimize=True)


def build_demo(asset_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create a reproducible clip and annotations; only generated RGB assets persist.

    Ego and LiDAR are coincident at 1.7m above world ground. World static surfaces
    remain fixed while the ego pose and one vehicle/pedestrian move with time.
    """
    asset_dir = Path(asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    cameras = [_camera(i) for i in range(7)]
    rng = np.random.default_rng(20261022)
    static_points, static_labels = [], []

    def append(points: np.ndarray, label: str) -> None:
        static_points.append(points)
        static_labels.extend([label] * len(points))

    road = np.column_stack((rng.uniform(-35, 72, 4250), rng.uniform(-8, 8, 4250),
                            rng.normal(0, .013, 4250)))
    append(road, "road")
    sidewalk = np.column_stack((rng.uniform(-35, 72, 650),
                                rng.choice([-1, 1], 650) * rng.uniform(8.1, 12.8, 650),
                                rng.normal(.16, .012, 650)))
    append(sidewalk, "sidewalk")
    for obj in _objects(0):
        if obj["label"] in ["building", "vegetation"]:
            append(_surface_points(rng, obj["center"], obj["size"], obj["samples"], obj["yaw"]), obj["label"])
    static = np.vstack(static_points)
    labels_static = np.asarray(static_labels)
    frames = []
    annotations: dict[str, Any] = {}
    for frame_index in range(12):
        frame_id = f"frame-{frame_index:03}"
        ego_yaw = frame_index * .0015
        pose = _pose(frame_index * .28, ego_yaw)
        inverse = np.linalg.inv(pose)
        objects = _objects(frame_index)
        samples, labels = [static], list(labels_static)
        for obj in objects:
            if obj["label"] in ["vehicle", "pedestrian"]:
                samples.append(_surface_points(rng, obj["center"], obj["size"], obj["samples"], obj["yaw"]))
                labels.extend([obj["label"]] * obj["samples"])
        world = np.vstack(samples)
        lidar = world @ inverse[:3, :3].T + inverse[:3, 3]
        visible = np.linalg.norm(lidar[:, :2], axis=1) < 65
        lidar = lidar[visible]
        labels_array = np.asarray(labels)[visible]
        intensities = np.array([{"road": .32, "sidewalk": .5, "vehicle": .86,
                                "vegetation": .42, "building": .63, "pedestrian": .75}[v]
                               for v in labels_array])
        points = np.column_stack((lidar, intensities)).round(3).tolist()
        images = {}
        for camera in cameras:
            filename = f"{frame_id}-{camera['id']}.jpg"
            _render(asset_dir / filename, camera, pose, objects, frame_index)
            images[camera["id"]] = f"/api/assets/demo/{filename}"
        frames.append({"id": frame_id, "timestamp": frame_index / 10,
                       "ego_to_world": pose.round(9).tolist(), "lidar_to_ego": np.eye(4).tolist(),
                       "points": points, "images": images})
        boxes = []
        for obj in objects:
            if obj["label"] not in ["vehicle", "pedestrian"]:
                continue
            center = inverse[:3, :3] @ np.asarray(obj["center"]) + inverse[:3, 3]
            boxes.append({"id": f"{frame_id}-{obj['id']}", "track_id": obj["id"],
                          "label": obj["label"], "center": center.round(4).tolist(),
                          "size": obj["size"], "yaw": round(obj["yaw"] - ego_yaw, 6)})
        vectors = []
        for i, y in enumerate([-7.75, -3.6, 0, 3.6, 7.75]):
            world_line = np.array([[x, y, .025] for x in np.linspace(-20, 60, 17)])
            line = world_line @ inverse[:3, :3].T + inverse[:3, 3]
            vectors.append({"id": f"lane-{i}", "label": "lane_boundary" if abs(y) > 7 else "lane",
                            "points": line.round(4).tolist(), "closed": False})
        annotations[frame_id] = {"boxes": boxes, "vectors": vectors, "image_polygons": [],
                                 "point_labels": {str(i): str(label) for i, label in enumerate(labels_array)},
                                 "revision": 0}
    clip = {"id": "demo", "name": "滨海大道 · 合成演示",
            "description": "12帧 / 7路环视 / 已知标定与位姿。全部图像、点云及标注均为程序生成的合成样例，用于编辑和几何链路验证，不是真实采集数据或模型推理结果。",
            "fps": 10, "cameras": cameras, "frames": frames, "source": "synthetic"}
    return clip, annotations
