"""Select synchronized registration input without changing annotation coordinates."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .models import Frame
from .storage import Store


@dataclass(frozen=True)
class CalibrationInput:
    points: np.ndarray
    initial_camera_from_input: np.ndarray
    primary_from_input: np.ndarray
    source: str
    path: Path | None

    def camera_from_primary(self, camera_from_input: np.ndarray) -> np.ndarray:
        return np.asarray(camera_from_input, dtype=float) @ np.linalg.inv(self.primary_from_input)

    def camera_from_input(self, camera_from_primary: np.ndarray) -> np.ndarray:
        return np.asarray(camera_from_primary, dtype=float) @ self.primary_from_input


def calibration_input(store: Store, frame: Frame,
                      camera_from_primary: np.ndarray) -> CalibrationInput:
    """Return points and transforms for registration.

    The persisted projection always maps the frame's primary point coordinates
    into the camera. A synchronized scan can use a different sensor coordinate
    frame, so candidates are explicitly converted back before publication.
    """
    effective = np.asarray(camera_from_primary, dtype=float)
    if frame.calibration_pointcloud_file is None:
        return CalibrationInput(store.frame_point_array(frame), effective, np.eye(4),
                                "primary_point_cloud", store.pointcloud_path(frame) if frame.pointcloud_file else None)
    world_from_primary = np.asarray(frame.ego_to_world) @ np.asarray(frame.lidar_to_ego)
    world_from_input = np.asarray(frame.ego_to_world) @ np.asarray(frame.calibration_lidar_to_ego)
    primary_from_input = np.linalg.inv(world_from_primary) @ world_from_input
    return CalibrationInput(
        store.calibration_point_array(frame), effective @ primary_from_input, primary_from_input,
        "synchronized_sensor_scan", store.calibration_pointcloud_path(frame),
    )
