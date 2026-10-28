"""Real subprocess protocol tests; the test executable is explicitly a fake Koide."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import pytest
from scipy.spatial.transform import Rotation

from omnilabel.koide_runtime import (
    KoideFrame, KoideRuntimeError, koide_status, prepare_koide_dataset,
    run_koide_calibration,
)
from omnilabel.models import Camera


@pytest.fixture
def koide_input(tmp_path):
    camera = Camera(id="front", name="Front", width=40, height=30,
                    intrinsics=[[30, 0, 20], [0, 31, 15], [0, 0, 1]],
                    camera_from_ego=np.eye(4).tolist(), distortion=[.01, .02, .001, -.001, 0])
    image_path = tmp_path / "original.jpg"
    Image.fromarray(np.arange(1200, dtype=np.uint8).reshape(30, 40)).save(image_path)
    points = np.column_stack([np.linspace(-3, 3, 64), np.ones(64), np.full(64, 7), np.linspace(0, 255, 64)])
    initial = np.eye(4)
    initial[:3, :3] = Rotation.from_euler("xyz", [.3, -.1, .4]).as_matrix()
    initial[:3, 3] = [.1, .2, -.3]
    return dict(camera=camera, image_path=image_path, points=points,
                initial_camera_from_lidar=initial, work_dir=tmp_path / "output with spaces $(literal)")


def configure_fake(monkeypatch, tmp_path: Path, body: str):
    script = tmp_path / "fake_koide.py"
    script.write_text(body, encoding="utf-8")
    monkeypatch.setenv("OMNILABEL_KOIDE_COMMAND", json.dumps([sys.executable, str(script), "{data_dir}"]))


def test_preprocessed_dataset_and_matrix_inverse_preserve_source(monkeypatch, tmp_path, koide_input):
    source = koide_input["image_path"].read_bytes()
    original_points = koide_input["points"].copy()
    expected = np.eye(4)
    expected[:3, :3] = Rotation.from_euler("xyz", [-.2, .15, .7]).as_matrix()
    expected[:3, 3] = [.4, -.6, .2]
    inverse = np.linalg.inv(expected)
    raw = [*inverse[:3, 3].tolist(), *Rotation.from_matrix(inverse[:3, :3]).as_quat().tolist()]
    configure_fake(monkeypatch, tmp_path,
        "import json, sys\nfrom pathlib import Path\n"
        "assert '--background' in sys.argv and '--auto_quit' in sys.argv\n"
        "p = Path(sys.argv[1]) / 'calib.json'\n"
        "c = json.loads(p.read_text())\n"
        f"c['results']['T_lidar_camera'] = {raw!r}\n"
        "p.write_text(json.dumps(c))\n")
    result = run_koide_calibration(**koide_input)
    assert np.allclose(result["matrix"], expected)
    assert result["engine"] == "koide" and not result["intrinsics_optimized"]
    assert koide_input["image_path"].read_bytes() == source
    assert np.array_equal(koide_input["points"], original_points)
    config = json.loads(Path(result["artifacts"]["input_json"]).read_text())
    assert config["camera"]["intrinsics"] == [30, 31, 20, 15]
    assert config["camera"]["distortion_coeffs"] == koide_input["camera"].distortion
    initial_lidar_camera = np.linalg.inv(koide_input["initial_camera_from_lidar"])
    values = config["results"]["init_T_lidar_camera"]
    assert np.allclose(values[:3], initial_lidar_camera[:3, 3])
    assert np.allclose(Rotation.from_quat(values[3:]).as_matrix(), initial_lidar_camera[:3, :3])
    body = (koide_input["work_dir"] / "frame_000000.ply").read_bytes().split(b"end_header\n", 1)
    assert b"binary_little_endian" in body[0] and b"property float intensity" in body[0]
    vertices = np.frombuffer(body[1], dtype="<f4").reshape(-1, 4)
    assert np.allclose(vertices[:, :3], original_points[:, :3])
    assert vertices[:, 3].min() == 0 and vertices[:, 3].max() < 1
    with Image.open(koide_input["work_dir"] / "frame_000000.png") as png:
        assert png.size == (40, 30) and png.mode == "L"


@pytest.mark.parametrize("body,code", [
    ("raise SystemExit(7)\n", "koide_failed"),
    ("pass\n", "koide_invalid_result"),
    ("import json,sys\nfrom pathlib import Path\np=Path(sys.argv[1])/'calib.json'\n"
     "c=json.loads(p.read_text())\nc['results']['T_lidar_camera']=[0,0,0,0,0,0,0]\np.write_text(json.dumps(c))\n", "koide_invalid_result"),
])
def test_failure_never_reports_an_initial_or_stale_transform(monkeypatch, tmp_path, koide_input, body, code):
    configure_fake(monkeypatch, tmp_path, body)
    koide_input["work_dir"].mkdir()
    (koide_input["work_dir"] / "calib.json").write_text('{"results":{"T_lidar_camera":[0,0,0,0,0,0,1]}}')
    with pytest.raises(KoideRuntimeError) as caught:
        run_koide_calibration(**koide_input)
    assert caught.value.code == code
    assert Path(caught.value.artifacts["log"]).is_file()


def test_timeout_stops_external_process(monkeypatch, tmp_path, koide_input):
    configure_fake(monkeypatch, tmp_path, "import time\ntime.sleep(60)\n")
    started = time.monotonic()
    with pytest.raises(KoideRuntimeError) as caught:
        run_koide_calibration(**koide_input, timeout_seconds=.2)
    assert caught.value.code == "koide_timeout"
    assert time.monotonic() - started < 4


def test_multi_frame_and_missing_real_intensity(tmp_path, koide_input):
    arguments = {key: value for key, value in koide_input.items() if key not in ("points", "image_path")}
    frames = [KoideFrame(koide_input["points"], koide_input["image_path"])] * 2
    config = prepare_koide_dataset(**arguments, frames=frames)
    assert config["meta"]["bag_names"] == ["frame_000000", "frame_000001"]
    koide_input["points"][:, 3] = 0
    with pytest.raises(ValueError, match="intensity"):
        prepare_koide_dataset(**arguments, frames=frames)


def test_unavailable_and_bad_server_command_are_explained(monkeypatch):
    monkeypatch.setenv("OMNILABEL_KOIDE_COMMAND", "ros2 run is not JSON")
    status = koide_status()
    assert not status["available"] and "JSON" in status["detail"]
    monkeypatch.delenv("OMNILABEL_KOIDE_COMMAND")
    monkeypatch.delenv("OMNILABEL_KOIDE_DOCKER_IMAGE", raising=False)
    monkeypatch.setattr("omnilabel.koide_runtime.shutil.which", lambda _: None)
    status = koide_status()
    assert not status["available"] and "Koide" in status["detail"]
