"""Native CLI contract tests; the fixture is not an accuracy validation of CRLF."""
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from omnilabel.models import Camera
from omnilabel.pjlab_runtime import _parse_result, assess_candidate, pjlab_status, run_pjlab_calibration, validate_mask


@pytest.fixture
def inputs(tmp_path):
    camera = Camera(id="front", name="Front", width=40, height=30,
                    intrinsics=[[30, 0, 20], [0, 31, 15], [0, 0, 1]],
                    distortion=[.01, -.02, 0, 0, .003], camera_from_ego=np.eye(4).tolist())
    image = tmp_path / "image.png"
    Image.new("RGB", (40, 30), (50, 60, 70)).save(image)
    mask = tmp_path / "mask.png"
    pixels = np.zeros((30, 40, 3), dtype=np.uint8)
    pixels[10:15, 5:35] = 255
    Image.fromarray(pixels).save(mask)
    return dict(points=np.array([[10, 0, 1, 30], [20, 1, 2, 100]], dtype=float),
                image_path=image, mask_path=mask, camera=camera, initial=np.eye(4), work_dir=tmp_path / "work")


def binary(monkeypatch, tmp_path, body):
    path = tmp_path / "native-fixture"
    path.write_text("#!/usr/bin/env python3\n" + body)
    path.chmod(0o755)
    monkeypatch.setenv("OMNILABEL_PJLAB_BINARY", str(path))


def test_native_arguments_original_coordinates_and_json(monkeypatch, tmp_path, inputs):
    binary(monkeypatch, tmp_path,
           "import sys,json\nfrom pathlib import Path\n"
           "assert len(sys.argv)==5\n"
           "assert Path(sys.argv[1]).name=='mask.png'\n"
           "assert b'FIELDS x y z intensity' in Path(sys.argv[2]).read_bytes()\n"
           "k=json.loads(Path(sys.argv[3]).read_text())['camera-intrinsic']['param']\n"
           "assert k['cam_K']['data'][1][1]==31\n"
           "assert k['cam_dist']['data'][0][-1]==.003\n"
           "print('before loss: .6\\nafter loss: .8\\n1 0 0\\n0 1 0\\n0 0 1\\nrefine extrinsic: \\nrotation: 0 0 0\\ntranslation: .12 -.03 .01')\n")
    before = inputs["points"].copy()
    result = run_pjlab_calibration(**inputs)
    assert np.allclose(np.asarray(result["matrix"])[:3, 3], [.12, -.03, .01])
    assert np.array_equal(inputs["points"], before)
    directory = Path(result["artifacts"]["directory"])
    raw = (directory / "scan.pcd").read_bytes().split(b"DATA binary\n")[1]
    assert np.allclose(np.frombuffer(raw, dtype="<f4").reshape(-1, 4), before)
    assert pjlab_status()["available"]
    # A second invocation cannot consume a previous pair's result files/log.
    second = run_pjlab_calibration(**inputs)
    assert second["artifacts"]["directory"] != str(directory)


@pytest.mark.parametrize("output", ["", "refine extrinsic:\ntranslation: 1 2 3", "1 0 0\n0 1 0\n0 0 -1\nrefine extrinsic:\ntranslation: 0 0 0", "nan 0 0\n0 1 0\n0 0 1\nrefine extrinsic:\ntranslation: 0 0 0"])
def test_no_missing_nonfinite_or_reflected_result(output):
    with pytest.raises(ValueError):
        _parse_result(output)


def test_native_quality_requires_improvement_and_bounded_motion():
    initial = np.eye(4)
    assert assess_candidate(initial, initial, .6, .8)["usable"]
    assert not assess_candidate(initial, initial, .8, .6)["usable"]
    matrix = initial.copy(); matrix[0, 3] = .9
    assert not assess_candidate(initial, matrix, .6, .8)["usable"]
    with pytest.raises(ValueError):
        assess_candidate(initial, initial, float('nan'), .8)


def test_failure_timeout_and_missing_runtime(monkeypatch, tmp_path, inputs):
    monkeypatch.delenv("OMNILABEL_PJLAB_BINARY", raising=False)
    assert not pjlab_status()["available"]
    with pytest.raises(ValueError, match="未部署"):
        run_pjlab_calibration(**inputs)
    binary(monkeypatch, tmp_path, "raise SystemExit(7)\n")
    with pytest.raises(ValueError, match="exit=7"):
        run_pjlab_calibration(**inputs)
    binary(monkeypatch, tmp_path, "import time\ntime.sleep(10)\n")
    monkeypatch.setenv("OMNILABEL_PJLAB_TIMEOUT", "1")
    with pytest.raises(ValueError, match="已终止"):
        run_pjlab_calibration(**inputs)


def test_mask_and_reflectivity_requirements(monkeypatch, tmp_path, inputs):
    for size, color in [((41, 30), 0), ((40, 30), 0), ((40, 30), 255)]:
        stream = BytesIO(); Image.new("RGB", size, (color,) * 3).save(stream, format="PNG")
        with pytest.raises(ValueError):
            validate_mask(stream.getvalue(), inputs["camera"])
    binary(monkeypatch, tmp_path, "pass\n")
    inputs["points"][:, 3] = 0
    with pytest.raises(ValueError, match="反射强度"):
        run_pjlab_calibration(**inputs)


def test_automatic_mask_generator_and_normalized_reflectivity(monkeypatch, tmp_path, inputs):
    import sys
    generator = tmp_path / "generate.py"
    generator.write_text("from PIL import Image,ImageDraw\nimport sys\n"
                         "image=Image.open(sys.argv[1])\n"
                         "mask=Image.new('RGB',image.size)\n"
                         "ImageDraw.Draw(mask).rectangle((5,10,35,15),fill='white')\n"
                         "mask.save(sys.argv[2])\n")
    monkeypatch.setenv("OMNILABEL_PJLAB_MASK_COMMAND", json.dumps(
        [sys.executable, str(generator), "{image}", "{mask}"]))
    binary(monkeypatch, tmp_path, "print('before loss: .6\\nafter loss: .8\\n1 0 0\\n0 1 0\\n0 0 1\\nrefine extrinsic:\\ntranslation: 0 0 0')\n")
    inputs["mask_path"] = None
    inputs["points"][:, 3] = [.1, .8]
    result = run_pjlab_calibration(**inputs)
    directory = Path(result["artifacts"]["directory"])
    cloud = np.frombuffer((directory / "scan.pcd").read_bytes().split(b"DATA binary\n")[1], dtype="<f4").reshape(-1, 4)
    assert np.allclose(cloud[:, 3], [25.5, 204])
    assert np.allclose(inputs["points"][:, 3], [.1, .8])
    assert (directory / "mask.log").is_file()
