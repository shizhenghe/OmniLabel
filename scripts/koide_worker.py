#!/usr/bin/env python3
"""Execute real Koide on an upstream-format data directory (standard library only)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import uuid


def checked(command: list[str], *, timeout: float = 8) -> None:
    result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(result.stdout.strip()[-2000:] or f"{command[0]} exit={result.returncode}")


def terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=2)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, help="calib.json and paired PNG/PLY directory")
    parser.add_argument("--docker-image", help="Prebuilt image containing Koide, ROS2 and xvfb-run")
    parser.add_argument("--check", action="store_true", help="Check executable, package or local Docker image only")
    parser.add_argument("--ros1", action="store_true", help="Use local ROS1 instead of ROS2")
    parser.add_argument("--timeout", type=float, default=900, help="Maximum worker runtime, in seconds")
    # Accepted so the backend may enforce these flags for every command; this
    # worker always adds them to the actual Koide command.
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--auto_quit", action="store_true")
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    docker_name = None
    process = None
    try:
        if args.docker_image:
            if not shutil.which("docker"):
                raise RuntimeError("docker 不可用；请先安装 Docker 并构建 docker/koide/Dockerfile。")
            # Never pull images implicitly during an annotation task.
            checked(["docker", "image", "inspect", args.docker_image])
        else:
            executable = "rosrun" if args.ros1 else "ros2"
            if not shutil.which(executable):
                raise RuntimeError(f"{executable} 不可用；请先加载安装了 Koide 的 ROS 环境。")
            checked(["rospack", "find", "direct_visual_lidar_calibration"] if args.ros1 else
                    ["ros2", "pkg", "prefix", "direct_visual_lidar_calibration"])
        if args.check:
            print(json.dumps({"available": True, "engine": "koide", "runtime": "docker" if args.docker_image else "ros"}))
            return 0
        if args.data_dir is None:
            parser.error("--data-dir is required without --check")
        data_dir = args.data_dir.resolve(strict=True)
        calib_path = data_dir / "calib.json"
        config = json.loads(calib_path.read_text(encoding="utf-8"))
        names = config["meta"]["bag_names"]
        if not names or "init_T_lidar_camera" not in config.get("results", {}):
            raise ValueError("calib.json must contain bag_names and results.init_T_lidar_camera")
        for name in names:
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError("bag_names must be simple file stems")
            for suffix in (".png", ".ply"):
                if not (data_dir / (name + suffix)).is_file():
                    raise ValueError(f"Missing {name + suffix}")
        config["results"].pop("T_lidar_camera", None)
        calib_path.write_text(json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if args.docker_image:
            docker_name = "omnilabel-koide-" + uuid.uuid4().hex
            # Do not make xvfb-run PID 1. Its signal-based Xvfb cleanup can
            # wait forever under Docker after a successful calibration. The
            # inner deadline makes it a child of coreutils timeout and also
            # leaves the outer Python deadline enough time to remove Docker.
            margin = min(5.0, args.timeout * 0.1)
            container_timeout = max(0.05, args.timeout - margin)
            command = ["docker", "run", "--rm", "--pull=never", "--network=none", "--name", docker_name,
                       "--mount", f"type=bind,src={data_dir},dst=/calibration", args.docker_image,
                       "timeout", "--signal=TERM", "--kill-after=5s", f"{container_timeout:g}s",
                       "xvfb-run", "-a", "-s", "-screen 0 1280x1024x24 +extension GLX",
                       "ros2", "run", "direct_visual_lidar_calibration", "calibrate", "/calibration",
                       "--background", "--auto_quit"]
        else:
            prefix = ["rosrun"] if args.ros1 else ["ros2", "run"]
            command = [*prefix, "direct_visual_lidar_calibration", "calibrate", str(data_dir), "--background", "--auto_quit"]
            if not os.environ.get("DISPLAY") and shutil.which("xvfb-run"):
                command = ["xvfb-run", "-a", *command]
        # A hidden Iridescence viewer still needs an OpenGL context.  Docker
        # uses Xvfb/Mesa; local ROS may use the existing display or xvfb-run.
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, start_new_session=True, shell=False)

        def interrupted(signum, _frame):
            raise InterruptedError(f"Koide worker interrupted by signal {signum}")

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        returncode = process.wait(timeout=args.timeout)
        if returncode:
            raise RuntimeError(f"Koide calibrate failed (exit={returncode})")
        result = json.loads(calib_path.read_text(encoding="utf-8"))
        if "T_lidar_camera" not in result.get("results", {}):
            raise RuntimeError("Koide did not write results.T_lidar_camera")
        print(json.dumps({"engine": "koide", "calib_json": str(calib_path), "status": "completed"}))
        return 0
    except subprocess.TimeoutExpired:
        print("Koide runtime timed out", file=sys.stderr)
        return 124
    except (OSError, RuntimeError, ValueError, KeyError, InterruptedError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        # Stop handling termination while cleaning up, including when the API
        # times out the outer worker.  Killing the Docker client alone would
        # leave the calibration process running inside its container.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if process is not None:
            terminate(process)
        if docker_name is not None:
            try:
                subprocess.run(["docker", "rm", "-f", docker_name], stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=4, check=False)
            except (OSError, subprocess.TimeoutExpired):
                pass


if __name__ == "__main__":
    raise SystemExit(main())
