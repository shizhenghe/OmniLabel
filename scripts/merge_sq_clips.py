#!/usr/bin/env python3
"""Merge the locally imported SQ part clips into one lazy, lossless clip."""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MERGED_ID = "sq-data-all"


def main() -> None:
    import sys
    sys.path.insert(0, str(ROOT / "backend"))
    from omnilabel.models import Clip, Frame
    from omnilabel.storage import Store
    source_store = Store(DATA)
    clip_root = DATA / "clips"
    source_paths = sorted(
        clip_root.glob("sq-data-part-*.json"),
        key=lambda path: int(re.search(r"part-(\d+)", path.name).group(1)),
    )
    if not source_paths:
        raise SystemExit("No data/clips/sq-data-part-*.json files found.")

    destination_clip = clip_root / f"{MERGED_ID}.json"
    points_root = DATA / "pointclouds"
    final_points = points_root / MERGED_ID
    annotation_root = DATA / "annotations"
    final_annotations = annotation_root / MERGED_ID
    if destination_clip.exists() or final_points.exists() or final_annotations.exists():
        raise SystemExit(f"Merged output already exists for {MERGED_ID}; refusing to overwrite it.")

    points_root.mkdir(parents=True, exist_ok=True)
    annotation_root.mkdir(parents=True, exist_ok=True)
    point_stage = Path(tempfile.mkdtemp(prefix=f".{MERGED_ID}-", dir=points_root))
    annotation_stage = Path(tempfile.mkdtemp(prefix=f".{MERGED_ID}-", dir=annotation_root))
    merged_frames = []
    source_summaries = []
    annotations_found = 0
    try:
        reference = None
        previous_timestamp = None
        total_points = 0
        for source_path in source_paths:
            part = json.loads(source_path.read_text(encoding="utf-8"))
            source_summaries.append({"id": part["id"], "name": part["name"], "description": part.get("description", ""),
                                     "fps": part["fps"], "frame_count": len(part["frames"]),
                                     "camera_count": len(part["cameras"])})
            cameras = part["cameras"]
            if reference is None:
                reference = {key: part[key] for key in ("fps", "cameras")}
            elif cameras != reference["cameras"] or abs(float(part["fps"]) - float(reference["fps"])) > 1e-6:
                raise ValueError(f"Camera calibration or fps differs in {source_path.name}")

            for frame in part["frames"]:
                timestamp = float(frame["timestamp"])
                if previous_timestamp is not None and timestamp <= previous_timestamp:
                    raise ValueError(f"Non-increasing timestamp at {source_path.name}:{frame['id']}")
                points = source_store.frame_point_array(Frame.model_validate(frame))
                frame.pop("points", None)
                relative_path = f"{frame['id']}.npy"
                np.save(point_stage / relative_path, points, allow_pickle=False)
                frame["pointcloud_file"] = f"{MERGED_ID}/{relative_path}"
                frame["point_count"] = len(points)
                merged_frames.append(frame)
                total_points += len(points)
                previous_timestamp = timestamp

                source_labels = DATA / "annotations" / part["id"] / f"{frame['id']}.json"
                if source_labels.is_file():
                    shutil.copy2(source_labels, annotation_stage / source_labels.name)
                    annotations_found += 1

        if len(merged_frames) > 2000:
            raise ValueError("Merged clip exceeds the supported frame count")
        manifest = {
            "id": MERGED_ID,
            "name": f"SQ test data · merged ({len(merged_frames)} frames)",
            "description": "Merged locally from SQ part clips; complete source point clouds are stored losslessly as per-frame NPY files.",
            "fps": float(reference["fps"]),
            "cameras": reference["cameras"],
            "frames": merged_frames,
            "source": "imported-merged",
        }
        # Validate the manifest through the application schema before publishing.
        validated = Clip.model_validate(manifest)
        final_points.parent.mkdir(parents=True, exist_ok=True)
        point_stage.rename(final_points)
        annotation_stage.rename(final_annotations)
        temporary = destination_clip.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(validated.model_dump(), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, destination_clip)
        summary_rows = [*source_summaries, {"id": MERGED_ID, "name": manifest["name"],
                                             "description": manifest["description"], "fps": manifest["fps"],
                                             "frame_count": len(merged_frames), "camera_count": len(manifest["cameras"])}]
        known_ids = {row["id"] for row in summary_rows}
        for path in sorted(clip_root.glob("*.json")):
            if path.name == destination_clip.name or path.stem in known_ids:
                continue
            other = Clip.model_validate_json(path.read_text(encoding="utf-8"))
            summary_rows.append({"id": other.id, "name": other.name, "description": other.description,
                                 "fps": other.fps, "frame_count": len(other.frames),
                                 "camera_count": len(other.cameras)})
        files = {path.name: [path.stat().st_size, path.stat().st_mtime_ns]
                 for path in sorted(clip_root.glob("*.json"))}
        from omnilabel.storage import CLIP_SUMMARY_CACHE_VERSION, atomic_json
        atomic_json(DATA / "clip_summaries.json", {"version": CLIP_SUMMARY_CACHE_VERSION,
                                                     "files": files, "summaries": summary_rows})
        print(f"Created {destination_clip}")
        print(f"Frames: {len(merged_frames)}; points: {total_points:,}; existing annotation files copied: {annotations_found}")
        print(f"Point clouds: {final_points}")
    except Exception:
        if point_stage.exists():
            shutil.rmtree(point_stage)
        if annotation_stage.exists():
            shutil.rmtree(annotation_stage)
        raise


if __name__ == "__main__":
    main()
