"""Actual official model API adapters. GPU dependencies are imported lazily.

Run these in model-specific environments, never in the web server's venv.
OpenOccupancy uses its own OccNet.forward_dummy inference path so inference
does not require fabricated ground-truth tensors used by its evaluation API.
"""
from __future__ import annotations

from copy import deepcopy
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

from .common import (InvalidInput, ModelUnavailable, as_numpy, current_annotations,
                     image_from_payload, mask_to_polygons, model_version, new_id,
                     points_array, validate_options)


def module_present(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ValueError, ImportError, ModuleNotFoundError):
        return False


class Adapter:
    id = ""
    name = ""
    tasks = []
    dependencies = []
    checkpoint_variable = ""
    config_variable = ""

    def __init__(self):
        self.model = None
        self.version = None
        self.failure = None
        self.device = os.environ.get("OMNILABEL_MODEL_DEVICE", "cuda:0")

    def status(self):
        missing = [name for name in self.dependencies if not module_present(name)]
        checkpoint = os.environ.get(self.checkpoint_variable, "")
        config = os.environ.get(self.config_variable, "") if self.config_variable else ""
        status, reason = "configured", "Weights configured; model has not been loaded"
        if missing:
            status, reason = "missing_dependencies", "Missing runtime packages: " + ", ".join(missing)
        elif not checkpoint or not Path(checkpoint).is_file():
            status, reason = "missing_checkpoint", "Set " + self.checkpoint_variable + " to the matching trusted checkpoint"
        elif self.config_variable and (not config or not Path(config).is_file()):
            status, reason = "missing_config", "Set " + self.config_variable + " to the checkpoint's model config"
        elif self.failure:
            status, reason = "error", self.failure
        elif self.model is not None:
            status, reason = "ready", "Official model loaded"
        return {"id": self.id, "name": self.name, "tasks": self.tasks, "available": status == "ready",
                "status": status, "reason": reason, "device": self.device,
                "supports": {"point_prompt": self.id == "sam2", "box_prompt": self.id == "sam2",
                             "text_prompt": False, "occupancy": self.id == "openoccupancy"}}

    def prerequisites(self):
        status = self.status()
        if status["status"] in ("missing_dependencies", "missing_checkpoint", "missing_config"):
            raise ModelUnavailable(status["reason"])

    def load(self):
        raise NotImplementedError

    def predict(self, payload):
        raise NotImplementedError


class SAM2Adapter(Adapter):
    id, name = "sam2", "SAM 2.1"
    tasks = ["image-segmentation"]
    dependencies = ["torch", "torchvision", "sam2"]
    checkpoint_variable = "SAM2_CHECKPOINT"

    def load(self):
        self.prerequisites()
        from sam2.build_sam import build_sam2
        checkpoint = os.environ[self.checkpoint_variable]
        config = os.environ.get("SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_t.yaml")
        self.model = build_sam2(config, checkpoint, device=self.device, apply_postprocessing=False)
        self.version = model_version(self.id, checkpoint)

    def predict(self, payload):
        if self.model is None: self.load()
        import torch
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        options = payload.get("options", {})
        threshold, label = validate_options(options)
        camera_id = options.get("camera_id")
        if not camera_id:
            camera_id = next(iter(payload["frame"].get("images", {})), None)
        image, camera = image_from_payload(payload, camera_id)
        image_array = np.asarray(image)
        prompt_points = np.asarray(options.get("points", []), dtype=np.float32)
        prompt_labels = np.asarray(options.get("point_labels", [1] * len(prompt_points)), dtype=np.int32)
        box = options.get("box")
        if prompt_points.size:
            if prompt_points.ndim != 2 or prompt_points.shape[1] != 2 or len(prompt_points) > 256 or len(prompt_labels) != len(prompt_points):
                raise InvalidInput("SAM2 point prompts must be up to 256 original-image pixel pairs with matching labels")
            if not np.isfinite(prompt_points).all() or not np.isin(prompt_labels, [0, 1]).all() or (prompt_points < 0).any() or (prompt_points[:, 0] >= camera["width"]).any() or (prompt_points[:, 1] >= camera["height"]).any():
                raise InvalidInput("SAM2 prompts must lie within the selected image; labels are 0 or 1")
        if box is not None:
            box = np.asarray(box, dtype=np.float32)
            if box.shape != (4,) or not np.isfinite(box).all() or not (0 <= box[0] < box[2] <= camera["width"] and 0 <= box[1] < box[3] <= camera["height"]):
                raise InvalidInput("SAM2 box prompt must be [x1,y1,x2,y2] inside the original image")
        masks = []
        with torch.inference_mode():
            if prompt_points.size or box is not None:
                predictor = SAM2ImagePredictor(self.model)
                predictor.set_image(image_array)
                predicted, scores, _ = predictor.predict(
                    point_coords=prompt_points if prompt_points.size else None,
                    point_labels=prompt_labels if prompt_points.size else None,
                    box=box, multimask_output=False)
                masks = [(mask, float(score)) for mask, score in zip(predicted, scores) if float(score) >= threshold]
            else:
                from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
                generator = SAM2AutomaticMaskGenerator(self.model, points_per_side=16,
                    pred_iou_thresh=max(threshold, 0.7), min_mask_region_area=0)
                generated = sorted(generator.generate(image_array), key=lambda item: item["predicted_iou"], reverse=True)
                maximum = options.get("max_masks", 32)
                if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= 64:
                    raise InvalidInput("max_masks must be between 1 and 64")
                masks = [(entry["segmentation"], float(entry["predicted_iou"])) for entry in generated[:maximum]]
        annotations = current_annotations(payload)
        conversions, confidence = [], {}
        for mask, score in masks:
            polygons, metadata = mask_to_polygons(mask)
            conversions.append(metadata)
            for polygon in polygons:
                annotation_id = new_id("sam2")
                annotations["image_polygons"].append({"id": annotation_id, "camera_id": camera_id, "label": label, "points": polygon})
                confidence[annotation_id] = score
        return {"annotations": annotations, "model_version": self.version,
                "confidence": confidence, "conversion": conversions,
                "warnings": ["SAM2 predicts class-agnostic masks. The selected label is assigned by the annotator, not inferred from text.",
                             "Simple polygon conversion approximates raster boundaries; holes are reported and require manual review."]}


def normalize_detection(result, class_names, threshold, annotations):
    """MMDetection3D 1.x boxes use gravity_center to avoid bottom-center errors."""
    if isinstance(result, tuple): result = result[0]
    instances = getattr(result, "pred_instances_3d", None)
    if instances is None:
        raise InvalidInput("The configured model did not return pred_instances_3d; choose a LiDAR detector config")
    boxes = instances.bboxes_3d
    centers = as_numpy(boxes.gravity_center)
    dimensions = as_numpy(boxes.dims)
    yaws = as_numpy(boxes.yaw)
    scores = as_numpy(instances.scores_3d)
    labels = as_numpy(instances.labels_3d).astype(int)
    if not (centers.shape == dimensions.shape == (len(scores), 3) and len(yaws) == len(scores) == len(labels)):
        raise InvalidInput("Detector returned inconsistent box arrays")
    # Inputs are always LiDAR; a camera-box checkpoint is a coordinate mismatch.
    if type(boxes).__name__ != "LiDARInstance3DBoxes":
        raise InvalidInput("This adapter requires LiDAR-coordinate detector boxes")
    confidence = {}
    for center, size, yaw, score, category in zip(centers, dimensions, yaws, scores, labels):
        if score < threshold: continue
        if not 0 <= category < len(class_names) or (size <= 0).any():
            raise InvalidInput("Detector class mapping or box dimensions are invalid")
        annotation_id, track_id = new_id("mmdet3d"), new_id("track")
        annotations["boxes"].append({"id": annotation_id, "track_id": track_id,
            "label": str(class_names[category]), "center": center.astype(float).tolist(),
            "size": size.astype(float).tolist(), "yaw": float(yaw)})
        confidence[annotation_id] = float(score)
    return confidence


class MMDetection3DAdapter(Adapter):
    id, name = "mmdetection3d", "MMDetection3D"
    tasks = ["3d-detection", "point-segmentation"]
    dependencies = ["torch", "mmengine", "mmcv", "mmdet3d"]
    checkpoint_variable, config_variable = "MMDET3D_CHECKPOINT", "MMDET3D_CONFIG"

    def load(self):
        self.prerequisites()
        from mmdet3d.apis import init_model
        self.model = init_model(os.environ[self.config_variable], os.environ[self.checkpoint_variable], device=self.device)
        self.version = model_version(self.id, os.environ[self.checkpoint_variable])

    def predict(self, payload):
        if self.model is None: self.load()
        from mmdet3d.apis import inference_detector, inference_segmentor
        options = payload.get("options", {})
        threshold, _ = validate_options(options)
        points = points_array(payload["frame"])
        classes = self.model.dataset_meta.get("classes", ())
        if not classes: raise InvalidInput("Checkpoint/config must declare its dataset class mapping")
        annotations = current_annotations(payload)
        task = options.get("task", os.environ.get("MMDET3D_TASK", "detection"))
        if task == "detection":
            result = inference_detector(self.model, points)
            confidence = normalize_detection(result, classes, threshold, annotations)
        elif task == "segmentation":
            result = inference_segmentor(self.model, points)
            if isinstance(result, tuple): result = result[0]
            labels = as_numpy(result.pred_pts_seg.pts_semantic_mask).reshape(-1)
            if len(labels) != len(points):
                raise InvalidInput("Segmentor changed point order/count; an explicit index adapter is required")
            if not np.equal(labels, labels.astype(int)).all() or ((labels < 0) | (labels >= len(classes))).any():
                raise InvalidInput("Segmentor labels do not match its declared dataset classes")
            annotations["point_labels"].update({str(i): str(classes[int(label)]) for i, label in enumerate(labels)})
            confidence = {}
        else:
            raise InvalidInput("MMDetection3D task must be detection or segmentation")
        return {"annotations": annotations, "model_version": self.version, "confidence": confidence,
                "warnings": ["Detector boxes are LiDAR coordinates with gravity-centered XYZ. Models are dataset-specific; verify calibration and class mapping."]}


OCC_CLASSES = ["free", "barrier", "bicycle", "bus", "car", "construction_vehicle", "motorcycle", "pedestrian",
               "traffic_cone", "trailer", "truck", "driveable_surface", "other_flat", "sidewalk", "terrain", "manmade", "vegetation"]


def occupancy_from_logits(logits, point_range, threshold, class_names=OCC_CLASSES, coordinates=None, full_shape=None, maximum=150000):
    """Convert actual coarse/fine network logits; this is model evidence, not GT."""
    value = as_numpy(logits).astype(np.float32)
    bounds = np.asarray(point_range, dtype=float)
    if bounds.shape != (6,) or not np.isfinite(bounds).all() or (bounds[3:] <= bounds[:3]).any():
        raise InvalidInput("OpenOccupancy point_cloud_range must contain valid XYZ bounds")
    if coordinates is None:
        if value.ndim == 5 and value.shape[0] == 1: value = value[0]
        if value.ndim != 4 or value.shape[0] != len(class_names):
            raise InvalidInput("Expected OpenOccupancy coarse logits [classes,X,Y,Z]")
        shape = np.array(value.shape[1:])
        scores = value.reshape(len(class_names), -1).T
        coordinate = None
    else:
        coordinate = as_numpy(coordinates).astype(np.int64)
        if coordinate.ndim != 2: raise InvalidInput("Fine occupancy coordinates must be 3xN")
        if coordinate.shape[0] == 3: coordinate = coordinate.T
        if value.ndim != 2 or value.shape != (len(coordinate), len(class_names)):
            raise InvalidInput("Expected fine occupancy logits [N,classes]")
        shape = np.asarray(full_shape, dtype=np.int64)
        if shape.shape != (3,) or (shape <= 0).any() or (coordinate < 0).any() or (coordinate >= shape).any():
            raise InvalidInput("Fine voxel coordinates are outside the configured occupancy grid")
        scores = value
    # Stable probability conversion; confidence thresholds are explicit.
    shifted = scores - scores.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    category = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    selected = np.flatnonzero(confidence >= threshold)
    total = len(selected)
    if len(selected) > maximum:
        # Occupied predictions receive display/export priority over model-free cells.
        order = np.lexsort((-confidence[selected], category[selected] == 0))
        selected = selected[order[:maximum]]
    voxel_dimensions = (bounds[3:] - bounds[:3]) / shape
    voxels = []
    for flat in selected:
        grid = coordinate[flat] if coordinate is not None else np.array(np.unravel_index(flat, tuple(shape)))
        voxels.append({"index": grid.astype(int).tolist(), "center": (bounds[:3] + (grid + .5) * voxel_dimensions).tolist(),
                       "state": "free" if category[flat] == 0 else "occupied", "label": str(class_names[category[flat]]),
                       "confidence": float(confidence[flat])})
    return {"coordinate_frame": "lidar", "origin": bounds[:3].tolist(), "grid_shape": shape.astype(int).tolist(),
            "voxel_size": float(voxel_dimensions.min()), "voxel_dimensions": voxel_dimensions.tolist(), "voxels": voxels,
            "source": "openoccupancy-model", "prediction_only": True, "truncated": total > maximum,
            "candidate_count": total, "occupied_count": sum(v["state"] == "occupied" for v in voxels),
            "free_count": sum(v["state"] == "free" for v in voxels), "unknown_policy": "Low-confidence or unreturned cells remain unknown"}


class OpenOccupancyAdapter(Adapter):
    id, name = "openoccupancy", "OpenOccupancy / CONet"
    tasks = ["occupancy", "point-segmentation"]
    dependencies = ["torch", "mmcv", "mmdet3d"]
    checkpoint_variable, config_variable = "OPENOCCUPANCY_CHECKPOINT", "OPENOCCUPANCY_CONFIG"

    def status(self):
        value = super().status()
        repo = os.environ.get("OPENOCCUPANCY_REPO", "")
        if value["status"] == "configured" and (not repo or not (Path(repo) / "projects/occ_plugin").is_dir()):
            value.update(status="missing_config", reason="Set OPENOCCUPANCY_REPO to the official source checkout", available=False)
        return value

    def load(self):
        self.prerequisites()
        import torch
        from mmcv import Config
        from mmcv.runner import load_checkpoint
        root = Path(os.environ["OPENOCCUPANCY_REPO"]).resolve()
        if str(root) not in sys.path: sys.path.insert(0, str(root))
        self.cfg = Config.fromfile(os.environ[self.config_variable])
        # Plugin path is the official registry module, not arbitrary client code.
        importlib.import_module("projects.occ_plugin")
        from mmdet3d.models import build_model
        self.cfg.model.pretrained = None
        self.model = build_model(self.cfg.model, train_cfg=None, test_cfg=self.cfg.get("test_cfg"))
        load_checkpoint(self.model, os.environ[self.checkpoint_variable], map_location="cpu")
        self.model.to(self.device).eval()
        self.version = model_version(self.id, os.environ[self.checkpoint_variable])

    def _camera_inputs(self, payload):
        import torch
        from projects.occ_plugin.datasets.pipelines.loading_bevdet import LoadMultiViewImageFromFiles_BEVDet
        config = self.cfg.data_config
        expected = int(config.get("Ncams", len(config["cams"])))
        configured = os.environ.get("OPENOCCUPANCY_CAMERA_ORDER")
        camera_ids = json.loads(configured) if configured else list(config["cams"])
        # Never silently truncate 7 uploaded cameras to a checkpoint's 6-camera rig.
        actual_ids = [c["id"] for c in payload["clip"]["cameras"]]
        if len(actual_ids) != expected or len(camera_ids) != expected or set(camera_ids) != set(actual_ids):
            raise InvalidInput("Camera count/order differs from the trained OpenOccupancy configuration. Supply an adapted config/checkpoint and OPENOCCUPANCY_CAMERA_ORDER; cameras are never silently dropped.")
        loader = LoadMultiViewImageFromFiles_BEVDet(config, is_train=False, img_norm_cfg=self.cfg.get("img_norm_cfg"), load_depth=False)
        images, rots, trans, intrinsic, post_rots, post_trans = [], [], [], [], [], []
        for camera_id in camera_ids:
            image, camera = image_from_payload(payload, camera_id)
            if any(abs(float(x)) > 1e-12 for x in camera.get("distortion", [])):
                raise InvalidInput("OpenOccupancy camera checkpoints require rectified images and matching intrinsic matrices")
            resize, dimensions, crop, flip, rotate = loader.sample_augmentation(image.height, image.width)
            image, rotation_2d, translation_2d = loader.img_transform(image, torch.eye(2), torch.zeros(2), resize, dimensions, crop, flip, rotate)
            cam_from_lidar = np.asarray(camera["camera_from_ego"]) @ np.asarray(payload["frame"]["lidar_to_ego"])
            lidar_from_cam = np.linalg.inv(cam_from_lidar)
            post_rotation, post_translation = torch.eye(3), torch.zeros(3)
            post_rotation[:2, :2], post_translation[:2] = rotation_2d, translation_2d
            images.append(loader.normalize_img(image, img_norm_cfg=loader.img_norm_cfg))
            rots.append(torch.tensor(lidar_from_cam[:3, :3], dtype=torch.float32))
            trans.append(torch.tensor(lidar_from_cam[:3, 3], dtype=torch.float32))
            intrinsic.append(torch.tensor(camera["intrinsics"], dtype=torch.float32))
            post_rots.append(post_rotation); post_trans.append(post_translation)
        result = [torch.stack(items)[None].to(self.device) for items in (images, rots, trans, intrinsic, post_rots, post_trans)]
        result.append(torch.eye(3, device=self.device)[None])  # no BEV training augmentation
        h, w = config["input_size"]
        result.append([torch.tensor([h], device=self.device), torch.tensor([w], device=self.device)])
        return result

    def predict(self, payload):
        if self.model is None: self.load()
        import torch
        options = payload.get("options", {})
        threshold, _ = validate_options(options)
        points = points_array(payload["frame"])
        modality = self.cfg.get("input_modality", {"use_lidar": True, "use_camera": False})
        lidar = None
        if modality.get("use_lidar"):
            # Official NuScenes encoder expects [x,y,z,intensity,time-lag];
            # a single uploaded scan has a true zero time-lag, not a fake sweep.
            lidar = [torch.from_numpy(np.column_stack((points, np.zeros(len(points), dtype=np.float32)))).to(self.device)]
        images = self._camera_inputs(payload) if modality.get("use_camera") else None
        point_range = self.cfg.point_cloud_range
        meta = [{"pc_range": point_range, "occ_size": self.cfg.occ_size,
                 "scene_token": payload["clip"]["id"], "lidar_token": payload["frame"]["id"]}]
        with torch.no_grad():
            output = self.model.forward_dummy(points=lidar, img_metas=meta, img_inputs=images)
        fine = output.get("output_voxels_fine")
        coordinates = output.get("output_coords_fine")
        if fine is not None and coordinates is not None:
            occupancy = occupancy_from_logits(fine[0], point_range, threshold, coordinates=coordinates[0], full_shape=self.cfg.occ_size)
        else:
            logits = fine[0] if fine is not None else output["output_voxels"][0]
            occupancy = occupancy_from_logits(logits, point_range, threshold)
        annotations = current_annotations(payload)
        lookup = {tuple(v["index"]): v for v in occupancy["voxels"]}
        grid = np.floor((points[:, :3] - np.asarray(point_range[:3])) / np.asarray(occupancy["voxel_dimensions"])).astype(int)
        for index, cell in enumerate(grid):
            voxel = lookup.get(tuple(cell))
            if voxel and voxel["state"] == "occupied": annotations["point_labels"][str(index)] = voxel["label"]
        return {"annotations": annotations, "model_version": self.version, "occupancy": occupancy,
                "warnings": ["OpenOccupancy results are learned occupancy predictions, not verified dense ground truth.",
                             "Published checkpoints are NuScenes-specific. This request uses uploaded calibration and a single scan; validate domain and sweep-density differences."]}


ADAPTERS = {adapter.id: adapter for adapter in (SAM2Adapter, MMDetection3DAdapter, OpenOccupancyAdapter)}
