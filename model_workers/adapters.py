"""Actual official model API adapters. GPU dependencies are imported lazily.

Run these in model-specific environments, never in the web server's venv.
OpenOccupancy uses its own OccNet.forward_dummy inference path so inference
does not require fabricated ground-truth tensors used by its evaluation API.
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import nullcontext
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from PIL import Image
import numpy as np

from .common import (InvalidInput, ModelUnavailable, as_numpy, current_annotations,
                     image_from_payload, mask_to_polygons, model_version, new_id,
                     points_array, validate_options)


def module_present(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ValueError, ImportError, ModuleNotFoundError):
        return False


def resolve_sam2_precision(torch, device, requested="auto"):
    """Turing uses FP16; native BF16 requires Ampere+ on the selected GPU.

    Some torch versions report emulated BF16 as supported on older cards. The
    capability check prevents treating that as native SAM2 BF16 support.
    """
    if requested not in ("auto", "float16", "bfloat16", "float32"):
        raise ModelUnavailable("SAM2_PRECISION must be auto, float16, bfloat16 or float32")
    if not str(device).startswith("cuda"):
        if requested not in ("auto", "float32"):
            raise ModelUnavailable("SAM2 float16/bfloat16 precision requires a CUDA device; use float32 on CPU")
        return "float32"
    with torch.cuda.device(device):
        major, _ = torch.cuda.get_device_capability(device)
        native_bf16 = major >= 8 and torch.cuda.is_bf16_supported()
    if requested == "bfloat16" and not native_bf16:
        raise ModelUnavailable("SAM2 bfloat16 requires native BF16 on Ampere or newer; use auto or float16 on RTX 2080/Turing")
    return ("bfloat16" if native_bf16 else "float16") if requested == "auto" else requested


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
    precision = None
    points_per_batch = 16

    def status(self):
        value = super().status()
        value["precision"] = self.precision or os.environ.get("SAM2_PRECISION", "auto")
        return value

    def load(self):
        self.prerequisites()
        import torch
        from sam2.build_sam import build_sam2
        self.precision = resolve_sam2_precision(torch, self.device, os.environ.get("SAM2_PRECISION", "auto"))
        try:
            self.points_per_batch = int(os.environ.get("SAM2_POINTS_PER_BATCH", "16"))
        except ValueError:
            raise ModelUnavailable("SAM2_POINTS_PER_BATCH must be an integer from 1 to 128") from None
        if not 1 <= self.points_per_batch <= 128:
            raise ModelUnavailable("SAM2_POINTS_PER_BATCH must be an integer from 1 to 128")
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
        autocast = nullcontext() if self.precision == "float32" else torch.autocast(
            device_type="cuda", dtype=getattr(torch, self.precision))
        with torch.inference_mode(), autocast:
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
                generator = SAM2AutomaticMaskGenerator(self.model, points_per_side=16, points_per_batch=self.points_per_batch,
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
        return {"annotations": annotations, "model_version": self.version, "inference_precision": self.precision,
                "confidence": confidence, "conversion": conversions,
                "warnings": ["SAM2 predicts class-agnostic masks. The selected label is assigned by the annotator, not inferred from text.",
                             "Simple polygon conversion approximates raster boundaries; holes are reported and require manual review."]}


def latr_coordinate_matrices(payload, camera_id):
    """Build LATR-ground -> LiDAR/camera transforms from OmniLabel calibration.

    LATR lanes are [right, forward, up]. OmniLabel ego/LiDAR axes are
    [forward, left, up], and frame vectors are stored in LiDAR coordinates.
    """
    camera = next((item for item in payload["clip"]["cameras"] if item.get("id") == camera_id), None)
    if camera is None:
        raise InvalidInput("Choose a camera present in this clip")
    frame = payload["frame"]
    try:
        lidar_to_ego = np.asarray(frame["lidar_to_ego"], dtype=np.float64)
        camera_from_ego = np.asarray(camera["camera_from_ego"], dtype=np.float64)
        direct = frame.get("projection_camera_from_lidar", {}).get(camera_id)
        if direct is None:
            direct = frame.get("annotation_camera_from_lidar", {}).get(camera_id)
        camera_from_lidar = (np.asarray(direct, dtype=np.float64) if direct is not None
                             else camera_from_ego @ lidar_to_ego)
        intrinsic = np.asarray(camera["intrinsics"], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        raise InvalidInput("LATR requires valid camera intrinsics and LiDAR/ego/camera extrinsics") from None
    if (lidar_to_ego.shape != (4, 4) or camera_from_lidar.shape != (4, 4)
            or intrinsic.shape != (3, 3)
            or not all(np.isfinite(value).all() for value in (lidar_to_ego, camera_from_lidar, intrinsic))):
        raise InvalidInput("LATR requires finite 4x4 extrinsics and a finite 3x3 intrinsic matrix")
    latr_to_ego = np.array([[0, 1, 0, 0], [-1, 0, 0, 0],
                            [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
    try:
        latr_to_lidar = np.linalg.inv(lidar_to_ego) @ latr_to_ego
    except np.linalg.LinAlgError:
        raise InvalidInput("LATR cannot use a singular LiDAR-to-ego transform") from None
    camera_from_latr = camera_from_lidar @ latr_to_lidar
    # ONCE LATR is trained for a forward camera. Side/rear views violate both
    # its query anchors and learned image distribution, so fail explicitly.
    if camera_from_latr[2, 1] <= 0.2:
        raise InvalidInput("LATR ONCE requires a forward-facing camera; select the front camera")
    return camera, intrinsic, latr_to_lidar, camera_from_latr


def decode_latr_predictions(output, anchor_y_steps, threshold, visibility_threshold,
                            label, annotations, latr_to_lidar):
    """Convert the final LATR decoder layer to OmniLabel LiDAR polylines."""
    try:
        lines = as_numpy(output["all_line_preds"])[-1, 0]
        logits = as_numpy(output["all_cls_scores"])[-1, 0]
    except (KeyError, IndexError, TypeError):
        raise InvalidInput("LATR returned an unsupported prediction structure") from None
    anchors = np.asarray(anchor_y_steps, dtype=np.float64)
    transform = np.asarray(latr_to_lidar, dtype=np.float64)
    if (lines.ndim != 2 or logits.ndim != 2 or len(lines) != len(logits)
            or anchors.ndim != 1 or len(anchors) < 2
            or lines.shape[1] != 3 * len(anchors) or logits.shape[1] < 2
            or transform.shape != (4, 4)):
        raise InvalidInput("LATR prediction dimensions do not match its configured lane anchors/classes")
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    categories = probabilities.argmax(axis=1)
    scores = probabilities.max(axis=1)
    confidence = {}
    for prediction, category, score in zip(lines, categories, scores):
        if int(category) == 0 or float(score) < threshold:
            continue
        visible = prediction[2 * len(anchors):] > visibility_threshold
        if int(visible.sum()) < 2:
            continue
        latr_points = np.column_stack((prediction[:len(anchors)][visible], anchors[visible],
                                       prediction[len(anchors):2 * len(anchors)][visible],
                                       np.ones(int(visible.sum()), dtype=np.float64)))
        lidar_points = (transform @ latr_points.T).T[:, :3]
        if not np.isfinite(lidar_points).all():
            raise InvalidInput("LATR returned non-finite lane coordinates")
        annotation_id = new_id("latr")
        annotations["vectors"].append({"id": annotation_id, "label": label,
                                       "points": lidar_points.astype(float).tolist(), "closed": False})
        confidence[annotation_id] = float(score)
    return confidence


class LATRAdapter(Adapter):
    id, name = "latr", "LATR (ONCE / OpenLane)"
    tasks = ["3d-lane-detection"]
    dependencies = ["torch", "torchvision", "mmcv", "mmdet", "mmdet3d", "mmseg",
                    "cv2", "scipy", "matplotlib", "geffnet", "fvcore"]
    checkpoint_variable, config_variable = "LATR_CHECKPOINT", "LATR_CONFIG"

    def __init__(self):
        super().__init__()
        self.cfg = None
        self.repo = None
        self.active_variant = None

    def variant_paths(self, variant):
        if variant not in ("once", "openlane"):
            raise InvalidInput("LATR weight_variant must be once or openlane")
        repo = Path(os.environ.get("LATR_REPO", "")) if os.environ.get("LATR_REPO") else None
        if repo is None:
            return None, None
        if variant == "once":
            checkpoint = os.environ.get("LATR_ONCE_CHECKPOINT", os.environ.get("LATR_CHECKPOINT", str(repo / "ckpt" / "once.pth")))
            config = os.environ.get("LATR_ONCE_CONFIG", os.environ.get("LATR_CONFIG", str(repo / "config" / "release_iccv" / "once.py")))
        else:
            checkpoint = os.environ.get("LATR_OPENLANE_CHECKPOINT", str(repo / "ckpt" / "openlane.pth"))
            config = os.environ.get("LATR_OPENLANE_CONFIG", str(repo / "config" / "release_iccv" / "latr_1000_baseline.py"))
        return Path(checkpoint), Path(config)

    def default_variant(self):
        variant = os.environ.get("LATR_DEFAULT_VARIANT", "once").lower()
        if variant not in ("once", "openlane"):
            raise ModelUnavailable("LATR_DEFAULT_VARIANT must be once or openlane")
        return variant

    def status(self):
        missing = [name for name in self.dependencies if not module_present(name)]
        repo = Path(os.environ.get("LATR_REPO", "")) if os.environ.get("LATR_REPO") else None
        variants = {}
        for variant in ("once", "openlane"):
            checkpoint, config = self.variant_paths(variant)
            variants[variant] = bool(checkpoint and checkpoint.is_file() and config and config.is_file())
        status, reason = "configured", "LATR weight variants configured; model has not been loaded"
        try:
            default = self.default_variant()
        except ModelUnavailable as exc:
            default, status, reason = "once", "error", str(exc)
        if missing:
            status, reason = "missing_dependencies", "Missing runtime packages: " + ", ".join(missing)
        elif repo is None or not (repo / "models" / "latr.py").is_file():
            status, reason = "missing_repository", "Set LATR_REPO to the trusted LATR source checkout"
        elif not variants.get(default):
            status, reason = "missing_checkpoint", "The default LATR variant requires its matching checkpoint and config"
        elif self.failure:
            status, reason = "error", self.failure
        elif self.model is not None:
            status, reason = "ready", "Official LATR model loaded (" + str(self.active_variant) + ")"
        value = {"id": self.id, "name": self.name, "tasks": self.tasks, "available": status == "ready",
                 "status": status, "reason": reason, "device": self.device,
                 "supports": {"point_prompt": False, "box_prompt": False, "text_prompt": False,
                              "occupancy": False, "lane_vectors": True,
                              "weight_variants": [name for name, ready in variants.items() if ready],
                              "active_weight_variant": self.active_variant}}
        return value

    def prerequisites(self, variant=None):
        variant = variant or self.default_variant()
        status = self.status()
        if status["status"] in ("missing_dependencies", "missing_repository"):
            raise ModelUnavailable(status["reason"])
        checkpoint, config = self.variant_paths(variant)
        if checkpoint is None or not checkpoint.is_file():
            raise ModelUnavailable("LATR " + variant + " checkpoint is missing")
        if config is None or not config.is_file():
            raise ModelUnavailable("LATR " + variant + " config is missing")
        return checkpoint, config

    def load(self, variant=None):
        variant = variant or self.default_variant()
        checkpoint_path, config_path = self.prerequisites(variant)
        import torch
        from mmcv.utils import Config
        device = torch.device(self.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ModelUnavailable("LATR requires a working CUDA runtime for its compiled deformable operators")
        self.repo = Path(os.environ["LATR_REPO"]).resolve()
        repo_text = str(self.repo)
        if repo_text not in sys.path:
            sys.path.insert(0, repo_text)
        try:
            from models.latr import LATR
        except (ImportError, ModuleNotFoundError) as exc:
            raise ModelUnavailable("LATR source dependencies are incomplete (" + type(exc).__name__ + ")") from None
        cfg = Config.fromfile(str(config_path))
        if cfg.latr_cfg.encoder.get("init_cfg", None) is not None:
            cfg.latr_cfg.encoder.init_cfg = None
        # Upstream LATRHead mutates project_crit with pop('type'). When the
        # argument is omitted it mutates the function's shared default dict,
        # so constructing a second model in one worker fails. Pass a fresh
        # per-load mapping to make ONCE/OpenLane switching repeatable.
        cfg.latr_cfg.head.update(
            loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
            loss_reg=dict(type="L1Loss", loss_weight=2.0),
            loss_vis=dict(type="BCEWithLogitsLoss", reduction="mean"),
            project_crit=dict(type="SmoothL1Loss", reduction="none"),
        )
        cfg.anchor_y_steps_dense = cfg.get("anchor_y_steps_dense", np.linspace(3, 103, 200))
        cfg.no_cuda = device.type != "cuda"
        expected_categories = 2 if variant == "once" else 21
        if int(cfg.num_category) != expected_categories:
            raise ModelUnavailable("LATR " + variant + " config does not match the selected dataset variant")
        # Release the previous GPU model before constructing a structurally
        # different variant (12 vs 40 queries) to avoid double VRAM usage.
        self.model, self.cfg, self.active_variant = None, None, None
        if device.type == "cuda":
            torch.cuda.empty_cache()
        model = LATR(cfg)
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {(key[7:] if key.startswith("module.") else key): value for key, value in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
        self.model, self.cfg = model.to(device).eval(), cfg
        self.active_variant = variant
        self.version = model_version(self.id + "-" + variant, checkpoint_path)
        self.failure = None

    def predict(self, payload):
        options = payload.get("options", {})
        variant = options.get("weight_variant", self.default_variant())
        if variant not in ("once", "openlane"):
            raise InvalidInput("LATR weight_variant must be once or openlane")
        if self.model is None or self.active_variant != variant:
            self.load(variant)
        import torch
        import cv2
        threshold, label = validate_options(options)
        camera_id = options.get("camera_id") or next(iter(payload["frame"].get("images", {})), None)
        if not camera_id:
            raise InvalidInput("LATR requires a selected forward camera image")
        image, _ = image_from_payload(payload, camera_id)
        camera, intrinsic, latr_to_lidar, camera_from_latr = latr_coordinate_matrices(payload, camera_id)
        pixels = np.asarray(image)
        distortion = np.asarray(camera.get("distortion", []), dtype=np.float64)
        if distortion.size and np.any(distortion != 0):
            pixels = cv2.undistort(pixels, intrinsic, distortion, None, intrinsic)
        height, width = pixels.shape[:2]
        crop_y = int(self.cfg.crop_y)
        if not 0 <= crop_y < height:
            raise InvalidInput("LATR crop_y is outside the selected image")
        resized = np.asarray(Image.fromarray(pixels[crop_y:]).resize(
            (int(self.cfg.resize_w), int(self.cfg.resize_h)), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        normalized = (resized - np.asarray(self.cfg.mean, dtype=np.float32)) / np.asarray(self.cfg.std, dtype=np.float32)
        image_tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        ratio_x = float(self.cfg.resize_w) / width
        ratio_y = float(self.cfg.resize_h) / (height - crop_y)
        crop_resize = np.array([[ratio_x, 0, 0], [0, ratio_y, -ratio_y * crop_y], [0, 0, 1]], dtype=np.float64)
        projection = crop_resize @ intrinsic @ camera_from_latr[:3]
        lidar2img = np.eye(4, dtype=np.float32)
        lidar2img[:3] = projection.astype(np.float32)
        network_height, network_width = int(self.cfg.resize_h), int(self.cfg.resize_w)
        extra = {
            "seg_label": torch.zeros((1, 1, network_height, network_width), dtype=torch.float32, device=self.device),
            "seg_idx_label": torch.zeros((1, int(self.cfg.max_lanes), network_height, network_width), dtype=torch.uint8, device=self.device),
            "lidar2img": torch.from_numpy(lidar2img).unsqueeze(0).to(self.device),
            "pad_shape": torch.tensor([[network_height, network_width]], dtype=torch.float32, device=self.device),
        }
        try:
            visibility_threshold = float(os.environ.get("LATR_VISIBILITY_THRESHOLD", "0"))
        except ValueError:
            raise ModelUnavailable("LATR_VISIBILITY_THRESHOLD must be a finite number") from None
        if not math.isfinite(visibility_threshold):
            raise ModelUnavailable("LATR_VISIBILITY_THRESHOLD must be a finite number")
        with torch.no_grad():
            output = self.model(image=image_tensor, extra_dict=extra, is_training=False)
        annotations = current_annotations(payload)
        confidence = decode_latr_predictions(output, self.cfg.anchor_y_steps, threshold,
                                             visibility_threshold, label, annotations, latr_to_lidar)
        return {"annotations": annotations, "model_version": self.version, "confidence": confidence,
                "weight_variant": variant,
                "warnings": ["LATR used the " + variant.upper() + " weight/config pair on the selected forward monocular image; review domain shift and calibration before saving.",
                             "Lane points were transformed from LATR [right, forward, up] coordinates into this frame's LiDAR coordinates."]}


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


ADAPTERS = {adapter.id: adapter for adapter in (SAM2Adapter, LATRAdapter, MMDetection3DAdapter, OpenOccupancyAdapter)}
