"""Two-model YOLOv8 ensemble for dental caries detection.

Model A (YOLOv8x) was trained on two classes: ``decay`` and ``filling``.
Model B (YOLOv8m) was trained on a single ``caries`` class.

Each model can optionally see a CLAHE-enhanced copy of the image (see
``ModelSpec.clahe``). Their class names are mapped onto a
shared vocabulary (``decay`` -> ``caries``) and overlapping boxes are merged
with weighted box fusion, so a lesion found by both models is counted once.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .fusion import Detection, weighted_box_fusion
from .preprocessing import apply_clahe

# Maps each model's raw class names onto the labels the app reports.
DEFAULT_LABEL_MAP = {"decay": "caries", "caries": "caries", "filling": "filling"}

COLOURS = {"caries": (40, 40, 220), "filling": (200, 140, 30)}  # BGR


@dataclass
class ModelSpec:
    name: str
    weights: Path
    clahe: bool = False  # enhance contrast before this model sees the image


@dataclass
class DetectionResult:
    detections: list[Detection] = field(default_factory=list)

    def count(self, label: str) -> int:
        return sum(d.label == label for d in self.detections)

    @property
    def caries_count(self) -> int:
        return self.count("caries")

    @property
    def filling_count(self) -> int:
        return self.count("filling")


class CariesDetector:
    def __init__(
        self,
        models: list[ModelSpec],
        conf: float = 0.25,
        fused_conf: float = 0.25,
        iou_threshold: float = 0.5,
        containment_threshold: float = 0.8,
        imgsz: int = 640,
        device: str | None = None,
        label_map: dict[str, str] | None = None,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: int = 8,
    ):
        from ultralytics import YOLO  # heavy import, keep it lazy

        missing = [str(m.weights) for m in models if not Path(m.weights).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Model weights not found: {missing}. Run `python scripts/download_models.py`."
            )
        self.models = [(m, YOLO(str(m.weights))) for m in models]
        self.conf = conf
        self.fused_conf = fused_conf
        self.iou_threshold = iou_threshold
        self.containment_threshold = containment_threshold
        self.imgsz = imgsz
        self.device = device
        self.label_map = label_map or DEFAULT_LABEL_MAP
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size

    def predict(self, image: np.ndarray) -> DetectionResult:
        """Run every model on a BGR image and fuse their detections."""
        raw: list[Detection] = []
        enhanced = None
        for spec, model in self.models:
            source = image
            if spec.clahe:
                if enhanced is None:  # compute once, share between models
                    enhanced = apply_clahe(image, self.clahe_clip_limit, self.clahe_tile_grid_size)
                source = enhanced
            result = model.predict(
                source=source, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False
            )[0]
            for xyxy, score, cls in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
                result.boxes.cls.cpu().numpy().astype(int),
            ):
                raw_label = result.names[int(cls)]
                label = self.label_map.get(raw_label, raw_label)
                raw.append(Detection(tuple(map(float, xyxy)), float(score), label, spec.name))

        fused = weighted_box_fusion(
            raw,
            iou_threshold=self.iou_threshold,
            containment_threshold=self.containment_threshold,
            num_models=len(self.models),
        )
        return DetectionResult([d for d in fused if d.score >= self.fused_conf])


def draw_detections(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Return a copy of ``image`` with labelled boxes drawn on it."""
    out = image.copy()
    scale = max(image.shape[:2]) / 640
    thickness = max(2, round(2 * scale))
    font_scale = max(0.5, 0.6 * scale)
    for det in detections:
        x1, y1, x2, y2 = map(int, det.box)
        colour = COLOURS.get(det.label, (0, 200, 0))
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, thickness)
        text = f"{det.label} {det.score:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        ty = max(y1, th + baseline + 4)
        cv2.rectangle(out, (x1, ty - th - baseline - 4), (x1 + tw + 6, ty), colour, -1)
        cv2.putText(out, text, (x1 + 3, ty - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out
