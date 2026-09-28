"""Glue between Django and the framework-independent ``detection`` package."""
import threading

import cv2
import numpy as np
from django.conf import settings
from django.core.files.base import ContentFile

from detection import CariesDetector, ModelSpec, draw_detections

from .models import Analysis

_detector = None
_lock = threading.Lock()


class InvalidImage(ValueError):
    pass


def get_detector():
    """Load both YOLO models once per process, on first use."""
    global _detector
    if _detector is None:
        with _lock:
            if _detector is None:
                cfg = settings.DETECTION
                clahe = set(cfg["CLAHE_MODELS"])
                _detector = CariesDetector(
                    models=[
                        ModelSpec("yolov8x", cfg["MODEL_A_WEIGHTS"], clahe="yolov8x" in clahe),
                        ModelSpec("yolov8m", cfg["MODEL_B_WEIGHTS"], clahe="yolov8m" in clahe),
                    ],
                    conf=cfg["MODEL_CONF"],
                    fused_conf=cfg["FUSED_CONF"],
                    iou_threshold=cfg["FUSION_IOU"],
                    containment_threshold=cfg["FUSION_CONTAINMENT"],
                    clahe_clip_limit=cfg["CLAHE_CLIP_LIMIT"],
                    clahe_tile_grid_size=cfg["CLAHE_TILE_GRID"],
                    device=cfg["DEVICE"],
                )
    return _detector


def analyse_upload(user, uploaded_file) -> Analysis:
    """Decode an uploaded photo in memory, run the ensemble and store the result."""
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImage("Could not read that file as an image.")

    detector = get_detector()
    result = detector.predict(image)

    ok, jpg = cv2.imencode(".jpg", draw_detections(image, result.detections), [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("Could not encode the result image.")

    analysis = Analysis(
        user=user,
        caries_count=result.caries_count,
        filling_count=result.filling_count,
        detections=[
            {"label": d.label, "score": round(d.score, 3), "box": [round(v, 1) for v in d.box], "source": d.source}
            for d in result.detections
        ],
        clahe_models=[spec.name for spec, _ in getattr(detector, "models", []) if spec.clahe],
    )
    analysis.result_image.save(f"{analysis.id}.jpg", ContentFile(jpg.tobytes()), save=False)
    analysis.save()
    return analysis
