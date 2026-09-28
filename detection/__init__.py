"""Framework-independent detection code: models, fusion and preprocessing.

Used by both the Django app (diagnosis/services.py) and the CLI (predict.py).
"""
from .ensemble import CariesDetector, DetectionResult, ModelSpec, draw_detections
from .fusion import Detection, weighted_box_fusion
from .preprocessing import apply_clahe

__all__ = [
    "CariesDetector",
    "Detection",
    "DetectionResult",
    "ModelSpec",
    "apply_clahe",
    "draw_detections",
    "weighted_box_fusion",
]
