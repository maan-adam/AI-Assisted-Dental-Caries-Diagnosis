"""Box-level fusion for combining detections from several models.

Kept free of torch / ultralytics imports so it is cheap to import and easy to
unit-test.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Detection:
    """A single bounding box in absolute pixel coordinates (x1, y1, x2, y2)."""

    box: tuple[float, float, float, float]
    score: float
    label: str
    source: str  # which model produced it (or "fused")


def _overlap(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x1 = np.maximum(a[0], b[:, 0])
    y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2])
    y2 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter, np.broadcast_to(area_a, area_b.shape), area_b


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between one box ``a`` (4,) and many boxes ``b`` (N, 4)."""
    inter, area_a, area_b = _overlap(a, b)
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def containment(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection over the *smaller* box: 1.0 when one box lies inside the other."""
    inter, area_a, area_b = _overlap(a, b)
    return inter / np.maximum(np.minimum(area_a, area_b), 1e-9)


def weighted_box_fusion(
    detections: list[Detection],
    iou_threshold: float = 0.5,
    containment_threshold: float = 0.8,
    num_models: int = 2,
) -> list[Detection]:
    """Merge overlapping boxes that share a label (simplified WBF).

    Two boxes are considered the same finding when their IoU is at least
    ``iou_threshold`` *or* one box lies almost entirely inside the other
    (``containment_threshold``). The second rule matters here because the two
    models were trained on datasets with different annotation styles: one
    boxes the lesion tightly, the other boxes the whole affected tooth.

    Boxes are clustered greedily in descending score order. Each cluster is
    replaced by one box whose coordinates are the score-weighted mean of its
    members. The fused score is the mean member score, scaled down when fewer
    models than ``num_models`` agreed, so a box found by both models ranks
    above a box found by only one.
    """
    fused: list[Detection] = []
    for label in sorted({d.label for d in detections}):
        group = sorted((d for d in detections if d.label == label), key=lambda d: -d.score)
        clusters: list[list[Detection]] = []
        centres: list[np.ndarray] = []
        for det in group:
            box = np.asarray(det.box, dtype=float)
            if centres:
                others = np.stack(centres)
                ious = iou(box, others)
                match = (ious >= iou_threshold) | (containment(box, others) >= containment_threshold)
                if match.any():
                    best = int(np.argmax(np.where(match, ious, -1.0)))
                    clusters[best].append(det)
                    centres[best] = _weighted_mean(clusters[best])
                    continue
            clusters.append([det])
            centres.append(box)

        for members, centre in zip(clusters, centres):
            n_sources = len({m.source for m in members})
            score = float(np.mean([m.score for m in members])) * min(n_sources, num_models) / num_models
            source = members[0].source if n_sources == 1 else "fused"
            fused.append(Detection(tuple(float(v) for v in centre), score, label, source))

    return sorted(fused, key=lambda d: -d.score)


def _weighted_mean(members: list[Detection]) -> np.ndarray:
    boxes = np.array([m.box for m in members], dtype=float)
    weights = np.array([m.score for m in members], dtype=float)
    return (boxes * weights[:, None]).sum(axis=0) / weights.sum()
