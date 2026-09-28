"""Image preprocessing applied before inference."""
from __future__ import annotations

import cv2
import numpy as np


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization on the lightness channel.

    Smartphone photos of teeth vary a lot in lighting: flash glare, shadows at
    the back of the mouth, warm indoor light. CLAHE evens out local contrast in
    small tiles, which helps dark lesions stand out from enamel. ``clip_limit``
    caps how much any tile's contrast is boosted, so sensor noise is not
    amplified.

    Only the L channel of LAB is equalised, so the colours (which carry
    information about staining and decay) are left unchanged.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected a BGR colour image of shape (H, W, 3)")
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    return cv2.cvtColor(cv2.merge((clahe.apply(lightness), a, b)), cv2.COLOR_LAB2BGR)
