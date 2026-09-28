import numpy as np
import pytest

from detection.preprocessing import apply_clahe


def low_contrast_image():
    rng = np.random.default_rng(0)
    gradient = np.tile(np.linspace(100, 140, 128, dtype=np.float32), (128, 1))
    noise = rng.normal(0, 2, (128, 128))
    gray = np.clip(gradient + noise, 0, 255).astype(np.uint8)
    return np.dstack([gray, gray, gray])


def test_clahe_keeps_shape_and_dtype():
    img = low_contrast_image()
    out = apply_clahe(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_clahe_increases_contrast():
    img = low_contrast_image()
    assert apply_clahe(img, clip_limit=3.0).std() > img.std()


def test_clahe_rejects_grayscale():
    with pytest.raises(ValueError):
        apply_clahe(np.zeros((10, 10), np.uint8))
