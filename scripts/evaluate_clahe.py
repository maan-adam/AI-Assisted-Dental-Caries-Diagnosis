"""Measure whether CLAHE preprocessing helps each model.

Runs Ultralytics validation twice per model on the same labelled validation
set: once on the original images and once on CLAHE-enhanced copies. Prints
precision, recall, mAP@50 and mAP@50-95 side by side.

Usage (use each model's own dataset, since their classes differ):
    python scripts/evaluate_clahe.py --weights models/yolov8m_caries.pt --data path/to/caries/data.yaml
    python scripts/evaluate_clahe.py --weights models/yolov8x_decay_filling.pt --data path/to/decay/data.yaml

``data.yaml`` is the file Roboflow includes in a "YOLOv8" export.
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detection.preprocessing import apply_clahe  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_split(data: dict, data_path: Path, split: str) -> Path:
    root = Path(data.get("path") or data_path.parent)
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()
    split_dir = Path(data[split])
    candidates = [split_dir if split_dir.is_absolute() else root / split_dir, data_path.parent / split_dir]
    # Roboflow exports often write "../valid/images" relative to the yaml.
    candidates.append((data_path.parent / split_dir.as_posix().replace("../", "")).resolve())
    for c in candidates:
        if c.is_dir():
            return c.resolve()
    raise FileNotFoundError(f"Could not find the '{split}' images folder. Tried: {candidates}")


def make_clahe_copy(images_dir: Path, out_root: Path, clip: float, tile: int) -> Path:
    """Copy images (CLAHE-enhanced) and their YOLO label files into out_root."""
    labels_dir = images_dir.parent / "labels"
    out_images, out_labels = out_root / "images", out_root / "labels"
    out_images.mkdir(parents=True)
    out_labels.mkdir(parents=True)
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        cv2.imwrite(str(out_images / img_path.name), apply_clahe(img, clip, tile))
        label = labels_dir / f"{img_path.stem}.txt"
        if label.is_file():
            shutil.copy(label, out_labels / label.name)
    return out_images


def validate(weights: Path, data_yaml: Path, device):
    from ultralytics import YOLO

    m = YOLO(str(weights)).val(data=str(data_yaml), device=device, verbose=False, plots=False)
    return {"P": m.box.mp, "R": m.box.mr, "mAP50": m.box.map50, "mAP50-95": m.box.map}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True, help="dataset data.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--clip-limit", type=float, default=2.0)
    ap.add_argument("--tile-grid", type=int, default=8)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    data = yaml.safe_load(args.data.read_text())
    images_dir = resolve_split(data, args.data, args.split)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        base_yaml = tmp / "original.yaml"
        base_yaml.write_text(yaml.safe_dump({**data, "path": "", "train": str(images_dir), "val": str(images_dir)}))
        clahe_images = make_clahe_copy(images_dir, tmp / "clahe", args.clip_limit, args.tile_grid)
        clahe_yaml = tmp / "clahe.yaml"
        clahe_yaml.write_text(yaml.safe_dump({**data, "path": "", "train": str(clahe_images), "val": str(clahe_images)}))

        results = {
            "original": validate(args.weights, base_yaml, args.device),
            f"CLAHE (clip={args.clip_limit}, tile={args.tile_grid})": validate(args.weights, clahe_yaml, args.device),
        }

    print(f"\n{args.weights.name} on {images_dir}\n")
    print(f"{'input':<28}{'P':>8}{'R':>8}{'mAP50':>9}{'mAP50-95':>10}")
    for name, r in results.items():
        print(f"{name:<28}{r['P']:>8.3f}{r['R']:>8.3f}{r['mAP50']:>9.3f}{r['mAP50-95']:>10.3f}")
    (orig, clahe) = results.values()
    print(f"\nChange in mAP50 with CLAHE: {clahe['mAP50'] - orig['mAP50']:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
