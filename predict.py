"""Run the caries-detection ensemble on images from the command line.

Examples:
    python predict.py docs/sample.jpg
    python predict.py path/to/folder --out outputs/ --json
    python predict.py docs/sample.jpg --clahe yolov8x           # CLAHE for YOLOv8x only
    python predict.py docs/sample.jpg --clahe yolov8x yolov8m   # CLAHE for both models
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

from detection import CariesDetector, ModelSpec, draw_detections

ROOT = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(path: Path):
    if path.is_dir():
        yield from sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    else:
        yield path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="image file or folder of images")
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="folder for annotated images")
    parser.add_argument("--model-a", type=Path, default=ROOT / "models" / "yolov8x_decay_filling.pt")
    parser.add_argument("--model-b", type=Path, default=ROOT / "models" / "yolov8m_caries.pt")
    parser.add_argument("--conf", type=float, default=0.25, help="per-model confidence threshold")
    parser.add_argument("--fused-conf", type=float, default=0.25, help="threshold after fusion")
    parser.add_argument("--clahe", nargs="*", default=[], choices=["yolov8x", "yolov8m"],
                        help="models that get a CLAHE-enhanced image (default: none)")
    parser.add_argument("--device", default=None, help='"cpu", "0", ...')
    parser.add_argument("--json", action="store_true", help="print detections as JSON")
    args = parser.parse_args()

    detector = CariesDetector(
        models=[
            ModelSpec("yolov8x", args.model_a, clahe="yolov8x" in args.clahe),
            ModelSpec("yolov8m", args.model_b, clahe="yolov8m" in args.clahe),
        ],
        conf=args.conf,
        fused_conf=args.fused_conf,
        device=args.device,
    )
    args.out.mkdir(parents=True, exist_ok=True)

    report = []
    for path in iter_images(args.source):
        image = cv2.imread(str(path))
        if image is None:
            print(f"skipping {path}: not a readable image", file=sys.stderr)
            continue
        result = detector.predict(image)
        out_path = args.out / f"{path.stem}_pred.jpg"
        cv2.imwrite(str(out_path), draw_detections(image, result.detections))
        report.append({
            "image": str(path),
            "output": str(out_path),
            "caries": result.caries_count,
            "fillings": result.filling_count,
            "detections": [
                {"label": d.label, "score": round(d.score, 3), "box": [round(v, 1) for v in d.box], "source": d.source}
                for d in result.detections
            ],
        })
        if not args.json:
            print(f"{path.name}: {result.caries_count} caries, {result.filling_count} fillings -> {out_path}")

    if args.json:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
