"""Download the trained model weights into ./models.

The weights are too large for a normal git repository (131 MB and 50 MB),
so they are hosted separately. Override the sources with environment
variables if you move them, e.g. to a GitHub Release:

    MODEL_A_URL=https://github.com/<user>/<repo>/releases/download/v1.0/yolov8x_decay_filling.pt
"""
import os
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

WEIGHTS = {
    "yolov8x_decay_filling.pt": os.environ.get(
        "MODEL_A_URL", "https://drive.google.com/uc?id=1J_3AWIQX08V-Uqc0ldvxM-ZaGmndlHHB"
    ),
    "yolov8m_caries.pt": os.environ.get(
        "MODEL_B_URL", "https://drive.google.com/uc?id=1ShMvwr_w263y4bEYXoy1fk93kxcx1NnN"
    ),
}


def download(url: str, dest: Path) -> None:
    if "drive.google.com" in url:
        import gdown

        if gdown.download(url, str(dest), quiet=False) is None:
            raise RuntimeError(f"gdown could not download {url}")
    else:
        import urllib.request

        urllib.request.urlretrieve(url, dest)


def main() -> int:
    MODELS_DIR.mkdir(exist_ok=True)
    for name, url in WEIGHTS.items():
        dest = MODELS_DIR / name
        if dest.is_file() and dest.stat().st_size > 0:
            print(f"✓ {name} already present")
            continue
        print(f"↓ {name}")
        download(url, dest)
    print(f"Weights are in {MODELS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
