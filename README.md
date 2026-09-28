# AI-Assisted Dental Caries Detection

[![tests](https://github.com/USERNAME/dental-caries-detection/actions/workflows/tests.yml/badge.svg)](https://github.com/USERNAME/dental-caries-detection/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-purple)
![Django](https://img.shields.io/badge/Django-5.x-0C4B33)

A web app that finds signs of tooth decay in ordinary photos of teeth. Two YOLOv8 detectors,
trained on differently annotated datasets, run on each image. Their predictions are merged with
**weighted box fusion**, so each finding is counted once and findings both models agree on
rank higher. Optional **CLAHE** preprocessing evens out the uneven lighting typical of phone photos.

It is meant as a first check at home that helps people decide whether to see a dentist. It is
**not** a medical diagnosis.

![Model A, Model B and the fused ensemble on the same photo](docs/ensemble_demo.jpg)

<sub>Model A boxes the whole affected area and Model B boxes the cavity tightly. The ensemble
merges them into a single finding with a combined confidence.</sub>

## Features

- **Two-model ensemble.** YOLOv8x (`decay`, `filling`) and YOLOv8m (`caries`) run on the same image. Their labels are mapped to one shared set, then fused.
- **Fusion built for mismatched annotation styles.** Boxes match when their IoU is high enough *or* when one box sits almost entirely inside the other. This way a tight cavity box and a whole-tooth box still count as one finding.
- **CLAHE preprocessing, configurable per model.** Contrast is equalised on the lightness channel only, so colour information is kept. A script measures its effect on mAP.
- **Django web app.** Registration with Django's password validators, login, upload, per-user history with delete, and the Django admin.
- **Privacy by default.** Uploaded photos are decoded in memory and never saved. Only the annotated result is stored, and it is served through a view that checks ownership rather than as a public media file.
- **Command-line inference** on a single image or a whole folder, with optional JSON output.
- **Configurable through environment variables** (thresholds, CLAHE, device, database); runs on SQLite with no setup.
- **Tested.** 23 tests cover fusion, preprocessing and the web flow. CI runs them without needing the model weights.

## Results

Each model's validation-set results from training:

| Model | Classes | Precision | Recall | mAP@50 |
|---|---|---|---|---|
| YOLOv8m | `caries` | 0.820 | 0.804 | **0.836** |
| YOLOv8x | `decay`, `filling` | 0.852 | 0.745 | 0.813 |

The two models were trained on different label sets, so their scores are not directly
comparable. YOLOv8x has the higher precision, while YOLOv8m has the higher recall. Combining
them lets each model catch lesions the other misses. The fused ensemble has not yet been
evaluated on a shared held-out test set.

**Data.** Annotated intraoral photos from Roboflow. The images were quality-checked and
resized, then augmented and split into train and validation sets.

## Screenshot

![Result page of the web app](docs/screenshot_result.jpg)

## How it works

```mermaid
flowchart LR
    A[Uploaded photo] --> P{CLAHE?<br/>per model}
    P --> B[YOLOv8x<br/>decay · filling]
    P --> C[YOLOv8m<br/>caries]
    B --> D[Map labels<br/>decay → caries]
    C --> D
    D --> E[Weighted box fusion<br/>IoU ≥ 0.5 or containment ≥ 0.8]
    E --> F[Annotated image<br/>+ caries / filling counts]
```

1. Each model gets either the original image or a CLAHE-enhanced copy, depending on the `CLAHE_MODELS` setting. Each model then predicts with confidence ≥ 0.25.
2. Labels from both models are mapped to one shared set (`decay` and `caries` both become `caries`).
3. Boxes with the same label are clustered. Each cluster becomes one box, placed at the confidence-weighted average position of its members.
4. The fused score is the members' mean confidence. It is scaled down when only one model found the box, so findings both models agree on rank higher.
5. Fused boxes below `FUSED_CONF` are dropped. The boxes are drawn on the **original** photo.

The detection code in [`detection/`](detection) does not depend on Django, so the web app
and the CLI share it.

### CLAHE

[`detection/preprocessing.py`](detection/preprocessing.py) applies Contrast Limited Adaptive
Histogram Equalization to the L channel in LAB colour space. Phone photos vary a lot in lighting
(flash glare, dark areas at the back of the mouth, warm indoor light), and CLAHE evens out local
contrast so dark lesions stand out from the enamel.

CLAHE is **off by default**. On a handful of sample photos it raised confidence for some
images and lowered it for others, so it should only be enabled once it has been measured on
labelled data. To measure it:

```bash
python scripts/evaluate_clahe.py --weights models/yolov8m_caries.pt --data path/to/data.yaml
```

This validates the model twice on the same images (original vs. CLAHE) and prints precision,
recall, mAP@50 and mAP@50-95 side by side. To enable CLAHE for one or both models, set
`CLAHE_MODELS=yolov8x` (or `yolov8x,yolov8m`).

## Quick start

```bash
git clone https://github.com/USERNAME/dental-caries-detection.git
cd dental-caries-detection
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/download_models.py   # fetches the two trained weight files (~180 MB) into models/
cp .env.example .env                # set SECRET_KEY; DEBUG=1 for local development
python manage.py migrate
python manage.py createsuperuser    # optional, for /admin
python manage.py runserver          # http://127.0.0.1:8000
```

### Command-line inference

```bash
python predict.py docs/sample.jpg                    # writes outputs/sample_pred.jpg
python predict.py path/to/images/ --json             # a whole folder, JSON report on stdout
python predict.py docs/sample.jpg --clahe yolov8x    # with CLAHE for YOLOv8x
```

### Docker

```bash
docker build -t caries-detector .
docker run -p 8000:8000 -e SECRET_KEY=change-me -e ALLOWED_HOSTS=localhost caries-detector
```

### Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The web tests replace the YOLO ensemble with a small fake detector, so they run in about a
second without the weight files.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `SECRET_KEY` | *(required)* | Django secret key |
| `DEBUG` | `0` | `1` for local development |
| `ALLOWED_HOSTS` | `localhost,127.0.0.1` | Comma-separated host names |
| `DATABASE_URL` | SQLite file | e.g. `mysql://user:pass@host/db` or `postgres://…` |
| `MODEL_CONF` | `0.25` | Per-model confidence threshold |
| `FUSED_CONF` | `0.25` | Threshold applied after fusion |
| `FUSION_IOU` | `0.5` | IoU at which two boxes are merged |
| `FUSION_CONTAINMENT` | `0.8` | Share of the smaller box that must lie inside the larger one to merge them |
| `CLAHE_MODELS` | *(empty)* | Models that get a CLAHE image: `yolov8x`, `yolov8m` |
| `CLAHE_CLIP_LIMIT` / `CLAHE_TILE_GRID` | `2.0` / `8` | CLAHE parameters |
| `DEVICE` | auto | `cpu`, `0` (first GPU), … |

## Project structure

```
├── config/                  # Django settings, URLs, WSGI
├── accounts/                # registration, login, logout
├── diagnosis/
│   ├── models.py            # Analysis: user, result image, counts, detections (JSON)
│   ├── services.py          # loads the ensemble once, runs it on an upload
│   ├── views.py             # upload, result, owner-only image, history, delete
│   └── admin.py
├── detection/               # framework-independent ML code
│   ├── ensemble.py          # runs both YOLO models, fuses, draws boxes
│   ├── fusion.py            # weighted box fusion (pure NumPy)
│   └── preprocessing.py     # CLAHE
├── templates/               # Django templates sharing one base layout
├── static/
├── scripts/
│   ├── download_models.py
│   └── evaluate_clahe.py    # mAP with vs. without CLAHE
├── predict.py               # CLI
├── tests/
├── Dockerfile
└── .github/workflows/tests.yml
```

## Limitations

- These models look at the visible tooth surface in ordinary photos. They cannot see caries between teeth or below the surface; finding those needs X-rays.
- Photo quality (lighting, focus, angle) has a large effect on the results.
- The output is a screening aid only. It is not a diagnosis and does not replace a dental examination.

## Credits

- Sample photo (`docs/sample.jpg`) by Suyash Dwivedi, via [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Dental_Caries_Cavity_2.JPG).
- Object detection with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics). The page layout was originally built with Mobirise.

## Team

This started as a graduation project by a team of five: Mohammed Al-Harthi, Yahya Mojaly,
Abdulrahman Alamro, Maan Adam and Yasser Al-Adwani.

This repository, including the ensemble fusion, CLAHE preprocessing, the Django app and the
tests, is maintained by **Maan Adam** · [LinkedIn](https://www.linkedin.com/in/maan-adam-b57320245)
