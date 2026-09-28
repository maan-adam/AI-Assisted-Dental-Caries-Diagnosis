import cv2
import numpy as np
import pytest
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse

from detection import Detection, DetectionResult
from diagnosis import services
from diagnosis.models import Analysis

PASSWORD = "c0rrect-horse-battery"


class FakeDetector:
    """Stands in for the YOLO ensemble so tests run without model weights."""

    models = []

    def predict(self, image):
        h, w = image.shape[:2]
        return DetectionResult([
            Detection((0, 0, w / 2, h / 2), 0.9, "caries", "fused"),
            Detection((w / 2, h / 2, w, h), 0.5, "filling", "yolov8x"),
        ])


@pytest.fixture(autouse=True)
def fake_detector(monkeypatch, settings, tmp_path):
    settings.MEDIA_ROOT = tmp_path
    settings.STORAGES = {
        "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
        "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    }
    monkeypatch.setattr(services, "_detector", FakeDetector())


def image_file(name="teeth.png"):
    ok, buf = cv2.imencode(".png", np.full((64, 64, 3), 200, np.uint8))
    assert ok
    return SimpleUploadedFile(name, buf.tobytes(), content_type="image/png")


@pytest.fixture
def user(db):
    return User.objects.create_user("maan", "maan@example.com", PASSWORD)


@pytest.fixture
def logged_in(client, user):
    client.force_login(user)
    return client


@pytest.mark.django_db
def test_public_pages(client):
    assert client.get(reverse("home")).status_code == 200
    assert client.get(reverse("about")).status_code == 200


@pytest.mark.django_db
def test_upload_requires_login(client):
    resp = client.get(reverse("diagnosis:upload"))
    assert resp.status_code == 302
    assert reverse("accounts:login") in resp["Location"]


@pytest.mark.django_db
def test_register_logs_in_and_hashes_password(client):
    resp = client.post(reverse("accounts:register"), {
        "username": "newuser", "email": "New@Example.com", "password1": PASSWORD, "password2": PASSWORD,
    })
    assert resp.status_code == 302
    user = User.objects.get(username="newuser")
    assert user.email == "new@example.com"
    assert user.password != PASSWORD and user.check_password(PASSWORD)


@pytest.mark.django_db
def test_register_rejects_weak_password(client):
    resp = client.post(reverse("accounts:register"), {
        "username": "u", "email": "u@example.com", "password1": "12345678", "password2": "12345678",
    })
    assert resp.status_code == 200
    assert not User.objects.filter(username="u").exists()


def test_register_rejects_duplicate_email(client, user):
    resp = client.post(reverse("accounts:register"), {
        "username": "other", "email": "MAAN@example.com", "password1": PASSWORD, "password2": PASSWORD,
    })
    assert b"already exists" in resp.content


def test_login_and_wrong_password(client, user):
    bad = client.post(reverse("accounts:login"), {"username": "maan", "password": "wrong"})
    assert bad.status_code == 200
    good = client.post(reverse("accounts:login"), {"username": "maan", "password": PASSWORD})
    assert good.status_code == 302


def test_upload_creates_analysis_and_shows_result(logged_in):
    resp = logged_in.post(reverse("diagnosis:upload"), {"img": image_file()})
    assert resp.status_code == 302
    analysis = Analysis.objects.get()
    assert (analysis.caries_count, analysis.filling_count) == (1, 1)

    page = logged_in.get(resp["Location"])
    assert b"Possible caries:</strong> 1" in page.content
    image = logged_in.get(reverse("diagnosis:result_image", args=[analysis.pk]))
    assert image.status_code == 200
    assert image["Content-Type"] == "image/jpeg"


def test_rejects_non_image(logged_in):
    bad = SimpleUploadedFile("teeth.jpg", b"not an image", content_type="image/jpeg")
    resp = logged_in.post(reverse("diagnosis:upload"), {"img": bad})
    assert resp.status_code == 400
    assert not Analysis.objects.exists()


def test_rejects_bad_extension(logged_in):
    resp = logged_in.post(reverse("diagnosis:upload"), {"img": image_file("notes.txt")})
    assert resp.status_code == 400


def test_results_are_private(client, logged_in):
    logged_in.post(reverse("diagnosis:upload"), {"img": image_file()})
    analysis = Analysis.objects.get()
    other = User.objects.create_user("bob", "bob@example.com", PASSWORD)
    client.force_login(other)
    assert client.get(reverse("diagnosis:result", args=[analysis.pk])).status_code == 404
    assert client.get(reverse("diagnosis:result_image", args=[analysis.pk])).status_code == 404


def test_history_and_delete(logged_in):
    logged_in.post(reverse("diagnosis:upload"), {"img": image_file()})
    analysis = Analysis.objects.get()
    assert analysis.created_at.strftime("%Y") in logged_in.get(reverse("diagnosis:history")).content.decode()
    path = analysis.result_image.path
    resp = logged_in.post(reverse("diagnosis:delete", args=[analysis.pk]))
    assert resp.status_code == 302
    assert not Analysis.objects.exists()
    import os
    assert not os.path.exists(path)
