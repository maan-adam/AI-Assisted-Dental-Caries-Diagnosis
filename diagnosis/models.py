import uuid

from django.conf import settings
from django.db import models


def result_path(instance, filename):
    return f"results/{instance.user_id}/{instance.id}.jpg"


class Analysis(models.Model):
    """One uploaded photo and what the ensemble found in it.

    Only the annotated result image is stored; the original upload is
    processed in memory and discarded.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="analyses")
    created_at = models.DateTimeField(auto_now_add=True)
    result_image = models.ImageField(upload_to=result_path)
    caries_count = models.PositiveIntegerField(default=0)
    filling_count = models.PositiveIntegerField(default=0)
    detections = models.JSONField(default=list)  # [{label, score, box, source}, ...]
    clahe_models = models.JSONField(default=list)  # which models saw a CLAHE image

    class Meta:
        ordering = ["-created_at"]
        verbose_name_plural = "analyses"

    def __str__(self):
        return f"{self.user} · {self.created_at:%Y-%m-%d %H:%M} · {self.caries_count} caries"
