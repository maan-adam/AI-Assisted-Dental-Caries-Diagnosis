from pathlib import Path

from django import forms
from django.conf import settings


class UploadForm(forms.Form):
    img = forms.FileField(label="Photo")

    def clean_img(self):
        file = self.cleaned_data["img"]
        ext = Path(file.name).suffix.lower().lstrip(".")
        if ext not in settings.ALLOWED_IMAGE_EXTENSIONS:
            raise forms.ValidationError("Unsupported file type. Use JPG, PNG, BMP or WEBP.")
        if file.size > settings.MAX_UPLOAD_SIZE:
            raise forms.ValidationError("File is too large (max 10 MB).")
        return file
