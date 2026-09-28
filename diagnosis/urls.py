from django.urls import path

from . import views

app_name = "diagnosis"

urlpatterns = [
    path("upload/", views.upload, name="upload"),
    path("history/", views.history, name="history"),
    path("result/<uuid:pk>/", views.result, name="result"),
    path("result/<uuid:pk>/image/", views.result_image, name="result_image"),
    path("result/<uuid:pk>/delete/", views.delete, name="delete"),
]
