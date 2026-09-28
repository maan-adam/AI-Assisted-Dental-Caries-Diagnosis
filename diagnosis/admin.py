from django.contrib import admin

from .models import Analysis


@admin.register(Analysis)
class AnalysisAdmin(admin.ModelAdmin):
    list_display = ("created_at", "user", "caries_count", "filling_count")
    list_filter = ("created_at",)
    search_fields = ("user__username", "user__email")
    readonly_fields = ("id", "user", "created_at", "result_image", "detections", "clahe_models")
