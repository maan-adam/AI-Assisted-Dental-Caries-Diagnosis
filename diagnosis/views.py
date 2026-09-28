from django.contrib.auth.decorators import login_required
from django.http import FileResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from .forms import UploadForm
from .models import Analysis
from .services import InvalidImage, analyse_upload


@login_required
def upload(request):
    form = UploadForm(request.POST or None, request.FILES or None)
    if request.method == "POST" and form.is_valid():
        try:
            analysis = analyse_upload(request.user, form.cleaned_data["img"])
        except InvalidImage as exc:
            form.add_error("img", str(exc))
        else:
            return redirect("diagnosis:result", pk=analysis.pk)
    status = 400 if form.errors else 200
    return render(request, "diagnosis/upload.html", {"form": form}, status=status)


@login_required
def result(request, pk):
    analysis = get_object_or_404(Analysis, pk=pk, user=request.user)
    return render(request, "diagnosis/result.html", {"analysis": analysis})


@login_required
def result_image(request, pk):
    """Serve a result image to its owner only (photos of mouths are private)."""
    analysis = get_object_or_404(Analysis, pk=pk, user=request.user)
    return FileResponse(analysis.result_image.open("rb"), content_type="image/jpeg")


@login_required
def history(request):
    return render(request, "diagnosis/history.html", {"analyses": request.user.analyses.all()[:50]})


@login_required
@require_POST
def delete(request, pk):
    analysis = get_object_or_404(Analysis, pk=pk, user=request.user)
    analysis.result_image.delete(save=False)
    analysis.delete()
    return redirect("diagnosis:history")
