import os
from django_microproject.detection import detection

from django.http import HttpResponse
from django.views.decorators.http import require_http_methods
from django.core.files.storage import FileSystemStorage


@require_http_methods(["POST"])
def convert(request):
    memory_file = request.FILES['file']
    fs = FileSystemStorage(location=f'{os.getcwd()}/tmp')
    file = fs.save(memory_file.name, memory_file)

    res = detection.detect_bboxes(fs.path(file))

    return HttpResponse(res)