from http import HTTPStatus

from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.shortcuts import render

from face_blur.services.video_storage import upload_to_blob_storage


# frontend views
def home(request):
    return render(request, "homepage.html")

def sample(request):
    return render(request, "sample.html")

# backend views
@require_POST
def upload(request):
    if request.method != "POST":
        return JsonResponse(
            {"error": "Method not allowed"},
            status=HTTPStatus.METHOD_NOT_ALLOWED,
        )

    uploaded_file = request.FILES.get("file")
    if not uploaded_file:
        return JsonResponse(
            {"error": "No file provided"},
            status=HTTPStatus.BAD_REQUEST,
        )

    try:
        result = upload_to_blob_storage(uploaded_file)

        return JsonResponse(
            {
                "file_name": result.original_file_name,
                "status": result.status,
                "date_created": result.date_created,
            },
            status=HTTPStatus.CREATED,
        )
    except Exception as e:
        return JsonResponse(
            {"error": str(e)},
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
        )