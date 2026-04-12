import json
from http import HTTPStatus

from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.http import require_POST, require_GET
from django.shortcuts import render

from face_blur.services.video_processing import trigger_video_processing, VideoProcessingTask
from face_blur.services.video_storage import *

# frontend views
def home(request):
    return render(request, "homepage.html")

def sample(request):
    return render(request, "sample.html")

# backend views
@require_POST
def upload_video(request):
    """
    Handle video file upload requests.
    Request:
        Method: POST
        Content-Type: multipart/form-data
        Body:
            - file: Video file to upload

    Responses:
        201 Created:
            {
                "file_name": str,
                "status": str,
                "date_created": datetime
            }
    """
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
        result = upload_video_to_blob(uploaded_file)

        return JsonResponse(
            {
                "file_name": result.original_file_name,
                "file_key": result.file_key,
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

@require_POST
def upload_whitelist(request):
    """
    Upload whitelist images for a previously uploaded video.

    Request:
        Method: POST
        Content-Type: multipart/form-data
        Body:
            - file_key (str): Identifier of the uploaded video
            - files (List[File]): One or more whitelist images

    Responses:
        201 Created:
            {
                "file_name": str,
                "file_key": str,
                "status": str,
                "date_created": datetime
            }
    """
    whitelist_images = request.FILES.getlist("files")
    video_file_key = request.POST.get("file_key")

    if not whitelist_images:
        return JsonResponse(
            {"error": "No whitelist images provided"},
            status=HTTPStatus.BAD_REQUEST,
        )

    if not video_file_key:
        return JsonResponse(
            {"error": "file_key is required"},
            status=HTTPStatus.BAD_REQUEST,
        )

    try:
        updated_record = upload_image_to_blob(whitelist_images, video_file_key)

        return JsonResponse(
            {
                "file_name": updated_record.original_file_name,
                "file_key": updated_record.file_key,
                "status": updated_record.status,
                "date_created": updated_record.date_created,
            },
            status=HTTPStatus.OK,
        )
    except FileNotFoundError:
        return JsonResponse(
            {"error": "Invalid file key"},
            status=HTTPStatus.NOT_FOUND,
        )
    except Exception as e:
        return JsonResponse(
            {"error": str(e)},
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
        )

@require_GET
def progress_stream(request, file_key: str):
    """
    Stream processing progress to frontend.
    """
    # check if file exists
    try:
        record = FileMetadata.objects.get(file_key=file_key)
        if record.status != FileMetadata.Status.WHITELIST_UPLOADED:
            raise RuntimeError(
                f"Video cannot be processed: expected status '{FileMetadata.Status.WHITELIST_UPLOADED}', got '{record.status}'."
            )
    except FileMetadata.DoesNotExist:
        raise FileNotFoundError(f"No FileMetadata record found for file_key={file_key}")

    task = trigger_video_processing(str(file_key))

    def event_stream():
        while True:
            task.progress_event.wait(timeout=30)
            task.progress_event.clear()
            yield f"data: {json.dumps(task.progress)}\n\n"
            if task.progress["percentage"] >= 100:
                break

    response = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response