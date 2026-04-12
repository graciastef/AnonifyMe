from django.urls import path
from . import views

urlpatterns = [
    # API endpoints
    path("api/videos/", views.upload_video, name="upload_videos"),
    path("api/whitelist-images/", views.upload_whitelist, name="upload_whitelist_images"),
    path("api/progress/<str:file_key>/", views.progress_stream, name="stream_progress"),

    # Frontend pages
    path("", views.home, name="home"),
    path("sample/", views.sample, name="sample"),
]