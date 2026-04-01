from django.urls import path
from . import views

urlpatterns = [
    # API endpoints
    path("api/upload/", views.upload, name="upload"),

    # Frontend pages
    path("", views.home, name="home"),
    path("sample/", views.sample, name="sample"),
]