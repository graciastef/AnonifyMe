from django.db import models
from django.core import serializers

# Create your models here.

class FileMetadata(models.Model):
    class Status(models.TextChoices):
        UPLOADED = "UPLOADED", "Uploaded"
        PROCESSING = "PROCESSING", "Processing"
        DONE = "DONE", "Done"

    original_file_name = models.CharField(max_length=500)
    upload_blob_url = models.URLField(max_length=2000)
    download_blob_url = models.URLField(null=True, blank=True, max_length=2000)
    date_created = models.DateTimeField(auto_now_add=True)
    date_processing_finished = models.DateTimeField(null=True, blank=True)
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.UPLOADED
    )
    #ClearML task id
    task_id = models.CharField(max_length=255, null=True, blank=True)

    def __str__(self):
        return serializers.serialize("json", [self])