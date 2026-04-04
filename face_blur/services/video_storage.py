import uuid
import os
from django.conf import settings
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContentSettings

from ..models import FileMetadata

credential = DefaultAzureCredential()

blob_service_client = BlobServiceClient(
    account_url=settings.AZURE_BLOB_URL,
    credential=credential
)

def upload_to_blob_storage(uploaded_file, file_key):
    container_client = blob_service_client.get_container_client(
        settings.AZURE_UPLOAD_CONTAINER_NAME
    )

    blob_file_name = f"{file_key}/{uploaded_file.name}"
    blob_client = container_client.get_blob_client(blob_file_name)

    # Upload file
    try:
        blob_client.upload_blob(
            uploaded_file.file,
            overwrite=True,
            content_settings=ContentSettings(
                content_type=uploaded_file.content_type
            )
        )
    except Exception as e:
        print(f"UploadToBlobStorage::Error {e}")
        raise RuntimeError("Upload Failed. Cannot connect to blob storage.")

    folder_url = f"{settings.AZURE_BLOB_URL}/{settings.AZURE_UPLOAD_CONTAINER_NAME}/{file_key}"
    return folder_url

def upload_video_to_blob(uploaded_file):
    file_key = uuid.uuid4()
    folder_url = upload_to_blob_storage(uploaded_file, file_key)

    try:
        upload_record = FileMetadata.objects.create(
            original_file_name=uploaded_file.name,
            file_key=file_key,
            upload_blob_url=folder_url,
            download_blob_url=None,
            status=FileMetadata.Status.UPLOADED,
            task_id=None,
        )
    except Exception as e:
        print(f"CreateFileMetadata object::Error {e}")
        raise RuntimeError("Failed to create metadata.")

    return upload_record

def upload_image_to_blob(whitelist_images, file_key):
    try:
        record = FileMetadata.objects.get(file_key=file_key)
        for img in whitelist_images:
            upload_to_blob_storage(img, f"{file_key}/{settings.AZURE_WHITELIST_FOLDER}")
        record.status = FileMetadata.Status.WHITELIST_UPLOADED
        record.save(update_fields=["status"])
        return record
    except FileMetadata.DoesNotExist:
        raise FileNotFoundError(f"No FileMetadata record found for file_key={file_key}")
    except Exception as e:
        print(f"UploadImageToBlob::Error {e}")
        raise RuntimeError("Failed to upload whitelist images.")

