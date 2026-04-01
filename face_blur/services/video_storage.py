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

def upload_to_blob_storage(uploaded_file):
    file_extension = os.path.splitext(uploaded_file.name)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"

    container_client = blob_service_client.get_container_client(
        settings.AZURE_UPLOAD_CONTAINER_NAME
    )

    blob_client = container_client.get_blob_client(f"{unique_filename}")

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

    file_url = f"{settings.AZURE_BLOB_URL}/{settings.AZURE_UPLOAD_CONTAINER_NAME}/{unique_filename}"
    print(f"file url: {file_url}, type: {type(file_url)}")
    try:
        upload_record = FileMetadata.objects.create(
            original_file_name=uploaded_file.name,
            upload_blob_url=file_url,
            download_blob_url=None,  # same for now unless you later generate SAS URL
            status=FileMetadata.Status.UPLOADED,
            task_id=None,
        )
    except Exception as e:
        print(f"CreateFileMetadata object::Error {e}")

    return upload_record