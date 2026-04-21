import uuid
import os
from datetime import datetime, timezone, timedelta

from django.conf import settings
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContentSettings, BlobSasPermissions, generate_blob_sas

_blob_service_client = None


def _get_client() -> BlobServiceClient:
    global _blob_service_client
    if _blob_service_client is None:
        _blob_service_client = BlobServiceClient(
            account_url=settings.AZURE_BLOB_URL,
            credential=DefaultAzureCredential(),
        )
    return _blob_service_client

def build_blob_path(file_key: str, filename: str = None, get_whitelist: bool = False, upload: bool = True) -> str:
    if not file_key:
        raise ValueError("file_key cannot be null")

    base_dir = settings.AZURE_UPLOAD_DIR if upload else settings.AZURE_DOWNLOAD_DIR
    parts = [base_dir, str(file_key)]

    if get_whitelist:
        parts.append(settings.AZURE_WHITELIST_FOLDER)
    if filename:
        parts.append(filename)

    path = "/".join(parts)
    return path if filename else path + "/"


def build_url(blob_path: str) -> str:
    return f"{settings.AZURE_BLOB_URL}/{settings.AZURE_CONTAINER_NAME}/{blob_path}"

def upload_to_blob_storage(uploaded_file, file_key: str, get_whitelist: bool = False) -> str:
    container_client = _get_client().get_container_client(settings.AZURE_CONTAINER_NAME)
    blob_path = build_blob_path(file_key, filename=uploaded_file.name, get_whitelist=get_whitelist)
    blob_client = container_client.get_blob_client(blob_path)

    try:
        blob_client.upload_blob(
            uploaded_file.file,
            overwrite=True,
            content_settings=ContentSettings(content_type=uploaded_file.content_type),
        )
    except Exception as e:
        print(f"UploadToBlobStorage::Error {e}")
        raise RuntimeError("Video Upload Failed. Cannot connect to blob storage.")

    return build_url(build_blob_path(file_key))  # return folder URL


def upload_video_to_blob(uploaded_file):
    from ..models import FileMetadata

    file_key = uuid.uuid4()
    folder_url = upload_to_blob_storage(uploaded_file, file_key)

    try:
        record = FileMetadata.objects.create(
            original_file_name=uploaded_file.name,
            file_key=file_key,
            upload_blob_url=folder_url,
            download_blob_url=None,
            status=FileMetadata.Status.UPLOADED,
            task_id=None,
        )
    except Exception as e:
        print(f"CreateFileMetadata::Error {e}")
        raise RuntimeError("Failed to create metadata.")

    return record


def upload_local_file_to_blob(
    local_file_path: str,
    file_key: str,
    content_type: str = "video/mp4",
) -> str:
    container_name = settings.AZURE_CONTAINER_NAME
    container_client = _get_client().get_container_client(container_name)

    filename = os.path.basename(local_file_path)
    blob_path = build_blob_path(file_key, filename=filename, upload=False)
    blob_client = container_client.get_blob_client(blob_path)

    try:
        with open(local_file_path, "rb") as f:
            blob_client.upload_blob(
                f,
                overwrite=True,
                content_settings=ContentSettings(content_type=content_type),
            )
    except Exception as e:
        print(f"UploadLocalFileToBlob::Error {e}")
        raise RuntimeError("Failed to upload local file to blob storage.")

    blob_url = build_url(blob_path)

    return blob_url, blob_path


def upload_image_to_blob(whitelist_images, file_key: str):
    from ..models import FileMetadata

    try:
        record = FileMetadata.objects.get(file_key=file_key)
        if record.status != FileMetadata.Status.UPLOADED:
            raise RuntimeError("Video has already been processed.")

        for img in whitelist_images:
            upload_to_blob_storage(img, file_key, get_whitelist=True)

        record.status = FileMetadata.Status.WHITELIST_UPLOADED
        record.save(update_fields=["status"])
        return record

    except FileMetadata.DoesNotExist:
        raise FileNotFoundError(f"No FileMetadata record found for file_key={file_key}")
    except Exception as e:
        print(f"UploadImageToBlob::Error {e}")
        raise RuntimeError("Failed to upload whitelist images.")

# ── download ──────────────────────────────────────────────────────────────────

def download_blob_by_name(file_key: str, file_name: str, local_dir: str, get_whitelisted: bool = False) -> str:
    container_client = _get_client().get_container_client(settings.AZURE_CONTAINER_NAME)
    blob_path = build_blob_path(file_key, filename=file_name, get_whitelist=get_whitelisted)
    blob_client = container_client.get_blob_client(blob_path)

    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, file_name)

    try:
        with open(local_path, "wb") as f:
            f.write(blob_client.download_blob().readall())
    except Exception as e:
        print(f"DownloadBlobByName::Error {e}")
        raise RuntimeError(f"Failed to download blob: {blob_path}")

    return local_path


def download_blobs(file_key: str, local_dir: str, get_whitelisted: bool = True) -> list[str]:
    container_client = _get_client().get_container_client(settings.AZURE_CONTAINER_NAME)
    prefix = build_blob_path(file_key, get_whitelist=get_whitelisted)

    os.makedirs(local_dir, exist_ok=True)

    downloaded = []
    try:
        blobs = container_client.list_blobs(name_starts_with=prefix)
        for blob in blobs:
            filename = os.path.basename(blob.name)
            if not filename:
                continue

            local_path = os.path.join(local_dir, filename)
            blob_client = container_client.get_blob_client(blob.name)

            with open(local_path, "wb") as f:
                f.write(blob_client.download_blob().readall())

            downloaded.append(local_path)

    except Exception as e:
        print(f"DownloadBlobs::Error {e}")
        raise RuntimeError(f"Failed to download blobs from prefix: {prefix}")

    return downloaded

def generate_download_sas_url(blob_path: str, expires_in_minutes: int = 63) -> str:
    now = datetime.now(timezone.utc)
    expiry = now + timedelta(minutes=expires_in_minutes)

    # 1. Ask Azure for a user delegation key using Entra-authenticated client
    user_delegation_key = _get_client().get_user_delegation_key(
        key_start_time=now,
        key_expiry_time=expiry,
    )

    # 2. Generate SAS for this specific blob
    sas_token = generate_blob_sas(
        account_name=settings.AZURE_STORAGE_ACCOUNT_NAME,
        container_name=settings.AZURE_CONTAINER_NAME,
        blob_name=blob_path,
        user_delegation_key=user_delegation_key,
        permission=BlobSasPermissions(read=True),
        expiry=expiry,
        start=now,
        content_disposition="attachment; filename=processed_video.mp4"
    )

    # 3. Build full downloadable URL
    blob_url = (
        f"{settings.AZURE_BLOB_URL}/"
        f"{settings.AZURE_CONTAINER_NAME}/"
        f"{blob_path}"
    )
    return f"{blob_url}?{sas_token}"
