"""
app/storage/gcs.py — GCP Cloud Storage helpers.

Provides async wrappers for downloading and uploading video files
to/from GCS using the google-cloud-storage Python SDK.
"""

import asyncio
import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from google.auth.exceptions import DefaultCredentialsError
from google.auth import compute_engine
from google.cloud import storage
from google.oauth2 import service_account
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from app.config import get_settings
import structlog

log = structlog.get_logger(__name__)

DEFAULT_CREDENTIALS_PATH = "/app/service_account.json"


def _is_cloud_run() -> bool:
    """Return True when running inside Cloud Run."""
    return bool(os.getenv("K_SERVICE"))


def _get_client() -> storage.Client:
    """Return a GCS client using a local key when configured, otherwise ADC."""
    settings = get_settings()
    env_credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    settings_credentials_path = settings.google_application_credentials
    credentials_path = env_credentials_path or settings_credentials_path

    if credentials_path and (
        env_credentials_path
        or settings_credentials_path != DEFAULT_CREDENTIALS_PATH
        or Path(credentials_path).expanduser().is_file()
    ):
        key_path = Path(credentials_path).expanduser()
        if not key_path.is_file():
            if _is_cloud_run():
                log.warning(
                    "gcs_credentials_file_missing_using_cloud_run_adc",
                    path=credentials_path,
                )
                return storage.Client(
                    project=settings.gcp_project_id,
                    credentials=compute_engine.Credentials(),
                )

            raise DefaultCredentialsError(
                "GOOGLE_APPLICATION_CREDENTIALS points to a missing service account file: "
                f"{credentials_path}"
            )

        credentials = service_account.Credentials.from_service_account_file(str(key_path))
        return storage.Client(project=settings.gcp_project_id, credentials=credentials)

    try:
        return storage.Client(project=settings.gcp_project_id)
    except DefaultCredentialsError as exc:
        raise DefaultCredentialsError(
            "Google Cloud credentials are not configured. Set GOOGLE_APPLICATION_CREDENTIALS "
            "to a readable service account JSON file for local Docker/dev, or run the service "
            "with a Cloud Run service account that can access the bucket."
        ) from exc


def _parse_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    """Parse gs://bucket/path into (bucket_name, blob_name)."""
    parsed = urlparse(gcs_uri)
    if parsed.scheme != "gs":
        raise ValueError(f"Invalid GCS URI: {gcs_uri}. Must start with gs://")
    bucket_name = parsed.netloc
    blob_name = parsed.path.lstrip("/")
    return bucket_name, blob_name


# ── Download ──────────────────────────────────────────────────────────────────

@retry(
    retry=retry_if_not_exception_type(DefaultCredentialsError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
async def download_from_gcs(gcs_uri: str, local_path: str) -> str:
    """
    Download a GCS object to a local file path.

    Args:
        gcs_uri:    GCS URI like gs://bucket/path/video.mp4
        local_path: Local filesystem path to save the file

    Returns:
        local_path on success
    """
    log.info("gcs_download_start", uri=gcs_uri, dest=local_path)

    def _download() -> None:
        bucket_name, blob_name = _parse_gcs_uri(gcs_uri)
        client = _get_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Ensure parent directory exists
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(local_path)

    # Run blocking GCS SDK call in thread pool
    await asyncio.get_event_loop().run_in_executor(None, _download)

    log.info("gcs_download_done", uri=gcs_uri, local=local_path)
    return local_path


# ── Upload ────────────────────────────────────────────────────────────────────

@retry(
    retry=retry_if_not_exception_type(DefaultCredentialsError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
async def upload_to_gcs(
    local_path: str,
    destination_blob_name: str,
    content_type: str = "video/mp4",
    bucket_name: Optional[str] = None,
) -> str:
    """
    Upload a local file to GCS and return its gs:// URI.

    Args:
        local_path:             Local file to upload
        destination_blob_name:  Path inside the bucket (e.g., output/job_id/clip.mp4)
        content_type:           MIME type of the file
        bucket_name:            Override bucket (default: from settings)

    Returns:
        GCS URI: gs://bucket/destination_blob_name
    """
    settings = get_settings()
    target_bucket = bucket_name or settings.gcp_bucket_name

    log.info("gcs_upload_start", local=local_path, dest=destination_blob_name)

    def _upload() -> None:
        client = _get_client()
        bucket = client.bucket(target_bucket)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(local_path, content_type=content_type)

    await asyncio.get_event_loop().run_in_executor(None, _upload)

    gcs_uri = f"gs://{target_bucket}/{destination_blob_name}"
    log.info("gcs_upload_done", uri=gcs_uri)
    return gcs_uri


async def upload_image_to_gcs(
    local_path: str,
    destination_blob_name: str,
    bucket_name: Optional[str] = None,
) -> str:
    """Convenience wrapper for image uploads (JPEG/PNG)."""
    ext = Path(local_path).suffix.lower()
    content_type = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    return await upload_to_gcs(local_path, destination_blob_name, content_type, bucket_name)


# ── Signed URL (for client-side access) ──────────────────────────────────────

async def generate_signed_url(gcs_uri: str, expiration_minutes: int = 60) -> str:
    """
    Generate a time-limited signed URL for a GCS object.
    Useful to give the frontend a direct download link.
    """
    import datetime

    def _sign() -> str:
        bucket_name, blob_name = _parse_gcs_uri(gcs_uri)
        client = _get_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        url = blob.generate_signed_url(
            expiration=datetime.timedelta(minutes=expiration_minutes),
            method="GET",
            version="v4",
        )
        return url

    return await asyncio.get_event_loop().run_in_executor(None, _sign)
