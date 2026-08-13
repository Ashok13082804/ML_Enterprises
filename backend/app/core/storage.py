"""
MLVerse X — MinIO Object Storage Client with Local Disk Fallback
"""
import io
import os
import logging
from pathlib import Path
from typing import Optional, BinaryIO
from minio import Minio
from minio.error import S3Error

from app.core.config import settings

logger = logging.getLogger(__name__)

_minio_client: Optional[Minio] = None
_use_local_storage = False
LOCAL_STORAGE_DIR = Path(__file__).resolve().parents[3] / "storage"


def get_minio_client() -> Minio:
    global _minio_client
    if _minio_client is None:
        _minio_client = Minio(
            f"{settings.MINIO_HOST}:{settings.MINIO_PORT}",
            access_key=settings.MINIO_ROOT_USER,
            secret_key=settings.MINIO_ROOT_PASSWORD,
            secure=settings.MINIO_SECURE,
        )
    return _minio_client


async def init_storage():
    """Create buckets if they don't exist, falling back to local disk if MinIO is offline."""
    global _use_local_storage
    buckets = [
        settings.MINIO_BUCKET_DATASETS,
        settings.MINIO_BUCKET_MODELS,
        settings.MINIO_BUCKET_REPORTS,
        settings.MINIO_BUCKET_AVATARS,
    ]
    try:
        client = get_minio_client()
        # Set a quick timeout to fail fast if host is offline
        client._http.connection_pool_kw['timeout'] = 1.0
        
        for bucket in buckets:
            if not client.bucket_exists(bucket):
                client.make_bucket(bucket)
                logger.info(f"Created MinIO bucket: {bucket}")
        logger.info("Successfully connected to MinIO storage.")
    except Exception as e:
        logger.warning(f"Failed to connect to MinIO storage at {settings.MINIO_HOST}:{settings.MINIO_PORT} (Error: {e}). Falling back to local disk storage.")
        _use_local_storage = True
        
        # Create local directories
        for bucket in buckets:
            bucket_dir = LOCAL_STORAGE_DIR / bucket
            bucket_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Local storage directories verified at: {LOCAL_STORAGE_DIR}")


def upload_file(
    bucket: str,
    object_name: str,
    data: BinaryIO,
    length: int,
    content_type: str = "application/octet-stream",
) -> str:
    """Upload a file to MinIO or local storage and return the object URL/path."""
    if _use_local_storage:
        file_path = LOCAL_STORAGE_DIR / bucket / object_name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(data.read())
        return f"/storage/{bucket}/{object_name}"

    client = get_minio_client()
    client.put_object(
        bucket,
        object_name,
        data,
        length,
        content_type=content_type,
    )
    return f"http://{settings.MINIO_HOST}:{settings.MINIO_PORT}/{bucket}/{object_name}"


def download_file(bucket: str, object_name: str) -> bytes:
    """Download a file from MinIO or local storage."""
    if _use_local_storage:
        file_path = LOCAL_STORAGE_DIR / bucket / object_name
        with open(file_path, "rb") as f:
            return f.read()

    client = get_minio_client()
    response = client.get_object(bucket, object_name)
    return response.read()


def delete_file(bucket: str, object_name: str) -> None:
    """Delete a file from MinIO or local storage."""
    if _use_local_storage:
        file_path = LOCAL_STORAGE_DIR / bucket / object_name
        if file_path.exists():
            file_path.unlink()
        return

    client = get_minio_client()
    client.remove_object(bucket, object_name)


def get_presigned_url(bucket: str, object_name: str, expires_hours: int = 1) -> str:
    """Get a presigned URL or local path for temporary access."""
    if _use_local_storage:
        return f"/storage/{bucket}/{object_name}"

    from datetime import timedelta
    client = get_minio_client()
    return client.presigned_get_object(
        bucket, object_name, expires=timedelta(hours=expires_hours)
    )
