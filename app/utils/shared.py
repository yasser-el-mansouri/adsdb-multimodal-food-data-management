"""
Shared utilities for the data pipeline.

This module provides common utilities that can be imported by any part of the pipeline.
"""

import os
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from pathlib import Path, PurePosixPath
from contextlib import contextmanager
from decimal import Decimal
import hashlib
import io
import re

# Optional imports
try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

try:
    from PIL import Image, ImageOps
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Import configuration
from .config import PipelineConfig


class S3Client:
    """MinIO/S3 client wrapper with common operations."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize S3 client."""
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required for S3Client. Install with: pip install boto3")
        
        self.config = config
        self.minio_user = config.get_env("MINIO_USER")
        self.minio_password = config.get_env("MINIO_PASSWORD")
        self.minio_endpoint = config.get_env("MINIO_ENDPOINT")
        
        if not all([self.minio_user, self.minio_password, self.minio_endpoint]):
            raise ValueError("MinIO credentials not found in environment variables")
        
        self.session = boto3.session.Session(
            aws_access_key_id=self.minio_user,
            aws_secret_access_key=self.minio_password,
            region_name="us-east-1"
        )
        
        self.client = self.session.client(
            "s3",
            endpoint_url=self.minio_endpoint,
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"})
        )
    
    def list_objects(self, bucket: str, prefix: str = "") -> List[str]:
        """List all object keys in a bucket with the given prefix."""
        keys = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith("/"):
                    keys.append(key)
        return keys
    
    def head_object(self, bucket: str, key: str) -> Optional[Dict[str, Any]]:
        """Get object metadata, return None if not found."""
        try:
            return self.client.head_object(Bucket=bucket, Key=key)
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return None
            raise
    
    def copy_object(self, src_bucket: str, src_key: str, dst_bucket: str, dst_key: str, 
                   overwrite: bool = True, metadata: Optional[Dict[str, str]] = None) -> str:
        """Copy an object between buckets."""
        if not overwrite and self.head_object(dst_bucket, dst_key) is not None:
            return "skip-exists"
        
        extra = {"MetadataDirective": "REPLACE"} if metadata else {}
        if metadata:
            extra["Metadata"] = metadata
        
        self.client.copy_object(
            Bucket=dst_bucket,
            Key=dst_key,
            CopySource={"Bucket": src_bucket, "Key": src_key},
            **extra
        )
        return "copied"
    
    def put_object(self, bucket: str, key: str, body: bytes, 
                   content_type: str = "application/octet-stream",
                   metadata: Optional[Dict[str, str]] = None) -> None:
        """Put an object to S3."""
        extra = {}
        if content_type:
            extra["ContentType"] = content_type
        if metadata:
            extra["Metadata"] = metadata
        
        self.client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            **extra
        )
    
    def get_object(self, bucket: str, key: str) -> Dict[str, Any]:
        """Get an object from S3."""
        response = self.client.get_object(Bucket=bucket, Key=key)
        return response


class Logger:
    """Enhanced logging utility."""
    
    def __init__(self, name: str, level: str = "INFO"):
        """Initialize logger."""
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message, extra=kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, extra=kwargs)


class ImageUtils:
    """Utilities for image processing shared across processors."""

    @staticmethod
    def check_cv2() -> bool:
        try:
            import cv2  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def check_imagehash() -> bool:
        try:
            import imagehash  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def compute_metrics(img) -> Dict[str, Any]:
        if not PIL_AVAILABLE:
            raise ImportError("PIL is required for ImageUtils. Install with: pip install Pillow")
        
        w, h = img.size
        aspect = (w / h) if h else 0
        metrics = {"w": w, "h": h, "aspect": float(aspect)}

        if ImageUtils.check_cv2():
            import cv2
            gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
            metrics["blur_varlap"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        return metrics

    @staticmethod
    def normalize_image(
        img,
        target_size: tuple[int, int] = (512, 512),
        target_mode: str = "RGB",
        target_format: str = "JPEG",
        target_quality: int = 90,
    ) -> bytes:
        if not PIL_AVAILABLE:
            raise ImportError("PIL is required for ImageUtils. Install with: pip install Pillow")
        
        img_rgb = img.convert(target_mode)
        img_fit = ImageOps.pad(
            img_rgb, target_size,
            method=Image.BICUBIC,
            color=None,
            centering=(0.5, 0.5)
        )
        buf = io.BytesIO()
        img_fit.save(buf, format=target_format, quality=target_quality, optimize=True)
        return buf.getvalue()

    @staticmethod
    def extract_recipe_id_from_key(key: str) -> str:
        name = PurePosixPath(key).name
        m = re.search(r"__([A-Za-z0-9_\-]+)_(\d+)\.", name)
        return m.group(1) if m else name


class KeyUtils:
    """Utilities for generating deterministic object keys."""

    @staticmethod
    def make_minio_key(path: str) -> str:
        p = PurePosixPath(path)
        h = hashlib.sha256(str(p).encode("utf-8")).hexdigest()[:16]
        return f"{h}__{p.name or 'file'}"

    @staticmethod
    def make_minio_key_image(prefix: str, recipe_id: str, index: int, ext: str) -> str:
        base_name = f"{recipe_id}_{index}.{ext or 'bin'}"
        h = hashlib.sha256(f"{recipe_id}:{index}".encode("utf-8")).hexdigest()[:32]
        return f"{prefix}/{h}__{base_name}"


def utc_timestamp() -> str:
    """Generate UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def to_builtin(obj: Any) -> Any:
    """Convert Decimal objects to built-in Python types."""
    if isinstance(obj, Decimal):
        return int(obj) if obj == obj.to_integral_value() else float(obj)
    if isinstance(obj, dict):
        return {k: to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_builtin(v) for v in obj]
    return obj


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    return re.sub(r"[^\w\-.]+", "_", filename)


def atomic_write_json(filepath: str, data: Dict[str, Any]) -> None:
    """Atomically write JSON data to file."""
    tmp_path = f"{filepath}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, filepath)


@contextmanager
def error_handler(logger: Logger, operation: str):
    """Context manager for error handling."""
    try:
        yield
    except Exception as e:
        logger.error(f"Error in {operation}: {str(e)}")
        raise
