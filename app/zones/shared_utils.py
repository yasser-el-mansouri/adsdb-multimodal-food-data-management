"""
Shared utilities for zone processors.

This module provides common utilities that can be imported by zone processors.
"""

import os
import json
import logging
import time
import psutil
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from pathlib import Path
from contextlib import contextmanager

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

# Import configuration
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
import config
PipelineConfig = config.PipelineConfig

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Try to load from app/.env first, then from root .env
    env_paths = [
        os.path.join(os.path.dirname(__file__), '..', '.env'),
        os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    ]
    for env_path in env_paths:
        if os.path.exists(env_path):
            load_dotenv(env_path)
            print(f"[INFO] Loaded environment variables from: {env_path}")
            break
    else:
        print("[WARNING] No .env file found, using system environment variables")
except ImportError:
    print("[WARNING] python-dotenv not available, using system environment variables")


class S3Client:
    """MinIO/S3 client wrapper with common operations."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize S3 client."""
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


class PerformanceMonitor:
    """Monitor pipeline performance and resource usage."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize performance monitor."""
        self.config = config
        self.enabled = config.get("monitoring.enabled", True)
        self.start_time = None
        self.start_memory = None
        self.start_disk = None
    
    def start(self):
        """Start monitoring."""
        if not self.enabled:
            return
        
        self.start_time = time.time()
        try:
            self.start_memory = psutil.virtual_memory().used
            self.start_disk = psutil.disk_usage('.').used
        except:
            self.start_memory = 0
            self.start_disk = 0
    
    def stop(self) -> Dict[str, Any]:
        """Stop monitoring and return metrics."""
        if not self.enabled or self.start_time is None:
            return {}
        
        end_time = time.time()
        try:
            end_memory = psutil.virtual_memory().used
            end_disk = psutil.disk_usage('.').used
        except:
            end_memory = 0
            end_disk = 0
        
        return {
            "execution_time": end_time - self.start_time,
            "memory_usage": end_memory - self.start_memory,
            "disk_usage": end_disk - self.start_disk,
            "peak_memory": psutil.virtual_memory().percent if hasattr(psutil, 'virtual_memory') else 0,
            "cpu_percent": psutil.cpu_percent() if hasattr(psutil, 'cpu_percent') else 0
        }


def utc_timestamp() -> str:
    """Generate UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def to_builtin(obj: Any) -> Any:
    """Convert Decimal objects to built-in Python types."""
    from decimal import Decimal
    if isinstance(obj, Decimal):
        return int(obj) if obj == obj.to_integral_value() else float(obj)
    if isinstance(obj, dict):
        return {k: to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_builtin(v) for v in obj]
    return obj


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    import re
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


def validate_config(config: PipelineConfig) -> List[str]:
    """Validate configuration and return list of issues."""
    issues = []
    
    # Check required environment variables
    required_env_vars = ["MINIO_USER", "MINIO_PASSWORD", "MINIO_ENDPOINT"]
    for var in required_env_vars:
        if not config.get_env(var):
            issues.append(f"Missing required environment variable: {var}")
    
    # Check configuration values
    if config.get("pipeline.batch_size", 0) <= 0:
        issues.append("Pipeline batch size must be positive")
    
    if config.get("pipeline.timeout", 0) <= 0:
        issues.append("Pipeline timeout must be positive")
    
    return issues
