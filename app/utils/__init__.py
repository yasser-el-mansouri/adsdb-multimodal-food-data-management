"""
Shared utilities for the data pipeline operations.

This module contains common functions and classes used across all pipeline stages.
"""

import os
import json
import logging
import yaml
import psutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Iterable
from decimal import Decimal
from contextlib import contextmanager

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv


class PipelineConfig:
    """Configuration manager for the pipeline."""
    
    def __init__(self, config_path: str = "app/config/pipeline.yaml"):
        """Initialize configuration from YAML file."""
        self.config_path = config_path
        self._config = self._load_config()
        self._load_env()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logging.warning(f"Config file {self.config_path} not found, using defaults")
            return self._get_default_config()
    
    def _load_env(self):
        """Load environment variables."""
        load_dotenv()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "pipeline": {
                "dry_run": False,
                "overwrite": True,
                "batch_size": 256,
                "timeout": 60
            },
            "storage": {
                "buckets": {
                    "landing_zone": "landing-zone",
                    "formatted_zone": "formatted-zone",
                    "trusted_zone": "trusted-zone",
                    "exploitation_zone": "exploitation-zone"
                }
            },
            "monitoring": {
                "enabled": True,
                "log_level": "INFO"
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def get_env(self, key: str, default: Any = None) -> Any:
        """Get environment variable."""
        return os.getenv(key, default)


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
    
    def list_objects(self, bucket: str, prefix: str = "") -> Iterable[str]:
        """List all object keys in a bucket with the given prefix."""
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith("/"):
                    yield key
    
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
        return self.client.get_object(Bucket=bucket, Key=key)


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
        self.start_memory = psutil.virtual_memory().used
        self.start_disk = psutil.disk_usage('.').used
    
    def stop(self) -> Dict[str, Any]:
        """Stop monitoring and return metrics."""
        if not self.enabled or self.start_time is None:
            return {}
        
        end_time = time.time()
        end_memory = psutil.virtual_memory().used
        end_disk = psutil.disk_usage('.').used
        
        return {
            "execution_time": end_time - self.start_time,
            "memory_usage": end_memory - self.start_memory,
            "disk_usage": end_disk - self.start_disk,
            "peak_memory": psutil.virtual_memory().percent,
            "cpu_percent": psutil.cpu_percent()
        }


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
