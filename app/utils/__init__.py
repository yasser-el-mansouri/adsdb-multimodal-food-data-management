"""
Utility modules for the data pipeline.

This package provides common utilities that can be imported by any part of the pipeline.
"""

import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    # Try to load from app/.env first, then from root .env
    env_paths = [
        os.path.join(os.path.dirname(__file__), ".env"),
        os.path.join(os.path.dirname(__file__), "..", ".env"),
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

# Import configuration and monitoring
from .config import PipelineConfig, validate_config
from .monitoring import PipelineMonitor, ResourceMonitor, create_monitoring_report, get_system_info

# Import all utilities from shared module
from .shared import (
    ImageUtils,
    KeyUtils,
    Logger,
    S3Client,
    atomic_write_json,
    error_handler,
    sanitize_filename,
    to_builtin,
    utc_timestamp,
)
