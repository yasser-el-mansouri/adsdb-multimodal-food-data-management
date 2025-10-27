"""
Configuration management for the data pipeline.

This module contains the PipelineConfig class and related configuration utilities.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path


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
            # Try to import yaml, fallback to JSON if not available
            try:
                import yaml
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            except ImportError:
                # Fallback: try to load as JSON
                json_path = self.config_path.replace('.yaml', '.json')
                if Path(json_path).exists():
                    with open(json_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                else:
                    logging.warning(f"YAML not available and JSON config not found, using defaults")
                    return self._get_default_config()
        except FileNotFoundError:
            logging.warning(f"Config file {self.config_path} not found, using defaults")
            return self._get_default_config()
    
    def _load_env(self):
        """Load environment variables."""
        # Simple environment loading without dotenv dependency
        # This will load from system environment variables
        pass
    
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
                },
                "prefixes": {
                    "temporal_landing": "temporal_landing",
                    "persistent_landing_images": "persistent_landing/images",
                    "persistent_landing_documents": "persistent_landing/documents",
                    "formatted_images": "images",
                    "formatted_documents": "documents",
                    "trusted_images": "images",
                    "trusted_documents": "documents",
                    "trusted_reports": "reports",
                    "chroma_exploitation": "chroma_exploitation"
                }
            },
            "huggingface": {
                "skip_files": [".gitattributes", ".gitignore", ".gitkeep"],
                "file_extensions": {
                    "image": ["jpg", "jpeg", "png", "gif", "webp", "bmp", "tiff"],
                    "document": ["json", "jsonl", "ndjson"]
                }
            },
            "image_processing": {
                "target_size": [512, 512],
                "target_mode": "RGB",
                "target_format": "JPEG",
                "target_quality": 90,
                "quality_thresholds": {
                    "min_width": 128,
                    "min_height": 128,
                    "min_aspect_ratio": 0.5,
                    "max_aspect_ratio": 3.0,
                    "blur_varlap_min": 50.0
                },
                "deduplication": {
                    "enabled": True,
                    "method": "perceptual_hash"
                }
            },
            "document_processing": {
                "skip_fields": ["url", "partition"],
                "always_tag": True,
                "nutrition": {
                    "totals_key_candidates": [
                        "nutr_values_per100g__from_recipes_with_nutritional_info",
                        "nutr_values_per100g",
                        "nutr_values"
                    ],
                    "per_ingredient_key_candidates": [
                        "nutr_per_ingredient__from_recipes_with_nutritional_info",
                        "nutr_per_ingredient"
                    ],
                    "numeric_fields": ["energy", "fat", "protein", "salt", "saturates", "sugars"],
                    "drop_totals_if_per_ingredient_present": True
                },
                "text_cleaning": {
                    "language": "english",
                    "remove_punctuation": True,
                    "remove_stopwords": True
                }
            },
            "chromadb": {
                "collection_name": "trusted_zone_documents",
                "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
                "metadata": {
                    "modality": "text",
                    "model": "Qwen/Qwen3-Embedding-0.6B",
                    "source": "minio"
                }
            },
            "monitoring": {
                "enabled": True,
                "log_level": "INFO",
                "performance_tracking": True,
                "resource_monitoring": True,
                "metrics": [
                    "execution_time",
                    "memory_usage",
                    "disk_usage",
                    "processed_records",
                    "error_count"
                ]
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