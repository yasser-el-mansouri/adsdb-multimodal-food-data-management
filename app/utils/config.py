"""
Configuration management for the data pipeline.

This module contains the PipelineConfig class and related configuration utilities.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


class PipelineConfig:
    """Configuration manager for the pipeline."""

    def __init__(self, config_path: str = "app/pipeline.yaml"):
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

                # Use utf-8-sig to gracefully handle BOM if present
                with open(self.config_path, "r", encoding="utf-8-sig") as f:
                    return yaml.safe_load(f)
            except ImportError:
                # Fallback: try to load as JSON
                json_path = self.config_path.replace(".yaml", ".json")
                if Path(json_path).exists():
                    with open(json_path, "r", encoding="utf-8") as f:
                        return json.load(f)
                else:
                    logging.warning(f"YAML not available and JSON config not found, using defaults")
                    return self._get_default_config()
            except Exception as e:
                logging.warning(f"Error loading YAML config: {e}, using defaults")
                return self._get_default_config()
        except FileNotFoundError:
            logging.warning(f"Config file {self.config_path} not found, using defaults")
            return self._get_default_config()
        except Exception as e:
            logging.warning(f"Error loading config: {e}, using defaults")
            return self._get_default_config()

    def _load_env(self):
        """Load environment variables."""
        # Simple environment loading without dotenv dependency
        # This will load from system environment variables
        pass

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "pipeline": {"dry_run": False, "overwrite": True, "batch_size": 256, "timeout": 60},
            "storage": {
                "buckets": {
                    "landing_zone": "landing-zone",
                    "formatted_zone": "formatted-zone",
                    "trusted_zone": "trusted-zone",
                    "exploitation_zone": "exploitation-zone",
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
                },
            },
            "huggingface": {
                "skip_files": [".gitattributes", ".gitignore", ".gitkeep"],
                "file_extensions": {
                    "image": ["jpg", "jpeg", "png", "gif", "webp", "bmp", "tiff"],
                    "document": ["json", "jsonl", "ndjson"],
                },
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
                    "blur_varlap_min": 50.0,
                },
                "deduplication": {"enabled": True, "method": "perceptual_hash"},
            },
            "document_processing": {
                "skip_fields": ["url", "partition"],
                "always_tag": True,
                "nutrition": {
                    "totals_key_candidates": ["nutrition_per100g", "nutrition_totals"],
                    "per_ingredient_key_candidates": ["nutrition_per_ingredient"],
                    "numeric_fields": ["fat", "salt", "saturates", "sugars"],
                    "drop_totals_if_per_ingredient_present": True,
                },
                "text_cleaning": {
                    "language": "english",
                    "remove_punctuation": True,
                    "remove_stopwords": True,
                },
            },
            "chromadb_documents": {
                "collection_name": "exploitation_documents",
                "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
                "metadata": {},
                "persist_dir": "app/zones/exploitation_zone/chroma_documents",
            },
            "chromadb_images": {
                "collection_name": "exploitation_images",
                "embedding_model": "clip-ViT-B-32",
                "metadata": {},
                "persist_dir": "app/zones/exploitation_zone/chroma_images",
            },
            "file_paths": {
                "image_index": "app/zones/landing_zone/image_index.json",
                "recipes_index": "app/zones/landing_zone/recipes_index.json",
                "recipe_ids_with_images": "app/zones/trusted_zone/recipe_ids_with_images.json",
            },
            "transfer": {
                "multipart_threshold": 8388608,
                "multipart_chunksize": 8388608,
                "max_concurrency": 4,
                "use_threads": True,
            },
            "testing": {"max_recipes": 1000, "max_processed": 50},
            "monitoring": {"enabled": True, "log_level": "INFO"},
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split(".")
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
