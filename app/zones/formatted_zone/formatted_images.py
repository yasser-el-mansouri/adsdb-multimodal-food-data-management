"""
Formatted Zone - Images Processing

This module handles the images format processing step for the Formatted Zone.
It processes and organizes images from the landing zone.
"""

import io
from pathlib import PurePosixPath
from typing import Any, Dict, Iterable, List, Set

import boto3
from PIL import Image, ImageOps

# Import shared utilities
from app.utils.shared import (
    ImageUtils,
    Logger,
    PipelineConfig,
    S3Client,
    error_handler,
    utc_timestamp,
)


class FormattedImagesProcessor:
    """Processor for formatted zone images."""

    def __init__(self, config: PipelineConfig):
        """Initialize the processor."""
        self.config = config
        self.logger = Logger("formatted_images", config.get("monitoring.log_level", "INFO"))
        self.s3_client = S3Client(config)
        # Configuration
        self.src_bucket = config.get("storage.buckets.landing_zone")
        self.src_prefix = config.get("storage.prefixes.persistent_landing_images")
        self.out_bucket = config.get("storage.buckets.formatted_zone")
        self.out_prefix = config.get("storage.prefixes.formatted_images")

        # Image processing configuration
        self.target_size = tuple(config.get("image_processing.target_size", [512, 512]))
        self.target_mode = config.get("image_processing.target_mode", "RGB")
        self.target_format = config.get("image_processing.target_format", "JPEG")
        self.target_quality = config.get("image_processing.target_quality", 90)

        # Quality thresholds
        quality_config = config.get("image_processing.quality_thresholds", {})
        self.min_width = quality_config.get("min_width", 128)
        self.min_height = quality_config.get("min_height", 128)
        self.min_aspect_ratio = quality_config.get("min_aspect_ratio", 0.5)
        self.max_aspect_ratio = quality_config.get("max_aspect_ratio", 3.0)
        self.blur_varlap_min = quality_config.get("blur_varlap_min", 50.0)

        # Deduplication
        self.deduplication_enabled = config.get("image_processing.deduplication.enabled", True)

        # Behavior flags
        self.dry_run = config.get("pipeline.dry_run", False)
        self.overwrite = config.get("pipeline.overwrite", True)

        # Check for optional dependencies
        self.cv2_available = ImageUtils.check_cv2()
        self.imagehash_available = ImageUtils.check_imagehash()

    def list_images(self, bucket: str, prefix: str) -> Iterable[str]:
        """List image files in bucket with prefix."""
        paginator = self.s3_client.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                if key.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff")
                ):
                    yield key

    def compute_metrics(self, img: Image.Image) -> Dict[str, Any]:
        """Compute image quality metrics."""
        return ImageUtils.compute_metrics(img)

    def normalize_image(self, img: Image.Image) -> bytes:
        """Normalize image to target format."""
        return ImageUtils.normalize_image(
            img,
            target_size=self.target_size,
            target_mode=self.target_mode,
            target_format=self.target_format,
            target_quality=self.target_quality,
        )

    def extract_recipe_id_from_key(self, key: str) -> str:
        """Extract recipe ID from image key."""
        return ImageUtils.extract_recipe_id_from_key(key)

    def process(self) -> Dict[str, Any]:
        """Main processing method."""
        try:
            with error_handler(self.logger, "formatted_images_processing"):
                # Group images by recipe ID
                recipe_images: Dict[str, List[str]] = {}

                for key in self.list_images(self.src_bucket, self.src_prefix):
                    recipe_id = self.extract_recipe_id_from_key(key)
                    recipe_images.setdefault(recipe_id, []).append(key)

                self.logger.info(f"Found {len(recipe_images)} recipes with images")

                # Process images with quality screening
                quality_stats = {
                    "evaluated": 0,
                    "kept": 0,
                    "corrupted": 0,
                    "too_small": 0,
                    "bad_aspect": 0,
                    "too_blurry": 0,
                    "dupes_removed": 0,
                }

                kept_per_recipe: Dict[str, List[str]] = {}
                seen_phashes: Dict[str, Set[str]] = {}

                for recipe_id, keys in recipe_images.items():
                    kept_per_recipe[recipe_id] = []
                    if self.deduplication_enabled and self.imagehash_available:
                        seen_phashes[recipe_id] = set()

                    for src_key in keys:
                        quality_stats["evaluated"] += 1

                        try:
                            # Load original image
                            obj = self.s3_client.get_object(bucket=self.src_bucket, key=src_key)
                            raw = obj["Body"].read()
                            img = Image.open(io.BytesIO(raw))
                            img.load()
                        except Exception as e:
                            quality_stats["corrupted"] += 1
                            self.logger.warning(f"Corrupted image {src_key}: {e}")
                            continue

                        # Basic quality checks
                        metrics = self.compute_metrics(img)
                        w, h, aspect = metrics["w"], metrics["h"], metrics["aspect"]

                        if w < self.min_width or h < self.min_height:
                            quality_stats["too_small"] += 1
                            continue

                        if not (self.min_aspect_ratio <= aspect <= self.max_aspect_ratio):
                            quality_stats["bad_aspect"] += 1
                            continue

                        # Blur check
                        if self.cv2_available and self.blur_varlap_min is not None:
                            blur_val = metrics.get("blur_varlap", 0.0)
                            if blur_val < self.blur_varlap_min:
                                quality_stats["too_blurry"] += 1
                                continue

                        # Deduplication check
                        if self.deduplication_enabled and self.imagehash_available:
                            import imagehash

                            ph = str(imagehash.phash(img))
                            if ph in seen_phashes[recipe_id]:
                                quality_stats["dupes_removed"] += 1
                                continue
                            seen_phashes[recipe_id].add(ph)

                        kept_per_recipe[recipe_id].append(src_key)
                        quality_stats["kept"] += 1

                self.logger.info(f"Quality screening results: {quality_stats}")

                # Copy processed images to formatted zone
                copied = skipped = 0

                if not self.dry_run:
                    for recipe_id, keys in kept_per_recipe.items():
                        for src_key in keys:
                            # Generate destination key
                            base = PurePosixPath(src_key).name
                            base_noext = base.rsplit(".", 1)[0]
                            dst_key = f"{self.out_prefix}/{base_noext}.jpg"

                            if (
                                not self.overwrite
                                and self.s3_client.head_object(self.out_bucket, dst_key) is not None
                            ):
                                self.logger.info(f"[SKIP] {dst_key} already exists")
                                skipped += 1
                                continue

                            try:
                                # Load and normalize image
                                obj = self.s3_client.get_object(bucket=self.src_bucket, key=src_key)
                                raw = obj["Body"].read()
                                img = Image.open(io.BytesIO(raw))
                                img.load()

                                # Normalize
                                out_bytes = self.normalize_image(img)

                                # Write normalized image
                                self.s3_client.put_object(
                                    bucket=self.out_bucket,
                                    key=dst_key,
                                    body=out_bytes,
                                    content_type="image/jpeg",
                                )
                                copied += 1

                            except Exception as e:
                                self.logger.warning(
                                    f"Failed to process {src_key} -> {dst_key}: {e}"
                                )
                                skipped += 1

                result = {
                    "total_recipes": len(recipe_images),
                    "total_images": sum(len(keys) for keys in recipe_images.values()),
                    "quality_stats": quality_stats,
                    "images_copied": copied,
                    "images_skipped": skipped,
                    "timestamp": utc_timestamp(),
                }

                self.logger.info(f"[STATS] Images copied: {copied}, skipped: {skipped}")
                return result

        except Exception as e:
            self.logger.error(f"Formatted images processing failed: {e}")
            raise


def main():
    """Main entry point for formatted images processing."""
    config = PipelineConfig()
    processor = FormattedImagesProcessor(config)

    try:
        result = processor.process()
        print(f"✅ Formatted images processing completed successfully")
        print(f"📊 Metrics: {result}")
    except Exception as e:
        print(f"❌ Formatted images processing failed: {e}")
        raise


if __name__ == "__main__":
    main()
