"""
Trusted Zone - Images Processing

This module handles the image processing step for the Trusted Zone.
It extracts recipe IDs from images and copies filtered images to the trusted zone.
"""

import io
import json
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


class TrustedImagesProcessor:
    """Processor for trusted zone images."""

    def __init__(self, config: PipelineConfig):
        """Initialize the processor."""
        self.config = config
        self.logger = Logger("trusted_images", config.get("monitoring.log_level", "INFO"))
        self.s3_client = S3Client(config)
        # Configuration
        self.src_bucket = config.get("storage.buckets.formatted_zone")
        self.src_prefix = config.get("storage.prefixes.formatted_images")
        self.out_bucket = config.get("storage.buckets.trusted_zone")
        self.out_prefix = config.get("storage.prefixes.trusted_images")
        self.report_prefix = config.get("storage.prefixes.trusted_reports")

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

        # Output file for documents processing - save in configured path
        self.recipe_ids_file = config.get(
            "file_paths.recipe_ids_with_images",
            "app/zones/trusted_zone/recipe_ids_with_images.json",
        )

        # Check for optional dependencies
        self.cv2_available = ImageUtils.check_cv2()
        self.imagehash_available = ImageUtils.check_imagehash()

    def list_images(self, bucket: str, prefix: str) -> Iterable[str]:
        """List image files in bucket with prefix."""
        paginator = self.s3_client.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith("/"):
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
            with error_handler(self.logger, "trusted_images_processing"):
                # Extract recipe IDs from image filenames
                img_ids: Set[str] = set()
                id_to_imgkeys: Dict[str, List[str]] = {}

                count_keys = 0
                for key in self.list_images(self.src_bucket, self.src_prefix):
                    count_keys += 1
                    rid = self.extract_recipe_id_from_key(key)
                    if not rid:
                        continue
                    img_ids.add(rid)
                    id_to_imgkeys.setdefault(rid, []).append(key)

                # Make copies deterministic
                for rid in id_to_imgkeys:
                    id_to_imgkeys[rid].sort()

                total_images = sum(len(v) for v in id_to_imgkeys.values())
                self.logger.info(f"Scanned image keys: {count_keys}")
                self.logger.info(f"Unique recipeIds with images: {len(img_ids)}")
                self.logger.info(f"Total image files matched to recipeIds: {total_images}")

                # Save recipe IDs for documents processing
                recipe_ids_data = {
                    "timestamp": utc_timestamp(),
                    "source": f"s3://{self.src_bucket}/{self.src_prefix}/",
                    "total_images_scanned": count_keys,
                    "unique_recipe_ids": len(img_ids),
                    "total_images_matched": total_images,
                    "recipe_ids_with_images": sorted(list(img_ids)),
                    "recipe_to_images": {rid: keys for rid, keys in id_to_imgkeys.items()},
                }

                if not self.dry_run:
                    with open(self.recipe_ids_file, "w", encoding="utf-8") as f:
                        json.dump(recipe_ids_data, f, ensure_ascii=False, indent=2)
                    self.logger.info(f"Saved recipe IDs to {self.recipe_ids_file}")

                # Quality screening and deduplication
                quality_stats = {
                    "evaluated": 0,
                    "kept": 0,
                    "corrupted": 0,
                    "too_small": 0,
                    "bad_aspect": 0,
                    "too_blurry": 0,
                    "dupes_removed": 0,
                }

                kept_per_rid: Dict[str, List[str]] = {}
                skips_log: List[Dict[str, str]] = []
                seen_phashes: Dict[str, Set[str]] = {}

                for rid, keys in id_to_imgkeys.items():
                    kept_per_rid[rid] = []
                    if self.deduplication_enabled and self.imagehash_available:
                        seen_phashes[rid] = set()

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
                            skips_log.append(
                                {"key": src_key, "reason": f"corrupted:{type(e).__name__}"}
                            )
                            continue

                        # Basic quality checks
                        metrics = self.compute_metrics(img)
                        w, h, aspect = metrics["w"], metrics["h"], metrics["aspect"]

                        if w < self.min_width or h < self.min_height:
                            quality_stats["too_small"] += 1
                            skips_log.append({"key": src_key, "reason": f"too_small:{w}x{h}"})
                            continue

                        if not (self.min_aspect_ratio <= aspect <= self.max_aspect_ratio):
                            quality_stats["bad_aspect"] += 1
                            skips_log.append({"key": src_key, "reason": f"bad_aspect:{aspect:.2f}"})
                            continue

                        # Blur check
                        if self.cv2_available and self.blur_varlap_min is not None:
                            blur_val = metrics.get("blur_varlap", 0.0)
                            if blur_val < self.blur_varlap_min:
                                quality_stats["too_blurry"] += 1
                                skips_log.append(
                                    {
                                        "key": src_key,
                                        "reason": f"blurry:varLap={blur_val:.2f} < {self.blur_varlap_min}",
                                    }
                                )
                                continue

                        # Deduplication check
                        if self.deduplication_enabled and self.imagehash_available:
                            import imagehash

                            ph = str(imagehash.phash(img))
                            if ph in seen_phashes[rid]:
                                quality_stats["dupes_removed"] += 1
                                skips_log.append(
                                    {"key": src_key, "reason": f"duplicate_phash:{ph}"}
                                )
                                continue
                            seen_phashes[rid].add(ph)

                        kept_per_rid[rid].append(src_key)
                        quality_stats["kept"] += 1

                self.logger.info(f"Quality screening results: {quality_stats}")

                # Copy images to trusted zone
                copied = skipped = 0

                if not self.dry_run:
                    for rid, keys in kept_per_rid.items():
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

                # Generate processing report
                report = {
                    "timestamp": utc_timestamp(),
                    "processing_step": "images",
                    "source_images_prefix": f"s3://{self.src_bucket}/{self.src_prefix}/",
                    "destination_images_prefix": f"s3://{self.out_bucket}/{self.out_prefix}/",
                    "total_images_scanned": count_keys,
                    "unique_recipe_ids_with_images": len(img_ids),
                    "total_images_matched": total_images,
                    "quality_screen": quality_stats,
                    "images_copied_normalized": copied,
                    "images_skipped": skipped,
                    "normalization": {
                        "target_size": self.target_size,
                        "target_mode": self.target_mode,
                        "target_format": self.target_format,
                        "target_quality": self.target_quality,
                    },
                    "recipe_ids_file": self.recipe_ids_file,
                    "dry_run": self.dry_run,
                    "overwrite": self.overwrite,
                }

                if not self.dry_run and skips_log:
                    # Save skips log to S3
                    csv_buf = io.StringIO()
                    csv_buf.write("key,reason\n")
                    for row in skips_log:
                        key = row["key"].replace(",", " ")
                        reason = row["reason"].replace(",", " ")
                        csv_buf.write(f"{key},{reason}\n")

                    self.s3_client.put_object(
                        bucket=self.out_bucket,
                        key=f"{self.report_prefix}/images_skips_{utc_timestamp()}.csv",
                        body=csv_buf.getvalue().encode("utf-8"),
                        content_type="text/csv",
                    )

                result = {
                    "total_images_scanned": count_keys,
                    "unique_recipe_ids": len(img_ids),
                    "total_images_matched": total_images,
                    "quality_stats": quality_stats,
                    "images_copied": copied,
                    "images_skipped": skipped,
                    "timestamp": utc_timestamp(),
                }

                self.logger.info(f"[STATS] Images copied: {copied}, skipped: {skipped}")
                return result

        except Exception as e:
            self.logger.error(f"Trusted images processing failed: {e}")
            raise


def main():
    """Main entry point for trusted images processing."""
    config = PipelineConfig()
    processor = TrustedImagesProcessor(config)

    try:
        result = processor.process()
        print(f"✅ Trusted images processing completed successfully")
        print(f"📊 Metrics: {result}")
    except Exception as e:
        print(f"❌ Trusted images processing failed: {e}")
        raise


if __name__ == "__main__":
    main()
