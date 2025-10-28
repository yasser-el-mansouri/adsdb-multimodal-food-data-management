"""
Landing Zone - Persistent Data Organization

This module handles the raw data organization step for the Landing Zone.
It organizes temporal data by type and applies naming conventions.
"""

import mimetypes
import os
import re
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional, Set

import boto3
from botocore.exceptions import ClientError

# Import shared utilities
from app.utils.shared import (
    Logger,
    PipelineConfig,
    S3Client,
    error_handler,
    sanitize_filename,
    utc_timestamp,
)


class PersistentLandingProcessor:
    """Processor for persistent landing zone data organization."""

    def __init__(self, config: PipelineConfig):
        """Initialize the processor."""
        self.config = config
        self.logger = Logger("persistent_landing", config.get("monitoring.log_level", "INFO"))
        self.s3_client = S3Client(config)
        # Configuration
        self.src_bucket = config.get("storage.buckets.landing_zone")
        self.dest_bucket = self.src_bucket
        self.img_prefix = config.get("storage.prefixes.persistent_landing_images")
        self.doc_prefix = config.get("storage.prefixes.persistent_landing_documents")
        self.hf_dataset = config.get_env("HF_DATASET")

        # Behavior flags
        self.delete_source_after_copy = True
        self.dry_run = config.get("pipeline.dry_run", False)
        self.overwrite = config.get("pipeline.overwrite", True)

        # File type detection
        self.image_mime_prefixes = ("image/",)
        self.image_exts = set(config.get("huggingface.file_extensions.image", []))
        self.doc_exts = set(config.get("huggingface.file_extensions.document", []))

    def guess_name_and_ext(self, key: str, head: Dict[str, Any]) -> tuple[str, str]:
        """Extract name and extension from key and headers."""
        p = PurePosixPath(key)
        name = p.name
        base = p.stem or "file"
        ext = p.suffix.lower().lstrip(".")

        if not ext:
            ctype = (head.get("ContentType") or "").split(";")[0].strip().lower()
            if ctype:
                guess = mimetypes.guess_extension(ctype) or ""
                ext = guess.lstrip(".")
                if ext == "jpe":
                    ext = "jpg"
        if ext == "jpeg":
            ext = "jpg"
        return base, ext or "bin"

    def is_image(self, head: Dict[str, Any], ext: str) -> bool:
        """Check if file is an image."""
        ctype = (head.get("ContentType") or "").lower()
        return ctype.startswith(self.image_mime_prefixes) or ext in self.image_exts

    def is_document_json(self, head: Dict[str, Any], ext: str) -> bool:
        """Check if file is a JSON document."""
        ctype = (head.get("ContentType") or "").split(";")[0].strip().lower()
        return ext in self.doc_exts or ctype == "application/json"

    def make_target_key(
        self, obj_type: str, dataset: str, ts: str, filename: str, ext: str, prefix: str
    ) -> str:
        """Generate target key with naming convention."""
        filename = sanitize_filename(filename)
        dataset = sanitize_filename(dataset)
        return f"{prefix}/{obj_type}${dataset}${ts}${filename}.{ext}"

    def copy_object(
        self,
        src_bucket: str,
        src_key: str,
        dst_bucket: str,
        dst_key: str,
        metadata: Optional[Dict[str, str]] = None,
        content_type: Optional[str] = None,
    ):
        """Copy object with metadata."""
        # Use S3Client's copy_object method
        self.s3_client.copy_object(
            src_bucket, src_key, dst_bucket, dst_key, overwrite=True, metadata=metadata
        )

        # Set content type if provided (S3Client doesn't support this yet)
        if content_type:
            self.s3_client.client.copy_object(
                CopySource={"Bucket": src_bucket, "Key": src_key},
                Bucket=dst_bucket,
                Key=dst_key,
                MetadataDirective="REPLACE",
                ContentType=content_type,
                Metadata=metadata or {},
            )

    def move_or_copy(self, src_bucket: str, src_key: str, dst_bucket: str, dst_key: str, **kwargs):
        """Move or copy object, optionally deleting source."""
        self.copy_object(src_bucket, src_key, dst_bucket, dst_key, **kwargs)
        if self.delete_source_after_copy:
            try:
                self.s3_client.client.delete_object(Bucket=src_bucket, Key=src_key)
            except ClientError as e:
                self.logger.warning(f"Failed to delete source {src_key}: {e}")

    # Index generation removed: recipe ID extraction is no longer handled here

    def process(self) -> Dict[str, Any]:
        """Main processing method."""
        try:
            with error_handler(self.logger, "persistent_landing_processing"):
                paginator = self.s3_client.client.get_paginator("list_objects_v2")
                ing_ts = utc_timestamp()

                pages = paginator.paginate(Bucket=self.src_bucket)
                total = moved_img = moved_doc = skipped = 0

                for page in pages:
                    for obj in page.get("Contents", []):
                        key = obj["Key"]
                        total += 1

                        if key.endswith("/") or key.startswith("."):
                            skipped += 1
                            continue

                        try:
                            head = self.s3_client.client.head_object(
                                Bucket=self.src_bucket, Key=key
                            )
                        except ClientError as e:
                            self.logger.warning(f"head_object failed for {key}: {e}")
                            skipped += 1
                            continue

                        base, ext = self.guess_name_and_ext(key, head)

                        if self.is_image(head, ext):
                            dst_key = self.make_target_key(
                                "image", self.hf_dataset, ing_ts, base, ext, prefix=self.img_prefix
                            )

                            if not self.dry_run:
                                self.move_or_copy(
                                    self.src_bucket,
                                    key,
                                    self.dest_bucket,
                                    dst_key,
                                    metadata={
                                        "src-bucket": self.src_bucket,
                                        "src-key": key,
                                        "dataset": self.hf_dataset,
                                        "ingestion-ts": ing_ts,
                                    },
                                    content_type=head.get("ContentType"),
                                )

                            moved_img += 1
                            self.logger.info(f"[IMG] {key} -> s3://{self.dest_bucket}/{dst_key}")

                        elif self.is_document_json(head, ext):
                            dst_key = self.make_target_key(
                                "document",
                                self.hf_dataset,
                                ing_ts,
                                base,
                                ext,
                                prefix=self.doc_prefix,
                            )

                            if not self.dry_run:
                                self.move_or_copy(
                                    self.src_bucket,
                                    key,
                                    self.dest_bucket,
                                    dst_key,
                                    metadata={
                                        "src-bucket": self.src_bucket,
                                        "src-key": key,
                                        "dataset": self.hf_dataset,
                                        "ingestion-ts": ing_ts,
                                    },
                                    content_type=head.get("ContentType") or "application/json",
                                )

                            moved_doc += 1
                            self.logger.info(f"[DOC] {key} -> s3://{self.dest_bucket}/{dst_key}")

                        else:
                            skipped += 1
                            self.logger.info(
                                f"[SKIP] {key} (ctype={head.get('ContentType')}, ext=.{ext})"
                            )

                result = {
                    "total_files": total,
                    "moved_images": moved_img,
                    "moved_documents": moved_doc,
                    "skipped_files": skipped,
                    "timestamp": utc_timestamp(),
                }

                self.logger.info(
                    f"[STATS] total={total} images={moved_img} documents={moved_doc} skipped={skipped}"
                )
                return result

        except Exception as e:
            self.logger.error(f"Persistent landing processing failed: {e}")
            raise


def main():
    """Main entry point for persistent landing processing."""
    config = PipelineConfig()
    processor = PersistentLandingProcessor(config)

    try:
        result = processor.process()
        print(f"✅ Persistent landing processing completed successfully")
        print(f"📊 Metrics: {result}")
    except Exception as e:
        print(f"❌ Persistent landing processing failed: {e}")
        raise


if __name__ == "__main__":
    main()
