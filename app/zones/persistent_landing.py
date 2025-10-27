"""
Landing Zone - Persistent Data Organization

This module handles the raw data organization step for the Landing Zone.
It organizes temporal data by type and applies naming conventions.
"""

import os
import json
import re
import mimetypes
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Dict, List, Set, Optional, Any

import boto3
from botocore.exceptions import ClientError

# Import shared utilities
from shared_utils import PipelineConfig, S3Client, Logger, PerformanceMonitor, utc_timestamp, sanitize_filename, atomic_write_json, error_handler


class PersistentLandingProcessor:
    """Processor for persistent landing zone data organization."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize the processor."""
        self.config = config
        self.logger = Logger("persistent_landing", config.get("monitoring.log_level", "INFO"))
        self.s3_client = S3Client(config)
        self.monitor = PerformanceMonitor(config)
        
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
        
        # Index files
        self.image_index_path = "image_index.json"
        self.recipes_index_path = "recipes_index.json"
    
    def utc_ts(self) -> str:
        """Generate UTC timestamp."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    
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
    
    def make_target_key(self, obj_type: str, dataset: str, ts: str, 
                       filename: str, ext: str, prefix: str) -> str:
        """Generate target key with naming convention."""
        filename = sanitize_filename(filename)
        dataset = sanitize_filename(dataset)
        return f"{prefix}/{obj_type}${dataset}${ts}${filename}.{ext}"
    
    def copy_object(self, src_bucket: str, src_key: str, dst_bucket: str, dst_key: str,
                   metadata: Optional[Dict[str, str]] = None, content_type: Optional[str] = None):
        """Copy object with metadata."""
        extra = {"MetadataDirective": "REPLACE"}
        if metadata:
            extra["Metadata"] = metadata
        if content_type:
            extra["ContentType"] = content_type
        
        self.s3_client.client.copy_object(
            CopySource={"Bucket": src_bucket, "Key": src_key},
            Bucket=dst_bucket,
            Key=dst_key,
            **extra
        )
    
    def move_or_copy(self, src_bucket: str, src_key: str, dst_bucket: str, dst_key: str, **kwargs):
        """Move or copy object, optionally deleting source."""
        self.copy_object(src_bucket, src_key, dst_bucket, dst_key, **kwargs)
        if self.delete_source_after_copy:
            try:
                self.s3_client.client.delete_object(Bucket=src_bucket, Key=src_key)
            except ClientError as e:
                self.logger.warning(f"Failed to delete source {src_key}: {e}")
    
    def extract_recipe_ids(self, bucket: str, key: str) -> Set[str]:
        """Extract recipe IDs from document file."""
        ids: Set[str] = set()
        ext = PurePosixPath(key).suffix.lower()
        
        try:
            if ext in (".jsonl", ".ndjson"):
                obj = self.s3_client.get_object(bucket=bucket, key=key)
                for line in obj["Body"].iter_lines():
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if isinstance(rec, dict) and "id" in rec:
                            ids.add(str(rec["id"]))
                    except Exception:
                        continue
                return ids
            
            obj = self.s3_client.get_object(bucket=bucket, key=key)
            body = obj["Body"].read()
            try:
                data = json.loads(body)
                if isinstance(data, list):
                    for rec in data:
                        if isinstance(rec, dict) and "id" in rec:
                            ids.add(str(rec["id"]))
                elif isinstance(data, dict):
                    if "id" in data:
                        ids.add(str(data["id"]))
                    for v in data.values():
                        if isinstance(v, list):
                            for rec in v:
                                if isinstance(rec, dict) and "id" in rec:
                                    ids.add(str(rec["id"]))
            except Exception:
                pass
        except ClientError:
            pass
        
        return ids
    
    def process(self) -> Dict[str, Any]:
        """Main processing method."""
        self.monitor.start()
        
        try:
            with error_handler(self.logger, "persistent_landing_processing"):
                paginator = self.s3_client.client.get_paginator("list_objects_v2")
                ing_ts = self.utc_ts()
                
                pages = paginator.paginate(Bucket=self.src_bucket)
                total = moved_img = moved_doc = skipped = 0
                
                images_index = {}
                recipes_index: Dict[str, Dict[str, List[str]]] = {}
                
                for page in pages:
                    for obj in page.get("Contents", []):
                        key = obj["Key"]
                        total += 1
                        
                        if key.endswith("/") or key.startswith("."):
                            skipped += 1
                            continue
                        
                        try:
                            head = self.s3_client.client.head_object(Bucket=self.src_bucket, Key=key)
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
                                    self.src_bucket, key, self.dest_bucket, dst_key,
                                    metadata={
                                        "src-bucket": self.src_bucket,
                                        "src-key": key,
                                        "dataset": self.hf_dataset,
                                        "ingestion-ts": ing_ts
                                    },
                                    content_type=head.get("ContentType")
                                )
                            
                            moved_img += 1
                            self.logger.info(f"[IMG] {key} -> s3://{self.dest_bucket}/{dst_key}")
                            
                            dst_name = PurePosixPath(dst_key).name
                            id_part = re.search(r'([A-Fa-f0-9]+[0-9]+)\.[^.]+$', dst_name)
                            image_id = id_part.group(1) if id_part else dst_name
                            
                            images_index[key] = {"id": image_id}
                        
                        elif self.is_document_json(head, ext):
                            dst_key = self.make_target_key(
                                "document", self.hf_dataset, ing_ts, base, ext, prefix=self.doc_prefix
                            )
                            
                            if not self.dry_run:
                                self.move_or_copy(
                                    self.src_bucket, key, self.dest_bucket, dst_key,
                                    metadata={
                                        "src-bucket": self.src_bucket,
                                        "src-key": key,
                                        "dataset": self.hf_dataset,
                                        "ingestion-ts": ing_ts
                                    },
                                    content_type=head.get("ContentType") or "application/json"
                                )
                            
                            moved_doc += 1
                            self.logger.info(f"[DOC] {key} -> s3://{self.dest_bucket}/{dst_key}")
                            
                            if not self.dry_run:
                                doc_ids = self.extract_recipe_ids(self.dest_bucket, dst_key)
                                
                                for rid in doc_ids:
                                    entry = recipes_index.setdefault(rid, {"docs": []})
                                    if dst_key not in entry["docs"]:
                                        entry["docs"].append(dst_key)
                                
                                self.logger.info(f"[IDX] IDs in doc: {len(doc_ids)}")
                        
                        else:
                            skipped += 1
                            self.logger.info(f"[SKIP] {key} (ctype={head.get('ContentType')}, ext=.{ext})")
                
                # Save indexes
                if images_index and not self.dry_run:
                    try:
                        atomic_write_json(self.image_index_path, images_index)
                        self.logger.info(f"[INDEX] Indexed images: {len(images_index)} -> {self.image_index_path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to write local index {self.image_index_path}: {e}")
                
                if recipes_index and not self.dry_run:
                    for rid, info in recipes_index.items():
                        info["docs"] = sorted(set(info.get("docs", [])))
                    try:
                        atomic_write_json(self.recipes_index_path, recipes_index)
                        self.logger.info(f"[INDEX] Indexed recipes: {len(recipes_index)} -> {self.recipes_index_path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to write {self.recipes_index_path}: {e}")
                
                metrics = self.monitor.stop()
                metrics.update({
                    "total_files": total,
                    "moved_images": moved_img,
                    "moved_documents": moved_doc,
                    "skipped_files": skipped,
                    "timestamp": utc_timestamp()
                })
                
                self.logger.info(f"[STATS] total={total} images={moved_img} documents={moved_doc} skipped={skipped}")
                return metrics
        
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
