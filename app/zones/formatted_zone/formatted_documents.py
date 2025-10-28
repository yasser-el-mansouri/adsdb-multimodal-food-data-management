"""
Formatted Zone - Documents Processing

This module handles the documents format processing step for the Formatted Zone.
It joins all recipes in a single file and removes irrelevant data.
"""

import io
import json
import re
from decimal import Decimal
from pathlib import PurePosixPath
from typing import Dict, List, Any, Iterable

import boto3
import ijson

# Import shared utilities
from app.utils.shared import PipelineConfig, S3Client, Logger, utc_timestamp, to_builtin, error_handler


class FormattedDocumentsProcessor:
    """Processor for formatted zone documents."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize the processor."""
        self.config = config
        self.logger = Logger("formatted_documents", config.get("monitoring.log_level", "INFO"))
        self.s3_client = S3Client(config)        
        # Configuration
        self.src_bucket = config.get("storage.buckets.landing_zone")
        self.src_prefix = config.get("storage.prefixes.persistent_landing_documents")
        self.out_bucket = config.get("storage.buckets.formatted_zone")
        self.out_key = f"{config.get('storage.prefixes.formatted_documents')}/recipes.jsonl"
        
        # Processing configuration
        self.skip_fields = set(config.get("document_processing.skip_fields", []))
        self.doc_exts = tuple(config.get("huggingface.file_extensions.document", []))
        self.always_tag = config.get("document_processing.always_tag", True)
        
        self.dry_run = config.get("pipeline.dry_run", False)
    
    def list_docs(self, bucket: str, prefix: str) -> Iterable[str]:
        """List document files in bucket with prefix."""
        paginator = self.s3_client.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                if key.lower().endswith(self.doc_exts):
                    yield key
    
    def iter_jsonl(self, bucket: str, key: str) -> Iterable[Dict[str, Any]]:
        """Iterate over JSONL file."""
        obj = self.s3_client.get_object(bucket=bucket, key=key)
        for line in obj["Body"].iter_lines():
            if not line:
                continue
            yield to_builtin(json.loads(line))
    
    def iter_json_array(self, bucket: str, key: str) -> Iterable[Dict[str, Any]]:
        """Iterate over JSON array file."""
        obj = self.s3_client.get_object(bucket=bucket, key=key)
        for item in ijson.items(obj["Body"], "item"):
            yield to_builtin(item)
    
    def detect_iter(self, bucket: str, key: str) -> Iterable[Dict[str, Any]]:
        """Detect file format and return appropriate iterator."""
        if key.lower().endswith((".jsonl", ".ndjson")):
            return self.iter_jsonl(bucket, key)
        
        obj = self.s3_client.get_object(bucket=bucket, key=key)
        head = obj["Body"].read(2048)
        obj["Body"].close()
        
        if head.lstrip()[:1] == b"[":
            return self.iter_json_array(bucket, key)
        
        obj = self.s3_client.get_object(bucket=bucket, key=key)
        body = obj["Body"].read()
        try:
            one = to_builtin(json.loads(body))
            def _once():
                if isinstance(one, dict):
                    yield one
            return _once()
        except Exception:
            return iter(())
    
    def short_tag_from_key(self, key: str) -> str:
        """Extract short tag from key."""
        stem = PurePosixPath(key).stem
        if "__" in stem:
            cand = stem.split("__")[-1]
        else:
            toks = re.split(r"[-_$]+", stem)
            cand = toks[-1] if toks else stem
        tag = re.sub(r"[^\w\-]+", "_", cand).strip("_").lower()
        return (tag[-40:] if len(tag) > 40 else tag) or "file"
    
    def merge_with_tags(self, dst: Dict[str, Any], src: Dict[str, Any], tag: str):
        """Merge source dict into destination with tag handling."""
        if not isinstance(src, dict):
            return
        
        for k, v in src.items():
            if k in self.skip_fields:
                continue
            if k == "id":
                if "id" not in dst:
                    dst["id"] = v
                continue
            
            base = f"{k}__from_{tag}" if self.always_tag else k
            new_k = base
            i = 2
            while new_k in dst:
                if not self.always_tag and new_k == k:
                    base = f"{k}__from_{tag}"
                    new_k = base
                    continue
                new_k = f"{base}_{i}"
                i += 1
            dst[new_k] = v
    
    def process(self) -> Dict[str, Any]:
        """Main processing method."""        
        try:
            with error_handler(self.logger, "formatted_documents_processing"):
                keys = list(self.list_docs(self.src_bucket, self.src_prefix))
                if not keys:
                    raise RuntimeError(f"No documents under s3://{self.src_bucket}/{self.src_prefix}")
                
                self.logger.info(f"Found {len(keys)} document file(s) under s3://{self.src_bucket}/{self.src_prefix}")
                
                joined: Dict[str, Dict[str, Any]] = {}
                total_seen = 0
                
                for idx, key in enumerate(keys, 1):
                    tag = self.short_tag_from_key(key)
                    relevant = 0
                    
                    for rec in self.detect_iter(self.src_bucket, key):
                        rid = rec.get("id")
                        if not rid:
                            continue
                        if rid not in joined:
                            joined[rid] = {}
                        self.merge_with_tags(joined[rid], rec, tag)
                        relevant += 1
                        total_seen += 1
                    
                    self.logger.info(f"[MERGE] ({idx}/{len(keys)}) {key} [tag={tag}] -> records processed: {relevant}")
                
                self.logger.info(f"[STATS] Total records processed: {total_seen}")
                self.logger.info(f"[STATS] Unique ids merged: {len(joined)}")
                
                if not self.dry_run:
                    # Write joined data as JSONL
                    buf = io.StringIO()
                    for rid in sorted(joined.keys()):
                        buf.write(json.dumps(joined[rid], ensure_ascii=False) + "\n")
                    payload = buf.getvalue().encode("utf-8")
                    
                    self.s3_client.put_object(
                        bucket=self.out_bucket,
                        key=self.out_key,
                        body=payload,
                        content_type="application/x-ndjson",
                        metadata={
                            "source-prefix": f"s3://{self.src_bucket}/{self.src_prefix}",
                            "note": "sample join by id with tagged collisions; plain ndjson"
                        }
                    )
                    self.logger.info(f"[OK] Wrote JSONL sample to s3://{self.out_bucket}/{self.out_key}")
                    
                    return {
                    "total_files": len(keys),
                    "total_records": total_seen,
                    "unique_records": len(joined),
                    "timestamp": utc_timestamp()
                }
        
        except Exception as e:
            self.logger.error(f"Formatted documents processing failed: {e}")
            raise


def main():
    """Main entry point for formatted documents processing."""
    config = PipelineConfig()
    processor = FormattedDocumentsProcessor(config)
    
    try:
        result = processor.process()
        print(f"✅ Formatted documents processing completed successfully")
        print(f"📊 Metrics: {result}")
    except Exception as e:
        print(f"❌ Formatted documents processing failed: {e}")
        raise


if __name__ == "__main__":
    main()
