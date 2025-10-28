"""
Landing Zone - Temporal Data Ingestion

This module handles the raw data ingestion step for the Landing Zone.
It ingests data from external sources (Hugging Face) and stores it in MinIO.
"""

import os
import json
import mimetypes
import requests
from pathlib import PurePosixPath
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from boto3.s3.transfer import TransferConfig
from huggingface_hub import HfApi, hf_hub_url
import ijson

# Import shared utilities
from app.utils.shared import PipelineConfig, S3Client, Logger, utc_timestamp, error_handler, KeyUtils


class TemporalLandingProcessor:
    """Processor for temporal landing zone data ingestion."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize the processor."""
        self.config = config
        self.logger = Logger("temporal_landing", config.get("monitoring.log_level", "INFO"))
        self.s3_client = S3Client(config)
        
        # Hugging Face configuration
        self.hf_token = config.get_env("HF_TOKEN")
        self.hf_orga = config.get_env("HF_ORGA")
        self.hf_dataset = config.get_env("HF_DATASET")
        self.hf_rev = config.get_env("HF_REV", "main")
        self.repo_id = f"{self.hf_orga}/{self.hf_dataset}"
        
        # Storage configuration
        self.bucket = config.get("storage.buckets.landing_zone")
        self.prefix = config.get("storage.prefixes.temporal_landing")
        self.timeout = config.get("pipeline.timeout", 60)
        
        # File filtering
        self.skip_files = set(config.get("huggingface.skip_files", []))
        
        # Transfer configuration
        transfer_config = config.get("transfer", {})
        self.transfer_config = TransferConfig(
            multipart_threshold=transfer_config.get("multipart_threshold", 8 * 1024 * 1024),
            multipart_chunksize=transfer_config.get("multipart_chunksize", 8 * 1024 * 1024),
            max_concurrency=transfer_config.get("max_concurrency", 4),
            use_threads=transfer_config.get("use_threads", True)
        )
        
        self.hf_api = HfApi()
    
    
    
    def upload_from_hf_url(self, hf_url: str, original_path: str, 
                          revision: Optional[str] = None, repo_id: Optional[str] = None) -> str:
        """Upload file from Hugging Face URL to MinIO."""
        headers = {"authorization": f"Bearer {self.hf_token}"} if self.hf_token else {}
        
        with requests.get(hf_url, stream=True, headers=headers, timeout=self.timeout) as r:
            r.raise_for_status()
            r.raw.decode_content = True
            
            key = f"{self.prefix}/{KeyUtils.make_minio_key(original_path)}"
            
            ctype, _ = mimetypes.guess_type(original_path)
            extra = {}
            if ctype:
                extra["ContentType"] = ctype
            
            meta = {"hf-original-path": original_path}
            if revision:
                meta["hf-revision"] = str(revision)
            if repo_id:
                meta["hf-repo-id"] = str(repo_id)
            if meta:
                extra["Metadata"] = meta
            
            self.s3_client.client.upload_fileobj(
                Fileobj=r.raw,
                Bucket=self.bucket,
                Key=key,
                Config=self.transfer_config,
                ExtraArgs=extra or None
            )
        
        return f"s3://{self.bucket}/{key}"
    
    def _safe_ext_from_url(self, url: str) -> str:
        """Extract safe extension from URL."""
        path = urlparse(url).path
        ext = PurePosixPath(path).suffix.lower().lstrip(".")
        if ext in {"jpg", "jpeg", "png", "gif", "webp", "bmp"}:
            return "jpg" if ext == "jpeg" else ext
        return ""
    
    def _safe_ext_from_ctype(self, ctype: Optional[str]) -> str:
        """Extract safe extension from content type."""
        if not ctype:
            return ""
        ctype = ctype.split(";")[0].strip().lower()
        mapping = {
            "image/jpeg": "jpg",
            "image/png": "png",
            "image/gif": "gif",
            "image/webp": "webp",
            "image/bmp": "bmp"
        }
        return mapping.get(ctype, "")
    
    def upload_image_from_url(self, url: str, recipe_id: str, index: int) -> Optional[str]:
        """Upload image from URL to MinIO."""
        headers = {}
        try:
            with requests.get(url, stream=True, timeout=self.timeout, headers=headers) as r:
                r.raise_for_status()
                r.raw.decode_content = True
                
                ext = self._safe_ext_from_url(url)
                if not ext:
                    ext = self._safe_ext_from_ctype(r.headers.get("Content-Type")) or "jpg"
                ctype = r.headers.get("Content-Type") or mimetypes.guess_type(f"f.{ext}")[0]
                
                key = KeyUtils.make_minio_key_image(self.prefix, recipe_id, index, ext)
                
                extra = {}
                if ctype:
                    extra["ContentType"] = ctype
                extra["Metadata"] = {
                    "source-url": url,
                    "recipe-id": str(recipe_id),
                    "image-index": str(index)
                }
                
                self.s3_client.client.upload_fileobj(
                    Fileobj=r.raw,
                    Bucket=self.bucket,
                    Key=key,
                    Config=self.transfer_config,
                    ExtraArgs=extra
                )
                return f"s3://{self.bucket}/{key}"
        
        except Exception as e:
            self.logger.warning(f"Failed to upload {url} (id={recipe_id}, idx={index}): {e}")
            return None
    
    def process_layer2_images(self, layer2_path: str) -> tuple[int, int]:
        """Process layer2.json file to upload images. Returns (processed_recipes, processed_images)."""
        layer2_url = hf_hub_url(
            repo_id=self.repo_id, 
            filename=layer2_path, 
            repo_type="dataset", 
            revision=self.hf_rev
        )
        headers = {"authorization": f"Bearer {self.hf_token}"} if self.hf_token else {}
        
        self.logger.info(f"Streaming from: {layer2_url}")
        with requests.get(layer2_url, stream=True, headers=headers, timeout=self.timeout) as r:
            r.raise_for_status()
            recipes = ijson.items(r.raw, "item")
            
            total_recipes = 0
            total_imgs = 0
            ok = 0
            
            for recipe in recipes:
                total_recipes += 1
                rid = str(recipe.get("id", "")).strip()
                images = recipe.get("images") or []
                
                for idx, img in enumerate(images):
                    url = img.get("url")
                    if not url:
                        continue
                    total_imgs += 1
                    if self.upload_image_from_url(url, rid, idx):
                        ok += 1
                
                # Limit for testing (configurable)
                max_processed = 1 #self.config.get("testing.max_processed", 50)
                max_recipes = 2 # self.config.get("testing.max_recipes", 1000)
                if ok > max_processed or total_recipes > max_recipes:
                    break
            
            self.logger.info(f"Layer2 processing: recipes={total_recipes}, imgs={total_imgs}, uploaded_ok={ok}")
            
            return total_recipes, ok
    
    def process(self) -> Dict[str, Any]:
        """Main processing method."""
        
        try:
            with error_handler(self.logger, "temporal_landing_processing"):
                files = self.hf_api.list_repo_files(
                    repo_id=self.repo_id, 
                    repo_type="dataset", 
                    revision=self.hf_rev
                )
                
                processed_files = 0
                skipped_files = 0
                processed_images = 0
                processed_recipes = 0
                
                for path in files:
                    fname = PurePosixPath(path).name
                    
                    if fname.startswith(".") or fname.endswith(".tar") or fname in self.skip_files:
                        skipped_files += 1
                        continue
                    
                    self.logger.info(f"Processing: {fname}")
                    
                    if fname == "layer2.json":
                        recipes_count, images_count = self.process_layer2_images(path)
                        processed_recipes += recipes_count
                        processed_images += images_count
                    else:
                        url = hf_hub_url(
                            repo_id=self.repo_id, 
                            filename=path, 
                            repo_type="dataset", 
                            revision=self.hf_rev
                        )
                        dest_uri = self.upload_from_hf_url(
                            hf_url=url,
                            original_path=path,
                            revision=self.hf_rev,
                            repo_id=self.repo_id
                        )
                        self.logger.info(f"Uploaded to: {dest_uri}")
                        processed_recipes += 1  # Count non-image files as recipes
                    
                    processed_files += 1
                
                result = {
                    "processed_files": processed_files,
                    "skipped_files": skipped_files,
                    "processed_images": processed_images,
                    "processed_recipes": processed_recipes,
                    "timestamp": utc_timestamp()
                }
                
                self.logger.info(f"Temporal landing processing completed: {result}")
                return result
        
        except Exception as e:
            self.logger.error(f"Temporal landing processing failed: {e}")
            raise


def main():
    """Main entry point for temporal landing processing."""
    config = PipelineConfig()
    processor = TemporalLandingProcessor(config)
    
    try:
        result = processor.process()
        print(f"✅ Temporal landing processing completed successfully")
        print(f"📊 Metrics: {result}")
    except Exception as e:
        print(f"❌ Temporal landing processing failed: {e}")
        raise


if __name__ == "__main__":
    main()
