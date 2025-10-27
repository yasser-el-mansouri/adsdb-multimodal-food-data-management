"""
Trusted Zone - Documents Processing

This module handles the document processing step for the Trusted Zone.
It filters documents to keep only those with images and applies quality controls.
"""

import io
import json
import math
import re
from pathlib import PurePosixPath
from typing import Dict, List, Any, Iterable, Set, Optional

import pandas as pd
from unidecode import unidecode

from app.utils import (
    PipelineConfig, S3Client, Logger, PerformanceMonitor, 
    utc_timestamp, error_handler
)


class TrustedDocumentsProcessor:
    """Processor for trusted zone documents."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize the processor."""
        self.config = config
        self.logger = Logger("trusted_documents", config.get("monitoring.log_level", "INFO"))
        self.s3_client = S3Client(config)
        self.monitor = PerformanceMonitor(config)
        
        # Configuration
        self.src_bucket = config.get("storage.buckets.formatted_zone")
        self.src_key = f"{config.get('storage.prefixes.formatted_documents')}/recipes.jsonl"
        self.out_bucket = config.get("storage.buckets.trusted_zone")
        self.out_key = f"{config.get('storage.prefixes.trusted_documents')}/recipes.jsonl"
        self.report_prefix = config.get("storage.prefixes.trusted_reports")
        
        # Input file from images processing
        self.recipe_ids_file = "recipe_ids_with_images.json"
        
        # Behavior flags
        self.dry_run = config.get("pipeline.dry_run", False)
        self.overwrite = config.get("pipeline.overwrite", True)
        
        # Document processing configuration
        doc_config = config.get("document_processing", {})
        self.skip_fields = set(doc_config.get("skip_fields", []))
        self.always_tag = doc_config.get("always_tag", True)
        
        # Nutrition configuration
        nutrition_config = doc_config.get("nutrition", {})
        self.nutrition_totals_candidates = nutrition_config.get("totals_key_candidates", [])
        self.nutrition_per_ingredient_candidates = nutrition_config.get("per_ingredient_key_candidates", [])
        self.nutrition_numeric_fields = nutrition_config.get("numeric_fields", [])
        self.drop_totals_if_per_ingredient_present = nutrition_config.get("drop_totals_if_per_ingredient_present", True)
        
        # Text cleaning configuration
        text_config = doc_config.get("text_cleaning", {})
        self.text_language = text_config.get("language", "english")
        self.remove_punctuation = text_config.get("remove_punctuation", True)
        self.remove_stopwords = text_config.get("remove_stopwords", True)
        
        # Initialize text processing
        self._init_text_processing()
        
        # Stats accumulator
        self.doc_stats = {
            "total_read": 0,
            "kept_after_id": 0,
            "dropped_missing_nutrition": 0,
            "dropped_outliers": 0,
            "written": 0,
            "nutrition_totals_dropped_due_to_duplication": 0,
            "text_tokens_avg": {
                "ingredients_raw": 0.0, "ingredients_clean": 0.0,
                "instructions_raw": 0.0, "instructions_clean": 0.0,
                "title_raw": 0.0, "title_clean": 0.0,
            }
        }
    
    def _init_text_processing(self):
        """Initialize text processing components."""
        if self.remove_stopwords:
            try:
                import nltk
                try:
                    nltk.data.find('corpora/stopwords')
                except LookupError:
                    nltk.download('stopwords', quiet=True)
                from nltk.corpus import stopwords
                self.stopwords = set(stopwords.words(self.text_language))
            except ImportError:
                self.logger.warning("NLTK not available, stopwords removal disabled")
                self.stopwords = set()
                self.remove_stopwords = False
        else:
            self.stopwords = set()
    
    def load_recipe_ids_with_images(self) -> Set[str]:
        """Load recipe IDs that have images from the images processing step."""
        try:
            with open(self.recipe_ids_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            recipe_ids = set(data.get('recipe_ids_with_images', []))
            self.logger.info(f"Loaded {len(recipe_ids)} recipe IDs with images")
            self.logger.info(f"Source: {data.get('source', 'unknown')}")
            self.logger.info(f"Generated at: {data.get('timestamp', 'unknown')}")
            
            return recipe_ids
        
        except FileNotFoundError:
            self.logger.error(f"Recipe IDs file not found: {self.recipe_ids_file}")
            self.logger.info("Make sure to run trusted_images.py first to generate the recipe IDs")
            return set()
        except Exception as e:
            self.logger.error(f"Failed to load recipe IDs: {e}")
            return set()
    
    def clean_text(self, text: str) -> str:
        """Clean text for processing."""
        if not text:
            return ""
        
        text = text.lower()
        text = unidecode(text)
        
        if self.remove_punctuation:
            text = re.sub(r"[^\w\s]", " ", text)
        
        text = re.sub(r"\s+", " ", text).strip()
        
        if self.remove_stopwords and self.stopwords:
            tokens = [t for t in text.split() if t not in self.stopwords]
            text = " ".join(tokens)
        
        return text
    
    def join_text_list(self, objs: List[Dict[str, Any]], key: str = "text") -> str:
        """Join text from list of objects."""
        if not isinstance(objs, list):
            return ""
        return " ".join([str(o.get(key, "")) for o in objs if isinstance(o, dict)])
    
    def pick_first_present(self, rec: Dict[str, Any], candidates: List[str]) -> Optional[str]:
        """Pick first existing key from candidates."""
        for k in candidates:
            if k in rec and rec[k] is not None:
                return k
        return None
    
    def iqr_bounds(self, vals: List[float], k: float = 1.5) -> Optional[Dict[str, float]]:
        """Compute IQR bounds for outlier detection."""
        if not vals:
            return None
        
        s = pd.Series(vals)
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = float(q1 - k * iqr), float(q3 + k * iqr)
        return {"q1": float(q1), "q3": float(q3), "lo": lo, "hi": hi}
    
    def token_len(self, text: str) -> int:
        """Count tokens in text."""
        return len((text or "").split())
    
    def read_jsonl_lines(self, bucket: str, key: str) -> Iterable[str]:
        """Stream JSONL lines from S3."""
        obj = self.s3_client.get_object(Bucket=bucket, Key=key)
        for raw in obj["Body"].iter_lines():
            if raw:  # skip empty
                yield raw
    
    def compute_iqr_thresholds(self, recipe_ids_with_images: Set[str]) -> Dict[str, Any]:
        """Compute IQR thresholds for nutritional fields."""
        self.logger.info("Pre-scanning documents to compute IQR thresholds for nutritional fields...")
        
        nutr_values_collections = {f: [] for f in self.nutrition_numeric_fields}
        total_prescan = 0
        
        for raw in self.read_jsonl_lines(self.src_bucket, self.src_key):
            try:
                rec = json.loads(raw)
            except Exception:
                continue
            total_prescan += 1
            
            # Only consider records we might keep later
            rid = rec.get("id")
            if not rid or rid not in recipe_ids_with_images:
                continue
            
            totals_key = self.pick_first_present(rec, self.nutrition_totals_candidates)
            if not totals_key:
                continue
            
            totals = rec.get(totals_key) or {}
            # Collect numeric fields if present and finite
            for f in self.nutrition_numeric_fields:
                v = totals.get(f)
                if isinstance(v, (int, float)) and math.isfinite(v):
                    nutr_values_collections[f].append(float(v))
        
        # Compute bounds
        iqr_thresholds = {}
        for f, vals in nutr_values_collections.items():
            iqr_thresholds[f] = self.iqr_bounds(vals) if vals else None
        
        self.logger.info(f"IQR thresholds computed: {iqr_thresholds}")
        return iqr_thresholds
    
    def process(self) -> Dict[str, Any]:
        """Main processing method."""
        self.monitor.start()
        
        try:
            with error_handler(self.logger, "trusted_documents_processing"):
                # Load recipe IDs that have images
                recipe_ids_with_images = self.load_recipe_ids_with_images()
                
                if not recipe_ids_with_images:
                    raise RuntimeError("No recipe IDs with images found. Cannot proceed with document filtering.")
                
                # Compute IQR thresholds
                iqr_thresholds = self.compute_iqr_thresholds(recipe_ids_with_images)
                
                # Process documents
                if self.dry_run:
                    return self._process_dry_run(recipe_ids_with_images, iqr_thresholds)
                else:
                    return self._process_full(recipe_ids_with_images, iqr_thresholds)
        
        except Exception as e:
            self.logger.error(f"Trusted documents processing failed: {e}")
            raise
    
    def _process_dry_run(self, recipe_ids_with_images: Set[str], iqr_thresholds: Dict[str, Any]) -> Dict[str, Any]:
        """Process documents in dry run mode."""
        self.logger.info("Counting documents that would be kept after id+nutrition checks...")
        
        total = kept_after_id = 0
        
        for raw in self.read_jsonl_lines(self.src_bucket, self.src_key):
            total += 1
            try:
                rec = json.loads(raw)
            except Exception:
                continue
            
            rid = rec.get("id")
            if not rid or rid not in recipe_ids_with_images:
                continue
            
            # Missing nutrition check
            totals_key = self.pick_first_present(rec, self.nutrition_totals_candidates)
            per_ingr_key = self.pick_first_present(rec, self.nutrition_per_ingredient_candidates)
            if not totals_key and not per_ingr_key:
                continue  # missing nutrition entirely
            
            # Outlier check on totals if present and thresholds exist
            if totals_key and iqr_thresholds and any(iqr_thresholds.values()):
                totals = rec.get(totals_key) or {}
                outlier = False
                for f in self.nutrition_numeric_fields:
                    thr = iqr_thresholds.get(f)
                    v = totals.get(f)
                    if thr and isinstance(v, (int, float)) and math.isfinite(v):
                        if not (thr["lo"] <= float(v) <= thr["hi"]):
                            outlier = True
                            break
                if outlier:
                    continue
            
            kept_after_id += 1
        
        self.logger.info(f"[DRY_RUN] total={total} kept_after_id={kept_after_id}")
        
        metrics = self.monitor.stop()
        metrics.update({
            "total_docs": total,
            "kept_docs": kept_after_id,
            "timestamp": utc_timestamp()
        })
        
        return metrics
    
    def _process_full(self, recipe_ids_with_images: Set[str], iqr_thresholds: Dict[str, Any]) -> Dict[str, Any]:
        """Process documents in full mode."""
        self.logger.info("Filtering and transforming documents to Trusted Zone...")
        
        # Create multipart writer for large files
        writer = MultipartJSONLWriter(
            self.s3_client, self.out_bucket, self.out_key,
            content_type="application/x-ndjson",
            metadata={
                "note": "ids with images + nutrition checks + text clean + de-dup nutrition",
                "ts": utc_timestamp()
            }
        )
        
        try:
            for raw in self.read_jsonl_lines(self.src_bucket, self.src_key):
                self.doc_stats["total_read"] += 1
                
                try:
                    rec = json.loads(raw)
                except Exception:
                    continue
                
                rid = rec.get("id")
                if not rid or rid not in recipe_ids_with_images:
                    continue
                
                # Check nutrition info but don't drop recipes without it
                totals_key = self.pick_first_present(rec, self.nutrition_totals_candidates)
                per_ingr_key = self.pick_first_present(rec, self.nutrition_per_ingredient_candidates)
                has_nutrition = bool(totals_key or per_ingr_key)
                rec["has_nutrition_data"] = has_nutrition
                
                # If totals exist, apply outlier filter per field
                outlier = False
                if totals_key and iqr_thresholds and any(iqr_thresholds.values()):
                    totals = rec.get(totals_key) or {}
                    for f in self.nutrition_numeric_fields:
                        thr = iqr_thresholds.get(f)
                        v = totals.get(f)
                        if thr and isinstance(v, (int, float)) and math.isfinite(v):
                            val = float(v)
                            if not (thr["lo"] <= val <= thr["hi"]):
                                outlier = True
                                break
                
                if outlier:
                    self.doc_stats["dropped_outliers"] += 1
                    continue
                
                self.doc_stats["kept_after_id"] += 1
                
                # Text cleaning for embeddings
                title_raw = (rec.get("title__from_layer1") or 
                           rec.get("title__from_recipes_with_nutritional_info") or 
                           rec.get("title") or "")
                ingr_raw = (self.join_text_list(rec.get("ingredients__from_layer1")) or 
                           self.join_text_list(rec.get("ingredients__from_recipes_with_nutritional_info")) or 
                           self.join_text_list(rec.get("ingredients")) or "")
                instr_raw = (self.join_text_list(rec.get("instructions__from_layer1")) or 
                           self.join_text_list(rec.get("instructions__from_recipes_with_nutritional_info")) or 
                           self.join_text_list(rec.get("instructions")) or "")
                
                rec["title_text_raw"] = title_raw
                rec["ingredients_text_raw"] = ingr_raw
                rec["instructions_text_raw"] = instr_raw
                
                rec["title_text_clean"] = self.clean_text(title_raw)
                rec["ingredients_text_clean"] = self.clean_text(ingr_raw)
                rec["instructions_text_clean"] = self.clean_text(instr_raw)
                
                # Update running token-length averages
                for k, v in {
                    "ingredients_raw": self.token_len(rec["ingredients_text_raw"]),
                    "ingredients_clean": self.token_len(rec["ingredients_text_clean"]),
                    "instructions_raw": self.token_len(rec["instructions_text_raw"]),
                    "instructions_clean": self.token_len(rec["instructions_text_clean"]),
                    "title_raw": self.token_len(rec["title_text_raw"]),
                    "title_clean": self.token_len(rec["title_text_clean"]),
                }.items():
                    # incremental average
                    n = max(1, self.doc_stats["kept_after_id"])
                    prev = self.doc_stats["text_tokens_avg"][k]
                    self.doc_stats["text_tokens_avg"][k] = prev + (v - prev) / n
                
                # Remove duplicated totals if per-ingredient available
                if self.drop_totals_if_per_ingredient_present and per_ingr_key and totals_key:
                    if totals_key in rec:
                        rec.pop(totals_key, None)
                        self.doc_stats["nutrition_totals_dropped_due_to_duplication"] += 1
                    rec["nutrition_normalized"] = True
                else:
                    rec["nutrition_normalized"] = bool(per_ingr_key and not totals_key)
                
                # Write transformed record
                writer.write_line(json.dumps(rec, ensure_ascii=False).encode("utf-8"))
                self.doc_stats["written"] += 1
        
        finally:
            writer.close()
        
        self.logger.info(f"[OK] Wrote filtered+transformed docs to s3://{self.out_bucket}/{self.out_key}")
        
        # Generate processing report
        report = {
            "timestamp": utc_timestamp(),
            "processing_step": "documents",
            "source_docs": f"s3://{self.src_bucket}/{self.src_key}",
            "destination_docs": f"s3://{self.out_bucket}/{self.out_key}",
            "recipe_ids_source": self.recipe_ids_file,
            "total_doc_count": self.doc_stats["total_read"],
            "kept_after_id": self.doc_stats["kept_after_id"],
            "written_doc_count": self.doc_stats["written"],
            "dropped_missing_nutrition": self.doc_stats["dropped_missing_nutrition"],
            "dropped_outliers": self.doc_stats["dropped_outliers"],
            "unique_recipe_ids_with_images": len(recipe_ids_with_images),
            "filtering_rate": f"{(self.doc_stats['written']/self.doc_stats['total_read']*100):.2f}%" if self.doc_stats['total_read'] > 0 else "0%",
            "nutrition": {
                "totals_key_candidates": self.nutrition_totals_candidates,
                "per_ingredient_key_candidates": self.nutrition_per_ingredient_candidates,
                "drop_totals_if_per_ingredient_present": self.drop_totals_if_per_ingredient_present,
                "iqr_thresholds": iqr_thresholds
            },
            "text_cleaning": {
                "lang": self.text_language,
                "stopwords_count": len(self.stopwords),
                "token_len_avg": self.doc_stats["text_tokens_avg"]
            },
            "nutrition_totals_dropped_due_to_duplication": self.doc_stats["nutrition_totals_dropped_due_to_duplication"],
            "dry_run": self.dry_run,
            "overwrite": self.overwrite
        }
        
        # Save report to S3
        self.s3_client.put_object(
            bucket=self.out_bucket,
            key=f"{self.report_prefix}/documents_processing_{utc_timestamp()}.json",
            body=json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"),
            content_type="application/json"
        )
        self.logger.info(f"[OK] Wrote report -> s3://{self.out_bucket}/{self.report_prefix}/")
        
        metrics = self.monitor.stop()
        metrics.update({
            "total_docs": self.doc_stats["total_read"],
            "kept_docs": self.doc_stats["kept_after_id"],
            "written_docs": self.doc_stats["written"],
            "dropped_outliers": self.doc_stats["dropped_outliers"],
            "timestamp": utc_timestamp()
        })
        
        return metrics


class MultipartJSONLWriter:
    """Streaming JSONL writer that uses S3 multipart uploads for large files."""
    
    def __init__(self, s3_client: S3Client, bucket: str, key: str, 
                 content_type: str = "application/x-ndjson", metadata: Optional[Dict[str, str]] = None):
        """Initialize multipart writer."""
        self.s3_client = s3_client
        self.bucket = bucket
        self.key = key
        self.buf = io.BytesIO()
        self.parts = []
        self.part_num = 1
        self.open = True
        
        min_part_size = 8 * 1024 * 1024  # 8 MB
        
        extra = {
            "Bucket": bucket,
            "Key": key,
            "ContentType": content_type,
            "Metadata": metadata or {},
        }
        resp = s3_client.client.create_multipart_upload(**extra)
        self.upload_id = resp["UploadId"]
        self.min_part_size = min_part_size
    
    def _flush_part(self):
        """Flush the current buffer as a multipart upload part."""
        self.buf.seek(0)
        body = self.buf.read()
        if not body:
            self.buf.seek(0)
            self.buf.truncate(0)
            return
        
        resp = self.s3_client.client.upload_part(
            Bucket=self.bucket, Key=self.key,
            UploadId=self.upload_id, PartNumber=self.part_num, Body=body
        )
        self.parts.append({"ETag": resp["ETag"], "PartNumber": self.part_num})
        self.part_num += 1
        self.buf.seek(0)
        self.buf.truncate(0)
    
    def write_line(self, raw_line_bytes: bytes):
        """Write a line to the buffer, flushing when buffer is full."""
        self.buf.write(raw_line_bytes)
        self.buf.write(b"\n")
        if self.buf.tell() >= self.min_part_size:
            self._flush_part()
    
    def close(self):
        """Close the writer and complete the multipart upload."""
        if not self.open:
            return
        
        try:
            # If there's leftover data, flush as a last part
            self._flush_part()
            if not self.parts:
                # No data kept: abort multipart, optionally create empty object
                self.s3_client.client.abort_multipart_upload(
                    Bucket=self.bucket, Key=self.key, UploadId=self.upload_id
                )
                # Optional: write a 0-byte file so the path exists
                self.s3_client.put_object(
                    bucket=self.bucket, key=self.key, body=b"",
                    content_type="application/x-ndjson",
                    metadata={"note": "empty after filtering", "ts": utc_timestamp()}
                )
            else:
                self.s3_client.client.complete_multipart_upload(
                    Bucket=self.bucket, Key=self.key, UploadId=self.upload_id,
                    MultipartUpload={"Parts": self.parts}
                )
        finally:
            self.open = False


def main():
    """Main entry point for trusted documents processing."""
    config = PipelineConfig()
    processor = TrustedDocumentsProcessor(config)
    
    try:
        result = processor.process()
        print(f"✅ Trusted documents processing completed successfully")
        print(f"📊 Metrics: {result}")
    except Exception as e:
        print(f"❌ Trusted documents processing failed: {e}")
        raise


if __name__ == "__main__":
    main()
