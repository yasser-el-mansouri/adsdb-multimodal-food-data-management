#!/usr/bin/env python
# coding: utf-8

# # Landing Zone — Persistent
# 
# This notebook handles the **raw data organization** step for the Landing Zone of our data pipeline.  
# Its primary goal is to:
# 
# 1. **Ingest raw data** from temporal Landing Zone
# 2. **Store the data** in MinIo by type of file, and apllying name convention
# 3. **Delete the data** from temporal Landing Zone if it's needed
# 4. **Making recipes indexes** for usage in trusted zone and for data governance

# ## 1. Setup and Configuration
# 

# In[1]:


import os
from dotenv import load_dotenv
import re
import time
import boto3
import mimetypes
import json
from datetime import datetime, timezone
from urllib.parse import unquote
from pathlib import PurePosixPath
from botocore.config import Config
from botocore.exceptions import ClientError

load_dotenv()

SRC_BUCKET      = "landing-zone"
DEST_BUCKET     = SRC_BUCKET
IMG_PREFIX      = "persistent_landing/images"
DOC_PREFIX      = "persistent_landing/documents"
HF_DATASET=os.getenv("HF_DATASET")

MINIO_USER=os.getenv("MINIO_USER")
MINIO_PASSWORD=os.getenv("MINIO_PASSWORD")
MINIO_ENDPOINT=os.getenv("MINIO_ENDPOINT")

DELETE_SOURCE_AFTER_COPY = True 

IMAGE_MIME_PREFIXES = ("image/",)
IMAGE_EXTS = {"jpg", "jpeg", "png", "gif", "webp", "bmp", "tiff"}
DOC_EXTS   = {"json", "jsonl", "ndjson"}

s3 = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_USER,
    aws_secret_access_key=MINIO_PASSWORD,
    region_name="us-east-1",
    config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
)


# This block loads environment variables and sets up a MinIO connection using **boto3**. It defines source and destination buckets, prefixes for images and documents, and basic file type filters for images and JSON files.
# 
# The script also includes a flag to optionally delete source files after copying, supporting cleanup operations. Overall, this section prepares the environment and storage connection for later steps that organize and move files within the dataset.

# ## 2. Helper Functions

# In[2]:


def utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def guess_name_and_ext(key: str, head: dict) -> tuple[str, str]:
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


def is_image(head: dict, ext: str) -> bool:
    ctype = (head.get("ContentType") or "").lower()
    return ctype.startswith(IMAGE_MIME_PREFIXES) or ext in IMAGE_EXTS


def is_document_json(head: dict, ext: str) -> bool:
    ctype = (head.get("ContentType") or "").split(";")[0].strip().lower()
    return ext in DOC_EXTS or ctype == "application/json"


def sanitize_filename(s: str) -> str:
    return re.sub(r"[^\w\-.]+", "_", s)


def make_target_key(obj_type: str, dataset: str, ts: str, filename: str, ext: str, prefix: str) -> str:
    filename = sanitize_filename(filename)
    dataset  = sanitize_filename(dataset)
    return f"{prefix}/{obj_type}${dataset}${ts}${filename}.{ext}"


def copy_object(src_bucket: str, src_key: str, dst_bucket: str, dst_key: str, metadata: dict | None = None, content_type: str | None = None):
    extra = {"MetadataDirective": "REPLACE"}
    if metadata:
        extra["Metadata"] = metadata
    if content_type:
        extra["ContentType"] = content_type

    s3.copy_object(
        CopySource={"Bucket": src_bucket, "Key": src_key},
        Bucket=dst_bucket,
        Key=dst_key,
        **extra,
    )


def move_or_copy(src_bucket: str, src_key: str, dst_bucket: str, dst_key: str, **kwargs):
    copy_object(src_bucket, src_key, dst_bucket, dst_key, **kwargs)
    if DELETE_SOURCE_AFTER_COPY:
        try:
            s3.delete_object(Bucket=src_bucket, Key=src_key)
        except ClientError as e:
            print(f"[WARN] failed to delete from origin {src_key}: {e}")


def atomic_write_json(path: str, data: dict):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def extract_recipe_ids(bucket: str, key: str) -> set[str]:
    ids: set[str] = set()
    ext = PurePosixPath(key).suffix.lower()

    try:
        if ext in (".jsonl", ".ndjson"):
            obj = s3.get_object(Bucket=bucket, Key=key)
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

        obj = s3.get_object(Bucket=bucket, Key=key)
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


# This block defines utility functions for classifying, naming, and moving files between MinIO buckets.
# 
# `utc_ts` generates a UTC timestamp used to version or label copied objects. `guess_name_and_ext` extracts or infers a file’s base name and extension from its key or content type, normalizing image formats like `.jpeg` to `.jpg`. The functions `is_image` and `is_document_json` classify files by checking MIME types and extensions.
# 
# `sanitize_filename` cleans names to avoid invalid characters, while `make_target_key` builds standardized keys using the pattern `type$dataset$timestamp$name.extension`, ensuring unique and descriptive filenames.
# 
# Finally, `copy_object` and `move_or_copy` handle the actual file transfer, replacing metadata if needed and optionally deleting the source file after copying, supporting a clean migration process.

# ## 3. Group data by type, apply name convention and store on persistent zone

# In[3]:


paginator = s3.get_paginator("list_objects_v2")
ing_ts = utc_ts()

pages = paginator.paginate(Bucket=SRC_BUCKET)
total = moved_img = moved_doc = skipped = 0

images_index = {}
IMAGE_INDEX_PATH = "image_index.json"

recipes_index: dict[str, dict] = {}
RECIPES_INDEX_PATH = "recipes_index.json"

for page in pages:
    for obj in page.get("Contents", []):
        key = obj["Key"]
        total += 1

        if key.endswith("/") or key.startswith("."):
            skipped += 1
            continue

        try:
            head = s3.head_object(Bucket=SRC_BUCKET, Key=key)
        except ClientError as e:
            print(f"[WARN] head_object failed in {key}: {e}")
            skipped += 1
            continue

        base, ext = guess_name_and_ext(key, head)

        if is_image(head, ext):
            dst_key = make_target_key("image", HF_DATASET, ing_ts, base, ext, prefix=IMG_PREFIX)
            move_or_copy(
                SRC_BUCKET, key, DEST_BUCKET, dst_key,
                metadata={
                    "src-bucket": SRC_BUCKET,
                    "src-key": key,
                    "dataset": HF_DATASET,
                    "ingestion-ts": ing_ts,
                },
                content_type=head.get("ContentType"),
            )
            moved_img += 1
            print(f"[IMG] {key} -> s3://{DEST_BUCKET}/{dst_key}")

            dst_name = PurePosixPath(dst_key).name
            id_part = re.search(r'([A-Fa-f0-9]+[0-9]+)\.[^.]+$', dst_name)
            image_id = id_part.group(1) if id_part else dst_name

            images_index[key] = {
                "id": image_id,
            }

        elif is_document_json(head, ext):
            dst_key = make_target_key("document", HF_DATASET, ing_ts, base, ext, prefix=DOC_PREFIX)
            move_or_copy(
                SRC_BUCKET, key, DEST_BUCKET, dst_key,
                metadata={
                    "src-bucket": SRC_BUCKET,
                    "src-key": key,
                    "dataset": HF_DATASET,
                    "ingestion-ts": ing_ts,
                },
                content_type=head.get("ContentType") or "application/json",
            )
            moved_doc += 1
            print(f"[DOC] {key} -> s3://{DEST_BUCKET}/{dst_key}")
            doc_ids = extract_recipe_ids(DEST_BUCKET, dst_key)

            for rid in doc_ids:
                entry = recipes_index.setdefault(rid, {"docs": []})
                if dst_key not in entry["docs"]:
                    entry["docs"].append(dst_key)

            print(f"[IDX] ids in doc: {len(doc_ids)}")

        else:
            skipped += 1
            print(f"[SKIP] {key} (ctype={head.get('ContentType')}, ext=.{ext})")

if images_index:
    try:
        atomic_write_json(IMAGE_INDEX_PATH, images_index)
        print(f"\n[INDEX] indexed images: {len(images_index)} -> {IMAGE_INDEX_PATH}")
    except Exception as e:
        print(f"[WARN] failed to write local index {IMAGE_INDEX_PATH}: {e}")

if recipes_index:
    for rid, info in recipes_index.items():
        info["docs"] = sorted(set(info.get("docs", [])))
    try:
        atomic_write_json(RECIPES_INDEX_PATH, recipes_index)
        print(f"[INDEX] indexed recipes: {len(recipes_index)} -> {RECIPES_INDEX_PATH}")
    except Exception as e:
        print(f"[WARN] failed to write {RECIPES_INDEX_PATH}: {e}")

print(f"\n[STATS] total={total}  images={moved_img}  documents={moved_doc}  skipped={skipped}")


# This block scans all objects in the source MinIO bucket and organizes them into structured destinations based on file type.
# 
# It uses a paginator to iterate through every stored object, skipping directories or hidden files. For each valid file, the script retrieves metadata, determines its type, and applies the standardized naming convention `type$dataset$timestamp$name.extension`.
# 
# Image files are moved under the image prefix, and JSON documents under the document prefix — both enriched with metadata such as source path, dataset name, and ingestion timestamp. Unsupported or unrecognized files are skipped.
# 
# Finally, summary statistics are printed, showing totals for processed, moved, and skipped files, providing a quick overview of the ingestion process.
