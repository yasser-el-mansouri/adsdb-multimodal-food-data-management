#!/usr/bin/env python
# coding: utf-8

# # Formatted Zone — Images
# 
# This notebook handles the **images format processing** step for the Formatted Zone of our data pipeline.  
# Its primary goal is to:
# 
# 1. **Load images** from the Landing Zone
# 2. **Convert all images** in the same format

# ## 1. Setup and Configuration
# 

# In[1]:


import os
from dotenv import load_dotenv
import io
import re
import boto3
from datetime import datetime, timezone
from pathlib import PurePosixPath
from botocore.config import Config
from botocore.exceptions import ClientError
from PIL import Image

load_dotenv()

SRC_BUCKET      = "landing-zone"
SRC_PREFIX      = "persistent_landing/images"      
DST_BUCKET      = "formatted-zone"
DST_PREFIX      = "images"
HF_DATASET      = os.getenv("HF_DATASET")

MINIO_USER=os.getenv("MINIO_USER")
MINIO_PASSWORD=os.getenv("MINIO_PASSWORD")
MINIO_ENDPOINT=os.getenv("MINIO_ENDPOINT")

TARGET_EXT      = "jpg" 
TARGET_PIL_FMT  = "JPEG"
TARGET_CTYPE    = "image/jpeg"

OVERWRITE = False

JPEG_SAVE_KW = dict(
    quality=90,
    optimize=True,
    progressive=True,
    subsampling="4:2:0",
)

session = boto3.session.Session(
    aws_access_key_id=MINIO_USER,
    aws_secret_access_key=MINIO_PASSWORD,
    region_name="us-east-1",
)
s3 = session.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
)


# This block loads environment variables and prepares MinIO access (via **boto3**) for an image formatting step. It defines source/destination buckets and prefixes, the dataset name, and target image settings: convert to **JPEG** (`.jpg`) with a fixed content type and PIL format.
# 
# It also sets overwrite behavior and JPEG save options (quality, optimization, progressive, subsampling) to standardize output size and quality. Finally, it creates a boto3 session/client pointed at the MinIO endpoint using credentials from `.env`.

# ## 2. Helper Functions

# In[ ]:


def utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def split_name(key: str):
    name = PurePosixPath(key).name
    base = PurePosixPath(name).stem
    ext  = PurePosixPath(name).suffix.lower().lstrip(".")

    if ext in ("jpeg", "jpe"):
        ext = "jpg"
    return base or "file", ext or "bin"


def is_probably_image(content_type: str | None, ext: str) -> bool:
    if content_type and content_type.lower().startswith("image/"):
        return True
    return ext in {"jpg", "jpeg", "png", "gif", "webp", "bmp", "tiff"}


def make_target_key(filename: str, ext: str) -> str:
    return f"{DST_PREFIX}/{filename}.{ext}"


def object_exists(bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code in ("404", "NoSuchKey", "NotFound"):
            return False
        elif error_code == "400":
            print(f"[WARN] Invalid key format: {key}")
            return False
        else:
            raise


def read_image_from_s3(bucket: str, key: str) -> Image.Image:
    obj = s3.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()
    img = Image.open(io.BytesIO(data))
    img.load()
    return img


def to_jpeg_bytes(img: Image.Image) -> bytes:
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        alpha_img = img.convert("RGBA")
        bg.paste(alpha_img, mask=alpha_img.split()[-1])
        out = io.BytesIO()
        bg.save(out, format=TARGET_PIL_FMT, **JPEG_SAVE_KW)
        return out.getvalue()
    else:
        if img.mode not in ("RGB",):
            img = img.convert("RGB")
        out = io.BytesIO()
        img.save(out, format=TARGET_PIL_FMT, **JPEG_SAVE_KW)
        return out.getvalue()


def upload_bytes(bucket: str, key: str, content: bytes, content_type: str, metadata: dict | None = None):
    extra = {"ContentType": content_type}
    if metadata:
        extra["Metadata"] = metadata
    s3.put_object(Bucket=bucket, Key=key, Body=content, **extra)


# This block defines utilities for identifying, converting, and uploading images during the formatting process.
# 
# `utc_ts` generates a timestamp, while `split_name` extracts a clean base name and extension from an S3 key, normalizing formats like `.jpeg` to `.jpg`. The function `is_probably_image` checks if a file is an image based on MIME type or extension.
# 
# `make_target_key` builds standardized output names using the pattern `image$dataset$filename.extension`, ensuring traceability and organization.
# 
# The functions `object_exists`, `read_image_from_s3`, and `to_jpeg_bytes` handle object checking, reading, and image conversion respectively — converting images to JPEG with consistent quality and handling transparency safely.
# 
# Finally, `upload_bytes` uploads the processed image bytes back to MinIO with appropriate content type and metadata.

# ## 3. Convert images in the same format

# In[3]:


ts = utc_ts()
paginator = s3.get_paginator("list_objects_v2")
pages = paginator.paginate(Bucket=SRC_BUCKET, Prefix=SRC_PREFIX)

total = converted = skipped = 0

for page in pages:
    for obj in page.get("Contents", []):
        key = obj["Key"]
        if key.endswith("/"):
            continue

        total += 1

        try:
            head = s3.head_object(Bucket=SRC_BUCKET, Key=key)
        except ClientError as e:
            print(f"[WARN] head failed {key}: {e}")
            skipped += 1
            continue

        base, ext = split_name(key)
        ctype = head.get("ContentType")

        if not is_probably_image(ctype, ext):
            print(f"[SKIP] is not an image: {key} (ctype={ctype}, ext=.{ext})")
            skipped += 1
            continue

        dst_key = make_target_key(base, TARGET_EXT)

        if not OVERWRITE and object_exists(DST_BUCKET, dst_key):
            print(f"[EXISTS] {dst_key}, jump (OVERWRITE=False)")
            skipped += 1
            continue

        try:
            img = read_image_from_s3(SRC_BUCKET, key)
            jpg_bytes = to_jpeg_bytes(img)
        except Exception as e:
            print(f"[WARN] failed to convert {key}: {e}")
            skipped += 1
            continue

        metadata = {
            "src-bucket": SRC_BUCKET,
            "src-key": key,
            "dataset": HF_DATASET,
            "unify-ts": ts,
            "target-format": TARGET_EXT,
        }
        try:
            upload_bytes(DST_BUCKET, dst_key, jpg_bytes, TARGET_CTYPE, metadata)
            converted += 1
            print(f"[OK] {key} -> s3://{DST_BUCKET}/{dst_key}")
        except Exception as e:
            print(f"[WARN] upload failed {dst_key}: {e}")
            skipped += 1

print(f"\n[UNIFY STATS] total_listed={total}  converted={converted}  skipped={skipped}")


# This block scans all images in the source bucket, converts them to a unified JPEG format, and uploads the results to the destination bucket.
# 
# It iterates through all files under the image prefix, skipping directories and non-image files. Each valid image is read from MinIO, converted to JPEG using consistent settings, and uploaded under a standardized name (`image$dataset$filename.jpg`) with metadata describing its origin, dataset, and processing timestamp.
# 
# The use of **JPEG (.jpg)** is intentional — it offers the best balance between **universal compatibility, file size, and processing speed**. Most visualization tools, web services, and ML pipelines natively support JPEG, making it a reliable and efficient standard for large-scale image handling.
# 
# If a target file already exists and overwriting is disabled, it is skipped to avoid duplication. The script logs progress and counts totals for processed, converted, and skipped files, providing a clear summary of the unification process.
