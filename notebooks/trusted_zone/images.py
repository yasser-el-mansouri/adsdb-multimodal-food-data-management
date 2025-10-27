#!/usr/bin/env python
# coding: utf-8

# # Trusted Zone — Images Processing
# 
# This notebook handles the **image processing** step for the Trusted Zone of our data pipeline.  
# Its primary goal is to:
# 
# 1. **Extract recipe IDs** from image filenames in the Formatted Zone
# 2. **Identify which recipes have images** and build a mapping
# 3. **Copy filtered images** to the Trusted Zone (only images with valid recipes)
# 4. **Generate a recipe IDs file** for the documents processing step
# 
# This notebook works in conjunction with `documents.ipynb` to ensure data integrity in the Trusted Zone.
# 

# ## 1. Setup and Configuration
# 

# In[1]:


import os, io, json, re
from pathlib import PurePosixPath
from datetime import datetime, timezone
from typing import Dict, List, Set, Iterable

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

# S3 / MinIO Configuration
MINIO_USER     = os.getenv("MINIO_USER")
MINIO_PASSWORD = os.getenv("MINIO_PASSWORD")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")

session = boto3.session.Session(
    aws_access_key_id=MINIO_USER,
    aws_secret_access_key=MINIO_PASSWORD,
    region_name="us-east-1"
)
s3 = session.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    config=Config(signature_version="s3v4", s3={"addressing_style": "path"})
)

# Paths and Buckets
FORM_BUCKET         = "formatted-zone"
FORM_IMAGES_PREFIX  = "images"

TRUST_BUCKET        = "trusted-zone"
TRUST_IMAGES_PREFIX = "images"
TRUST_REPORT_PREFIX = "reports"

# Output file for documents processing
RECIPE_IDS_FILE = "recipe_ids_with_images.json"

# Behavior flags
DRY_RUN   = False
OVERWRITE = True

def utc_ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


# In[2]:


# --- Quality & normalization config ---
from PIL import Image, ImageOps
import numpy as np

# Target canonical spec for Trusted Zone images
TARGET_SIZE = (512, 512)         # WxH, letterboxed
TARGET_MODE = "RGB"              # normalize mode
TARGET_FMT  = "JPEG"             # write as .jpg
TARGET_QUALITY = 90

# Basic quality thresholds
MIN_W, MIN_H = 128, 128          # remove tiny images
MIN_ASPECT, MAX_ASPECT = 0.5, 3.0  # w/h range guard

# blur screening using OpenCV (auto-off if not installed)
try:
    import cv2
    CV2_AVAILABLE = True
    BLUR_VARLAP_MIN = 50.0       # tune if needed
except Exception:
    CV2_AVAILABLE = False
    BLUR_VARLAP_MIN = None

# Near-duplicate removal (per recipe) via perceptual hash
try:
    import imagehash
    DEDUPE_PER_RECIPE = True
except Exception:
    imagehash = None
    DEDUPE_PER_RECIPE = False

def compute_metrics(img: Image.Image) -> dict:
    """Return dict with width/height, aspect, (optional) blur metric."""
    w, h = img.size
    aspect = (w / h) if h else 0
    metrics = {"w": w, "h": h, "aspect": float(aspect)}
    if CV2_AVAILABLE:
        gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
        metrics["blur_varlap"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return metrics

def normalize_image(img: Image.Image) -> bytes:
    """Convert to TARGET_MODE, letterbox to TARGET_SIZE, write JPEG bytes."""
    img_rgb = img.convert(TARGET_MODE)
    # Letterbox to keep aspect ratio
    img_fit = ImageOps.pad(img_rgb, TARGET_SIZE, method=Image.BICUBIC, color=None, centering=(0.5, 0.5))
    buf = io.BytesIO()
    img_fit.save(buf, format=TARGET_FMT, quality=TARGET_QUALITY, optimize=True)
    return buf.getvalue()


# ## 2. S3 Helper Functions
# 
# These utility functions provide a clean interface for S3 operations, handling common patterns like listing objects, checking existence, and copying files between buckets.
# 

# In[3]:


def s3_list_keys(bucket: str, prefix: str) -> Iterable[str]:
    """List all object keys in a bucket with the given prefix."""
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key = obj["Key"]
            if not key.endswith("/"):
                yield key

def s3_head(bucket: str, key: str):
    """Get object metadata, return None if not found."""
    try:
        return s3.head_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
            return None
        raise

def s3_copy_object(src_bucket: str, src_key: str, dst_bucket: str, dst_key: str, overwrite: bool = True):
    """Copy an object between buckets with optional overwrite control."""
    if not overwrite and s3_head(dst_bucket, dst_key) is not None:
        return "skip-exists"
    return s3.copy_object(
        Bucket=dst_bucket,
        Key=dst_key,
        CopySource={"Bucket": src_bucket, "Key": src_key},
        MetadataDirective="COPY"
    )


# ## 3. Extract Recipe IDs from Image Filenames
# 
# Each image stored in the Formatted Zone follows a structured naming convention that encodes metadata, including the recipe identifier.  
# The general pattern is:
# 
# **fileType$dataSource$ingestionTimestamp$hash__recipeId_positionOnImagesUrlArrayFromLayer2.extension**
# 
# From these filenames, we extract the `recipeId` using a regular expression.  
# This allows us to associate every image with its corresponding recipe entry, even when multiple images exist for the same recipe.  
# The result of this step is two structures:
# 
# - `img_ids`: a set of **unique recipe IDs** that have at least one image.
# - `id_to_imgkeys`: a dictionary mapping each `recipeId` to **all its image keys** (to preserve one-to-many relationships).
# 

# In[4]:


# Regular expression to extract recipe ID from image filenames
# Recognizes names like:
#   images/type$src$ts$hash__000018c8a5_0.jpg
#   images/type$src$ts$hash__abcd_ef-12_3.JPEG
# ID part: letters, digits, underscore, dash
ID_REGEX = re.compile(
    r"__([A-Za-z0-9_\-]+)_(\d+)\.(?:jpe?g|png|webp|gif|bmp|tiff)$",
    re.IGNORECASE
)

def recipe_id_from_image_key(key: str) -> str | None:
    """Extract recipe ID from an image S3 key."""
    name = PurePosixPath(key).name
    m = ID_REGEX.search(name)
    return m.group(1) if m else None

print("Extracting recipe IDs from image filenames...")

img_ids: Set[str] = set()                    # unique IDs (for filtering)
id_to_imgkeys: Dict[str, List[str]] = {}     # ALL images per ID

count_keys = 0
for key in s3_list_keys(FORM_BUCKET, FORM_IMAGES_PREFIX + "/"):
    count_keys += 1
    rid = recipe_id_from_image_key(key)
    if not rid:
        continue
    img_ids.add(rid)
    id_to_imgkeys.setdefault(rid, []).append(key)

# Make copies deterministic (optional)
for rid in id_to_imgkeys:
    id_to_imgkeys[rid].sort()

total_images = sum(len(v) for v in id_to_imgkeys.values())
print(f"[INFO] scanned image keys: {count_keys}")
print(f"[INFO] unique recipeIds with images: {len(img_ids)}")
print(f"[INFO] total image files matched to recipeIds: {total_images}")


# ## 4. Save Recipe IDs for Documents Processing
# 
# We save the extracted recipe IDs to a JSON file that will be used by the `documents.ipynb` notebook to filter the recipe documents. This creates a clean separation between image and document processing while maintaining the necessary coupling.
# 

# In[5]:


# Prepare data for documents processing
recipe_ids_data = {
    "timestamp": utc_ts(),
    "source": f"s3://{FORM_BUCKET}/{FORM_IMAGES_PREFIX}/",
    "total_images_scanned": count_keys,
    "unique_recipe_ids": len(img_ids),
    "total_images_matched": total_images,
    "recipe_ids_with_images": sorted(list(img_ids)),
    "recipe_to_images": {rid: keys for rid, keys in id_to_imgkeys.items()}
}

# Save to local file
with open(RECIPE_IDS_FILE, 'w', encoding='utf-8') as f:
    json.dump(recipe_ids_data, f, ensure_ascii=False, indent=2)

print(f"[OK] saved recipe IDs to {RECIPE_IDS_FILE}")
print(f"[INFO] {len(img_ids)} recipe IDs will be used for document filtering")


# ## 5. Quality screening and per-recipe deduplication
# 

# **This step adds:**
# 
# - integrity check (corruption)
# - min size and aspect-range checks
# - optional blur screen (auto-disabled if OpenCV not installed)
# - per-recipe near-duplicate removal via perceptual hash
# - structured skips_log for the report

# In[6]:


print("Screening images for quality and duplicates...")

quality_stats = {
    "evaluated": 0,
    "kept": 0,
    "corrupted": 0,
    "too_small": 0,
    "bad_aspect": 0,
    "too_blurry": 0,
    "dupes_removed": 0,
}

# Keep decisions per recipe id
kept_per_rid: Dict[str, List[str]] = {}
skips_log: List[dict] = []

# For dedupe: track perceptual hashes per recipe
seen_phashes: Dict[str, set] = {}

for rid, keys in id_to_imgkeys.items():
    kept_per_rid[rid] = []
    if DEDUPE_PER_RECIPE and imagehash is not None:
        seen_phashes[rid] = set()

    for src_key in keys:
        quality_stats["evaluated"] += 1

        # Load original image bytes
        try:
            obj = s3.get_object(Bucket=FORM_BUCKET, Key=src_key)
            raw = obj["Body"].read()
            img = Image.open(io.BytesIO(raw))
            img.load()
        except Exception as e:
            quality_stats["corrupted"] += 1
            skips_log.append({"key": src_key, "reason": f"corrupted:{type(e).__name__}"})
            continue

        # Basic metrics
        m = compute_metrics(img)
        w, h, aspect = m["w"], m["h"], m["aspect"]

        if w < MIN_W or h < MIN_H:
            quality_stats["too_small"] += 1
            skips_log.append({"key": src_key, "reason": f"too_small:{w}x{h}"})
            continue

        if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
            quality_stats["bad_aspect"] += 1
            skips_log.append({"key": src_key, "reason": f"bad_aspect:{aspect:.2f}"})
            continue

        if CV2_AVAILABLE and BLUR_VARLAP_MIN is not None:
            fm = m.get("blur_varlap", 0.0)
            if fm < BLUR_VARLAP_MIN:
                quality_stats["too_blurry"] += 1
                skips_log.append({"key": src_key, "reason": f"blurry:varLap={fm:.2f} < {BLUR_VARLAP_MIN}"})
                continue

        # Per-recipe near-duplicate check via phash
        if DEDUPE_PER_RECIPE and imagehash is not None:
            ph = str(imagehash.phash(img))
            if ph in seen_phashes[rid]:
                quality_stats["dupes_removed"] += 1
                skips_log.append({"key": src_key, "reason": f"duplicate_phash:{ph}"})
                continue
            seen_phashes[rid].add(ph)

        kept_per_rid[rid].append(src_key)
        quality_stats["kept"] += 1

print("[STATS] quality screening:", quality_stats)


# ## 5. Copy Images to Trusted Zone
# 
# This step takes the **subset of images that passed the quality screen** and writes them into the Trusted Zone in a **canonical format**. We do not perform a raw copy. We normalize each image so that downstream components see consistent inputs.
# 
# **What happens here**
# - Read each kept image from the Formatted Zone using its original key.
# - Convert the image to `RGB`.
# - Resize with **letterboxing** to a fixed canvas so aspect ratio is preserved.
# - Encode to `JPEG` with a fixed quality setting.
# - Write the normalized file to `trusted-zone/images/` using the **same basename** but with a `.jpg` extension.
# 
# **Why we normalize**
# - A single mode and size simplifies model-agnostic processing and caching.
# - Letterboxing avoids deformation and keeps visual content intact.
# - JPEG reduces storage while keeping visual quality stable.
# 
# **Canonical spec**
# - Target size: `512 × 512` pixels
# - Color mode: `RGB`
# - Format: `JPEG`
# - Quality: `90`
# 
# **Input set**
# - Only images present in `kept_per_rid` are processed. This list is produced by the previous step after integrity, size, aspect, blur, and near-duplicate checks.
# 
# **Naming and traceability**
# - Output key: `trusted-zone/images/<original_basename_without_ext>.jpg`
# - The original basename is preserved so the mapping to the source remains evident.
# 
# **Idempotency and flags**
# - `OVERWRITE=False` makes the step skip files that already exist in the Trusted Zone.
# - `DRY_RUN=True` prints the planned operations without writing any object.
# 
# **Outputs**
# - Normalized images under `trusted-zone/images/`.
# - Per-file write counters are included in the final report as `images_copied_normalized` and `images_skipped`.
# 
# **Notes**
# - If OpenCV is installed the blur screen is active. If not, the notebook still runs and simply omits that filter.
# - Per-recipe near-duplicate removal uses perceptual hashing when the `imagehash` package is available.

# ### Why we use image hashing
# 
# We use image hashing to find and remove duplicate or nearly identical images. Recipes could include repeated pictures with different filenames, sizes, or slight edits. A perceptual hash gives each image a small numerical fingerprint based on its visual content, not its raw bytes. If two images look the same, their hashes will also be the same. This lets us keep only one clean copy per recipe and avoid storing redundant or repeated images in the Trusted Zone.
# 

# In[7]:


print("Copying images to Trusted Zone...")

copied = skipped = 0

if DRY_RUN:
    print("[DRY_RUN] Would copy the following images:")
    for rid, keys in kept_per_rid.items():
        for src_key in keys:
            dst_key = f"{TRUST_IMAGES_PREFIX}/{PurePosixPath(src_key).name}"
            print(f"  {src_key} -> {dst_key}")
    copied = total_images
else:
    # Normalize and write only the KEPT images from the quality screen
    for rid, keys in kept_per_rid.items():
        for src_key in keys:
            # Destination uses same basename but normalized to .jpg
            base = PurePosixPath(src_key).name
            base_noext = base.rsplit(".", 1)[0]
            dst_key = f"{TRUST_IMAGES_PREFIX}/{base_noext}.jpg"

            if not OVERWRITE and s3_head(TRUST_BUCKET, dst_key) is not None:
                print(f"[SKIP] {dst_key} already exists")
                skipped += 1
                continue

            try:
                # Load original
                obj = s3.get_object(Bucket=FORM_BUCKET, Key=src_key)
                raw = obj["Body"].read()
                img = Image.open(io.BytesIO(raw))
                img.load()

                # Normalize
                out_bytes = normalize_image(img)

                # Write normalized JPG
                s3.put_object(
                    Bucket=TRUST_BUCKET,
                    Key=dst_key,
                    Body=out_bytes,
                    ContentType="image/jpeg"
                )
                copied += 1
            except ClientError as e:
                print(f"[WARN] write failed {src_key} -> {dst_key}: {e}")
                skipped += 1


print(f"[STATS] images copied={copied} skipped={skipped}")


# ## 6. Generate Processing Report
# 
# Finally, we generate a comprehensive report of the image processing step and save it to the Trusted Zone for audit and monitoring purposes.
# 

# In[ ]:


report = {
    "timestamp": utc_ts(),
    "processing_step": "images",
    "source_images_prefix": f"s3://{FORM_BUCKET}/{FORM_IMAGES_PREFIX}/",
    "destination_images_prefix": f"s3://{TRUST_BUCKET}/{TRUST_IMAGES_PREFIX}/",

    # From earlier steps
    "total_images_scanned": count_keys,
    "unique_recipe_ids_with_images": len(img_ids),
    "total_images_matched": total_images,

    # New: quality screen stats
    "quality_screen": {
        "evaluated": quality_stats["evaluated"],
        "kept": quality_stats["kept"],
        "corrupted": quality_stats["corrupted"],
        "too_small": quality_stats["too_small"],
        "bad_aspect": quality_stats["bad_aspect"],
        "too_blurry": quality_stats["too_blurry"],
        "dupes_removed": quality_stats["dupes_removed"],
        "blur_threshold": BLUR_VARLAP_MIN,
        "min_wh": [MIN_W, MIN_H],
        "aspect_range": [MIN_ASPECT, MAX_ASPECT],
        "dedupe_enabled": DEDUPE_PER_RECIPE,
    },

    # Output write stats
    "images_copied_normalized": copied,
    "images_skipped": skipped,

    # Notebook config snapshot for traceability
    "normalization": {
        "target_size": TARGET_SIZE,
        "target_mode": TARGET_MODE,
        "target_format": TARGET_FMT,
        "target_quality": TARGET_QUALITY
    },

    "recipe_ids_file": RECIPE_IDS_FILE,
    "dry_run": DRY_RUN,
    "overwrite": OVERWRITE
}

if not DRY_RUN and skips_log:
    csv_buf = io.StringIO()
    csv_buf.write("key,reason\n")
    for row in skips_log:
        key = row["key"].replace(",", " ")
        reason = row["reason"].replace(",", " ")
        csv_buf.write(f"{key},{reason}\n")
    s3.put_object(
        Bucket=TRUST_BUCKET,
        Key=f"{TRUST_REPORT_PREFIX}/images_skips_{utc_ts()}.csv",
        Body=csv_buf.getvalue().encode("utf-8"),
        ContentType="text/csv"
    )

if DRY_RUN:
    print("[DRY_RUN] Would save the following report:")
    print(json.dumps(report, indent=2))


