#!/usr/bin/env python
# coding: utf-8

# # Landing Zone — Temporal
# 
# This notebook handles the **raw data ingestion** step for the Landing Zone of our data pipeline.  
# Its primary goal is to:
# 
# 1. **Ingest raw data** from external sources
# 2. **Store the data** in MinIo being images and json documents

# ## 1. Setup and Configuration
# 

# In[4]:


import os
from os.path import basename
import json
import requests
from dotenv import load_dotenv
from pathlib import Path, PurePosixPath
import hashlib
import mimetypes
import boto3
from botocore.config import Config
from boto3.s3.transfer import TransferConfig
from urllib.parse import urlparse
import ijson
from huggingface_hub import HfApi, hf_hub_url
from datasets import load_dataset


load_dotenv()

HF_TOKEN=os.getenv("HF_TOKEN")
HF_ORGA=os.getenv("HF_ORGA")
HF_DATASET=os.getenv("HF_DATASET")
HF_REV=os.getenv("HF_REV")
REPO_ID=f"{HF_ORGA}/{HF_DATASET}"

MINIO_USER=os.getenv("MINIO_USER")
MINIO_PASSWORD=os.getenv("MINIO_PASSWORD")
MINIO_ENDPOINT=os.getenv("MINIO_ENDPOINT")
MINIO_BUCKET = "landing-zone"
PREFIX = "temporal_landing"
TIMEOUT = 60

SKIP = {".gitattributes", ".gitignore", ".gitkeep"}


s3 = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_USER,
    aws_secret_access_key=MINIO_PASSWORD,
    region_name="us-east-1",
    config=Config(
        signature_version="s3v4",
        s3={"addressing_style": "path"}
    ),
)

transfer_cfg = TransferConfig(
    multipart_threshold=8 * 1024 * 1024,
    multipart_chunksize=8 * 1024 * 1024,
    max_concurrency=4,
    use_threads=True,
)


# This block loads environment variables and sets up the connection to MinIO using **boto3**. It also defines basic constants like bucket name, prefixes, and timeouts used later in the process.
# 

# ## 2. Helper Functions

# In[5]:


def make_minio_key(path: str) -> str:
    p = PurePosixPath(path)
    h = hashlib.sha256(str(p).encode("utf-8")).hexdigest()[:16]
    return f"{h}__{p.name or 'file'}"


def get_original_filename_from_key(key: str) -> str:
    name_part = Path(key).name
    if "__" in name_part:
        return name_part.split("__", 1)[1]
    return name_part


def upload_from_hf_url(hf_url: str, hf_token: str, original_path: str, *, revision=None, repo_id=None):
    headers = {"authorization": f"Bearer {hf_token}"} if hf_token else {}
    with requests.get(hf_url, stream=True, headers=headers, timeout=60) as r:
        r.raise_for_status()
        r.raw.decode_content = True

        key = f"temporal_landing/{make_minio_key(original_path)}"

        ctype, _ = mimetypes.guess_type(original_path)
        extra = {}
        if ctype:
            extra["ContentType"] = ctype

        meta = {
            "hf-original-path": original_path,
        }
        if revision:
            meta["hf-revision"] = str(revision)
        if repo_id:
            meta["hf-repo-id"] = str(repo_id)
        if meta:
            extra["Metadata"] = meta

        s3.upload_fileobj(
            Fileobj=r.raw,
            Bucket=MINIO_BUCKET,
            Key=key,
            Config=transfer_cfg,
            ExtraArgs=extra or None,
        )

    return f"s3://{MINIO_BUCKET}/{key}"


def _safe_ext_from_url(url: str) -> str:
    path = urlparse(url).path
    ext = PurePosixPath(path).suffix.lower().lstrip(".")
    if ext in {"jpg", "jpeg", "png", "gif", "webp", "bmp"}:
        return "jpg" if ext == "jpeg" else ext
    return ""


def _safe_ext_from_ctype(ctype: str | None) -> str:
    if not ctype:
        return ""
    ctype = ctype.split(";")[0].strip().lower()
    mapping = {
        "image/jpeg": "jpg",
        "image/png": "png",
        "image/gif": "gif",
        "image/webp": "webp",
        "image/bmp": "bmp",
    }
    return mapping.get(ctype, "")


def make_minio_key_image(recipe_id: str, index: int, ext: str, prefix: str = PREFIX) -> str:
    base_name = f"{recipe_id}_{index}.{ext or 'bin'}"
    h = hashlib.sha256(f"{recipe_id}:{index}".encode("utf-8")).hexdigest()[:32]
    return f"{prefix}/{h}__{base_name}"


def upload_image_from_url(url: str, recipe_id: str, index: int) -> str | None:
    headers = {}
    try:
        with requests.get(url, stream=True, timeout=TIMEOUT, headers=headers) as r:
            r.raise_for_status()
            r.raw.decode_content = True

            ext = _safe_ext_from_url(url)
            if not ext:
                ext = _safe_ext_from_ctype(r.headers.get("Content-Type")) or "jpg"
            ctype = r.headers.get("Content-Type") or mimetypes.guess_type(f"f.{ext}")[0]

            key = make_minio_key_image(recipe_id, index, ext)

            extra = {}
            if ctype:
                extra["ContentType"] = ctype
            extra["Metadata"] = {
                "source-url": url,
                "recipe-id": str(recipe_id),
                "image-index": str(index),
            }

            s3.upload_fileobj(
                Fileobj=r.raw,
                Bucket=MINIO_BUCKET,
                Key=key,
                Config=transfer_cfg,
                ExtraArgs=extra,
            )
            return f"s3://{MINIO_BUCKET}/{key}"

    except Exception as e:
        print(f"[WARN] Failure when uploading {url} (id={recipe_id}, idx={index}): {e}")
        return None


def process_layer2_images(repo_id: str, revision: str, hf_token: str | None, layer2_path: str):
    layer2_url = hf_hub_url(repo_id=repo_id, filename=layer2_path, repo_type="dataset", revision=revision)
    headers = {"authorization": f"Bearer {hf_token}"} if hf_token else {}

    print(f"[layer2] streaming from: {layer2_url}")
    with requests.get(layer2_url, stream=True, headers=headers, timeout=TIMEOUT) as r:
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
                if upload_image_from_url(url, rid, idx):
                    ok += 1

            # TODO: this is a hack to stop the process after a certain number of recipes just for testing
            if ok > 50 or total_recipes > 1000:
                break

        print(f"[layer2] recipes={total_recipes}, imgs={total_imgs}, uploaded_ok={ok}")


# This block defines helper functions to generate file keys, manage uploads, and process image data. The naming convention for stored files follows a structured pattern like `type$dataset$timestamp$name.extension`. This model ensures that every object in MinIO is uniquely identifiable, traceable to its source, and easy to classify by type or dataset.
# 
# The use of hashing and separators (`__` or `$`) keeps keys consistent and avoids collisions, while still preserving human-readable elements such as the original name or dataset reference. This approach provides both organization and scalability when handling large numbers of files.

# ## 3. Ingest raw data from external source

# In[6]:


api = HfApi()
files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset", revision=HF_REV)

for path in files:
    fname = basename(path)
    if fname.startswith(".") or fname.endswith(".tar") or fname in SKIP:
        continue

    print(fname)

    if fname == "layer2.json":
        process_layer2_images(REPO_ID, HF_REV, HF_TOKEN, path)  
    else:
        url=hf_hub_url(repo_id=REPO_ID, filename=path, repo_type="dataset", revision=HF_REV)
        dest_uri = upload_from_hf_url(
            hf_url=url,
            hf_token=HF_TOKEN,
            original_path=path,
            revision=HF_REV,
            repo_id=REPO_ID,
        )
        print("Uploaded at", dest_uri)


# This block lists all files from the Hugging Face dataset and uploads them to MinIO, skipping temporary or irrelevant files. For regular files, it generates the Hugging Face URL and uploads the content using the helper functions defined earlier.
# 
# The file `layer2.json` is treated differently — it isn’t uploaded because it only contains metadata about images that are already extracted and stored separately. Instead, its content is processed directly to upload the referenced images, avoiding redundant storage.
