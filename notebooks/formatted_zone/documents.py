#!/usr/bin/env python
# coding: utf-8

# # Formatted Zone — Documents
# 
# This notebook handles the **documents format processing** step for the Formatted Zone of our data pipeline.  
# Its primary goal is to:
# 
# 1. **Load documents** from the Landing Zone
# 2. **Join all recipes** in a single file
# 3. **Skip data** that is no relevant
# 4. **Upload data** in a efficent way to work with

# ## 1. Setup and Configuration
# 

# In[1]:


import os, io, re, json
from decimal import Decimal
from pathlib import PurePosixPath
import boto3, ijson
from dotenv import load_dotenv
from botocore.config import Config

load_dotenv()

DOC_SRC_BUCKET =  "landing-zone"
DOC_SRC_PREFIX = "persistent_landing/documents"
OUT_S3_BUCKET  = "formatted-zone"
OUT_S3_KEY     = "documents/recipes.jsonl"

SKIP_FIELDS = {"url", "partition"}

DOC_EXTS = (".json", ".jsonl", ".ndjson")
ALWAYS_TAG = True
MINIO_USER = os.getenv("MINIO_USER")
MINIO_PASSWORD = os.getenv("MINIO_PASSWORD")
MINIO_ENDPOINT =os.getenv("MINIO_ENDPOINT")

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


# This block loads environment variables and sets up MinIO (via **boto3**) to read raw documents and write a formatted output. It defines source and destination S3 paths, basic filters (file extensions to read and fields to skip), and a flag for tagging. Then it creates a boto3 session/client pointed at the MinIO endpoint using credentials from `.env`, so later steps can stream JSON/JSONL inputs from `landing-zone` and write the consolidated result to `formatted-zone/recipes.jsonl`.

# ## 2. Helper Functions

# In[2]:


def to_builtin(x):
    if isinstance(x, Decimal):
        return int(x) if x == x.to_integral_value() else float(x)
    if isinstance(x, dict):  return {k: to_builtin(v) for k, v in x.items()}
    if isinstance(x, list):  return [to_builtin(v) for v in x]
    return x


def list_docs(bucket: str, prefix: str):
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            if key.lower().endswith(DOC_EXTS):
                yield key


def iter_jsonl(bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    for line in obj["Body"].iter_lines():
        if not line:
            continue
        yield to_builtin(json.loads(line))


def iter_json_array(bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    for item in ijson.items(obj["Body"], "item"):
        yield to_builtin(item)


def detect_iter(bucket, key):
    if key.lower().endswith((".jsonl", ".ndjson")):
        return iter_jsonl(bucket, key)
    obj = s3.get_object(Bucket=bucket, Key=key)
    head = obj["Body"].read(2048)
    obj["Body"].close()
    if head.lstrip()[:1] == b"[":
        return iter_json_array(bucket, key)
    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()
    try:
        one = to_builtin(json.loads(body))
        def _once():
            if isinstance(one, dict):
                yield one
        return _once()
    except Exception:
        return iter(())


def short_tag_from_key(key: str) -> str:
    stem = PurePosixPath(key).stem
    if "__" in stem:
        cand = stem.split("__")[-1]
    else:
        toks = re.split(r"[-_$]+", stem)
        cand = toks[-1] if toks else stem
    tag = re.sub(r"[^\w\-]+", "_", cand).strip("_").lower()
    return (tag[-40:] if len(tag) > 40 else tag) or "file"


def merge_with_tags(dst: dict, src: dict, tag: str):
    if not isinstance(src, dict):
        return
    for k, v in src.items():
        if k in SKIP_FIELDS:
            continue
        if k == "id":
            if "id" not in dst:
                dst["id"] = v
            continue
        base = f"{k}__from_{tag}" if ALWAYS_TAG else k
        new_k = base
        i = 2
        while new_k in dst:
            if not ALWAYS_TAG and new_k == k:
                base = f"{k}__from_{tag}"
                new_k = base
                continue
            new_k = f"{base}_{i}"
            i += 1
        dst[new_k] = v


# This block provides tools for reading and merging JSON documents stored in MinIO.
# 
# The conversion function `to_builtin` standardizes data types, turning objects like `Decimal` into basic Python numbers. The iterators (`list_docs`, `iter_jsonl`, `iter_json_array`, and `detect_iter`) automatically detect the file format—JSON, JSONL, or NDJSON—and stream records efficiently, allowing the script to handle large datasets without loading everything into memory.
# 
# `short_tag_from_key` extracts a short, clean tag from each file name, used to identify the source of data when merging. Finally, `merge_with_tags` combines multiple JSON records into one, renaming fields with a tag suffix to avoid collisions and preserve the origin of each field. This ensures consistent, traceable merging across datasets.

# ## 3. Join recipes removing irrelevant data in a efficent strucutre

# In[3]:


keys = list(list_docs(DOC_SRC_BUCKET, DOC_SRC_PREFIX))
if not keys:
    raise RuntimeError(f"No documents under s3://{DOC_SRC_BUCKET}/{DOC_SRC_PREFIX}")
print(f"[INFO] Found {len(keys)} document file(s) under s3://{DOC_SRC_BUCKET}/{DOC_SRC_PREFIX}")

joined: dict[str, dict] = {}
total_seen = 0

for idx, key in enumerate(keys, 1):
    tag = short_tag_from_key(key)
    relevant = 0
    for rec in detect_iter(DOC_SRC_BUCKET, key):
        rid = rec.get("id")
        if not rid:
            continue
        if rid not in joined:
            joined[rid] = {}
        merge_with_tags(joined[rid], rec, tag)
        relevant += 1
        total_seen += 1
    print(f"[MERGE] ({idx}/{len(keys)}) {key} [tag={tag}] -> records processed: {relevant}")

print(f"[STATS] Total records processed: {total_seen}")
print(f"[STATS] Unique ids merged:       {len(joined)}")

buf = io.StringIO()
for rid in sorted(joined.keys()):
    buf.write(json.dumps(joined[rid], ensure_ascii=False) + "\n")
payload = buf.getvalue().encode("utf-8")

s3.put_object(
    Bucket=OUT_S3_BUCKET,
    Key=OUT_S3_KEY,
    Body=payload,
    ContentType="application/x-ndjson",
    Metadata={"source-prefix": f"s3://{DOC_SRC_BUCKET}/{DOC_SRC_PREFIX}",
                "note": "sample join by id with tagged collisions; plain ndjson"},
)
print(f"[OK] Wrote JSONL sample to s3://{OUT_S3_BUCKET}/{OUT_S3_KEY}")


# This block gathers all document keys under the source prefix, fails fast if none are found, and then streams each file to merge records by `id`. For every file, it derives a short source tag and uses it to rename colliding fields, preserving provenance. The result is a dictionary keyed by `id` with tagged fields from all inputs.
# 
# Finally, it serializes the merged data as **NDJSON** and writes it to `formatted-zone/documents/recipes.jsonl` with metadata about the source prefix. The NDJSON format is used because it is more compact than regular JSON, reducing storage size and improving processing efficiency when handling large datasets.
