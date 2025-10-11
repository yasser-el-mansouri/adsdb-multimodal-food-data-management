#!/bin/sh
set -eu

ALIAS="myminio"
ENDPOINT="http://minio:9000"
ACCESS_KEY="${MINIO_ROOT_USER}"
SECRET_KEY="${MINIO_ROOT_PASSWORD}"

echo ">> Waiting for MinIO to be ready..."
for i in $(seq 1 30); do
  if mc alias set "${ALIAS}" "${ENDPOINT}" "${ACCESS_KEY}" "${SECRET_KEY}" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

create_if_missing() {
  BUCKET="$1"
  echo ">> Ensure bucket ${BUCKET}"
  if ! mc ls "${ALIAS}/${BUCKET}" >/dev/null 2>&1; then
    mc mb -p "${ALIAS}/${BUCKET}"
  else
    echo "   - ${BUCKET} already exists"
  fi
}

create_if_missing "landing-zone"
create_if_missing "formatted-zone"
create_if_missing "trusted-zone"
create_if_missing "exploitation-zone"

echo ">> Enable versioning (idempotent)"
mc version enable "${ALIAS}/landing-zone" || true
mc version enable "${ALIAS}/formatted-zone" || true
mc version enable "${ALIAS}/trusted-zone" || true
mc version enable "${ALIAS}/exploitation-zone" || true

echo ">> Set anonymous access to private (idempotent)"
mc anonymous set private "${ALIAS}/landing-zone" || true
mc anonymous set private "${ALIAS}/formatted-zone" || true
mc anonymous set private "${ALIAS}/trusted-zone" || true
mc anonymous set private "${ALIAS}/exploitation-zone" || true

echo ">> List buckets (recursive) — won't fail the script if empty"
mc ls --recursive "${ALIAS}/landing-zone" || true
mc ls --recursive "${ALIAS}/formatted-zone" || true
mc ls --recursive "${ALIAS}/trusted-zone" || true
mc ls --recursive "${ALIAS}/exploitation-zone" || true

echo ">> Init OK"
