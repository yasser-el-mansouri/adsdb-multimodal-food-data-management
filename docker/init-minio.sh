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

echo ">> Create bucket landing-zone (idempotent)"
mc mb -p "${ALIAS}/landing-zone" || true

echo ">> Enable versioning"
mc version enable "${ALIAS}/landing-zone" || true

echo ">> Create prefixes"
printf "" | mc pipe "${ALIAS}/landing-zone/temporal-landing/.keep" || true
printf "" | mc pipe "${ALIAS}/landing-zone/persistent-landing/.keep" || true

mc anonymous set private "${ALIAS}/landing-zone" || true
mc ls --recursive "${ALIAS}/landing-zone" || true
echo ">> Init OK"
