# adsdb-multimodal-food-data-management
This project aims to design and implement a multi-modal data management pipeline for the domain of food and cooking. Leveraging DataOps principles, the system ingests, processes, and organizes structured multimodal data (images, text, and possibly video/audio), enabling a range of downstream analytical and generative tasks.

## Quick Start (MinIO + init script)

1. **Start the container**

```bash
docker compose up -d
```

2. **Make sure the init script is executable**
   (Use **Git Bash** on Windows or any Unix shell)

```bash
chmod +x init-minio.sh
```

### Windows line-endings gotcha (CRLF/BOM)

On Windows, the script may fail due to line endings or BOM. Fix it with:

```bash
sed -i 's/\r$//' init-minio.sh
sed -i '1s/^\xEF\xBB\xBF//' init-minio.sh
chmod +x init-minio.sh
```

> After fixing the script, re-run:
>
> ```bash
> docker compose down -v
> docker compose up -d
> ```
