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

Github repository: https://github.com/yasser-el-mansouri/adsdb-multimodal-food-data-management
HuggingFace datasets source: https://huggingface.co/datasets/ADSDB-DYS/adsdb-multimodal-food-data-management/tree/main

## 🧩 Task 3 — Multimodal Retrieval-Augmented Generation (RAG)

Task 3 is the generative part of the project.  
It uses both text and image data from the trusted zone to produce coherent, grounded answers.  
This is done through a combination of:
- **ChromaDB** (for similarity retrieval using OpenCLIP embeddings)
- **MinIO** (for image and text storage)
- **Ollama** (running a multimodal model called **LLaVA**)

The notebook to run is `task3.ipynb`.

---

### 1. Set up Ollama and the required models

1. Install Ollama from:  
   https://ollama.ai/download

2. Once installed, pull the multimodal model used in Task 3:  

```bash
ollama pull llava:7b
```


This version of LLaVA is able to process both text and images.  
You can check that it was installed correctly with:


```bash
ollama list
```

You should see something like:

```bash
llava:7b
```


3. Keep Ollama running in the background while executing the notebook.  
The code will send image and text prompts to this local model server.

this is achieved with:
```bash
ollama serve
```

---

### 2. Make sure MinIO and ChromaDB are ready

- **MinIO** should already contain the `trusted-zone` bucket populated from the previous tasks.
- **ChromaDB** should have two collections:

  - trusted_zone_text
  - trusted_zone_images

These are automatically created and use **OpenCLIP** as the embedding function for multimodal similarity search.

If any of these are missing, re-run `documents.ipynb` and `images.ipynb` first.

