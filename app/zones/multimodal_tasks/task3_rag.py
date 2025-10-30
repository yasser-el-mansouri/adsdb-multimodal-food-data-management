"""
Multimodal Tasks - Task 3: RAG with LLaVA

This module handles multimodal RAG (Retrieval-Augmented Generation) operations.
It provides functionality for:
- Text-to-text retrieval from ChromaDB
- Image-to-image and text-to-image retrieval
- Prompt building for LLaVA vision model
- Generation using Ollama's LLaVA model
- Tracing and logging results to MinIO
"""

import base64
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import torch
import open_clip
from PIL import Image
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import (
    OpenCLIPEmbeddingFunction,
    SentenceTransformerEmbeddingFunction,
)

# Import shared utilities
from app.utils.shared import Logger, PipelineConfig, S3Client, error_handler, utc_timestamp


class Task3RAGProcessor:
    """Processor for multimodal task 3: RAG with LLaVA."""

    def __init__(self, config: PipelineConfig):
        """Initialize the processor."""
        self.config = config
        self.logger = Logger("task3_rag", config.get("monitoring.log_level", "INFO"))
        self.s3_client = S3Client(config)

        # Get multimodal tasks configuration
        task3_config = config.get("multimodal_tasks", {})
        
        # ChromaDB configuration - separate directories for text and images
        self.text_chroma_persist_dir = task3_config.get(
            "text_chroma_persist_dir", "app/zones/exploitation_zone/chroma_documents"
        )
        self.image_chroma_persist_dir = task3_config.get(
            "image_chroma_persist_dir", "app/zones/exploitation_zone/chroma_images"
        )
        
        # Collection names
        self.text_collection_name = task3_config.get("text_collection_name", "exploitation_documents")
        self.image_collection_name = task3_config.get("image_collection_name", "exploitation_images")
        
        # Embedding models
        self.text_embedding_model = task3_config.get(
            "text_embedding_model", "Qwen/Qwen3-Embedding-0.6B"
        )
        self.image_embedding_model = task3_config.get("image_embedding_model", "ViT-B-32")
        self.image_pretrained = task3_config.get("image_pretrained", "laion2b_s34b_b79k")
        
        # Ollama configuration
        self.ollama_host = task3_config.get("ollama_host", "http://localhost:11434")
        self.ollama_model = task3_config.get("ollama_model", "llava")
        
        # Query configuration
        self.default_k = task3_config.get("default_k", 5)
        self.max_retrieved_images = task3_config.get("max_retrieved_images", 3)
        
        # Storage configuration
        storage_config = config.get("storage", {})
        self.results_bucket = storage_config.get("buckets", {}).get("trusted_zone", "trusted-zone")
        
        # Device configuration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {self.device}")

        # Initialize separate ChromaDB clients for text and images
        self.text_chroma_client = PersistentClient(path=self.text_chroma_persist_dir)
        self.image_chroma_client = PersistentClient(path=self.image_chroma_persist_dir)
        
        # Initialize embedding functions
        self.ef_text = SentenceTransformerEmbeddingFunction(model_name=self.text_embedding_model)
        self.ef_img = OpenCLIPEmbeddingFunction(model_name=self.image_embedding_model)
        
        # Initialize OpenCLIP model for custom image encoding
        self.clip_model, self.clip_preprocess, _ = open_clip.create_model_and_transforms(
            self.image_embedding_model,
            pretrained=self.image_pretrained,
            device=self.device,
        )
        
        self.logger.info(f"Text ChromaDB directory: {self.text_chroma_persist_dir}")
        self.logger.info(f"Image ChromaDB directory: {self.image_chroma_persist_dir}")
        self.logger.info(f"Text embedding model: {self.text_embedding_model}")
        self.logger.info(f"Image embedding model: {self.image_embedding_model}")
        self.logger.info(f"Ollama host: {self.ollama_host}")
        self.logger.info(f"Ollama model: {self.ollama_model}")
        
        # Check Ollama connectivity at initialization
        self._check_ollama_connection()

    def _check_ollama_connection(self):
        """
        Check if Ollama is running and accessible.
        Provides helpful error messages if not.
        """
        try:
            self.logger.info("Checking Ollama server connectivity...")
            resp = httpx.get(f"{self.ollama_host}/api/tags", timeout=5.0)
            resp.raise_for_status()
            self.logger.info("✓ Ollama server is running and accessible")
        except httpx.ConnectError as e:
            error_msg = (
                f"\n{'='*70}\n"
                f"ERROR: Cannot connect to Ollama server at {self.ollama_host}\n"
                f"{'='*70}\n"
                f"Ollama must be running before executing task3_rag.\n\n"
                f"To start Ollama:\n"
                f"  1. Open a new terminal\n"
                f"  2. Run: ollama serve\n\n"
                f"To install the LLaVA model (if not already installed):\n"
                f"  ollama pull {self.ollama_model}\n\n"
                f"Once Ollama is running, try again.\n"
                f"{'='*70}\n"
            )
            self.logger.error(error_msg)
            raise ConnectionError(error_msg) from e
        except Exception as e:
            self.logger.warning(f"Could not verify Ollama connection: {e}")
            self.logger.warning(
                f"If task3_rag fails, ensure Ollama is running:\n"
                f"  Terminal 1: ollama serve\n"
                f"  Terminal 2: ollama pull {self.ollama_model}"
            )

    def get_text_collection(self):
        """Get the text collection."""
        try:
            return self.text_chroma_client.get_collection(
                name=self.text_collection_name,
                embedding_function=self.ef_text,
            )
        except Exception as e:
            self.logger.error(f"Failed to get text collection '{self.text_collection_name}': {e}")
            raise

    def get_image_collection(self):
        """Get the image collection."""
        try:
            return self.image_chroma_client.get_collection(
                name=self.image_collection_name,
                embedding_function=self.ef_img,
            )
        except Exception as e:
            self.logger.error(f"Failed to get image collection '{self.image_collection_name}': {e}")
            raise

    @torch.no_grad()
    def encode_image_to_vec(self, pil_img: Image.Image) -> List[float]:
        """
        Produce an OpenCLIP image embedding compatible with the vectors stored in ChromaDB.
        """
        img_tensor = self.clip_preprocess(pil_img).unsqueeze(0).to(self.device)
        img_features = self.clip_model.encode_image(img_tensor)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        return img_features.squeeze(0).cpu().tolist()

    def retrieve_text(self, query: str, k: int = None) -> Dict[str, Any]:
        """
        Retrieve top-k similar text documents from ChromaDB.
        """
        if k is None:
            k = self.default_k
            
        try:
            col = self.get_text_collection()
            
            result = col.query(
                query_texts=[query],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )

            hits = []
            docs = result.get("documents", [[]])[0]
            metas = result.get("metadatas", [[]])[0]
            dists = result.get("distances", [[]])[0]

            for text_doc, meta, dist in zip(docs, metas, dists):
                hits.append({
                    "text": text_doc,
                    "meta": meta,
                    "score": float(dist),
                })

            self.logger.info(f"Text retrieval for query '{query}' returned {len(hits)} results")
            
            return {
                "query": query,
                "hits": hits,
            }
        except Exception as e:
            self.logger.error(f"Text retrieval failed for '{query}': {e}")
            raise

    def retrieve_images(
        self,
        query: Optional[str] = None,
        image_path: Optional[str] = None,
        k: int = None,
    ) -> Dict[str, Any]:
        """
        Retrieve top-k similar images from ChromaDB.
        
        Modes:
          1. text→image search: query is provided
          2. image→image search: image_path is provided
        """
        if k is None:
            k = self.default_k
            
        try:
            col = self.get_image_collection()

            if image_path is not None:
                # IMAGE → IMAGE mode
                pil_img = Image.open(image_path).convert("RGB")
                img_vec = self.encode_image_to_vec(pil_img)

                result = col.query(
                    query_embeddings=[img_vec],
                    n_results=k,
                    include=["metadatas", "distances"],
                )
                self.logger.info(f"Image retrieval for '{image_path}' returned results")
            else:
                # TEXT → IMAGE mode
                result = col.query(
                    query_texts=[query],
                    n_results=k,
                    include=["metadatas", "distances"],
                )
                self.logger.info(f"Text-to-image retrieval for '{query}' returned results")

            metas = result.get("metadatas", [[]])[0]
            dists = result.get("distances", [[]])[0]

            hits = []
            for meta, dist in zip(metas, dists):
                hits.append({
                    "image_s3_bucket": meta.get("bucket"),
                    "image_s3_key": meta.get("object_key"),
                    "score": float(dist),
                })

            return {"hits": hits}
        except Exception as e:
            self.logger.error(f"Image retrieval failed: {e}")
            raise

    def load_top_images_as_base64(self, image_hits: List[Dict], max_images: int = None) -> List[Dict]:
        """
        Downloads images from MinIO and converts them to base64 for LLaVA.
        """
        if max_images is None:
            max_images = self.max_retrieved_images
            
        b64_list = []

        for i, hit in enumerate(image_hits):
            if i >= max_images:
                break

            bucket = hit["image_s3_bucket"]
            key = hit["image_s3_key"]

            try:
                # Download from MinIO
                obj = self.s3_client.client.get_object(Bucket=bucket, Key=key)
                img_bytes = obj["Body"].read()

                # Normalize to JPEG bytes in memory
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG", quality=90)
                jpeg_bytes = buf.getvalue()

                # Base64 encode for Ollama
                b64_img = base64.b64encode(jpeg_bytes).decode("utf-8")
                b64_list.append({
                    "bucket": bucket,
                    "key": key,
                    "b64": b64_img,
                })
                
                self.logger.debug(f"Loaded image {i+1}/{max_images}: {key}")
            except Exception as e:
                self.logger.warning(f"Failed to load image {key}: {e}")
                continue

        return b64_list

    def build_llava_prompt(
        self,
        user_query: str,
        text_hits: List[Dict],
        image_hits_meta: List[Dict],
        user_image_provided: bool = False,
    ) -> str:
        """
        Build the instruction prompt for LLaVA.
        """
        # Determine context role based on whether user provided an image
        if user_image_provided:
            context_role = (
                "You are a culinary vision assistant. The user gave you an image of a dish. "
                "You MUST start by understanding that uploaded image. "
                "Then, if helpful, you can use the retrieved references as supporting context."
            )
        else:
            context_role = (
                "You are a culinary assistant. The user asked a question about food. "
                "Use the retrieved references below to answer. "
                "The user HAS NOT provided an image, the images you see are not the user's image."
            )

        # Summarize text candidates
        text_context = ""
        if text_hits:
            text_parts = []
            for i, h in enumerate(text_hits[:3], 1):
                snippet = h["text"]
                if len(snippet) > 200:
                    snippet = snippet[:200] + " ..."
                text_parts.append(f"- TEXT {i} (score {h['score']:.4f}): {snippet}")
            text_context = "\n".join(text_parts)
        else:
            text_context = "(no text candidates found)"

        # Summarize image candidates
        image_context = ""
        if image_hits_meta:
            image_parts = []
            for i, meta in enumerate(image_hits_meta, 1):
                image_parts.append(f"- IMAGE {i}: bucket={meta['bucket']}, key={meta['key']}")
            image_context = "\n".join(image_parts)
        else:
            image_context = "(no image candidates found)"

        prompt = f"""
{context_role}

User request:
\"\"\"{user_query}\"\"\"

===== CONTEXT YOU CAN USE =====
You have access to:
1. TEXT_CANDIDATES — recipes, titles, or ingredient lists retrieved from our database that are semantically similar to the user's request.
2. IMAGE_CANDIDATES — images visually related to the user's request.
3. {"Optionally, a USER-UPLOADED IMAGE, which should be treated as the target dish when present." if user_image_provided else ""}

TEXT_CANDIDATES:
{text_context}

IMAGE_CANDIDATES (visual references from the database):
{image_context}

{"If the user has uploaded an image, the FIRST image you receive corresponds to that dish." if user_image_provided else ""}

===== INSTRUCTIONS =====
Your goal is to **intelligently assist the user with any food- or recipe-related query.**

1. **Dish Identification (Image-Based Queries)**
   - If the user uploaded an image, describe what dish or ingredients it likely shows.
   - Use both visual cues and TEXT_CANDIDATES to infer the dish category.

2. **Recipe Generation (User Asks for a Recipe)**
   - Provide: Dish name, Ingredient list with quantities, Ordered preparation steps
   - Keep recipes consistent: ingredients in steps must appear in the ingredient list.

3. **Food or Cooking Knowledge Questions**
   - Answer clearly using your knowledge, optionally referencing TEXT_CANDIDATES.

===== RULES =====
- Always remain truthful to the data
- NEVER fabricate unrelated dishes or ingredients
- Keep quantities sensible (grams, ml, cups, etc.)
- End with a short **Inspiration** section listing which TEXT_CANDIDATES or IMAGE_CANDIDATES you used

Example structure:
Dish Name (if applicable)
Ingredients
Steps
Explanation (if conceptual)
Inspiration
""".strip()

        return prompt

    def generate_with_llava(self, prompt: str, b64_images: List[str], timeout: float = 120.0) -> str:
        """
        Calls Ollama's LLaVA model with a prompt and images.
        """
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "images": b64_images,
            "stream": False,
        }
        
        self.logger.info(f"Sending request to LLaVA with {len(b64_images)} images")

        try:
            resp = httpx.post(
                f"{self.ollama_host}/api/generate",
                json=payload,
                timeout=timeout
            )
            resp.raise_for_status()
            data = resp.json()
            answer = data.get("response", "").strip()
            
            self.logger.info(f"Received response from LLaVA ({len(answer)} chars)")
            return answer
        except httpx.ConnectError as e:
            error_msg = (
                f"\n{'='*70}\n"
                f"ERROR: Cannot connect to Ollama server at {self.ollama_host}\n"
                f"{'='*70}\n"
                f"Ollama server is not responding. Please ensure:\n\n"
                f"1. Ollama is running:\n"
                f"   Open a terminal and run: ollama serve\n\n"
                f"2. The LLaVA model is installed:\n"
                f"   ollama pull {self.ollama_model}\n\n"
                f"3. The Ollama host is correct: {self.ollama_host}\n"
                f"   (You can change this in pipeline.yaml: multimodal_tasks.ollama_host)\n"
                f"{'='*70}\n"
            )
            self.logger.error(error_msg)
            raise ConnectionError(error_msg) from e
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                error_msg = (
                    f"\n{'='*70}\n"
                    f"ERROR: LLaVA model not found (404)\n"
                    f"{'='*70}\n"
                    f"The Ollama server is running, but the model '{self.ollama_model}' is not available.\n\n"
                    f"To install the model:\n"
                    f"  ollama pull {self.ollama_model}\n\n"
                    f"To list installed models:\n"
                    f"  ollama list\n"
                    f"{'='*70}\n"
                )
                self.logger.error(error_msg)
                raise RuntimeError(error_msg) from e
            else:
                self.logger.error(f"LLaVA request failed with status {e.response.status_code}: {e}")
                raise
        except httpx.TimeoutException as e:
            self.logger.error(
                f"LLaVA request timed out after {timeout}s. "
                f"The model may still be loading or the query is complex."
            )
            raise
        except Exception as e:
            self.logger.error(f"LLaVA generation failed: {e}")
            raise

    def save_trace_to_minio(self, record: Dict[str, Any]) -> str:
        """
        Save the RAG trace to MinIO for provenance.
        """
        query_hash = hashlib.sha1(record['query'].encode()).hexdigest()[:8]
        key = f"results/generative/rag_multimodal/{utc_timestamp()}_{query_hash}.json"
        
        try:
            self.s3_client.client.put_object(
                Bucket=self.results_bucket,
                Key=key,
                Body=json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8"),
                ContentType="application/json",
            )
            self.logger.info(f"Saved trace to MinIO: {key}")
            return key
        except Exception as e:
            self.logger.error(f"Failed to save trace to MinIO: {e}")
            raise

    def run_multimodal_rag(
        self,
        user_query: str,
        k: int = None,
        user_image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full multimodal RAG pipeline.
        
        Args:
            user_query: The user's text question/request
            k: Number of items to retrieve per modality
            user_image_path: Optional path to user's input image
            
        Returns:
            Dict with query, answer, retrieved context, and metadata
        """
        if k is None:
            k = self.default_k
            
        self.logger.info(f"Starting multimodal RAG for query: '{user_query}'")
        
        # 1. Retrieve relevant documents and images
        text_res = self.retrieve_text(user_query, k=k)
        text_hits = text_res["hits"]

        if user_image_path:
            self.logger.info(f"Using user-provided image: {user_image_path}")
            img_res = self.retrieve_images(image_path=user_image_path, k=k)
        else:
            img_res = self.retrieve_images(query=user_query, k=k)

        image_hits = img_res["hits"]

        # 2. Prepare images for LLaVA
        top_images_b64 = self.load_top_images_as_base64(image_hits, max_images=self.max_retrieved_images)
        b64_list_for_llava = [img["b64"] for img in top_images_b64]

        # If user uploaded an image, include it first
        if user_image_path:
            pil_img = Image.open(user_image_path).convert("RGB")
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=90)
            user_img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            b64_list_for_llava.insert(0, user_img_b64)
            self.logger.info("Prepended user image to LLaVA input")

        # 3. Build prompt
        prompt = self.build_llava_prompt(
            user_query,
            text_hits,
            top_images_b64,
            user_image_provided=(user_image_path is not None)
        )

        # 4. Generate answer using LLaVA
        llava_answer = self.generate_with_llava(prompt, b64_list_for_llava)

        # 5. Create record
        record = {
            "query": user_query,
            "user_image_path": user_image_path,
            "prompt": prompt,
            "answer": llava_answer,
            "model_info": self.ollama_model,
            "retrieved_text": text_hits[:3],
            "retrieved_images": top_images_b64,
            "timestamp_utc": utc_timestamp(),
        }

        # 6. Save trace to MinIO
        s3_key = self.save_trace_to_minio(record)
        record["s3_key"] = s3_key

        self.logger.info(f"Multimodal RAG completed successfully")
        return record

    def process(self) -> Dict[str, Any]:
        """
        Main processing method - runs demo queries.
        """
        try:
            with error_handler(self.logger, "task3_rag_processing"):
                self.logger.info("Starting task 3 RAG processing")
                
                # Check collections
                text_col = self.get_text_collection()
                image_col = self.get_image_collection()
                
                text_count = text_col.count()
                image_count = image_col.count()
                
                self.logger.info(f"Text collection has {text_count} documents")
                self.logger.info(f"Image collection has {image_count} images")
                
                # Run a demo query
                demo_query = "Give me a pasta recipe with tomato and basil"
                self.logger.info(f"Running demo query: '{demo_query}'")
                
                result = self.run_multimodal_rag(demo_query, k=5)
                
                processing_result = {
                    "timestamp": utc_timestamp(),
                    "collection_stats": {
                        "text_count": text_count,
                        "image_count": image_count,
                    },
                    "demo_query": demo_query,
                    "demo_result_s3_key": result.get("s3_key"),
                    "answer_length": len(result.get("answer", "")),
                    "status": "success",
                }
                
                self.logger.info(f"Task 3 RAG processing completed: {processing_result}")
                return processing_result

        except Exception as e:
            self.logger.error(f"Task 3 RAG processing failed: {e}")
            raise


def main():
    """Main entry point for task 3 RAG processing."""
    print("\n" + "="*70)
    print("Task 3: Multimodal RAG with LLaVA")
    print("="*70)
    print("\n⚠️  IMPORTANT: This task requires Ollama to be running!")
    print("\nIf not already running, start Ollama in another terminal:")
    print("  1. ollama serve")
    print("  2. ollama pull llava")
    print("\nProceeding with task initialization...\n")
    
    config = PipelineConfig()
    
    try:
        processor = Task3RAGProcessor(config)
        result = processor.process()
        print(f"\n✅ Task 3 RAG processing completed successfully")
        print(f"📊 Metrics: {result}")
    except ConnectionError as e:
        print(f"\n❌ Task 3 RAG processing failed: Ollama connection error")
        print(f"\n{e}")
        raise
    except Exception as e:
        print(f"\n❌ Task 3 RAG processing failed: {e}")
        raise


if __name__ == "__main__":
    main()

