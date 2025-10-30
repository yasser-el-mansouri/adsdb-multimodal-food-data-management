"""
Multimodal Tasks - Task 1: Retrieval Processing

This module handles retrieval operations for both text and image queries.
It provides functionality for:
- Image-to-image similarity search
- Text-to-text similarity search
- Query operations on ChromaDB collections
"""

import io
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

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


class Task1RetrievalProcessor:
    """Processor for multimodal task 1: retrieval operations."""

    def __init__(self, config: PipelineConfig):
        """Initialize the processor."""
        self.config = config
        self.logger = Logger("task1_retrieval", config.get("monitoring.log_level", "INFO"))
        self.s3_client = S3Client(config)

        # Get multimodal tasks configuration
        ops_config = config.get("multimodal_tasks", {})
        
        # ChromaDB configuration - separate directories for text and images
        self.text_chroma_persist_dir = ops_config.get(
            "text_chroma_persist_dir", "app/zones/exploitation_zone/chroma_documents"
        )
        self.image_chroma_persist_dir = ops_config.get(
            "image_chroma_persist_dir", "app/zones/exploitation_zone/chroma_images"
        )
        
        # Collection names
        self.text_collection_name = ops_config.get("text_collection_name", "trusted_zone_documents")
        self.image_collection_name = ops_config.get("image_collection_name", "trusted_zone_images")
        
        # Embedding models
        self.text_embedding_model = ops_config.get(
            "text_embedding_model", "Qwen/Qwen3-Embedding-0.6B"
        )
        self.image_embedding_model = ops_config.get("image_embedding_model", "ViT-B-32")
        self.image_pretrained = ops_config.get("image_pretrained", "laion2b_s34b_b79k")
        
        # Query configuration
        self.default_k = ops_config.get("default_k", 5)
        
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

    def get_text_collection(self):
        """Get or create the text collection."""
        try:
            return self.text_chroma_client.get_collection(
                name=self.text_collection_name,
                embedding_function=self.ef_text,
            )
        except Exception as e:
            self.logger.error(f"Failed to get text collection '{self.text_collection_name}': {e}")
            raise

    def get_image_collection(self):
        """Get or create the image collection."""
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
        
        Args:
            pil_img: PIL Image to encode
            
        Returns:
            List of floats representing the image embedding
        """
        img_tensor = self.clip_preprocess(pil_img).unsqueeze(0).to(self.device)
        img_features = self.clip_model.encode_image(img_tensor)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        return img_features.squeeze(0).cpu().tolist()

    def retrieve_text(self, query: str, k: int = None) -> Dict[str, Any]:
        """
        Retrieve top-k similar text docs from ChromaDB for a given natural language query.
        
        Args:
            query: Natural language query string
            k: Number of results to return (default from config)
            
        Returns:
            Dict with 'query' and 'hits' keys, where hits = [{"text", "meta", "score"}, ...]
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
            self.logger.error(f"Text retrieval failed for query '{query}': {e}")
            raise

    def retrieve_images(self, image_path: str, k: int = None) -> Dict[str, Any]:
        """
        Retrieve top-k similar IMAGES from ChromaDB using image-to-image search.
        
        Args:
            image_path: Path to the query image
            k: Number of results to return (default from config)
            
        Returns:
            Dict with 'hits' key, where hits = [{"image_s3_bucket", "image_s3_key", "score"}, ...]
        """
        if k is None:
            k = self.default_k
            
        try:
            col = self.get_image_collection()
            
            # Load and encode the query image
            pil_img = Image.open(image_path).convert("RGB")
            img_vec = self.encode_image_to_vec(pil_img)
            
            result = col.query(
                query_embeddings=[img_vec],
                n_results=k,
                include=["metadatas", "distances"],
            )

            metas = result.get("metadatas", [[]])[0]
            dists = result.get("distances", [[]])[0]

            hits = []
            for meta, dist in zip(metas, dists):
                hits.append({
                    "image_s3_bucket": meta.get("bucket"),
                    "image_s3_key": meta.get("object_key"),
                    "score": float(dist),
                })

            self.logger.info(f"Image retrieval for '{image_path}' returned {len(hits)} results")
            
            return {"hits": hits}
        except Exception as e:
            self.logger.error(f"Image retrieval failed for '{image_path}': {e}")
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the ChromaDB collections."""
        stats = {}
        
        try:
            # List all collections from both ChromaDB clients
            text_collections = self.text_chroma_client.list_collections()
            image_collections = self.image_chroma_client.list_collections()
            
            all_collection_names = [col.name for col in text_collections] + [col.name for col in image_collections]
            stats["total_collections"] = len(all_collection_names)
            stats["collection_names"] = all_collection_names
            
            # Get stats for text collection
            try:
                text_col = self.get_text_collection()
                stats["text_collection"] = {
                    "name": self.text_collection_name,
                    "count": text_col.count(),
                }
            except Exception as e:
                stats["text_collection"] = {"error": str(e)}
            
            # Get stats for image collection
            try:
                image_col = self.get_image_collection()
                stats["image_collection"] = {
                    "name": self.image_collection_name,
                    "count": image_col.count(),
                }
            except Exception as e:
                stats["image_collection"] = {"error": str(e)}
                
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {e}")
            stats["error"] = str(e)
            
        return stats

    def run_demo_queries(self) -> Dict[str, Any]:
        """Run demonstration queries to verify the retrieval system."""
        demo_results = {
            "text_queries": [],
            "image_queries": [],
        }
        
        # Demo text queries
        text_queries = [
            "crunchy onion potato bake",
            "chicken noodle soup",
        ]
        
        for query in text_queries:
            try:
                result = self.retrieve_text(query, k=3)
                demo_results["text_queries"].append({
                    "query": query,
                    "num_results": len(result["hits"]),
                    "top_score": result["hits"][0]["score"] if result["hits"] else None,
                })
                self.logger.info(f"Demo text query '{query}' successful")
            except Exception as e:
                self.logger.warning(f"Demo text query '{query}' failed: {e}")
                demo_results["text_queries"].append({
                    "query": query,
                    "error": str(e),
                })
        
        return demo_results

    def process(self) -> Dict[str, Any]:
        """Main processing method."""
        try:
            with error_handler(self.logger, "task1_retrieval_processing"):
                self.logger.info("Starting task 1 retrieval processing")
                
                # Get collection statistics
                stats = self.get_collection_stats()
                self.logger.info(f"Collection stats: {stats}")
                
                # Run demo queries
                demo_results = self.run_demo_queries()
                self.logger.info(f"Demo queries completed")
                
                result = {
                    "timestamp": utc_timestamp(),
                    "collection_stats": stats,
                    "demo_results": demo_results,
                    "status": "success",
                }
                
                self.logger.info(f"Task 1 retrieval processing completed: {result}")
                return result

        except Exception as e:
            self.logger.error(f"Task 1 retrieval processing failed: {e}")
            raise


def main():
    """Main entry point for task 1 retrieval processing."""
    config = PipelineConfig()
    processor = Task1RetrievalProcessor(config)

    try:
        result = processor.process()
        print(f"✅ Task 1 retrieval processing completed successfully")
        print(f"📊 Metrics: {result}")
    except Exception as e:
        print(f"❌ Task 1 retrieval processing failed: {e}")
        raise


if __name__ == "__main__":
    main()

