"""
Exploitation Zone - Task 2 - Multi modal search

This module shows an example of multi modal search in the Exploitation Zone.
It does separate text and image searches that return both text and image results.
"""

import io
import os
from typing import Any, Dict, Iterable, List, Optional

import chromadb
import numpy as np
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from PIL import Image

# Import shared utilities
from app.utils.shared import Logger, PipelineConfig, S3Client, utc_timestamp




class ExploitationMultiModalSearcher:
    """Processor for multi modal search"""

    def __init__(self, config: PipelineConfig):
        """Initialize the processor."""
        self.config = config
        self.logger = Logger("task_2_multi_modal_search", config.get("monitoring.log_level", "INFO"))
        self.s3_client = S3Client(config)

        # Configuration
        self.src_bucket = config.get("storage.buckets.trusted_zone")
        self.src_prefix = config.get("storage.prefixes.trusted_images")

        # ChromaDB configuration
        chromadb_config = config.get("chromadb_multimodal", {})
        self.persist_dir = chromadb_config.get(
            "persist_dir", "app/zones/exploitation_zone/chroma_exploitation"
        )

        chromadb_config_multi = config.get("chromadb_multimodal", {})
        self.collection_name_multi = chromadb_config_multi.get("collection_name", "exploitation_multimodal")
        self.embedding_model_multi = chromadb_config_multi.get("embedding_model", "ViT-B-32")
        self.metadata_multi = chromadb_config_multi.get("metadata", {})

        # Initialize ChromaDB client and embedding function
        self.chroma_client = PersistentClient(path=self.persist_dir)
        self.ef_multimodal = OpenCLIPEmbeddingFunction()

        # Initialize ChromaDB collection
        multi_collection_kwargs = {"name": self.collection_name_multi, "embedding_function": self.ef_multimodal}

        # Only add metadata if it's not empty
        if self.metadata_multi:
            multi_collection_kwargs["metadata"] = self.metadata_multi

        self.multi_col = self.chroma_client.get_or_create_collection(**multi_collection_kwargs)

        self.logger.info(f"ChromaDB directory: {self.persist_dir}")

    def summarize(self, label, arr):
        if not arr:
            print(f"No {label} results found.")
            return None, None
        return min(arr), max(arr)
    
    # --- Print multi-modal search summary ---
    def print_summary(self, res: Dict[str, Any]) -> str:
        metas = res["metadatas"][0]
        dists = res["distances"][0]

        image_dists = [d for m, d in zip(metas, dists) if m.get("type") == "image"]
        text_dists  = [d for m, d in zip(metas, dists) if m.get("type") == "text"]

        closest_img, farthest_img = self.summarize("image", image_dists)
        closest_txt, farthest_txt = self.summarize("text", text_dists)

        # --- Print the summary neatly ---
        def safe_fmt(x):
            return f"{x:.3f}" if x is not None else "N/A"
        res = (
            f"Closest image match has distance  {safe_fmt(closest_img)}\n"
            f"Farthest image match has distance {safe_fmt(farthest_img)}\n"
            f"Closest recipe match has distance {safe_fmt(closest_txt)}\n"
            f"Farthest recipe match has distance {safe_fmt(farthest_txt)}"
        )
        return res

    def text_search(self, query: str = None, n_results: int = 127) -> Dict[str, Any]:
        """Perform text-based search on the multimodal collection."""
        if query is None:
            query = "fettuccine alfredo pasta dish with creamy sauce"

        count = self.multi_col.count()
        self.logger.info(f"Collection 'exploitation_multimodal' has {count} entries")

        all_data = self.multi_col.get(limit=10000)
        metas = all_data["metadatas"]

        image_count = sum(1 for m in metas if m.get("type") == "image")
        text_count  = sum(1 for m in metas if m.get("type") == "text")

        self.logger.info(f"Total items: {len(metas)} | Images: {image_count} | Texts: {text_count}")

        res = self.multi_col.query(
            query_texts=[query],
            n_results=n_results,
            include=["metadatas", "documents", "distances"]
        )

        res_string = self.print_summary(res)

        self.logger.info(f"Text search completed. result:\n {res_string}")
        
        return {
            "query": query,
            "query_type": "text",
            "total_count": count,
            "image_count": image_count,
            "text_count": text_count,
            "n_results": n_results,
            "summary": res_string,
            # Store detailed results but don't log them
            "num_results": len(res["ids"][0]) if res.get("ids") else 0,
            "closest_distance": min(res["distances"][0]) if res.get("distances") else None,
            "farthest_distance": max(res["distances"][0]) if res.get("distances") else None
        }

    def image_search(self, image_path: str = None, n_results: int = 127) -> Dict[str, Any]:
        """Perform image-based search on the multimodal collection."""
        # Find image path if not provided
        if image_path is None:
            image_candidates = [
                "calico-beans.jpg",
                "app/zones/exploitation_zone/calico-beans.jpg",
                "notebooks/exploitation_zone/tasks/calico-beans.jpg"
            ]
            for candidate in image_candidates:
                if os.path.exists(candidate):
                    image_path = candidate
                    break
            
            if image_path is None:
                self.logger.warning(f"No query image found. Checked: {image_candidates}")
                return {"status": "skipped", "reason": "no query image found"}
        
        count = self.multi_col.count()
        self.logger.info(f"Collection 'exploitation_multimodal' has {count} entries")

        all_data = self.multi_col.get(limit=10000)
        metas = all_data["metadatas"]

        image_count = sum(1 for m in metas if m.get("type") == "image")
        text_count  = sum(1 for m in metas if m.get("type") == "text")

        self.logger.info(f"Total items: {len(metas)} | Images: {image_count} | Texts: {text_count}")

        query = np.array(Image.open(image_path).convert("RGB"))

        res = self.multi_col.query(
            query_images=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        res_string = self.print_summary(res)

        self.logger.info(f"Image search completed. result:\n {res_string}")
        
        return {
            "query": image_path,
            "query_type": "image",
            "total_count": count,
            "image_count": image_count,
            "text_count": text_count,
            "n_results": n_results,
            "summary": res_string,
            # Store detailed results but don't log them
            "num_results": len(res["ids"][0]) if res.get("ids") else 0,
            "closest_distance": min(res["distances"][0]) if res.get("distances") else None,
            "farthest_distance": max(res["distances"][0]) if res.get("distances") else None
        }

    def process(self) -> Dict[str, Any]:
        """
        Main processing method that orchestrates the multimodal search tasks.
        
        Returns:
            Dictionary containing results from both text and image searches
        """
        self.logger.info("Starting Task 2: Multimodal Search")

        results = {
            "status": "completed",
            "collection_name": self.collection_name_multi,
            "persist_dir": self.persist_dir,
        }

        try:
            # Perform text search with a default query
            text_query = "fettuccine alfredo pasta dish with creamy sauce"
            self.logger.info(f"Performing text search with query: '{text_query}'")
            text_results = self.text_search(text_query)
            results["text_search"] = text_results
            
            # Print text search summary
            print("\n" + "="*70)
            print("🔤 TEXT SEARCH RESULTS")
            print("="*70)
            print(f"Query: {text_query}")
            print(f"\n{text_results['summary']}")
            print("="*70)

            # Perform image search with a default image
            image_results = self.image_search()
            results["image_search"] = image_results
            
            # Print image search summary
            if image_results.get("status") != "skipped":
                print("\n" + "="*70)
                print("🖼️  IMAGE SEARCH RESULTS")
                print("="*70)
                print(f"Query: {image_results['query']}")
                print(f"\n{image_results['summary']}")
                print("="*70 + "\n")

            self.logger.info("Task 2: Multimodal Search completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during multimodal search: {e}")
            results["status"] = "error"
            results["error"] = str(e)
            raise

        return results


def main():
    """Main entry point for exploitation images processing."""
    config = PipelineConfig()
    searcher = ExploitationMultiModalSearcher(config)

    try:
        results = searcher.process()
        print("✅ Task 2: Multimodal Search completed successfully")
        print(f"📊 Results:")
        
        if "text_search" in results:
            print(f"\n🔤 Text Search Results:")
            print(f"   Query: {results['text_search']['query']}")
            print(f"   {results['text_search']['summary']}")
        
        if "image_search" in results and results["image_search"].get("status") != "skipped":
            print(f"\n🖼️  Image Search Results:")
            print(f"   Query: {results['image_search']['query']}")
            print(f"   {results['image_search']['summary']}")
            
    except Exception as e:
        print(f"❌ Task 2: Multimodal search failed: {e}")
        raise


if __name__ == "__main__":
    main()