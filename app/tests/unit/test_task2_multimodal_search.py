"""
Unit tests for Multimodal Tasks - Task 2: Multimodal Search Processing

These tests verify the ExploitationMultiModalSearcher functionality.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, mock_open

import numpy as np
from PIL import Image

# Add app directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.config import PipelineConfig
from zones.multimodal_tasks.task2 import ExploitationMultiModalSearcher


class TestExploitationMultiModalSearcher(unittest.TestCase):
    """Unit tests for the ExploitationMultiModalSearcher."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "test_config.yaml")

        # Create a test config file
        test_config = """
pipeline:
  dry_run: true
  overwrite: true
  batch_size: 32

storage:
  buckets:
    landing_zone: "test-landing-zone"
    formatted_zone: "test-formatted-zone"
    trusted_zone: "test-trusted-zone"
    exploitation_zone: "test-exploitation-zone"
  prefixes:
    trusted_images: "images"

chromadb_multimodal:
  collection_name: "test_multimodal_collection"
  embedding_model: "ViT-B-32"
  metadata: {}
  persist_dir: "test_multimodal_chroma_dir"

monitoring:
  enabled: true
  log_level: "INFO"
"""
        with open(self.config_file, "w") as f:
            f.write(test_config)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch.dict(
        os.environ,
        {
            "MINIO_USER": "test_user",
            "MINIO_PASSWORD": "test_password",
            "MINIO_ENDPOINT": "http://localhost:9000",
        },
    )
    @patch("zones.multimodal_tasks.task2.PersistentClient")
    @patch("zones.multimodal_tasks.task2.OpenCLIPEmbeddingFunction")
    def test_processor_initialization(self, mock_clip_ef, mock_chroma):
        """Test that processor initializes correctly."""
        # Mock ChromaDB client
        mock_chroma_instance = MagicMock()
        mock_collection = MagicMock()
        mock_chroma_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_chroma_instance

        # Mock OpenCLIP embedding function
        mock_clip_ef.return_value = MagicMock()

        config = PipelineConfig(self.config_file)
        processor = ExploitationMultiModalSearcher(config)

        # Verify initialization
        self.assertEqual(processor.persist_dir, "test_multimodal_chroma_dir")
        self.assertEqual(processor.collection_name_multi, "test_multimodal_collection")
        self.assertEqual(processor.embedding_model_multi, "ViT-B-32")
        self.assertEqual(processor.src_bucket, "test-trusted-zone")
        self.assertEqual(processor.src_prefix, "images")

        # Verify ChromaDB was initialized with correct path
        mock_chroma.assert_called_once_with(path="test_multimodal_chroma_dir")
        mock_chroma_instance.get_or_create_collection.assert_called_once()

    @patch.dict(
        os.environ,
        {
            "MINIO_USER": "test_user",
            "MINIO_PASSWORD": "test_password",
            "MINIO_ENDPOINT": "http://localhost:9000",
        },
    )
    @patch("zones.multimodal_tasks.task2.PersistentClient")
    @patch("zones.multimodal_tasks.task2.OpenCLIPEmbeddingFunction")
    def test_summarize_with_results(self, mock_clip_ef, mock_chroma):
        """Test summarize method with results."""
        # Mock setup
        mock_chroma.return_value = MagicMock()
        mock_clip_ef.return_value = MagicMock()

        config = PipelineConfig(self.config_file)
        processor = ExploitationMultiModalSearcher(config)

        # Test with results
        distances = [0.5, 0.8, 1.2, 0.3]
        min_dist, max_dist = processor.summarize("test", distances)

        self.assertEqual(min_dist, 0.3)
        self.assertEqual(max_dist, 1.2)

    @patch.dict(
        os.environ,
        {
            "MINIO_USER": "test_user",
            "MINIO_PASSWORD": "test_password",
            "MINIO_ENDPOINT": "http://localhost:9000",
        },
    )
    @patch("zones.multimodal_tasks.task2.PersistentClient")
    @patch("zones.multimodal_tasks.task2.OpenCLIPEmbeddingFunction")
    def test_summarize_without_results(self, mock_clip_ef, mock_chroma):
        """Test summarize method without results."""
        # Mock setup
        mock_chroma.return_value = MagicMock()
        mock_clip_ef.return_value = MagicMock()

        config = PipelineConfig(self.config_file)
        processor = ExploitationMultiModalSearcher(config)

        # Test with empty list
        min_dist, max_dist = processor.summarize("test", [])

        self.assertIsNone(min_dist)
        self.assertIsNone(max_dist)

    @patch.dict(
        os.environ,
        {
            "MINIO_USER": "test_user",
            "MINIO_PASSWORD": "test_password",
            "MINIO_ENDPOINT": "http://localhost:9000",
        },
    )
    @patch("zones.multimodal_tasks.task2.PersistentClient")
    @patch("zones.multimodal_tasks.task2.OpenCLIPEmbeddingFunction")
    def test_print_summary(self, mock_clip_ef, mock_chroma):
        """Test print_summary method."""
        # Mock setup
        mock_chroma.return_value = MagicMock()
        mock_clip_ef.return_value = MagicMock()

        config = PipelineConfig(self.config_file)
        processor = ExploitationMultiModalSearcher(config)

        # Create mock results with mixed types
        mock_results = {
            "metadatas": [[
                {"type": "image"},
                {"type": "text"},
                {"type": "image"},
                {"type": "text"},
            ]],
            "distances": [[0.5, 0.3, 0.8, 0.4]]
        }

        summary = processor.print_summary(mock_results)

        # Verify summary format
        self.assertIn("Closest image match", summary)
        self.assertIn("Farthest image match", summary)
        self.assertIn("Closest recipe match", summary)
        self.assertIn("Farthest recipe match", summary)
        self.assertIn("0.500", summary)  # Closest image
        self.assertIn("0.800", summary)  # Farthest image
        self.assertIn("0.300", summary)  # Closest text
        self.assertIn("0.400", summary)  # Farthest text

    @patch.dict(
        os.environ,
        {
            "MINIO_USER": "test_user",
            "MINIO_PASSWORD": "test_password",
            "MINIO_ENDPOINT": "http://localhost:9000",
        },
    )
    @patch("zones.multimodal_tasks.task2.PersistentClient")
    @patch("zones.multimodal_tasks.task2.OpenCLIPEmbeddingFunction")
    def test_text_search(self, mock_clip_ef, mock_chroma):
        """Test text_search method."""
        # Mock collection
        mock_collection = MagicMock()
        mock_collection.count.return_value = 100
        mock_collection.get.return_value = {
            "metadatas": [
                {"type": "image"} for _ in range(70)
            ] + [
                {"type": "text"} for _ in range(30)
            ]
        }
        mock_collection.query.return_value = {
            "metadatas": [[
                {"type": "image"},
                {"type": "text"},
                {"type": "image"},
            ]],
            "documents": [["doc1", "doc2", "doc3"]],
            "distances": [[0.5, 0.3, 0.8]]
        }

        # Mock ChromaDB
        mock_chroma_instance = MagicMock()
        mock_chroma_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_chroma_instance
        mock_clip_ef.return_value = MagicMock()

        config = PipelineConfig(self.config_file)
        processor = ExploitationMultiModalSearcher(config)

        # Perform text search
        query = "test query"
        results = processor.text_search(query, n_results=10)

        # Verify results structure
        self.assertEqual(results["query"], query)
        self.assertEqual(results["query_type"], "text")
        self.assertEqual(results["total_count"], 100)
        self.assertEqual(results["image_count"], 70)
        self.assertEqual(results["text_count"], 30)
        self.assertEqual(results["n_results"], 10)
        self.assertIn("summary", results)
        self.assertIn("num_results", results)
        self.assertIn("closest_distance", results)
        self.assertIn("farthest_distance", results)

        # Verify collection methods were called
        mock_collection.count.assert_called_once()
        mock_collection.get.assert_called_once()
        mock_collection.query.assert_called_once_with(
            query_texts=[query],
            n_results=10,
            include=["metadatas", "documents", "distances"]
        )

    @patch.dict(
        os.environ,
        {
            "MINIO_USER": "test_user",
            "MINIO_PASSWORD": "test_password",
            "MINIO_ENDPOINT": "http://localhost:9000",
        },
    )
    @patch("zones.multimodal_tasks.task2.PersistentClient")
    @patch("zones.multimodal_tasks.task2.OpenCLIPEmbeddingFunction")
    @patch("zones.multimodal_tasks.task2.Image")
    @patch("zones.multimodal_tasks.task2.np")
    def test_image_search(self, mock_np, mock_pil_image, mock_clip_ef, mock_chroma):
        """Test image_search method."""
        # Mock collection
        mock_collection = MagicMock()
        mock_collection.count.return_value = 100
        mock_collection.get.return_value = {
            "metadatas": [
                {"type": "image"} for _ in range(70)
            ] + [
                {"type": "text"} for _ in range(30)
            ]
        }
        mock_collection.query.return_value = {
            "metadatas": [[
                {"type": "image"},
                {"type": "text"},
                {"type": "image"},
            ]],
            "documents": [["doc1", "doc2", "doc3"]],
            "distances": [[0.5, 0.3, 0.8]]
        }

        # Mock ChromaDB
        mock_chroma_instance = MagicMock()
        mock_chroma_instance.get_or_create_collection.return_value = mock_collection
        mock_chroma.return_value = mock_chroma_instance
        mock_clip_ef.return_value = MagicMock()

        # Mock image loading
        mock_image = MagicMock()
        mock_image.convert.return_value = mock_image
        mock_pil_image.open.return_value = mock_image
        
        # Mock numpy array conversion
        mock_array = MagicMock()
        mock_np.array.return_value = mock_array

        config = PipelineConfig(self.config_file)
        processor = ExploitationMultiModalSearcher(config)

        # Perform image search
        image_path = "test_image.jpg"
        results = processor.image_search(image_path, n_results=10)

        # Verify results structure
        self.assertEqual(results["query"], image_path)
        self.assertEqual(results["query_type"], "image")
        self.assertEqual(results["total_count"], 100)
        self.assertEqual(results["image_count"], 70)
        self.assertEqual(results["text_count"], 30)
        self.assertEqual(results["n_results"], 10)
        self.assertIn("summary", results)
        self.assertIn("num_results", results)
        self.assertIn("closest_distance", results)
        self.assertIn("farthest_distance", results)

        # Verify image was loaded and converted
        mock_pil_image.open.assert_called_once_with(image_path)
        mock_image.convert.assert_called_once_with("RGB")
        mock_np.array.assert_called_once()

        # Verify collection methods were called
        mock_collection.count.assert_called_once()
        mock_collection.get.assert_called_once()
        mock_collection.query.assert_called_once()

    @patch.dict(
        os.environ,
        {
            "MINIO_USER": "test_user",
            "MINIO_PASSWORD": "test_password",
            "MINIO_ENDPOINT": "http://localhost:9000",
        },
    )
    @patch("zones.multimodal_tasks.task2.PersistentClient")
    @patch("zones.multimodal_tasks.task2.OpenCLIPEmbeddingFunction")
    @patch("zones.multimodal_tasks.task2.os.path.exists")
    @patch.object(ExploitationMultiModalSearcher, "text_search")
    @patch.object(ExploitationMultiModalSearcher, "image_search")
    def test_process_with_image(self, mock_image_search, mock_text_search, 
                                 mock_exists, mock_clip_ef, mock_chroma):
        """Test process method when query image exists."""
        # Mock setup
        mock_chroma.return_value = MagicMock()
        mock_clip_ef.return_value = MagicMock()
        mock_exists.side_effect = [True]  # First candidate exists

        # Mock search results
        mock_text_search.return_value = {
            "query": "test query",
            "query_type": "text",
            "summary": "text summary"
        }
        mock_image_search.return_value = {
            "query": "calico-beans.jpg",
            "query_type": "image",
            "summary": "image summary"
        }

        config = PipelineConfig(self.config_file)
        processor = ExploitationMultiModalSearcher(config)

        # Run process
        results = processor.process()

        # Verify results
        self.assertEqual(results["status"], "completed")
        self.assertEqual(results["collection_name"], "test_multimodal_collection")
        self.assertIn("text_search", results)
        self.assertIn("image_search", results)

        # Verify search methods were called
        mock_text_search.assert_called_once_with("fettuccine alfredo pasta dish with creamy sauce")
        mock_image_search.assert_called_once_with()  # Called without args - finds image path internally

    @patch.dict(
        os.environ,
        {
            "MINIO_USER": "test_user",
            "MINIO_PASSWORD": "test_password",
            "MINIO_ENDPOINT": "http://localhost:9000",
        },
    )
    @patch("zones.multimodal_tasks.task2.PersistentClient")
    @patch("zones.multimodal_tasks.task2.OpenCLIPEmbeddingFunction")
    @patch("zones.multimodal_tasks.task2.os.path.exists")
    @patch.object(ExploitationMultiModalSearcher, "text_search")
    def test_process_without_image(self, mock_text_search, mock_exists, mock_clip_ef, mock_chroma):
        """Test process method when no query image exists."""
        # Mock setup
        mock_chroma.return_value = MagicMock()
        mock_clip_ef.return_value = MagicMock()
        mock_exists.return_value = False  # No image exists

        # Mock text search results
        mock_text_search.return_value = {
            "query": "test query",
            "query_type": "text",
            "summary": "text summary"
        }

        config = PipelineConfig(self.config_file)
        processor = ExploitationMultiModalSearcher(config)

        # Run process
        results = processor.process()

        # Verify results
        self.assertEqual(results["status"], "completed")
        self.assertIn("text_search", results)
        self.assertIn("image_search", results)
        self.assertEqual(results["image_search"]["status"], "skipped")
        self.assertEqual(results["image_search"]["reason"], "no query image found")

        # Verify only text search was called
        mock_text_search.assert_called_once()

    @patch.dict(
        os.environ,
        {
            "MINIO_USER": "test_user",
            "MINIO_PASSWORD": "test_password",
            "MINIO_ENDPOINT": "http://localhost:9000",
        },
    )
    @patch("zones.multimodal_tasks.task2.PersistentClient")
    @patch("zones.multimodal_tasks.task2.OpenCLIPEmbeddingFunction")
    def test_process_with_error(self, mock_clip_ef, mock_chroma):
        """Test process method with error."""
        # Mock setup
        mock_chroma.return_value = MagicMock()
        mock_clip_ef.return_value = MagicMock()

        config = PipelineConfig(self.config_file)
        processor = ExploitationMultiModalSearcher(config)

        # Mock text_search to raise an error
        with patch.object(processor, 'text_search', side_effect=Exception("Test error")):
            with self.assertRaises(Exception) as context:
                processor.process()
            
            self.assertIn("Test error", str(context.exception))


if __name__ == "__main__":
    unittest.main()

