"""
Unit tests for Multimodal Tasks - Task 1: Retrieval Processing

These tests verify the Task1RetrievalProcessor functionality.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, PropertyMock

import numpy as np
from PIL import Image

# Add app directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.config import PipelineConfig
from zones.multimodal_tasks.task1_retrieval import Task1RetrievalProcessor


class TestTask1RetrievalProcessor(unittest.TestCase):
    """Unit tests for the Task1RetrievalProcessor."""

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

multimodal_tasks:
  text_chroma_persist_dir: "test_text_chroma_dir"
  image_chroma_persist_dir: "test_image_chroma_dir"
  text_collection_name: "test_text_collection"
  image_collection_name: "test_image_collection"
  text_embedding_model: "all-MiniLM-L6-v2"
  image_embedding_model: "ViT-B-32"
  image_pretrained: "laion2b_s34b_b79k"
  default_k: 5

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
    @patch("zones.multimodal_tasks.task1_retrieval.PersistentClient")
    @patch("zones.multimodal_tasks.task1_retrieval.open_clip.create_model_and_transforms")
    def test_processor_initialization(self, mock_clip, mock_chroma):
        """Test that processor initializes correctly."""
        # Mock ChromaDB client (will be called twice - once for text, once for images)
        mock_text_chroma_instance = MagicMock()
        mock_image_chroma_instance = MagicMock()
        mock_chroma.side_effect = [mock_text_chroma_instance, mock_image_chroma_instance]
        
        # Mock CLIP model
        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_clip.return_value = (mock_model, mock_preprocess, None)

        config = PipelineConfig(self.config_file)
        processor = Task1RetrievalProcessor(config)

        # Verify initialization
        self.assertEqual(processor.text_chroma_persist_dir, "test_text_chroma_dir")
        self.assertEqual(processor.image_chroma_persist_dir, "test_image_chroma_dir")
        self.assertEqual(processor.text_collection_name, "test_text_collection")
        self.assertEqual(processor.image_collection_name, "test_image_collection")
        self.assertEqual(processor.text_embedding_model, "all-MiniLM-L6-v2")
        self.assertEqual(processor.image_embedding_model, "ViT-B-32")
        self.assertEqual(processor.default_k, 5)

        # Verify ChromaDB was initialized with correct paths (called twice)
        self.assertEqual(mock_chroma.call_count, 2)
        mock_chroma.assert_any_call(path="test_text_chroma_dir")
        mock_chroma.assert_any_call(path="test_image_chroma_dir")

    @patch.dict(
        os.environ,
        {
            "MINIO_USER": "test_user",
            "MINIO_PASSWORD": "test_password",
            "MINIO_ENDPOINT": "http://localhost:9000",
        },
    )
    @patch("zones.multimodal_tasks.task1_retrieval.PersistentClient")
    @patch("zones.multimodal_tasks.task1_retrieval.open_clip.create_model_and_transforms")
    def test_get_text_collection(self, mock_clip, mock_chroma):
        """Test getting text collection."""
        # Mock ChromaDB (called twice in __init__ - once for text, once for images)
        mock_collection = MagicMock()
        mock_text_chroma_instance = MagicMock()
        mock_text_chroma_instance.get_collection.return_value = mock_collection
        mock_image_chroma_instance = MagicMock()
        mock_chroma.side_effect = [mock_text_chroma_instance, mock_image_chroma_instance]
        
        # Mock CLIP
        mock_clip.return_value = (MagicMock(), MagicMock(), None)

        config = PipelineConfig(self.config_file)
        processor = Task1RetrievalProcessor(config)

        # Get collection
        collection = processor.get_text_collection()

        # Verify
        self.assertEqual(collection, mock_collection)
        mock_text_chroma_instance.get_collection.assert_called_once()
        call_kwargs = mock_text_chroma_instance.get_collection.call_args[1]
        self.assertEqual(call_kwargs["name"], "test_text_collection")

    @patch.dict(
        os.environ,
        {
            "MINIO_USER": "test_user",
            "MINIO_PASSWORD": "test_password",
            "MINIO_ENDPOINT": "http://localhost:9000",
        },
    )
    @patch("zones.multimodal_tasks.task1_retrieval.PersistentClient")
    @patch("zones.multimodal_tasks.task1_retrieval.open_clip.create_model_and_transforms")
    def test_get_image_collection(self, mock_clip, mock_chroma):
        """Test getting image collection."""
        # Mock ChromaDB (called twice in __init__ - once for text, once for images)
        mock_collection = MagicMock()
        mock_text_chroma_instance = MagicMock()
        mock_image_chroma_instance = MagicMock()
        mock_image_chroma_instance.get_collection.return_value = mock_collection
        mock_chroma.side_effect = [mock_text_chroma_instance, mock_image_chroma_instance]
        
        # Mock CLIP
        mock_clip.return_value = (MagicMock(), MagicMock(), None)

        config = PipelineConfig(self.config_file)
        processor = Task1RetrievalProcessor(config)

        # Get collection
        collection = processor.get_image_collection()

        # Verify
        self.assertEqual(collection, mock_collection)
        call_kwargs = mock_image_chroma_instance.get_collection.call_args[1]
        self.assertEqual(call_kwargs["name"], "test_image_collection")

    @patch.dict(
        os.environ,
        {
            "MINIO_USER": "test_user",
            "MINIO_PASSWORD": "test_password",
            "MINIO_ENDPOINT": "http://localhost:9000",
        },
    )
    @patch("zones.multimodal_tasks.task1_retrieval.PersistentClient")
    @patch("zones.multimodal_tasks.task1_retrieval.open_clip.create_model_and_transforms")
    @patch("zones.multimodal_tasks.task1_retrieval.torch")
    def test_encode_image_to_vec(self, mock_torch, mock_clip, mock_chroma):
        """Test encoding an image to vector."""
        # Mock ChromaDB (called twice in __init__)
        mock_chroma.side_effect = [MagicMock(), MagicMock()]
        
        # Mock CLIP model
        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        
        # Mock the image preprocessing
        mock_tensor = MagicMock()
        mock_tensor.unsqueeze.return_value.to.return_value = mock_tensor
        mock_preprocess.return_value = mock_tensor
        
        # Mock the model output
        mock_features = MagicMock()
        mock_features.norm.return_value = MagicMock()
        type(mock_features).__truediv__ = lambda self, other: mock_features
        mock_features.squeeze.return_value.cpu.return_value.tolist.return_value = [0.1] * 512
        mock_model.encode_image.return_value = mock_features
        
        mock_clip.return_value = (mock_model, mock_preprocess, None)
        
        # Mock torch.no_grad
        mock_torch.no_grad = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

        config = PipelineConfig(self.config_file)
        processor = Task1RetrievalProcessor(config)

        # Create a test image
        test_image = Image.new("RGB", (256, 256), color="red")
        
        # Encode image
        vector = processor.encode_image_to_vec(test_image)

        # Verify
        self.assertIsInstance(vector, list)
        self.assertEqual(len(vector), 512)
        mock_preprocess.assert_called_once_with(test_image)

    @patch.dict(
        os.environ,
        {
            "MINIO_USER": "test_user",
            "MINIO_PASSWORD": "test_password",
            "MINIO_ENDPOINT": "http://localhost:9000",
        },
    )
    @patch("zones.multimodal_tasks.task1_retrieval.PersistentClient")
    @patch("zones.multimodal_tasks.task1_retrieval.open_clip.create_model_and_transforms")
    def test_retrieve_text(self, mock_clip, mock_chroma):
        """Test text retrieval."""
        # Mock ChromaDB
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Recipe 1 text", "Recipe 2 text"]],
            "metadatas": [[{"id": "1", "title": "Recipe 1"}, {"id": "2", "title": "Recipe 2"}]],
            "distances": [[0.1, 0.2]],
        }
        
        mock_text_chroma_instance = MagicMock()
        mock_text_chroma_instance.get_collection.return_value = mock_collection
        mock_image_chroma_instance = MagicMock()
        mock_chroma.side_effect = [mock_text_chroma_instance, mock_image_chroma_instance]
        
        # Mock CLIP
        mock_clip.return_value = (MagicMock(), MagicMock(), None)

        config = PipelineConfig(self.config_file)
        processor = Task1RetrievalProcessor(config)

        # Retrieve text
        result = processor.retrieve_text("chicken soup", k=2)

        # Verify
        self.assertEqual(result["query"], "chicken soup")
        self.assertEqual(len(result["hits"]), 2)
        self.assertEqual(result["hits"][0]["text"], "Recipe 1 text")
        self.assertEqual(result["hits"][0]["score"], 0.1)
        self.assertEqual(result["hits"][1]["text"], "Recipe 2 text")
        self.assertEqual(result["hits"][1]["score"], 0.2)
        
        # Verify query was called correctly
        mock_collection.query.assert_called_once_with(
            query_texts=["chicken soup"],
            n_results=2,
            include=["documents", "metadatas", "distances"],
        )

    @patch.dict(
        os.environ,
        {
            "MINIO_USER": "test_user",
            "MINIO_PASSWORD": "test_password",
            "MINIO_ENDPOINT": "http://localhost:9000",
        },
    )
    @patch("zones.multimodal_tasks.task1_retrieval.PersistentClient")
    @patch("zones.multimodal_tasks.task1_retrieval.open_clip.create_model_and_transforms")
    @patch("zones.multimodal_tasks.task1_retrieval.Image")
    def test_retrieve_images(self, mock_image_module, mock_clip, mock_chroma):
        """Test image retrieval."""
        # Mock ChromaDB
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "metadatas": [[
                {"bucket": "test-bucket", "object_key": "image1.jpg"},
                {"bucket": "test-bucket", "object_key": "image2.jpg"}
            ]],
            "distances": [[0.15, 0.25]],
        }
        
        mock_text_chroma_instance = MagicMock()
        mock_image_chroma_instance = MagicMock()
        mock_image_chroma_instance.get_collection.return_value = mock_collection
        mock_chroma.side_effect = [mock_text_chroma_instance, mock_image_chroma_instance]
        
        # Mock CLIP
        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        
        mock_tensor = MagicMock()
        mock_tensor.unsqueeze.return_value.to.return_value = mock_tensor
        mock_preprocess.return_value = mock_tensor
        
        mock_features = MagicMock()
        mock_features.norm.return_value = MagicMock()
        type(mock_features).__truediv__ = lambda self, other: mock_features
        mock_features.squeeze.return_value.cpu.return_value.tolist.return_value = [0.1] * 512
        mock_model.encode_image.return_value = mock_features
        
        mock_clip.return_value = (mock_model, mock_preprocess, None)
        
        # Mock PIL Image
        mock_pil_image = MagicMock()
        mock_pil_image.convert.return_value = mock_pil_image
        mock_image_module.open.return_value = mock_pil_image

        config = PipelineConfig(self.config_file)
        processor = Task1RetrievalProcessor(config)

        # Retrieve images
        result = processor.retrieve_images("test_image.jpg", k=2)

        # Verify
        self.assertEqual(len(result["hits"]), 2)
        self.assertEqual(result["hits"][0]["image_s3_bucket"], "test-bucket")
        self.assertEqual(result["hits"][0]["image_s3_key"], "image1.jpg")
        self.assertEqual(result["hits"][0]["score"], 0.15)
        self.assertEqual(result["hits"][1]["score"], 0.25)
        
        # Verify image was opened and processed
        mock_image_module.open.assert_called_once_with("test_image.jpg")

    @patch.dict(
        os.environ,
        {
            "MINIO_USER": "test_user",
            "MINIO_PASSWORD": "test_password",
            "MINIO_ENDPOINT": "http://localhost:9000",
        },
    )
    @patch("zones.multimodal_tasks.task1_retrieval.PersistentClient")
    @patch("zones.multimodal_tasks.task1_retrieval.open_clip.create_model_and_transforms")
    def test_get_collection_stats(self, mock_clip, mock_chroma):
        """Test getting collection statistics."""
        # Mock collections
        mock_text_collection = MagicMock()
        mock_text_collection.count.return_value = 1000
        type(mock_text_collection).name = PropertyMock(return_value="test_text_collection")
        
        mock_image_collection = MagicMock()
        mock_image_collection.count.return_value = 500
        type(mock_image_collection).name = PropertyMock(return_value="test_image_collection")
        
        # Mock ChromaDB (separate clients for text and images)
        mock_text_chroma_instance = MagicMock()
        mock_text_chroma_instance.list_collections.return_value = [mock_text_collection]
        mock_text_chroma_instance.get_collection.return_value = mock_text_collection
        
        mock_image_chroma_instance = MagicMock()
        mock_image_chroma_instance.list_collections.return_value = [mock_image_collection]
        mock_image_chroma_instance.get_collection.return_value = mock_image_collection
        
        mock_chroma.side_effect = [mock_text_chroma_instance, mock_image_chroma_instance]
        
        # Mock CLIP
        mock_clip.return_value = (MagicMock(), MagicMock(), None)

        config = PipelineConfig(self.config_file)
        processor = Task1RetrievalProcessor(config)

        # Get stats
        stats = processor.get_collection_stats()

        # Verify
        self.assertEqual(stats["total_collections"], 2)
        self.assertEqual(len(stats["collection_names"]), 2)
        self.assertIn("test_text_collection", stats["collection_names"])
        self.assertIn("test_image_collection", stats["collection_names"])
        self.assertEqual(stats["text_collection"]["count"], 1000)
        self.assertEqual(stats["image_collection"]["count"], 500)

    @patch.dict(
        os.environ,
        {
            "MINIO_USER": "test_user",
            "MINIO_PASSWORD": "test_password",
            "MINIO_ENDPOINT": "http://localhost:9000",
        },
    )
    @patch("zones.multimodal_tasks.task1_retrieval.PersistentClient")
    @patch("zones.multimodal_tasks.task1_retrieval.open_clip.create_model_and_transforms")
    def test_run_demo_queries(self, mock_clip, mock_chroma):
        """Test running demo queries."""
        # Mock ChromaDB
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Recipe text"]],
            "metadatas": [[{"id": "1", "title": "Recipe"}]],
            "distances": [[0.1]],
        }
        
        mock_text_chroma_instance = MagicMock()
        mock_text_chroma_instance.get_collection.return_value = mock_collection
        mock_image_chroma_instance = MagicMock()
        mock_chroma.side_effect = [mock_text_chroma_instance, mock_image_chroma_instance]
        
        # Mock CLIP
        mock_clip.return_value = (MagicMock(), MagicMock(), None)

        config = PipelineConfig(self.config_file)
        processor = Task1RetrievalProcessor(config)

        # Run demo queries
        results = processor.run_demo_queries()

        # Verify
        self.assertIn("text_queries", results)
        self.assertIn("image_queries", results)
        self.assertGreater(len(results["text_queries"]), 0)
        
        # Check first text query result
        first_query = results["text_queries"][0]
        self.assertIn("query", first_query)
        self.assertIn("num_results", first_query)

    @patch.dict(
        os.environ,
        {
            "MINIO_USER": "test_user",
            "MINIO_PASSWORD": "test_password",
            "MINIO_ENDPOINT": "http://localhost:9000",
        },
    )
    @patch("zones.multimodal_tasks.task1_retrieval.PersistentClient")
    @patch("zones.multimodal_tasks.task1_retrieval.open_clip.create_model_and_transforms")
    def test_process(self, mock_clip, mock_chroma):
        """Test the main process method."""
        # Mock ChromaDB
        mock_collection = MagicMock()
        mock_collection.count.return_value = 100
        type(mock_collection).name = PropertyMock(return_value="test_collection")
        mock_collection.query.return_value = {
            "documents": [["Recipe text"]],
            "metadatas": [[{"id": "1", "title": "Recipe"}]],
            "distances": [[0.1]],
        }
        
        mock_text_chroma_instance = MagicMock()
        mock_text_chroma_instance.list_collections.return_value = [mock_collection]
        mock_text_chroma_instance.get_collection.return_value = mock_collection
        
        mock_image_chroma_instance = MagicMock()
        mock_image_chroma_instance.list_collections.return_value = []
        
        mock_chroma.side_effect = [mock_text_chroma_instance, mock_image_chroma_instance]
        
        # Mock CLIP
        mock_clip.return_value = (MagicMock(), MagicMock(), None)

        config = PipelineConfig(self.config_file)
        processor = Task1RetrievalProcessor(config)

        # Process
        result = processor.process()

        # Verify
        self.assertIn("timestamp", result)
        self.assertIn("collection_stats", result)
        self.assertIn("demo_results", result)
        self.assertEqual(result["status"], "success")

    @patch.dict(
        os.environ,
        {
            "MINIO_USER": "test_user",
            "MINIO_PASSWORD": "test_password",
            "MINIO_ENDPOINT": "http://localhost:9000",
        },
    )
    @patch("zones.multimodal_tasks.task1_retrieval.PersistentClient")
    @patch("zones.multimodal_tasks.task1_retrieval.open_clip.create_model_and_transforms")
    def test_retrieve_text_uses_default_k(self, mock_clip, mock_chroma):
        """Test that retrieve_text uses default k when not specified."""
        # Mock ChromaDB
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Recipe 1", "Recipe 2", "Recipe 3", "Recipe 4", "Recipe 5"]],
            "metadatas": [[{"id": str(i)} for i in range(5)]],
            "distances": [[0.1 * i for i in range(5)]],
        }
        
        mock_text_chroma_instance = MagicMock()
        mock_text_chroma_instance.get_collection.return_value = mock_collection
        mock_image_chroma_instance = MagicMock()
        mock_chroma.side_effect = [mock_text_chroma_instance, mock_image_chroma_instance]
        
        # Mock CLIP
        mock_clip.return_value = (MagicMock(), MagicMock(), None)

        config = PipelineConfig(self.config_file)
        processor = Task1RetrievalProcessor(config)

        # Retrieve without specifying k (should use default_k=5)
        result = processor.retrieve_text("test query")

        # Verify default k was used
        mock_collection.query.assert_called_once()
        call_args = mock_collection.query.call_args[1]
        self.assertEqual(call_args["n_results"], 5)


if __name__ == "__main__":
    unittest.main()

