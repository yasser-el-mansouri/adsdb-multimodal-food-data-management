"""
Unit tests for Multimodal Tasks - Task 3: RAG with LLaVA

These tests verify the Task3RAGProcessor functionality.
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
from zones.multimodal_tasks.task3_rag import Task3RAGProcessor


class TestTask3RAGProcessor(unittest.TestCase):
    """Unit tests for the Task3RAGProcessor."""

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
  max_retrieved_images: 3
  ollama_host: "http://localhost:11434"
  ollama_model: "llava"

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
    @patch("zones.multimodal_tasks.task3_rag.PersistentClient")
    @patch("zones.multimodal_tasks.task3_rag.open_clip.create_model_and_transforms")
    @patch("zones.multimodal_tasks.task3_rag.httpx.get")
    def test_processor_initialization(self, mock_httpx_get, mock_clip, mock_chroma):
        """Test that processor initializes correctly."""
        # Mock Ollama connectivity check
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_httpx_get.return_value = mock_response
        
        # Mock ChromaDB client (will be called twice - once for text, once for images)
        mock_text_chroma_instance = MagicMock()
        mock_image_chroma_instance = MagicMock()
        mock_chroma.side_effect = [mock_text_chroma_instance, mock_image_chroma_instance]

        # Mock CLIP model
        mock_model = MagicMock()
        mock_preprocess = MagicMock()
        mock_clip.return_value = (mock_model, mock_preprocess, None)

        config = PipelineConfig(self.config_file)
        processor = Task3RAGProcessor(config)

        # Verify initialization
        self.assertEqual(processor.text_chroma_persist_dir, "test_text_chroma_dir")
        self.assertEqual(processor.image_chroma_persist_dir, "test_image_chroma_dir")
        self.assertEqual(processor.text_collection_name, "test_text_collection")
        self.assertEqual(processor.image_collection_name, "test_image_collection")
        self.assertEqual(processor.text_embedding_model, "all-MiniLM-L6-v2")
        self.assertEqual(processor.image_embedding_model, "ViT-B-32")
        self.assertEqual(processor.ollama_host, "http://localhost:11434")
        self.assertEqual(processor.ollama_model, "llava")
        self.assertEqual(processor.default_k, 5)
        self.assertEqual(processor.max_retrieved_images, 3)

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
    @patch("zones.multimodal_tasks.task3_rag.PersistentClient")
    @patch("zones.multimodal_tasks.task3_rag.open_clip.create_model_and_transforms")
    @patch("zones.multimodal_tasks.task3_rag.httpx.get")
    def test_retrieve_text(self, mock_httpx_get, mock_clip, mock_chroma):
        """Test text retrieval."""
        # Mock Ollama connectivity check
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_httpx_get.return_value = mock_response
        
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
        processor = Task3RAGProcessor(config)

        # Retrieve text
        result = processor.retrieve_text("pasta recipe", k=2)

        # Verify
        self.assertEqual(result["query"], "pasta recipe")
        self.assertEqual(len(result["hits"]), 2)
        self.assertEqual(result["hits"][0]["text"], "Recipe 1 text")
        self.assertEqual(result["hits"][0]["score"], 0.1)

    @patch.dict(
        os.environ,
        {
            "MINIO_USER": "test_user",
            "MINIO_PASSWORD": "test_password",
            "MINIO_ENDPOINT": "http://localhost:9000",
        },
    )
    @patch("zones.multimodal_tasks.task3_rag.PersistentClient")
    @patch("zones.multimodal_tasks.task3_rag.open_clip.create_model_and_transforms")
    @patch("zones.multimodal_tasks.task3_rag.httpx.get")
    def test_retrieve_images(self, mock_httpx_get, mock_clip, mock_chroma):
        """Test image retrieval."""
        # Mock Ollama connectivity check
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_httpx_get.return_value = mock_response
        
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
        mock_clip.return_value = (mock_model, mock_preprocess, None)

        config = PipelineConfig(self.config_file)
        processor = Task3RAGProcessor(config)

        # Retrieve images
        result = processor.retrieve_images(query="pasta dish", k=2)

        # Verify
        self.assertEqual(len(result["hits"]), 2)
        self.assertEqual(result["hits"][0]["image_s3_bucket"], "test-bucket")
        self.assertEqual(result["hits"][0]["image_s3_key"], "image1.jpg")

    @patch.dict(
        os.environ,
        {
            "MINIO_USER": "test_user",
            "MINIO_PASSWORD": "test_password",
            "MINIO_ENDPOINT": "http://localhost:9000",
        },
    )
    @patch("zones.multimodal_tasks.task3_rag.PersistentClient")
    @patch("zones.multimodal_tasks.task3_rag.open_clip.create_model_and_transforms")
    @patch("zones.multimodal_tasks.task3_rag.httpx.get")
    def test_load_top_images_as_base64(self, mock_httpx_get, mock_clip, mock_chroma):
        """Test loading images from MinIO and converting to base64."""
        # Mock Ollama connectivity check
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_httpx_get.return_value = mock_response
        
        # Mock ChromaDB (called twice in __init__)
        mock_chroma.side_effect = [MagicMock(), MagicMock()]
        
        # Mock CLIP
        mock_clip.return_value = (MagicMock(), MagicMock(), None)

        config = PipelineConfig(self.config_file)
        processor = Task3RAGProcessor(config)

        # Mock S3 client
        mock_s3_response = MagicMock()
        # Create a simple test image
        test_img = Image.new("RGB", (100, 100), color="red")
        import io
        buf = io.BytesIO()
        test_img.save(buf, format="JPEG")
        mock_s3_response["Body"].read.return_value = buf.getvalue()
        
        processor.s3_client.client.get_object = MagicMock(return_value=mock_s3_response)

        # Test data
        image_hits = [
            {"image_s3_bucket": "test-bucket", "image_s3_key": "image1.jpg"},
            {"image_s3_bucket": "test-bucket", "image_s3_key": "image2.jpg"},
        ]

        # Load images
        result = processor.load_top_images_as_base64(image_hits, max_images=2)

        # Verify
        self.assertEqual(len(result), 2)
        self.assertIn("bucket", result[0])
        self.assertIn("key", result[0])
        self.assertIn("b64", result[0])
        self.assertIsInstance(result[0]["b64"], str)

    @patch.dict(
        os.environ,
        {
            "MINIO_USER": "test_user",
            "MINIO_PASSWORD": "test_password",
            "MINIO_ENDPOINT": "http://localhost:9000",
        },
    )
    @patch("zones.multimodal_tasks.task3_rag.PersistentClient")
    @patch("zones.multimodal_tasks.task3_rag.open_clip.create_model_and_transforms")
    @patch("zones.multimodal_tasks.task3_rag.httpx.get")
    def test_build_llava_prompt(self, mock_httpx_get, mock_clip, mock_chroma):
        """Test building the LLaVA prompt."""
        # Mock Ollama connectivity check
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_httpx_get.return_value = mock_response
        
        # Mock ChromaDB (called twice in __init__)
        mock_chroma.side_effect = [MagicMock(), MagicMock()]
        
        # Mock CLIP
        mock_clip.return_value = (MagicMock(), MagicMock(), None)

        config = PipelineConfig(self.config_file)
        processor = Task3RAGProcessor(config)

        # Test data
        text_hits = [
            {"text": "Recipe 1: Pasta with tomato", "score": 0.1, "meta": {}},
            {"text": "Recipe 2: Basil pesto pasta", "score": 0.2, "meta": {}},
        ]
        image_hits_meta = [
            {"bucket": "test-bucket", "key": "image1.jpg"},
            {"bucket": "test-bucket", "key": "image2.jpg"},
        ]

        # Build prompt
        prompt = processor.build_llava_prompt(
            "Give me a pasta recipe",
            text_hits,
            image_hits_meta,
            user_image_provided=False
        )

        # Verify
        self.assertIn("pasta recipe", prompt.lower())
        self.assertIn("TEXT", prompt)
        self.assertIn("IMAGE", prompt)
        self.assertIn("Recipe 1", prompt)

    @patch.dict(
        os.environ,
        {
            "MINIO_USER": "test_user",
            "MINIO_PASSWORD": "test_password",
            "MINIO_ENDPOINT": "http://localhost:9000",
        },
    )
    @patch("zones.multimodal_tasks.task3_rag.PersistentClient")
    @patch("zones.multimodal_tasks.task3_rag.open_clip.create_model_and_transforms")
    @patch("zones.multimodal_tasks.task3_rag.httpx.get")
    @patch("zones.multimodal_tasks.task3_rag.httpx.post")
    def test_generate_with_llava(self, mock_httpx_post, mock_httpx_get, mock_clip, mock_chroma):
        """Test calling LLaVA for generation."""
        # Mock Ollama connectivity check
        mock_get_response = MagicMock()
        mock_get_response.raise_for_status.return_value = None
        mock_httpx_get.return_value = mock_get_response
        
        # Mock ChromaDB (called twice in __init__)
        mock_chroma.side_effect = [MagicMock(), MagicMock()]
        
        # Mock CLIP
        mock_clip.return_value = (MagicMock(), MagicMock(), None)

        # Mock httpx response
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Here's a pasta recipe..."}
        mock_httpx_post.return_value = mock_response

        config = PipelineConfig(self.config_file)
        processor = Task3RAGProcessor(config)

        # Generate
        prompt = "Give me a pasta recipe"
        b64_images = ["base64_image_1", "base64_image_2"]
        
        result = processor.generate_with_llava(prompt, b64_images)

        # Verify
        self.assertEqual(result, "Here's a pasta recipe...")
        mock_httpx_post.assert_called_once()
        call_args = mock_httpx_post.call_args
        self.assertEqual(call_args[1]["json"]["model"], "llava")
        self.assertEqual(call_args[1]["json"]["prompt"], prompt)
        self.assertEqual(call_args[1]["json"]["images"], b64_images)

    @patch.dict(
        os.environ,
        {
            "MINIO_USER": "test_user",
            "MINIO_PASSWORD": "test_password",
            "MINIO_ENDPOINT": "http://localhost:9000",
        },
    )
    @patch("zones.multimodal_tasks.task3_rag.PersistentClient")
    @patch("zones.multimodal_tasks.task3_rag.open_clip.create_model_and_transforms")
    @patch("zones.multimodal_tasks.task3_rag.httpx.get")
    def test_save_trace_to_minio(self, mock_httpx_get, mock_clip, mock_chroma):
        """Test saving trace to MinIO."""
        # Mock Ollama connectivity check
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_httpx_get.return_value = mock_response
        
        # Mock ChromaDB (called twice in __init__)
        mock_chroma.side_effect = [MagicMock(), MagicMock()]
        
        # Mock CLIP
        mock_clip.return_value = (MagicMock(), MagicMock(), None)

        config = PipelineConfig(self.config_file)
        processor = Task3RAGProcessor(config)

        # Mock S3 put_object
        processor.s3_client.client.put_object = MagicMock(return_value={})

        # Test data
        record = {
            "query": "test query",
            "answer": "test answer",
            "timestamp_utc": "2024-01-01T00:00:00Z",
        }

        # Save trace
        s3_key = processor.save_trace_to_minio(record)

        # Verify
        self.assertIn("results/generative/rag_multimodal/", s3_key)
        self.assertTrue(s3_key.endswith(".json"))
        processor.s3_client.client.put_object.assert_called_once()


if __name__ == "__main__":
    unittest.main()

