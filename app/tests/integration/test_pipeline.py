"""
Integration tests for the data pipeline.

These tests verify that the pipeline components work together correctly.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add app directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.config import PipelineConfig

# Import processors from the new zone structure
from zones.landing_zone.temporal_landing import TemporalLandingProcessor
from zones.landing_zone.persistent_landing import PersistentLandingProcessor
from zones.formatted_zone.formatted_documents import FormattedDocumentsProcessor
from zones.formatted_zone.formatted_images import FormattedImagesProcessor
from zones.trusted_zone.trusted_images import TrustedImagesProcessor
from zones.trusted_zone.trusted_documents import TrustedDocumentsProcessor
from zones.exploitation_zone.exploitation_documents import ExploitationDocumentsProcessor
from zones.multimodal_tasks.task1_retrieval import Task1RetrievalProcessor


class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""

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
  timeout: 30

storage:
  buckets:
    landing_zone: "test-landing-zone"
    formatted_zone: "test-formatted-zone"
    trusted_zone: "test-trusted-zone"
    exploitation_zone: "test-exploitation-zone"
  
  prefixes:
    temporal_landing: "temporal_landing"
    persistent_landing_images: "persistent_landing/images"
    persistent_landing_documents: "persistent_landing/documents"
    formatted_images: "images"
    formatted_documents: "documents"
    trusted_images: "images"
    trusted_documents: "documents"
    trusted_reports: "reports"
    chroma_exploitation: "chroma_exploitation"

huggingface:
  skip_files:
    - ".gitattributes"
    - ".gitignore"
    - ".gitkeep"
  
  file_extensions:
    image: ["jpg", "jpeg", "png", "gif", "webp", "bmp", "tiff"]
    document: ["json", "jsonl", "ndjson"]

image_processing:
  target_size: [256, 256]
  target_mode: "RGB"
  target_format: "JPEG"
  target_quality: 85
  
  quality_thresholds:
    min_width: 64
    min_height: 64
    min_aspect_ratio: 0.3
    max_aspect_ratio: 4.0
    blur_varlap_min: 30.0
  
  deduplication:
    enabled: true
    method: "perceptual_hash"

document_processing:
  skip_fields: ["url", "partition"]
  always_tag: true
  
  nutrition:
    totals_key_candidates:
      - "nutr_values_per100g__from_recipes_with_nutritional_info"
      - "nutr_values_per100g"
      - "nutr_values"
    
    per_ingredient_key_candidates:
      - "nutr_per_ingredient__from_recipes_with_nutritional_info"
      - "nutr_per_ingredient"
    
    numeric_fields: ["energy", "fat", "protein", "salt", "saturates", "sugars"]
    drop_totals_if_per_ingredient_present: true
  
  text_cleaning:
    language: "english"
    remove_punctuation: true
    remove_stopwords: true

chromadb_documents:
  collection_name: "test_collection"
  embedding_model: "all-MiniLM-L6-v2"
  persist_dir: "app/zones/exploitation_zone/chroma_documents"
  metadata:
    modality: "text"
    model: "all-MiniLM-L6-v2"
    source: "minio"

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
  performance_tracking: true
  resource_monitoring: true
"""
        with open(self.config_file, "w") as f:
            f.write(test_config)

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.config_file):
            os.remove(self.config_file)

    @patch.dict(
        os.environ,
        {
            "MINIO_USER": "test_user",
            "MINIO_PASSWORD": "test_password",
            "MINIO_ENDPOINT": "http://localhost:9000",
            "HF_TOKEN": "test_token",
            "HF_ORGA": "test_org",
            "HF_DATASET": "test_dataset",
            "CHROMA_PERSIST_DIR": "/tmp/test_chroma",
        },
    )
    def test_temporal_landing_processor_initialization(self):
        """Test temporal landing processor initialization."""
        config = PipelineConfig(self.config_file)
        processor = TemporalLandingProcessor(config)

        self.assertEqual(processor.hf_orga, "test_org")
        self.assertEqual(processor.hf_dataset, "test_dataset")
        self.assertEqual(processor.bucket, "test-landing-zone")
        self.assertEqual(processor.prefix, "temporal_landing")

    @patch.dict(
        os.environ,
        {
            "MINIO_USER": "test_user",
            "MINIO_PASSWORD": "test_password",
            "MINIO_ENDPOINT": "http://localhost:9000",
            "HF_DATASET": "test_dataset",
        },
    )
    def test_persistent_landing_processor_initialization(self):
        """Test persistent landing processor initialization."""
        config = PipelineConfig(self.config_file)
        processor = PersistentLandingProcessor(config)

        self.assertEqual(processor.src_bucket, "test-landing-zone")
        self.assertEqual(processor.img_prefix, "persistent_landing/images")
        self.assertEqual(processor.doc_prefix, "persistent_landing/documents")
        self.assertEqual(processor.hf_dataset, "test_dataset")

    @patch.dict(
        os.environ,
        {
            "MINIO_USER": "test_user",
            "MINIO_PASSWORD": "test_password",
            "MINIO_ENDPOINT": "http://localhost:9000",
        },
    )
    def test_formatted_documents_processor_initialization(self):
        """Test formatted documents processor initialization."""
        config = PipelineConfig(self.config_file)
        processor = FormattedDocumentsProcessor(config)

        self.assertEqual(processor.src_bucket, "test-landing-zone")
        self.assertEqual(processor.out_bucket, "test-formatted-zone")
        self.assertEqual(processor.out_key, "documents/recipes.jsonl")
        self.assertEqual(processor.skip_fields, {"url", "partition"})

    @patch.dict(
        os.environ,
        {
            "MINIO_USER": "test_user",
            "MINIO_PASSWORD": "test_password",
            "MINIO_ENDPOINT": "http://localhost:9000",
        },
    )
    def test_formatted_images_processor_initialization(self):
        """Test formatted images processor initialization."""
        config = PipelineConfig(self.config_file)
        processor = FormattedImagesProcessor(config)

        self.assertEqual(processor.src_bucket, "test-landing-zone")
        self.assertEqual(processor.out_bucket, "test-formatted-zone")
        self.assertEqual(processor.out_prefix, "images")
        self.assertEqual(processor.target_size, (256, 256))
        self.assertEqual(processor.target_quality, 85)

    @patch.dict(
        os.environ,
        {
            "MINIO_USER": "test_user",
            "MINIO_PASSWORD": "test_password",
            "MINIO_ENDPOINT": "http://localhost:9000",
        },
    )
    def test_trusted_images_processor_initialization(self):
        """Test trusted images processor initialization."""
        config = PipelineConfig(self.config_file)
        processor = TrustedImagesProcessor(config)

        self.assertEqual(processor.src_bucket, "test-formatted-zone")
        self.assertEqual(processor.out_bucket, "test-trusted-zone")
        self.assertEqual(processor.out_prefix, "images")
        self.assertEqual(processor.target_size, (256, 256))

    @patch.dict(
        os.environ,
        {
            "MINIO_USER": "test_user",
            "MINIO_PASSWORD": "test_password",
            "MINIO_ENDPOINT": "http://localhost:9000",
        },
    )
    def test_trusted_documents_processor_initialization(self):
        """Test trusted documents processor initialization."""
        config = PipelineConfig(self.config_file)
        processor = TrustedDocumentsProcessor(config)

        self.assertEqual(processor.src_bucket, "test-formatted-zone")
        self.assertEqual(processor.out_bucket, "test-trusted-zone")
        self.assertEqual(processor.out_key, "documents/recipes.jsonl")
        self.assertEqual(processor.text_language, "english")

    @patch.dict(
        os.environ,
        {
            "MINIO_USER": "test_user",
            "MINIO_PASSWORD": "test_password",
            "MINIO_ENDPOINT": "http://localhost:9000",
        },
    )
    def test_exploitation_documents_processor_initialization(self):
        """Test exploitation documents processor initialization."""
        config = PipelineConfig(self.config_file)
        processor = ExploitationDocumentsProcessor(config)

        self.assertEqual(processor.src_bucket, "test-trusted-zone")
        self.assertEqual(processor.collection_name, "test_collection")
        self.assertEqual(processor.embedding_model, "all-MiniLM-L6-v2")
        # persist_dir now comes from config, not environment variable
        self.assertEqual(processor.persist_dir, "app/zones/exploitation_zone/chroma_documents")

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
    def test_task1_retrieval_processor_initialization(self, mock_clip, mock_chroma):
        """Test task 1 retrieval processor initialization."""
        # Mock ChromaDB and CLIP
        mock_chroma.return_value = MagicMock()
        mock_clip.return_value = (MagicMock(), MagicMock(), None)
        
        config = PipelineConfig(self.config_file)
        processor = Task1RetrievalProcessor(config)

        self.assertEqual(processor.text_chroma_persist_dir, "test_text_chroma_dir")
        self.assertEqual(processor.image_chroma_persist_dir, "test_image_chroma_dir")
        self.assertEqual(processor.text_collection_name, "test_text_collection")
        self.assertEqual(processor.image_collection_name, "test_image_collection")
        self.assertEqual(processor.text_embedding_model, "all-MiniLM-L6-v2")
        self.assertEqual(processor.image_embedding_model, "ViT-B-32")
        self.assertEqual(processor.default_k, 5)

    def test_configuration_consistency(self):
        """Test that configuration is consistent across processors."""
        config = PipelineConfig(self.config_file)

        # Test that all processors can be initialized with the same config
        processors = [
            TemporalLandingProcessor,
            PersistentLandingProcessor,
            FormattedDocumentsProcessor,
            FormattedImagesProcessor,
            TrustedImagesProcessor,
            TrustedDocumentsProcessor,
            ExploitationDocumentsProcessor,
        ]
        
        # Task1 processor needs special mocking
        task1_processor = Task1RetrievalProcessor

        for processor_class in processors:
            with patch.dict(
                os.environ,
                {
                    "MINIO_USER": "test_user",
                    "MINIO_PASSWORD": "test_password",
                    "MINIO_ENDPOINT": "http://localhost:9000",
                    "HF_TOKEN": "test_token",
                    "HF_ORGA": "test_org",
                    "HF_DATASET": "test_dataset",
                    "CHROMA_PERSIST_DIR": "/tmp/test_chroma",
                },
            ):
                try:
                    processor = processor_class(config)
                    self.assertIsNotNone(processor)
                except Exception as e:
                    self.fail(f"Failed to initialize {processor_class.__name__}: {e}")
            
            # Test task1 processor with mocking
            with patch("zones.multimodal_tasks.task1_retrieval.PersistentClient"):
                with patch("zones.multimodal_tasks.task1_retrieval.open_clip.create_model_and_transforms") as mock_clip:
                    mock_clip.return_value = (MagicMock(), MagicMock(), None)
                    try:
                        processor = task1_processor(config)
                        self.assertIsNotNone(processor)
                    except Exception as e:
                        self.fail(f"Failed to initialize {task1_processor.__name__}: {e}")

    def test_pipeline_stage_dependencies(self):
        """Test that pipeline stages have correct dependencies."""
        config = PipelineConfig(self.config_file)

        # Test that each stage can access its required configuration
        with patch.dict(
            os.environ,
            {
                "MINIO_USER": "test_user",
                "MINIO_PASSWORD": "test_password",
                "MINIO_ENDPOINT": "http://localhost:9000",
                "HF_TOKEN": "test_token",
                "HF_ORGA": "test_org",
                "HF_DATASET": "test_dataset",
                "CHROMA_PERSIST_DIR": "/tmp/test_chroma",
            },
        ):
            # Test temporal landing
            temporal = TemporalLandingProcessor(config)
            self.assertEqual(temporal.bucket, "test-landing-zone")

            # Test persistent landing
            persistent = PersistentLandingProcessor(config)
            self.assertEqual(persistent.src_bucket, "test-landing-zone")

            # Test formatted documents
            formatted_docs = FormattedDocumentsProcessor(config)
            self.assertEqual(formatted_docs.src_bucket, "test-landing-zone")
            self.assertEqual(formatted_docs.out_bucket, "test-formatted-zone")

            # Test formatted images
            formatted_imgs = FormattedImagesProcessor(config)
            self.assertEqual(formatted_imgs.src_bucket, "test-landing-zone")
            self.assertEqual(formatted_imgs.out_bucket, "test-formatted-zone")

            # Test trusted images
            trusted_imgs = TrustedImagesProcessor(config)
            self.assertEqual(trusted_imgs.src_bucket, "test-formatted-zone")
            self.assertEqual(trusted_imgs.out_bucket, "test-trusted-zone")

            # Test trusted documents
            trusted_docs = TrustedDocumentsProcessor(config)
            self.assertEqual(trusted_docs.src_bucket, "test-formatted-zone")
            self.assertEqual(trusted_docs.out_bucket, "test-trusted-zone")

            # Test exploitation documents
            exploitation = ExploitationDocumentsProcessor(config)
            self.assertEqual(exploitation.src_bucket, "test-trusted-zone")


class TestPipelineDataFlow(unittest.TestCase):
    """Test data flow between pipeline stages."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "test_config.yaml")

        # Create a minimal test config
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
    temporal_landing: "temporal_landing"
    persistent_landing_images: "persistent_landing/images"
    persistent_landing_documents: "persistent_landing/documents"
    formatted_images: "images"
    formatted_documents: "documents"
    trusted_images: "images"
    trusted_documents: "documents"
    trusted_reports: "reports"
    chroma_exploitation: "chroma_exploitation"

huggingface:
  skip_files: []
  file_extensions:
    image: ["jpg", "jpeg", "png"]
    document: ["json", "jsonl"]

image_processing:
  target_size: [256, 256]
  target_mode: "RGB"
  target_format: "JPEG"
  target_quality: 85
  quality_thresholds:
    min_width: 64
    min_height: 64
    min_aspect_ratio: 0.3
    max_aspect_ratio: 4.0
    blur_varlap_min: 30.0
  deduplication:
    enabled: true
    method: "perceptual_hash"

document_processing:
  skip_fields: ["url", "partition"]
  always_tag: true
  nutrition:
    totals_key_candidates: ["nutr_values"]
    per_ingredient_key_candidates: ["nutr_per_ingredient"]
    numeric_fields: ["energy", "fat", "protein"]
    drop_totals_if_per_ingredient_present: true
  text_cleaning:
    language: "english"
    remove_punctuation: true
    remove_stopwords: true

chromadb_documents:
  collection_name: "test_collection"
  embedding_model: "all-MiniLM-L6-v2"
  persist_dir: "app/zones/exploitation_zone/chroma_documents"
  metadata:
    modality: "text"
    model: "all-MiniLM-L6-v2"
    source: "minio"

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
        if os.path.exists(self.config_file):
            os.remove(self.config_file)

    def test_pipeline_data_flow(self):
        """Test that data flows correctly between pipeline stages."""
        config = PipelineConfig(self.config_file)

        with patch.dict(
            os.environ,
            {
                "MINIO_USER": "test_user",
                "MINIO_PASSWORD": "test_password",
                "MINIO_ENDPOINT": "http://localhost:9000",
                "HF_TOKEN": "test_token",
                "HF_ORGA": "test_org",
                "HF_DATASET": "test_dataset",
                "CHROMA_PERSIST_DIR": "/tmp/test_chroma",
            },
        ):
            # Test that each stage can be initialized
            # This verifies that the data flow configuration is correct

            # Stage 1: Temporal Landing -> Landing Zone
            temporal = TemporalLandingProcessor(config)
            self.assertEqual(temporal.bucket, "test-landing-zone")

            # Stage 2: Persistent Landing (Landing Zone -> Landing Zone, organized)
            persistent = PersistentLandingProcessor(config)
            self.assertEqual(persistent.src_bucket, "test-landing-zone")
            self.assertEqual(persistent.dest_bucket, "test-landing-zone")

            # Stage 3: Formatted Documents (Landing Zone -> Formatted Zone)
            formatted_docs = FormattedDocumentsProcessor(config)
            self.assertEqual(formatted_docs.src_bucket, "test-landing-zone")
            self.assertEqual(formatted_docs.out_bucket, "test-formatted-zone")

            # Stage 4: Formatted Images (Landing Zone -> Formatted Zone)
            formatted_imgs = FormattedImagesProcessor(config)
            self.assertEqual(formatted_imgs.src_bucket, "test-landing-zone")
            self.assertEqual(formatted_imgs.out_bucket, "test-formatted-zone")

            # Stage 5: Trusted Images (Formatted Zone -> Trusted Zone)
            trusted_imgs = TrustedImagesProcessor(config)
            self.assertEqual(trusted_imgs.src_bucket, "test-formatted-zone")
            self.assertEqual(trusted_imgs.out_bucket, "test-trusted-zone")

            # Stage 6: Trusted Documents (Formatted Zone -> Trusted Zone)
            trusted_docs = TrustedDocumentsProcessor(config)
            self.assertEqual(trusted_docs.src_bucket, "test-formatted-zone")
            self.assertEqual(trusted_docs.out_bucket, "test-trusted-zone")

            # Stage 7: Exploitation Documents (Trusted Zone -> ChromaDB)
            exploitation = ExploitationDocumentsProcessor(config)
            self.assertEqual(exploitation.src_bucket, "test-trusted-zone")
            
            # Stage 8: Task 1 Retrieval (ChromaDB -> Query Results)
            with patch("zones.multimodal_tasks.task1_retrieval.PersistentClient"):
                with patch("zones.multimodal_tasks.task1_retrieval.open_clip.create_model_and_transforms") as mock_clip:
                    mock_clip.return_value = (MagicMock(), MagicMock(), None)
                    task1 = Task1RetrievalProcessor(config)
                    self.assertEqual(task1.text_chroma_persist_dir, "test_text_chroma_dir")
                    self.assertEqual(task1.image_chroma_persist_dir, "test_image_chroma_dir")


if __name__ == "__main__":
    unittest.main()
