"""
Unit tests for the data pipeline utilities.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.utils import (
    Logger,
    PipelineConfig,
    S3Client,
    atomic_write_json,
    sanitize_filename,
    to_builtin,
    utc_timestamp,
    validate_config,
)


class TestPipelineConfig(unittest.TestCase):
    """Test cases for PipelineConfig class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "test_config.yaml")

        # Create a test config file
        test_config = """
pipeline:
  dry_run: true
  overwrite: false
  batch_size: 128
  timeout: 30

storage:
  buckets:
    landing_zone: "test-landing-zone"
    formatted_zone: "test-formatted-zone"
    trusted_zone: "test-trusted-zone"
    exploitation_zone: "test-exploitation-zone"

monitoring:
  enabled: true
  log_level: "DEBUG"
"""
        with open(self.config_file, "w") as f:
            f.write(test_config)

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.config_file):
            os.remove(self.config_file)

    def test_config_loading(self):
        """Test configuration loading from YAML file."""
        config = PipelineConfig(self.config_file)

        self.assertEqual(config.get("pipeline.dry_run"), True)
        self.assertEqual(config.get("pipeline.overwrite"), False)
        self.assertEqual(config.get("pipeline.batch_size"), 128)
        self.assertEqual(config.get("storage.buckets.landing_zone"), "test-landing-zone")

    def test_config_defaults(self):
        """Test default configuration when file doesn't exist."""
        config = PipelineConfig("nonexistent.yaml")

        self.assertEqual(config.get("pipeline.dry_run"), False)
        self.assertEqual(config.get("pipeline.batch_size"), 256)

    def test_config_nested_access(self):
        """Test nested configuration access."""
        config = PipelineConfig(self.config_file)

        self.assertEqual(config.get("storage.buckets.landing_zone"), "test-landing-zone")
        self.assertEqual(config.get("monitoring.log_level"), "DEBUG")

    def test_config_missing_key(self):
        """Test accessing missing configuration keys."""
        config = PipelineConfig(self.config_file)

        self.assertIsNone(config.get("nonexistent.key"))
        self.assertEqual(config.get("nonexistent.key", "default"), "default")


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""

    def test_utc_timestamp(self):
        """Test UTC timestamp generation."""
        timestamp = utc_timestamp()

        self.assertIsInstance(timestamp, str)
        self.assertTrue(timestamp.endswith("Z"))
        self.assertEqual(len(timestamp), 20)  # Format: YYYY-MM-DDTHH-MM-SSZ

    def test_to_builtin_decimal(self):
        """Test conversion of Decimal to built-in types."""
        from decimal import Decimal

        # Test integer Decimal
        result = to_builtin(Decimal("123"))
        self.assertEqual(result, 123)
        self.assertIsInstance(result, int)

        # Test float Decimal
        result = to_builtin(Decimal("123.45"))
        self.assertEqual(result, 123.45)
        self.assertIsInstance(result, float)

    def test_to_builtin_dict(self):
        """Test conversion of dict with Decimals."""
        from decimal import Decimal

        data = {
            "int_val": Decimal("123"),
            "float_val": Decimal("123.45"),
            "nested": {"decimal": Decimal("456.78")},
        }

        result = to_builtin(data)

        self.assertEqual(result["int_val"], 123)
        self.assertEqual(result["float_val"], 123.45)
        self.assertEqual(result["nested"]["decimal"], 456.78)
        self.assertIsInstance(result["int_val"], int)
        self.assertIsInstance(result["float_val"], float)
        self.assertIsInstance(result["nested"]["decimal"], float)

    def test_to_builtin_list(self):
        """Test conversion of list with Decimals."""
        from decimal import Decimal

        data = [Decimal("123"), Decimal("456.78")]
        result = to_builtin(data)

        self.assertEqual(result, [123, 456.78])
        self.assertIsInstance(result[0], int)
        self.assertIsInstance(result[1], float)

    def test_sanitize_filename(self):
        """Test filename sanitization."""
        # Test normal filename
        self.assertEqual(sanitize_filename("normal_file.txt"), "normal_file.txt")

        # Test filename with special characters
        self.assertEqual(sanitize_filename("file with spaces.txt"), "file_with_spaces.txt")
        self.assertEqual(
            sanitize_filename("file@with#special$chars.txt"), "file_with_special_chars.txt"
        )

        # Test empty filename
        self.assertEqual(sanitize_filename(""), "")

    def test_atomic_write_json(self):
        """Test atomic JSON file writing."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_file = f.name

        try:
            test_data = {"key": "value", "number": 123}
            atomic_write_json(temp_file, test_data)

            # Verify file was written correctly
            self.assertTrue(os.path.exists(temp_file))

            with open(temp_file, "r") as f:
                loaded_data = f.read()

            # Should be valid JSON
            import json

            parsed_data = json.loads(loaded_data)
            self.assertEqual(parsed_data, test_data)

        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)


class TestLogger(unittest.TestCase):
    """Test cases for Logger class."""

    def test_logger_creation(self):
        """Test logger creation."""
        logger = Logger("test_logger", "INFO")

        self.assertIsNotNone(logger.logger)
        self.assertEqual(logger.logger.name, "test_logger")

    def test_logger_methods(self):
        """Test logger methods."""
        logger = Logger("test_logger", "DEBUG")

        # These should not raise exceptions
        logger.info("Test info message")
        logger.warning("Test warning message")
        logger.error("Test error message")
        logger.debug("Test debug message")


class TestValidation(unittest.TestCase):
    """Test cases for configuration validation."""

    def test_validate_config_missing_env_vars(self):
        """Test validation with missing environment variables."""
        config = PipelineConfig("nonexistent.yaml")

        # Mock missing environment variables
        with patch.dict(os.environ, {}, clear=True):
            issues = validate_config(config)

            self.assertGreater(len(issues), 0)
            self.assertTrue(any("MINIO_USER" in issue for issue in issues))
            self.assertTrue(any("MINIO_PASSWORD" in issue for issue in issues))
            self.assertTrue(any("MINIO_ENDPOINT" in issue for issue in issues))

    def test_validate_config_valid(self):
        """Test validation with valid configuration."""
        config = PipelineConfig("nonexistent.yaml")

        # Mock valid environment variables
        with patch.dict(
            os.environ,
            {
                "MINIO_USER": "test_user",
                "MINIO_PASSWORD": "test_password",
                "MINIO_ENDPOINT": "http://localhost:9000",
            },
        ):
            issues = validate_config(config)

            # Should have no issues
            self.assertEqual(len(issues), 0)


if __name__ == "__main__":
    unittest.main()
