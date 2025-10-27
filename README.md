# Data Pipeline Operations Environment

This repository contains a modularized operations environment for the ADSDB Multimodal Food Data Management pipeline. The system transforms raw data from external sources through multiple processing zones, ultimately creating a searchable knowledge base of recipes with images.

## 🏗️ Architecture Overview

The pipeline follows a data lake architecture with four main zones:

1. **Landing Zone**: Raw data ingestion and initial organization
2. **Formatted Zone**: Data cleaning, joining, and standardization
3. **Trusted Zone**: Quality-controlled, filtered data ready for analysis
4. **Exploitation Zone**: Vector embeddings and search capabilities

## 📁 Project Structure

```
app/
├── config/
│   └── pipeline.yaml          # Pipeline configuration
├── utils/
│   └── __init__.py           # Shared utilities and common functions
├── zones/
│   ├── temporal_landing.py   # Raw data ingestion
│   ├── persistent_landing.py # Data organization
│   ├── formatted_documents.py # Document processing
│   ├── formatted_images.py   # Image processing
│   ├── trusted_images.py     # Image quality control
│   ├── trusted_documents.py  # Document quality control
│   └── exploitation_documents.py # Vector embeddings
├── tests/
│   ├── unit/                 # Unit tests
│   └── integration/         # Integration tests
├── monitoring/
│   └── __init__.py          # Performance monitoring
├── orchestrate.py           # Main pipeline orchestrator
├── cli.py                   # Command-line interface
└── .env.sample             # Environment variables template
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- MinIO server running locally or remotely
- Hugging Face account (for dataset access)
- Required Python packages (see requirements.txt)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd adsdb-multimodal-food-data-management
   ```

2. **Install dependencies**
   ```bash
   pip install -r notebooks/requirements.txt
   pip install typer rich pytest
   ```

3. **Configure environment**
   ```bash
   cp app/.env.sample .env
   # Edit .env with your actual values
   ```

4. **Initialize the pipeline**
   ```bash
   python -m app.cli init
   ```

5. **Validate configuration**
   ```bash
   python -m app.cli validate
   ```

### Running the Pipeline

#### Full Pipeline
```bash
python -m app.cli run
```

#### Specific Stages
```bash
# Run only the landing zone
python -m app.cli run --stages temporal_landing,persistent_landing

# Run a single stage
python -m app.cli run --stage trusted_images
```

#### Dry Run Mode
```bash
python -m app.cli run --dry-run
```

#### Verbose Logging
```bash
python -m app.cli run --verbose
```

## ⚙️ Configuration

### Environment Variables (.env)

```bash
# Hugging Face Configuration
HF_TOKEN=your_huggingface_token_here
HF_ORGA=your_organization_name
HF_DATASET=your_dataset_name
HF_REV=main

# MinIO Configuration
MINIO_USER=minio
MINIO_PASSWORD=minio12345
MINIO_ENDPOINT=http://localhost:9000

# ChromaDB Configuration
CHROMA_PERSIST_DIR=./chroma_exploitation

# Pipeline Configuration
PIPELINE_LOG_LEVEL=INFO
PIPELINE_DRY_RUN=false
PIPELINE_OVERWRITE=true
```

### Pipeline Configuration (pipeline.yaml)

The pipeline configuration file contains all non-sensitive settings:

- **Pipeline execution settings**: dry_run, overwrite, batch_size, timeout
- **Storage configuration**: bucket names and prefixes
- **Processing parameters**: image quality thresholds, text cleaning settings
- **Monitoring configuration**: log levels, performance tracking
- **Testing configuration**: test data sizes, coverage thresholds

## 🔄 Pipeline Stages

### 1. Temporal Landing
- **Purpose**: Ingest raw data from Hugging Face datasets
- **Input**: External Hugging Face dataset
- **Output**: Raw files in MinIO landing zone
- **Key Features**: 
  - Handles various file formats (JSON, JSONL, images)
  - Processes layer2.json for image URLs
  - Applies naming conventions with hashing

### 2. Persistent Landing
- **Purpose**: Organize raw data by type and apply naming conventions
- **Input**: Raw files from temporal landing
- **Output**: Organized files in persistent landing structure
- **Key Features**:
  - File type detection (images vs documents)
  - Standardized naming: `type$dataset$timestamp$name.ext`
  - Creates indexes for images and recipes

### 3. Formatted Documents
- **Purpose**: Join all recipes and remove irrelevant data
- **Input**: Organized documents from persistent landing
- **Output**: Single JSONL file with joined recipes
- **Key Features**:
  - Merges recipes by ID with tagged field collisions
  - Removes specified fields (url, partition)
  - Handles multiple JSON formats (JSON, JSONL, arrays)

### 4. Formatted Images
- **Purpose**: Process and organize images
- **Input**: Organized images from persistent landing
- **Output**: Processed images in formatted zone
- **Key Features**:
  - Quality screening (size, aspect ratio, blur detection)
  - Per-recipe deduplication using perceptual hashing
  - Image normalization to standard format

### 5. Trusted Images
- **Purpose**: Extract recipe IDs and copy filtered images
- **Input**: Processed images from formatted zone
- **Output**: Quality-controlled images in trusted zone
- **Key Features**:
  - Recipe ID extraction from filenames
  - Quality filtering and deduplication
  - Generates recipe IDs file for document processing
  - Image normalization to 512x512 JPEG

### 6. Trusted Documents
- **Purpose**: Filter documents to keep only those with images
- **Input**: Joined documents from formatted zone
- **Output**: Filtered documents in trusted zone
- **Key Features**:
  - Filters by recipe IDs that have images
  - Applies nutrition quality controls (IQR outlier detection)
  - Text cleaning and normalization
  - Removes duplicated nutrition data

### 7. Exploitation Documents
- **Purpose**: Generate embeddings and store in ChromaDB
- **Input**: Filtered documents from trusted zone
- **Output**: Vector embeddings in ChromaDB
- **Key Features**:
  - Uses SentenceTransformer for embeddings
  - Batch processing for efficiency
  - Metadata extraction (FSA nutrition lights)
  - Semantic search capabilities

## 🧪 Testing

### Unit Tests
```bash
python -m pytest app/tests/unit/ -v
```

### Integration Tests
```bash
python -m pytest app/tests/integration/ -v
```

### All Tests
```bash
python -m app.cli test
```

## 📊 Monitoring

### Pipeline Status
```bash
python -m app.cli status
```

### Real-time Monitoring
```bash
python -m app.cli monitor
```

### Performance Metrics
The pipeline automatically collects:
- Execution time per stage
- Memory usage
- Disk usage
- CPU utilization
- Error tracking
- Resource monitoring

## 🔧 CLI Commands

| Command | Description |
|---------|-------------|
| `run` | Execute the pipeline |
| `status` | Show pipeline status and available stages |
| `validate` | Validate configuration and environment |
| `test` | Run all tests |
| `monitor` | Show real-time monitoring |
| `init` | Initialize the pipeline environment |

### Command Options

- `--stages`: Comma-separated list of stages to run
- `--stage`: Single stage to run
- `--dry-run`: Run in dry-run mode (no actual processing)
- `--verbose`: Enable verbose logging
- `--config`: Specify configuration file path

## 🐳 Docker Support

The pipeline is designed to work with the existing Docker setup:

```bash
# Start MinIO and other services
docker-compose up -d

# Run the pipeline
python -m app.cli run
```

## 📈 Performance Considerations

### Batch Processing
- Documents are processed in configurable batches (default: 256)
- Images are processed individually with quality checks
- ChromaDB embeddings are batched for efficiency

### Memory Management
- Streaming processing for large files
- Multipart uploads for large outputs
- Memory monitoring and optimization

### Error Handling
- Comprehensive error handling at each stage
- Graceful degradation for optional dependencies
- Detailed error logging and reporting

## 🔍 Troubleshooting

### Common Issues

1. **MinIO Connection Failed**
   - Check MinIO endpoint and credentials
   - Ensure MinIO server is running
   - Verify network connectivity

2. **Hugging Face Authentication**
   - Verify HF_TOKEN is set correctly
   - Check dataset permissions
   - Ensure dataset exists and is accessible

3. **ChromaDB Issues**
   - Check CHROMA_PERSIST_DIR permissions
   - Ensure sufficient disk space
   - Verify embedding model availability

4. **Memory Issues**
   - Reduce batch_size in configuration
   - Monitor memory usage during execution
   - Consider processing smaller datasets

### Debug Mode

Enable debug logging for detailed troubleshooting:

```bash
python -m app.cli run --verbose
```

## 📝 Logging

The pipeline uses structured logging with multiple levels:

- **INFO**: General pipeline progress
- **WARNING**: Non-critical issues
- **ERROR**: Critical errors that may affect processing
- **DEBUG**: Detailed debugging information

Logs include:
- Timestamps
- Stage information
- Performance metrics
- Error details
- Resource usage

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs for error details
3. Validate your configuration
4. Check the test suite for examples

## 🔄 Continuous Integration

The pipeline is designed to work in CI/CD environments:

- Automated testing
- Configuration validation
- Performance monitoring
- Error reporting
- Artifact generation

## 📊 Metrics and Reporting

The pipeline generates comprehensive reports including:

- Processing statistics
- Quality metrics
- Performance data
- Error summaries
- Resource usage

Reports are saved in JSON format and can be integrated with monitoring systems.