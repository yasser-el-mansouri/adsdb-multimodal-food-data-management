# ADSDB Multimodal Food Data Management Pipeline

This repository contains a comprehensive multimodal data pipeline for food recipe management. The system transforms raw data from external sources through multiple processing zones, ultimately creating a searchable knowledge base of recipes with images using ChromaDB.

## 🚀 Quick Start

### All Users

Run the pipeline with these commands:

```bash
# Using CLI (recommended)
python app/cli.py run

# Or using orchestrator directly (always works)
python app/orchestrate.py
```

### 🪟 Windows Users

**Important**: Windows PowerShell/CMD may have issues with emoji display in the CLI. Use these commands:

```bash
# Set UTF-8 encoding for proper display
chcp 65001

# Then run CLI commands normally
python app/cli.py run
```

## 🏗️ Architecture Overview

The pipeline follows a data lake architecture with four main zones:

1. **Landing Zone**: Raw data ingestion and initial organization
2. **Formatted Zone**: Data cleaning, joining, and standardization
3. **Trusted Zone**: Quality-controlled, filtered data ready for analysis
4. **Exploitation Zone**: Vector embeddings and search capabilities (documents + images)

## 📁 Project Structure

```
adsdb-multimodal-food-data-management/
├── app/                           # Main application code
│   ├── pipeline.yaml             # Centralized pipeline configuration
│   ├── requirements.txt           # Python dependencies
│   ├── cli.py                    # Command-line interface
│   ├── orchestrate.py            # Main pipeline orchestrator
│   ├── utils/                    # Shared utilities
│   │   ├── __init__.py           # Utility exports
│   │   ├── config.py             # Configuration management
│   │   ├── monitoring.py         # Performance monitoring
│   │   └── shared.py             # Shared utilities (S3Client, Logger, etc.)
│   └── zones/                    # Processing zones
│       ├── landing_zone/          # Raw data ingestion
│       │   ├── temporal_landing.py
│       │   └── persistent_landing.py
│       ├── formatted_zone/        # Data cleaning and joining
│       │   ├── formatted_documents.py
│       │   └── formatted_images.py
│       ├── trusted_zone/          # Quality control
│       │   ├── trusted_documents.py
│       │   └── trusted_images.py
│       └── exploitation_zone/     # Vector embeddings
│           ├── exploitation_documents.py
│           └── exploitation_images.py
├── notebooks/                     # Jupyter notebooks for exploration
│   ├── landing_zone/
│   ├── formatted_zone/
│   ├── trusted_zone/
│   └── exploitation_zone/
├── docker/                       # Docker configuration
│   ├── docker-compose.yml
│   ├── init-minio.sh
│   └── _minio_data/             # MinIO data directory
├── tests/                        # Test suite
│   ├── unit/
│   └── integration/
├── app/zones/landing_zone/       # Index files
│   ├── image_index.json
│   └── recipes_index.json
├── app/zones/trusted_zone/       # Recipe IDs with images
│   └── recipe_ids_with_images.json
├── app/zones/exploitation_zone/   # ChromaDB data
│   ├── chroma_documents/
│   └── chroma_images/
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** with pip
- **MinIO server** running locally or remotely
- **Hugging Face account** (for dataset access)
- **Git** for version control

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd adsdb-multimodal-food-data-management
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r app/requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp app/env.example .env
   # Edit .env with your actual values
   ```

4. **Initialize the pipeline**
   ```bash
   python app/cli.py init
   ```

5. **Validate configuration**
   ```bash
   python app/cli.py validate
   ```

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

```

## 🔄 Pipeline Stages

### 1. Temporal Landing (`temporal_landing`)
- **Purpose**: Ingest raw data from Hugging Face datasets
- **Input**: External Hugging Face dataset
- **Output**: Raw files in MinIO landing zone
- **Key Features**: 
  - Handles various file formats (JSON, JSONL, images)
  - Processes layer2.json for image URLs
  - Applies naming conventions with hashing
  - Creates image and recipe indexes

### 2. Persistent Landing (`persistent_landing`)
- **Purpose**: Organize raw data by type and apply naming conventions
- **Input**: Raw files from temporal landing
- **Output**: Organized files in persistent landing structure
- **Key Features**:
  - File type detection (images vs documents)
  - Standardized naming: `type$dataset$timestamp$name.ext`
  - Separates images and documents

### 3. Formatted Documents (`formatted_documents`)
- **Purpose**: Join all recipes and remove irrelevant data
- **Input**: Organized documents from persistent landing
- **Output**: Single JSONL file with joined recipes
- **Key Features**:
  - Merges recipes by ID with tagged field collisions
  - Removes specified fields (url, partition)
  - Handles multiple JSON formats (JSON, JSONL, arrays)

### 4. Formatted Images (`formatted_images`)
- **Purpose**: Process and organize images
- **Input**: Organized images from persistent landing
- **Output**: Processed images in formatted zone
- **Key Features**:
  - Quality screening (size, aspect ratio, blur detection)
  - Per-recipe deduplication using perceptual hashing
  - Image normalization to standard format

### 5. Trusted Images (`trusted_images`)
- **Purpose**: Extract recipe IDs and copy filtered images
- **Input**: Processed images from formatted zone
- **Output**: Quality-controlled images in trusted zone
- **Key Features**:
  - Recipe ID extraction from filenames
  - Quality filtering and deduplication
  - Generates recipe IDs file for document processing
  - Image normalization to 512x512 JPEG

### 6. Trusted Documents (`trusted_documents`)
- **Purpose**: Filter documents to keep only those with images
- **Input**: Joined documents from formatted zone
- **Output**: Filtered documents in trusted zone
- **Key Features**:
  - Filters by recipe IDs that have images
  - Applies nutrition quality controls (IQR outlier detection)
  - Text cleaning and normalization
  - Removes duplicated nutrition data

### 7. Exploitation Documents (`exploitation_documents`)
- **Purpose**: Generate embeddings and store in ChromaDB
- **Input**: Filtered documents from trusted zone
- **Output**: Vector embeddings in ChromaDB
- **Key Features**:
  - Uses SentenceTransformer for embeddings
  - Batch processing for efficiency
  - Metadata extraction (FSA nutrition lights)
  - Semantic search capabilities

### 8. Exploitation Images (`exploitation_images`)
- **Purpose**: Generate image embeddings and store in ChromaDB
- **Input**: Quality-controlled images from trusted zone
- **Output**: Vector embeddings in ChromaDB
- **Key Features**:
  - Uses OpenCLIP for image embeddings
  - Cross-modal search capabilities
  - Batch processing for efficiency
  - Image metadata extraction

## 🖥️ Running the Pipeline

### All Users

The pipeline can be run using the CLI or orchestrator directly:

```bash
# Using CLI (recommended)
python app/cli.py run

# Or using orchestrator directly (always works)
python app/orchestrate.py
```

### Windows Users (PowerShell/CMD)
**Important**: Windows terminals may have issues with emoji display. Use these commands:

```bash
# Set UTF-8 encoding for proper display
chcp 65001

# Then run CLI commands
python app/cli.py run
```

### CLI Commands

#### Full Pipeline
```bash
# Using CLI (recommended)
python app/cli.py run

# Or using orchestrator directly (always works)
python app/orchestrate.py
```

#### Specific Zones
```bash
# Run only the landing zone
python app/cli.py run --stages temporal_landing,persistent_landing

# Run only the formatted zone
python app/cli.py run --stages formatted_documents,formatted_images

# Run only the trusted zone
python app/cli.py run --stages trusted_images,trusted_documents

# Run only the exploitation zone
python app/cli.py run --stages exploitation_documents,exploitation_images

# Run from formatted zone onwards (faster for development)
python app/cli.py run --stages formatted_documents,formatted_images,trusted_images,trusted_documents,exploitation_documents,exploitation_images
```

#### Single Stages
```bash
# Run a single stage
python app/cli.py run --stage trusted_images
python app/cli.py run --stage exploitation_documents
python app/cli.py run --stage exploitation_images
```

#### Pipeline Options
```bash
# Dry run mode (no actual processing)
python app/cli.py run --dry-run

# Verbose logging
python app/cli.py run --verbose

# Custom configuration file
python app/cli.py run --config custom_pipeline.yaml
```

### CLI Command Reference

| Command | Description |
|---------|-------------|
| `run` | Execute the pipeline |
| `status` | Show pipeline status and available stages |
| `validate` | Validate configuration and environment |
| `test` | Run all tests |
| `metrics` | Show real-time system metrics |
| `report` | Display latest pipeline execution report |
| `init` | Initialize the pipeline environment |

### Command Options

- `--stages`: Comma-separated list of stages to run
- `--stage`: Single stage to run
- `--dry-run`: Run in dry-run mode (no actual processing)
- `--verbose`: Enable verbose logging
- `--config`: Specify configuration file path

## ⚙️ Configuration

### Pipeline Configuration (pipeline.yaml)

The centralized configuration file contains all non-sensitive settings:

```yaml
# Storage configuration
storage:
  buckets:
    landing_zone: "landing-zone"
    formatted_zone: "formatted-zone"
    trusted_zone: "trusted-zone"
  prefixes:
    temporal_landing: "temporal_landing"
    persistent_landing: "persistent_landing"
    formatted_documents: "formatted/documents"
    formatted_images: "formatted/images"
    trusted_documents: "trusted/documents"
    trusted_images: "trusted/images"

# ChromaDB configuration
chromadb_documents:
  collection_name: "exploitation_documents"
  embedding_model: "Qwen/Qwen3-Embedding-0.6B"
  persist_dir: "app/zones/exploitation_zone/chroma_documents"

chromadb_images:
  collection_name: "exploitation_images"
  embedding_model: "OpenCLIP"
  persist_dir: "app/zones/exploitation_zone/chroma_images"

# Pipeline settings
pipeline:
  batch_size: 256
  timeout: 3600
  dry_run: false
  overwrite: true
```

## 🐳 Docker Support

The pipeline works with Docker for easy deployment:

```bash
# Start MinIO and other services
cd docker
docker-compose up -d

# Run the pipeline
python app/cli.py run
```

### Docker Services
- **MinIO**: S3-compatible object storage
- **Data persistence**: All data stored in `docker/_minio_data/`

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
python app/cli.py test
```

### Test Coverage
```bash
python -m pytest --cov=app app/tests/ --cov-report=html
```

## 📊 Monitoring and Reporting

### Real-time Monitoring
```bash
# Show system metrics
python app/cli.py metrics

# Show latest execution report
python app/cli.py report
```

### Performance Metrics
The pipeline automatically collects:
- Execution time per stage
- Memory usage
- Disk usage
- CPU utilization
- Error tracking
- Resource monitoring

### Quality Reports

#### Code Quality Report
```bash
# Generate code quality report using pylint
pylint --output-format=text app/ > quality_report.txt

# Or with HTML output
pylint --output-format=html app/ > quality_report.html
```

#### Pipeline Execution Reports
Reports are automatically generated in JSON format:
- `pipeline_metrics_YYYYMMDD_HHMMSS.json`
- Contains detailed execution statistics
- Performance metrics per stage
- Error summaries

## 📓 Jupyter Notebooks

The `notebooks/` directory contains exploration notebooks for each zone:

```bash
# Start Jupyter Lab
jupyter lab

# Or Jupyter Notebook
jupyter notebook
```

### Notebook Structure
- `landing_zone/`: Raw data exploration
- `formatted_zone/`: Data cleaning and joining
- `trusted_zone/`: Quality control analysis
- `exploitation_zone/`: Vector embeddings and search

## 📝 Logging

The pipeline uses structured logging with multiple levels:

- **INFO**: General pipeline progress
- **WARNING**: Non-critical issues
- **ERROR**: Critical errors that may affect processing
- **DEBUG**: Detailed debugging information

### Log Configuration
```bash
# Set log level via environment
export PIPELINE_LOG_LEVEL=DEBUG

# Or via CLI
python app/cli.py run --verbose
```

### Log Files
- Console output with timestamps
- Structured JSON logs for monitoring
- Error tracking and reporting
- Performance metrics logging

## 🔍 Troubleshooting

### Common Issues

1. **MinIO Connection Failed**
   ```bash
   # Check MinIO status
   docker ps | grep minio
   
   # Test connection
   python -c "from app.utils.shared import S3Client; S3Client(PipelineConfig()).test_connection()"
   ```

2. **Hugging Face Authentication**
   ```bash
   # Verify token
   python -c "import os; print('HF_TOKEN set:', bool(os.getenv('HF_TOKEN')))"
   ```

3. **ChromaDB Issues**
   ```bash
   # Check directory permissions
   ls -la app/zones/exploitation_zone/
   
   # Test ChromaDB
   python -c "import chromadb; print('ChromaDB version:', chromadb.__version__)"
   ```

4. **Memory Issues**
   ```bash
   # Monitor memory usage
   python app/cli.py metrics
   
   # Reduce batch size in pipeline.yaml
   ```

5. **Import Errors**
   ```bash
   # Check Python path
   python -c "import sys; print('Python path:', sys.path)"
   
   # Test imports
   python -c "from app.utils.config import PipelineConfig; print('Config OK')"
   ```

### Debug Mode

Enable debug logging for detailed troubleshooting:

```bash
# Set debug level
export PIPELINE_LOG_LEVEL=DEBUG

# Run with verbose output
python app/cli.py run --verbose
```

### Windows-Specific Issues

1. **PowerShell Execution Policy**
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

2. **Path Issues**
   ```bash
   # Use forward slashes in paths
   python app/cli.py run --config app/pipeline.yaml
   ```

3. **Encoding Issues**
   ```bash
   # Set UTF-8 encoding
   chcp 65001
   ```

## 📈 Performance Considerations

### Batch Processing
- Documents: 256 records per batch (configurable)
- Images: Individual processing with quality checks
- ChromaDB embeddings: Batched for efficiency

### Memory Management
- Streaming processing for large files
- Multipart uploads for large outputs
- Memory monitoring and optimization

### Error Handling
- Comprehensive error handling at each stage
- Graceful degradation for optional dependencies
- Detailed error logging and reporting

## 🔄 Continuous Integration

The pipeline is designed to work in CI/CD environments:

- Automated testing with pytest
- Configuration validation
- Performance monitoring
- Error reporting
- Quality reports generation

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs for error details
3. Validate your configuration with `python app/cli.py validate`
4. Check the test suite for examples
5. Generate quality reports for code analysis

## 📊 Metrics and Reporting

The pipeline generates comprehensive reports including:

- Processing statistics per stage
- Quality metrics (images processed, documents filtered)
- Performance data (execution time, memory usage)
- Error summaries and debugging information
- Resource usage monitoring
- Code quality analysis

Reports are saved in JSON format and can be integrated with monitoring systems like Grafana, Prometheus, or custom dashboards.

## 🔧 Development

### Adding New Stages
1. Create processor class in appropriate zone folder
2. Implement `process()` method
3. Add to orchestrator stages list
4. Update configuration as needed
5. Add tests

### Configuration Management
- All configuration in `app/pipeline.yaml`
- Environment variables for sensitive data
- Validation with `python app/cli.py validate`

### Code Quality
- Follow PEP 8 style guidelines
- Use type hints
- Comprehensive error handling
- Unit and integration tests
- Regular quality reports with pylint