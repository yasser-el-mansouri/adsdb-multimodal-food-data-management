# Multimodal Tasks Zone

This zone handles multimodal tasks such as retrieval, querying, and analysis on both text and image data using ChromaDB vector stores and generative AI models.

## Tasks

### Task 1: Multimodal Retrieval (`task1_retrieval.py`)

Performs multimodal retrieval operations on recipe data:
- **Text-to-text search**: Find similar recipes based on text queries
- **Image-to-image search**: Find visually similar recipe images

**Configuration** (`pipeline.yaml`):
```yaml
multimodal_tasks:
  text_chroma_persist_dir: 'app/zones/exploitation_zone/chroma_documents'
  image_chroma_persist_dir: 'app/zones/exploitation_zone/chroma_images'
  text_collection_name: 'exploitation_documents'
  image_collection_name: 'exploitation_images'
  text_embedding_model: 'Qwen/Qwen3-Embedding-0.6B'
  image_embedding_model: 'ViT-B-32'
  default_k: 5
```

**Usage**:
```bash
# Via CLI
python -m app.cli run --stage task1_retrieval

# Or directly
python -m app.zones.multimodal_tasks.task1_retrieval
```

**Requirements**:
- ChromaDB collections must be created by running `exploitation_documents` and `exploitation_images` stages first
- No external services required

---

### Task 2: Multimodal Search (`task2.py`)

Performs unified multimodal search operations on a combined collection containing both text and image data:
- **Text-based search**: Query with text to find relevant recipes AND images
- **Image-based search**: Query with an image to find similar images AND relevant recipes

Unlike Task 1 which searches separate collections, Task 2 uses a single multimodal collection where text and images are indexed together, enabling cross-modal retrieval.

**Configuration** (`pipeline.yaml`):
```yaml
chromadb_multimodal:
  collection_name: 'exploitation_multimodal'
  embedding_model: 'OpenCLIP'
  metadata: {}
  persist_dir: 'app/zones/exploitation_zone/chroma_exploitation'
```

**Usage**:
```bash
# Via CLI
python -m app.cli run --stage task2_multimodal_search

# Or directly
python -m app.zones.multimodal_tasks.task2
```

**Example Queries**:
- Text query: "fettuccine alfredo pasta dish with creamy sauce"
- Image query: Using `calico-beans.jpg` as the query image

**Requirements**:
- ChromaDB multimodal collection must be populated with both text and image embeddings
- Query images should be available (e.g., `calico-beans.jpg`, `fettuccine-alfredo.jpg`)
- No external services required (uses OpenCLIP embeddings)

**Output**:
The processor returns distance statistics for both image and text matches:
- Closest/farthest image match distances
- Closest/farthest recipe text match distances

This allows you to see how well text and images are aligned in the embedding space and which modality provides better matches for a given query.

---

### Task 3: Multimodal RAG with LLaVA (`task3_rag.py`)

Implements Retrieval-Augmented Generation (RAG) using:
1. **Text retrieval** from recipe documents
2. **Image retrieval** from recipe images  
3. **LLaVA** (Large Language and Vision Assistant) for multimodal generation via Ollama

**Configuration** (`pipeline.yaml`):
```yaml
multimodal_tasks:
  # ... (same as above) ...
  max_retrieved_images: 3
  ollama_host: 'http://localhost:11434'
  ollama_model: 'llava'
```

**Usage**:
```bash
# IMPORTANT: Start Ollama first!
# Terminal 1: Start Ollama server
ollama serve

# Terminal 2: Pull LLaVA model (if not already installed)
ollama pull llava

# Terminal 3: Run the task
python -m app.cli run --stage task3_rag
```

**Requirements**:
- ✅ ChromaDB collections (from exploitation stages)
- ✅ **Ollama server must be running** (`ollama serve`)
- ✅ **LLaVA model must be installed** (`ollama pull llava`)

**Error Handling**:

The task includes comprehensive error handling for Ollama connectivity:

1. **Connectivity check at initialization**: Verifies Ollama is accessible before processing
2. **Clear error messages**: Provides actionable instructions when Ollama is not available
3. **Model availability check**: Detects when Ollama is running but LLaVA isn't installed

**Example Error Messages**:

```
❌ Ollama not running:
======================================================================
ERROR: Cannot connect to Ollama server at http://localhost:11434
======================================================================
Ollama must be running before executing task3_rag.

To start Ollama:
  1. Open a new terminal
  2. Run: ollama serve

To install the LLaVA model (if not already installed):
  ollama pull llava
======================================================================

❌ LLaVA model not installed:
======================================================================
ERROR: LLaVA model not found (404)
======================================================================
The Ollama server is running, but the model 'llava' is not available.

To install the model:
  ollama pull llava

To list installed models:
  ollama list
======================================================================
```

---

## Architecture

Both tasks follow the same processor pattern:

```python
class TaskProcessor:
    def __init__(self, config: PipelineConfig)
        # Initialize ChromaDB clients, embedding models, etc.
    
    def process(self) -> Dict[str, Any]:
        # Main processing logic
        # Returns metrics dictionary
```

### Text Embeddings
- Model: `Qwen/Qwen3-Embedding-0.6B` (SentenceTransformer)
- Used for text-to-text retrieval

### Image Embeddings
- Model: `ViT-B-32` with `laion2b_s34b_b79k` pretrained weights (OpenCLIP)
- Used for image-to-image and text-to-image retrieval

### Vector Store
- **ChromaDB** (persistent storage)
- Separate collections for text documents and images
- Configured via `text_chroma_persist_dir` and `image_chroma_persist_dir`

---

## Testing

Comprehensive tests are available:

### Unit Tests
```bash
# Task 1
pytest app/tests/unit/test_task1_retrieval.py -v

# Task 2
pytest app/tests/unit/test_task2_multimodal_search.py -v

# Task 3
pytest app/tests/unit/test_task3_rag.py -v
```

### Integration Tests
```bash
pytest app/tests/integration/test_pipeline.py::TestPipelineIntegration::test_task1_retrieval_processor_initialization -v
pytest app/tests/integration/test_pipeline.py::TestPipelineIntegration::test_task2_multimodal_search_processor_initialization -v
pytest app/tests/integration/test_pipeline.py::TestPipelineIntegration::test_task3_rag_processor_initialization -v
```

All tests use mocking to avoid dependencies on external services and actual data.

---

## Pipeline Integration

Both tasks are integrated into the main pipeline orchestrator:

```python
# In app/orchestrate.py
self.stages = [
    # ... other stages ...
    ("task1_retrieval", Task1RetrievalProcessor),
    ("task2_multimodal_search", ExploitationMultiModalSearcher),
    ("task3_rag", Task3RAGProcessor),
]
```

View all available stages:
```bash
python -m app.cli status
```

---

## Development Notes

### Adding New Multimodal Tasks

1. Create a new processor file: `app/zones/multimodal_tasks/taskN_*.py`
2. Implement the processor class following the pattern
3. Add configuration to `pipeline.yaml` under `multimodal_tasks`
4. Import and register in `app/orchestrate.py`
5. Add to CLI status in `app/cli.py`
6. Create unit tests in `app/tests/unit/test_taskN_*.py`
7. Add integration test in `app/tests/integration/test_pipeline.py`

### External Service Dependencies

If your task requires external services (like Ollama for Task 3):
- Add connectivity checks in `__init__` method
- Provide clear, actionable error messages
- Document requirements in this README
- Add warnings to CLI (`app/cli.py` run command)
- Update tests to mock external service calls

---

## Troubleshooting

### Task 1 Issues

**Q: "Collection not found" error**  
A: Run the exploitation stages first:
```bash
python -m app.cli run --stage exploitation_documents
python -m app.cli run --stage exploitation_images
```

**Q: Empty results**  
A: Check that embeddings were generated successfully. Verify ChromaDB directory exists and has data.

### Task 2 Issues

**Q: "Collection has 0 entries" or empty results**  
A: Run the exploitation stages first to populate the multimodal collection:
```bash
python -m app.cli run --stages exploitation_documents,exploitation_images
```

**Q: "No text results found" in image search**  
A: This is expected if `n_results` is too small. The task is configured to retrieve all 127 items (92 images + 35 texts) by default.

**Q: Image file not found**  
A: Ensure query images (e.g., `calico-beans.jpg`) are available in the correct locations:
- `notebooks/exploitation_zone/tasks/`
- Current working directory
- Or provide absolute paths

### Task 3 Issues

**Q: "Connection refused" error**  
A: Start Ollama server:
```bash
ollama serve
```

**Q: "404 Not Found" error**  
A: Install the LLaVA model:
```bash
ollama pull llava
```

**Q: Slow generation**  
A: LLaVA inference can be slow on CPU. Consider:
- Using GPU if available
- Reducing `max_retrieved_images` in config
- Increasing timeout in code

**Q: Missing images in MinIO**  
A: Some image references in ChromaDB might point to non-existent MinIO objects. This is logged as a warning and the task continues with available images.

---

## Related Documentation

- [Testing Guide](TESTING.md) - Comprehensive testing documentation
- [Pipeline Configuration](../../pipeline.yaml) - Full configuration reference
- [Main README](../../../README.md) - Project overview

