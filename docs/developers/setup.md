# Developer Setup for DocMind AI

## Prerequisites

- Python 3.11+ (tested with 3.11, 3.12).
- uv for package management.
- Git.
- Optional: Docker, NVIDIA CUDA 12.8+ for GPU testing with RTX 4090.

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/BjornMelin/docmind-ai-llm.git
cd docmind-ai-llm
```

### 2. Create Virtual Environment

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

### 3. GPU Development Setup (RTX 4090 with vLLM FlashInfer)

```bash
# Install PyTorch 2.7.1 with CUDA 12.8
uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --extra-index-url https://download.pytorch.org/whl/cu128

# Install vLLM with FlashInfer
uv pip install "vllm[flashinfer]>=0.10.1" \
    --extra-index-url https://download.pytorch.org/whl/cu128

# Install GPU extras
uv sync --extra gpu
```

### 4. Testing Dependencies

```bash
uv sync --extra test
```

### 5. spaCy Model Installation

DocMind AI uses spaCy for Named Entity Recognition (NER), linguistic analysis, knowledge graph entity extraction, and document preprocessing.

#### Quick Setup (Recommended)

```bash
# Install the small English model (recommended)
uv run python -m spacy download en_core_web_sm
```

#### Available Models

| Model | Size | Description | Use Case |
|-------|------|-------------|----------|
| `en_core_web_sm` | ~15MB | Small English model | General purpose, fast |
| `en_core_web_md` | ~50MB | Medium English model | Better accuracy |
| `en_core_web_lg` | ~560MB | Large English model | Highest accuracy |

#### Installation Commands

```bash
# Small model (default for DocMind AI)
uv run python -m spacy download en_core_web_sm

# Medium model (better accuracy)
uv run python -m spacy download en_core_web_md

# Large model (best accuracy, slower)
uv run python -m spacy download en_core_web_lg
```

## Running Locally

```bash
streamlit run src/app.py
```

## Testing

```bash
pytest
```

## Linting/Formatting

```bash
ruff check .
ruff format .
```

## Environment Variables

See `.env.example` for configs like OLLAMA_BASE_URL.

## spaCy Setup Details

### Offline Installation

For environments without internet access, you can pre-download and install models:

#### Step 1: Download Model Archive

On a machine with internet access, download the model:

```bash
# Download the model wheel file
uv run python -m spacy download en_core_web_sm --user

# Find the downloaded model location
uv run python -m spacy info en_core_web_sm
```

#### Step 2: Locate Model Files

Find the installed model directory:

```bash
# In your virtual environment
find .venv -name "en_core_web_sm*" -type d
```

#### Step 3: Transfer for Offline Installation

1. Copy the entire model directory to your offline environment
2. Place it in the same relative path in your offline environment's `.venv`
3. Verify installation:

```bash
uv run python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('Model loaded successfully')"
```

### Troubleshooting spaCy Issues

#### Model Not Found Error

If you encounter `OSError: [E050] Can't find model 'en_core_web_sm'`:

1. **Check Installation**: Verify the model is installed

   ```bash
   uv run python -m spacy info en_core_web_sm
   ```

2. **Reinstall Model**: Download and install again

   ```bash
   uv run python -m spacy download en_core_web_sm --force-reinstall
   ```

3. **Check Virtual Environment**: Ensure you're using the correct environment

   ```bash
   which python
   uv run python -c "import spacy; print(spacy.__file__)"
   ```

#### Download Failures

If model download fails:

1. **Check Network**: Ensure internet connectivity
2. **Use Alternative Installation**:

   ```bash
   # Alternative download method
   pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
   ```

3. **Manual Installation**: Download wheel from GitHub releases

#### Permission Issues

On some systems, you may need additional permissions:

```bash
# Linux/macOS
sudo uv run python -m spacy download en_core_web_sm

# Windows (run as administrator)
uv run python -m spacy download en_core_web_sm
```

### Integration with DocMind AI

DocMind AI automatically handles spaCy model loading through the `ensure_spacy_model()` function in `src/utils/core.py`. This function:

1. **Attempts to load** the specified model (default: `en_core_web_sm`)
2. **Auto-downloads** if the model is not found locally
3. **Provides fallbacks** with proper error handling
4. **Uses `uv run`** when available for consistency with project tooling

#### Configuration

You can configure which spaCy model to use by modifying the application settings or by calling the function with a different model name:

```python
from src.utils.core import ensure_spacy_model

# Use small model (default)
nlp = ensure_spacy_model("en_core_web_sm")

# Use medium model for better accuracy
nlp = ensure_spacy_model("en_core_web_md")

# Use large model for best accuracy
nlp = ensure_spacy_model("en_core_web_lg")
```

### Security Considerations

The updated spaCy integration follows security best practices:

- ✅ **No custom URL dependencies**: Uses standard spaCy model distribution
- ✅ **Official sources only**: Downloads from spaCy's official repositories
- ✅ **Package manager integration**: Works with `uv` and standard Python packaging
- ✅ **No hardcoded credentials**: No API keys or authentication required
- ✅ **Local installation**: Models are cached locally for offline use

### Performance Impact

Model size affects both accuracy and performance:

- **Small model** (`en_core_web_sm`): Fastest loading and processing
- **Medium model** (`en_core_web_md`): Balanced accuracy/performance
- **Large model** (`en_core_web_lg`): Best accuracy, slower processing

For most document analysis tasks, the small model provides sufficient accuracy with optimal performance.

## Verification

### spaCy Setup Verification

To verify your spaCy setup is working correctly:

```bash
# Test model loading and basic functionality
uv run python -c "
from src.utils.core import ensure_spacy_model

# Load and test the model
nlp = ensure_spacy_model('en_core_web_sm')
if nlp:
    doc = nlp('Apple Inc. is a technology company.')
    print('✅ spaCy is working correctly')
    print(f'Entities found: {[(ent.text, ent.label_) for ent in doc.ents]}')
else:
    print('❌ spaCy model loading failed')
"
```

Expected output:

```text
✅ spaCy is working correctly
Entities found: [('Apple Inc.', 'ORG')]
```

### Complete Setup Verification

To verify your complete development setup:

```bash
# 1. Check Python environment
python --version
which python

# 2. Check package installation
uv run python -c "import streamlit, spacy, torch; print('✅ All packages imported successfully')"

# 3. Check GPU support (if applicable)
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 4. Run basic tests
pytest tests/ -v --tb=short
```

## Development Workflow

For contributions, follow these steps:

1. **Set up environment** as described above
2. **Create feature branch**: `git checkout -b feat/your-feature`
3. **Make changes** following the coding standards in [Development Guide](development-guide.md)
4. **Run tests**: `pytest`
5. **Run linting**: `ruff check . && ruff format .`
6. **Commit changes**: Use clear, descriptive commit messages
7. **Create pull request**: Follow [CONTRIBUTING.md](../../CONTRIBUTING.md)

## Getting Help

If you encounter issues during setup:

1. **spaCy issues**: Check the [spaCy documentation](https://spacy.io/usage/models)
2. **GPU issues**: See [GPU and Performance Guide](gpu-and-performance.md)
3. **General setup**: Review the DocMind AI logs for detailed error messages
4. **Contributing**: Follow [Development Guide](development-guide.md)
5. **Project issues**: Create an issue on GitHub with setup details

For additional technical documentation, see:

- [Architecture Overview](architecture.md) - System design and components
- [GPU and Performance](gpu-and-performance.md) - Hardware optimization
- [Model Configuration](model-configuration.md) - AI model setup
- [Development Guide](development-guide.md) - Development practices
