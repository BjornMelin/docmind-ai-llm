# spaCy Model Setup Guide

This document provides comprehensive instructions for setting up spaCy language models with DocMind AI, including offline installation scenarios.

## Overview

DocMind AI uses spaCy for:

- Named Entity Recognition (NER)

- Linguistic analysis and text processing

- Knowledge graph entity extraction

- Document preprocessing

## Quick Setup (Recommended)

For most users, the standard installation is sufficient:

```bash

# Install the small English model (recommended)
uv run python -m spacy download en_core_web_sm
```

This downloads and installs the model to your local environment, making it available for offline use.

## Available Models

| Model | Size | Description | Use Case |
|-------|------|-------------|----------|
| `en_core_web_sm` | ~15MB | Small English model | General purpose, fast |
| `en_core_web_md` | ~50MB | Medium English model | Better accuracy |
| `en_core_web_lg` | ~560MB | Large English model | Highest accuracy |

### Installation Commands

```bash

# Small model (default for DocMind AI)
uv run python -m spacy download en_core_web_sm

# Medium model (better accuracy)
uv run python -m spacy download en_core_web_md

# Large model (best accuracy, slower)
uv run python -m spacy download en_core_web_lg
```

## Offline Installation

For environments without internet access, you can pre-download and install models:

### Step 1: Download Model Archive

On a machine with internet access, download the model:

```bash

# Download the model wheel file
uv run python -m spacy download en_core_web_sm --user

# Find the downloaded model location
uv run python -m spacy info en_core_web_sm
```

### Step 2: Locate Model Files

Find the installed model directory:

```bash

# In your virtual environment
find .venv -name "en_core_web_sm*" -type d
```

### Step 3: Transfer for Offline Installation

1. Copy the entire model directory to your offline environment
2. Place it in the same relative path in your offline environment's `.venv`
3. Verify installation:

```bash
uv run python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('Model loaded successfully')"
```

## Troubleshooting

### Model Not Found Error

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

### Download Failures

If model download fails:

1. **Check Network**: Ensure internet connectivity
2. **Use Alternative Installation**:

   ```bash
   # Alternative download method
   pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
   ```

3. **Manual Installation**: Download wheel from GitHub releases

### Permission Issues

On some systems, you may need additional permissions:

```bash

# Linux/macOS
sudo uv run python -m spacy download en_core_web_sm

# Windows (run as administrator)
uv run python -m spacy download en_core_web_sm
```

## Integration with DocMind AI

DocMind AI automatically handles spaCy model loading through the `ensure_spacy_model()` function in `src/utils/core.py`. This function:

1. **Attempts to load** the specified model (default: `en_core_web_sm`)
2. **Auto-downloads** if the model is not found locally
3. **Provides fallbacks** with proper error handling
4. **Uses `uv run`** when available for consistency with project tooling

### Configuration

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

## Security Considerations

The updated spaCy integration follows security best practices:

- ✅ **No custom URL dependencies**: Uses standard spaCy model distribution

- ✅ **Official sources only**: Downloads from spaCy's official repositories

- ✅ **Package manager integration**: Works with `uv` and standard Python packaging

- ✅ **No hardcoded credentials**: No API keys or authentication required

- ✅ **Local installation**: Models are cached locally for offline use

## Performance Impact

Model size affects both accuracy and performance:

- **Small model** (`en_core_web_sm`): Fastest loading and processing

- **Medium model** (`en_core_web_md`): Balanced accuracy/performance

- **Large model** (`en_core_web_lg`): Best accuracy, slower processing

For most document analysis tasks, the small model provides sufficient accuracy with optimal performance.

## Verification

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

## Getting Help

If you encounter issues:

1. Check the [spaCy documentation](https://spacy.io/usage/models)
2. Review the DocMind AI logs for detailed error messages
3. Ensure your Python environment matches the project requirements
4. For offline environments, verify model files are properly transferred

This setup ensures DocMind AI can perform linguistic analysis while maintaining security and offline functionality.
