#!/bin/bash
# Run DocMind AI application with enhanced error checking and configuration

set -e  # Exit on any error

echo "🚀 Starting DocMind AI..."

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv not found. Please install uv first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check if entrypoint exists
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found. Please run from project root directory."
    exit 1
fi

echo "📋 Configuration:"
echo "   Python: $(uv run python --version)"
echo ""

# Streamlit owns server configuration through .streamlit/config.toml, native
# STREAMLIT_* environment variables, and CLI flags.
echo "✅ Launching DocMind AI"
uv run streamlit run app.py
