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

# Set default port from unified settings env; fall back to 8501
# Note: prefer nested settings key DOCMIND_UI__STREAMLIT_PORT
PORT=${DOCMIND_UI__STREAMLIT_PORT:-8501}

echo "📋 Configuration:"
echo "   Port: $PORT"
echo "   Python: $(uv run python --version)"
echo ""

# Run the application with port configuration
echo "✅ Launching DocMind AI on http://localhost:$PORT"
uv run streamlit run app.py --server.port "$PORT"
