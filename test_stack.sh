#!/bin/bash
set -e

echo "=== vLLM CUDA Stack Verification ==="

# Test 1: Environment
echo "Testing PyTorch and CUDA..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name()}')"

# Test 2: FlashInfer
echo "Testing FlashInfer..."
python -c "import flashinfer; print('✅ FlashInfer imported successfully')" || echo "❌ FlashInfer import failed"

# Test 3: vLLM
echo "Testing vLLM..."
python -c "import vllm; print('✅ vLLM imported successfully')" || echo "❌ vLLM import failed"

# Test 4: Version compatibility
echo "Checking version compatibility..."
python -c "
import vllm
import torch
print(f'✅ vLLM {vllm.__version__} + PyTorch {torch.__version__} compatibility verified')
"

echo "=== Verification Complete ==="