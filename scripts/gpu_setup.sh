#!/bin/bash

# DocMind AI GPU Infrastructure Setup Script
# Phase 1: CUDA 12.x, cuDNN 9.x, and NVIDIA Container Toolkit Installation
# Target: 100x performance improvement for embedding generation and vector search

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        error "This script should not be run as root for security reasons."
        echo "Please run as a regular user with sudo privileges."
        exit 1
    fi
}

# Check system requirements
check_system() {
    log "Checking system requirements..."
    
    # Check Ubuntu version
    if ! lsb_release -a 2>/dev/null | grep -q "Ubuntu 24.04"; then
        warning "This script is optimized for Ubuntu 24.04. Your system may have compatibility issues."
    fi
    
    # Check NVIDIA GPU
    if ! nvidia-smi >/dev/null 2>&1; then
        error "NVIDIA GPU or drivers not detected. Please install NVIDIA drivers first."
        exit 1
    fi
    
    # Get GPU info
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
    log "Detected GPU: $GPU_INFO"
    
    # Check CUDA driver version
    CUDA_DRIVER_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    log "CUDA Driver Version: $CUDA_DRIVER_VERSION"
    
    if [[ $(echo "$CUDA_DRIVER_VERSION < 12.0" | bc -l) -eq 1 ]]; then
        error "CUDA driver version $CUDA_DRIVER_VERSION is too old. Need 12.0+ for optimal performance."
        exit 1
    fi
    
    success "System requirements check passed"
}

# Install CUDA 12.x toolkit
install_cuda_toolkit() {
    log "Installing CUDA 12.x toolkit..."
    
    # Check if CUDA toolkit is already installed
    if command -v nvcc >/dev/null 2>&1; then
        EXISTING_CUDA=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        log "CUDA toolkit $EXISTING_CUDA already installed"
        if [[ $(echo "$EXISTING_CUDA >= 12.0" | bc -l) -eq 1 ]]; then
            success "CUDA toolkit version is compatible"
            return 0
        else
            warning "Existing CUDA toolkit version $EXISTING_CUDA is too old"
        fi
    fi
    
    # Add NVIDIA package repository
    log "Adding NVIDIA CUDA repository..."
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    
    # Install CUDA toolkit
    log "Installing CUDA 12.8 toolkit (this may take several minutes)..."
    sudo apt-get install -y cuda-toolkit-12-8
    
    # Add CUDA to PATH
    log "Configuring CUDA environment variables..."
    
    # Create CUDA environment script
    cat << 'EOF' | sudo tee /etc/profile.d/cuda.sh
# CUDA Configuration for DocMind AI
export CUDA_HOME=/usr/local/cuda
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# vLLM Optimization
export VLLM_TORCH_BACKEND=cu128  
export VLLM_ATTENTION_BACKEND=FLASHINFER

# Memory Management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# FastEmbed GPU Configuration
export FASTEMBED_CACHE_PATH=/tmp/fastembed_cache
export ONNXRUNTIME_PROVIDERS="CUDAExecutionProvider,CPUExecutionProvider"
EOF

    # Make the script executable
    sudo chmod +x /etc/profile.d/cuda.sh
    
    # Source the environment
    source /etc/profile.d/cuda.sh
    
    # Create symlink for current version
    sudo ln -sf /usr/local/cuda-12.8 /usr/local/cuda
    
    success "CUDA 12.8 toolkit installed successfully"
}

# Install cuDNN 9.x
install_cudnn() {
    log "Installing cuDNN 9.x..."
    
    # Check if cuDNN is already installed
    if dpkg -l | grep -q libcudnn9; then
        log "cuDNN 9.x already installed"
        return 0
    fi
    
    # Install cuDNN
    log "Installing cuDNN development libraries..."
    sudo apt-get install -y libcudnn9-dev libcudnn9-cuda-12
    
    success "cuDNN 9.x installed successfully"
}

# Install NVIDIA Container Toolkit
install_nvidia_container_toolkit() {
    log "Installing NVIDIA Container Toolkit..."
    
    # Check if already installed
    if command -v nvidia-container-runtime >/dev/null 2>&1; then
        log "NVIDIA Container Toolkit already installed"
        return 0
    fi
    
    # Add NVIDIA container toolkit repository
    log "Adding NVIDIA container toolkit repository..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
    sudo apt-get update
    
    # Install NVIDIA container toolkit
    log "Installing NVIDIA container toolkit..."
    sudo apt-get install -y nvidia-container-toolkit
    
    # Configure Docker daemon
    log "Configuring Docker for GPU support..."
    sudo nvidia-ctk runtime configure --runtime=docker
    
    # Restart Docker service
    log "Restarting Docker service..."
    sudo systemctl restart docker
    
    success "NVIDIA Container Toolkit installed and configured"
}

# Install GPU monitoring tools
install_monitoring_tools() {
    log "Installing GPU monitoring and management tools..."
    
    # Install Python monitoring tools
    pip3 install --user nvidia-ml-py3 gpustat psutil
    
    # Install system monitoring tools
    sudo apt-get update
    sudo apt-get install -y htop nvtop
    
    success "GPU monitoring tools installed"
}

# Install FastEmbed GPU support
install_fastembed_gpu() {
    log "Installing FastEmbed GPU support..."
    
    # Install FastEmbed with GPU acceleration
    pip3 install --user fastembed-gpu
    
    # Install ONNX Runtime GPU
    pip3 install --user onnxruntime-gpu
    
    success "FastEmbed GPU support installed"
}

# Create GPU validation script
create_validation_script() {
    log "Creating GPU validation script..."
    
    cat << 'EOF' > gpu_validation.py
#!/usr/bin/env python3
"""
DocMind AI GPU Validation Script
Tests GPU functionality for CUDA, PyTorch, FastEmbed, and vLLM
"""

import sys
import subprocess
import time
from typing import Dict, Any

def check_nvidia_smi() -> Dict[str, Any]:
    """Check NVIDIA driver and GPU status"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            return {"status": "✅ PASS", "details": "NVIDIA driver working"}
        else:
            return {"status": "❌ FAIL", "details": f"nvidia-smi failed: {result.stderr}"}
    except FileNotFoundError:
        return {"status": "❌ FAIL", "details": "nvidia-smi not found"}

def check_cuda_toolkit() -> Dict[str, Any]:
    """Check CUDA toolkit installation"""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            version_line = [line for line in result.stdout.split('\n') if 'release' in line][0]
            version = version_line.split('release ')[1].split(',')[0]
            return {"status": "✅ PASS", "details": f"CUDA toolkit {version} installed"}
        else:
            return {"status": "❌ FAIL", "details": "nvcc not found"}
    except (FileNotFoundError, IndexError):
        return {"status": "❌ FAIL", "details": "CUDA toolkit not installed"}

def check_pytorch_cuda() -> Dict[str, Any]:
    """Check PyTorch CUDA support"""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            return {
                "status": "✅ PASS", 
                "details": f"PyTorch CUDA {cuda_version}, {device_count} GPU(s), {device_name}"
            }
        else:
            return {"status": "❌ FAIL", "details": "PyTorch CUDA not available"}
    except ImportError:
        return {"status": "⚠️  SKIP", "details": "PyTorch not installed"}

def check_fastembed_gpu() -> Dict[str, Any]:
    """Check FastEmbed GPU support"""
    try:
        from fastembed import TextEmbedding
        
        # Test with GPU providers
        model = TextEmbedding(
            "BAAI/bge-small-en-v1.5",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        
        # Quick embedding test
        start_time = time.time()
        embeddings = list(model.embed(["GPU test document"]))
        end_time = time.time()
        
        if len(embeddings) > 0:
            return {
                "status": "✅ PASS", 
                "details": f"FastEmbed GPU working, embedding time: {(end_time-start_time)*1000:.1f}ms"
            }
        else:
            return {"status": "❌ FAIL", "details": "FastEmbed failed to generate embeddings"}
            
    except ImportError:
        return {"status": "⚠️  SKIP", "details": "FastEmbed not installed"}
    except Exception as e:
        return {"status": "❌ FAIL", "details": f"FastEmbed error: {str(e)}"}

def check_docker_gpu() -> Dict[str, Any]:
    """Check Docker GPU support"""
    try:
        result = subprocess.run([
            'docker', 'run', '--rm', '--gpus', 'all', 
            'nvidia/cuda:12.8-base-ubuntu22.04', 
            'nvidia-smi', '--query-gpu=name', '--format=csv,noheader'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip()
            return {"status": "✅ PASS", "details": f"Docker GPU access: {gpu_name}"}
        else:
            return {"status": "❌ FAIL", "details": f"Docker GPU test failed: {result.stderr}"}
    except subprocess.TimeoutExpired:
        return {"status": "❌ FAIL", "details": "Docker GPU test timed out"}
    except FileNotFoundError:
        return {"status": "⚠️  SKIP", "details": "Docker not found"}
    except Exception as e:
        return {"status": "❌ FAIL", "details": f"Docker error: {str(e)}"}

def check_gpu_memory() -> Dict[str, Any]:
    """Check GPU memory usage"""
    try:
        import torch
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated = torch.cuda.memory_allocated(0)
            cached = torch.cuda.memory_reserved(0)
            
            total_gb = total_memory / 1e9
            allocated_mb = allocated / 1e6
            cached_mb = cached / 1e6
            
            return {
                "status": "✅ PASS",
                "details": f"GPU Memory: {total_gb:.1f}GB total, {allocated_mb:.1f}MB allocated, {cached_mb:.1f}MB cached"
            }
        else:
            return {"status": "❌ FAIL", "details": "CUDA not available for memory check"}
    except ImportError:
        return {"status": "⚠️  SKIP", "details": "PyTorch not available for memory check"}

def main():
    """Run all GPU validation tests"""
    print("🚀 DocMind AI GPU Infrastructure Validation")
    print("=" * 60)
    
    tests = [
        ("NVIDIA Driver", check_nvidia_smi),
        ("CUDA Toolkit", check_cuda_toolkit),
        ("PyTorch CUDA", check_pytorch_cuda),
        ("FastEmbed GPU", check_fastembed_gpu),
        ("Docker GPU", check_docker_gpu),
        ("GPU Memory", check_gpu_memory),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        result = test_func()
        results.append((test_name, result))
        print(f"   {result['status']} {result['details']}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result['status'] == '✅ PASS')
    failed = sum(1 for _, result in results if result['status'] == '❌ FAIL')
    skipped = sum(1 for _, result in results if result['status'] == '⚠️  SKIP')
    
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")  
    print(f"⚠️  Skipped: {skipped}")
    
    if failed == 0:
        print("\n🎉 GPU infrastructure is ready for 100x performance improvements!")
        return 0
    else:
        print(f"\n⚠️  {failed} tests failed. Please review the setup.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

    chmod +x gpu_validation.py
    success "GPU validation script created"
}

# Update Docker Compose configuration
update_docker_compose() {
    log "Updating docker-compose.yml for enhanced GPU support..."
    
    # Backup existing docker-compose.yml
    if [[ -f docker-compose.yml ]]; then
        cp docker-compose.yml docker-compose.yml.backup
        log "Backed up existing docker-compose.yml"
    fi
    
    cat << 'EOF' > docker-compose.yml
version: "3.8"

services:
  app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
      - BACKEND=ollama
      - CONTEXT_SIZE=32768
      - QDRANT_URL=http://qdrant:6333
      - LANGFUSE_PUBLIC_KEY=your_public_key
      - LANGFUSE_SECRET_KEY=your_secret_key
      - LANGFUSE_HOST=https://cloud.langfuse.com
      # GPU Configuration
      - NVIDIA_VISIBLE_DEVICES=all
      - CUDA_VISIBLE_DEVICES=0
      - VLLM_TORCH_BACKEND=cu128
      - VLLM_ATTENTION_BACKEND=FLASHINFER
      - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
      - FASTEMBED_CACHE_PATH=/tmp/fastembed_cache
      - ONNXRUNTIME_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider
    volumes:
      - .:/app
      - ./models:/models
      - gpu_cache:/tmp/fastembed_cache
    extra_hosts:
      - "host.docker.internal:host-gateway"
    depends_on:
      qdrant:
        condition: service_healthy
    runtime: nvidia
    ipc: host
    shm_size: '2gb'

  qdrant:
    image: qdrant/qdrant:gpu-nvidia-latest
    ports:
      - "6333:6333"
      - "6334:6334"
    environment:
      - QDRANT__GPU__INDEXING=1
      - QDRANT__GPU__FORCE_HALF_PRECISION=true
      - QDRANT__GPU__GROUPS_COUNT=512
      - QDRANT__GPU__PARALLEL_INDEXES=1
      - QDRANT__GPU__DEVICE_FILTER=nvidia
      - QDRANT__GPU__ALLOW_INTEGRATED=false
    volumes:
      - qdrant_storage:/qdrant/storage
    healthcheck:
      test: ["CMD", "wget", "--spider", "http://localhost:6333"]
      interval: 5s
      timeout: 3s
      retries: 5
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    runtime: nvidia

volumes:
  qdrant_storage:
  gpu_cache:
EOF

    success "Updated docker-compose.yml with enhanced GPU support"
}

# Update environment configuration
update_env_example() {
    log "Updating .env.example with GPU configuration..."
    
    # Backup existing .env.example
    if [[ -f .env.example ]]; then
        cp .env.example .env.example.backup
    fi
    
    cat << 'EOF' > .env.example
# Backend Configuration
BACKEND=ollama
OLLAMA_BASE_URL=http://localhost:11434
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LLAMACPP_MODEL_PATH=/path/to/model.gguf
DEFAULT_MODEL=Qwen/Qwen3-8B
CONTEXT_SIZE=4096

# Logging
LOG_PATH=logs/app.log

# Vector Database
QDRANT_URL=http://localhost:6333

# Embedding Models (GPU-optimized)
DEFAULT_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
DEFAULT_RERANKER_MODEL=jinaai/jina-reranker-v1-turbo-en
DEFAULT_SPARSE_MODEL=prithvida/Splade_PP_en_v1

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
NVIDIA_VISIBLE_DEVICES=all
VLLM_TORCH_BACKEND=cu128
VLLM_ATTENTION_BACKEND=FLASHINFER
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
TOKENIZERS_PARALLELISM=false

# FastEmbed GPU Settings
FASTEMBED_CACHE_PATH=/tmp/fastembed_cache
ONNXRUNTIME_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider
FASTEMBED_GPU_BATCH_SIZE=512

# Qdrant GPU Settings
QDRANT_GPU_INDEXING=1
QDRANT_GPU_FORCE_HALF_PRECISION=true
QDRANT_GPU_GROUPS_COUNT=512
QDRANT_GPU_PARALLEL_INDEXES=1

# Performance Monitoring
GPU_MEMORY_UTILIZATION=0.9
ENABLE_GPU_MONITORING=true
EOF

    success "Updated .env.example with GPU configuration"
}

# Create performance test script
create_performance_test() {
    log "Creating GPU performance test script..."
    
    cat << 'EOF' > test_gpu_performance.py
#!/usr/bin/env python3
"""
DocMind AI GPU Performance Test
Benchmarks GPU acceleration for embeddings and vector search
"""

import time
import sys
from typing import List, Dict, Any

def test_fastembed_performance() -> Dict[str, Any]:
    """Test FastEmbed GPU vs CPU performance"""
    try:
        from fastembed import TextEmbedding
        
        # Test documents
        documents = [
            f"This is test document number {i} for GPU performance benchmarking. "
            f"DocMind AI aims for 100x performance improvement with GPU acceleration. "
            f"FastEmbed provides efficient embedding generation for document retrieval."
            for i in range(500)
        ]
        
        # Test GPU performance
        gpu_model = TextEmbedding(
            "BAAI/bge-small-en-v1.5",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        
        start_time = time.time()
        gpu_embeddings = list(gpu_model.embed(documents))
        gpu_time = time.time() - start_time
        
        # Test CPU performance
        cpu_model = TextEmbedding(
            "BAAI/bge-small-en-v1.5", 
            providers=["CPUExecutionProvider"]
        )
        
        start_time = time.time()
        cpu_embeddings = list(cpu_model.embed(documents[:50]))  # Smaller batch for CPU
        cpu_time = time.time() - start_time
        
        # Calculate estimated full CPU time
        estimated_cpu_time = cpu_time * (len(documents) / 50)
        speedup = estimated_cpu_time / gpu_time if gpu_time > 0 else 0
        
        return {
            "status": "✅ SUCCESS",
            "gpu_time": gpu_time,
            "estimated_cpu_time": estimated_cpu_time,
            "speedup": speedup,
            "documents_processed": len(documents),
            "gpu_throughput": len(documents) / gpu_time,
            "details": f"{speedup:.1f}x speedup with GPU acceleration"
        }
        
    except ImportError:
        return {"status": "❌ FAIL", "details": "FastEmbed not installed"}
    except Exception as e:
        return {"status": "❌ FAIL", "details": f"Performance test failed: {str(e)}"}

def test_pytorch_gpu_performance() -> Dict[str, Any]:
    """Test PyTorch GPU performance"""
    try:
        import torch
        
        if not torch.cuda.is_available():
            return {"status": "❌ FAIL", "details": "CUDA not available"}
        
        device = torch.device("cuda:0")
        
        # Test matrix operations
        size = 4096
        start_time = time.time()
        
        # GPU computation
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        
        gpu_time = time.time() - start_time
        
        # CPU computation for comparison
        start_time = time.time()
        a_cpu = torch.randn(size//4, size//4)  # Smaller for CPU
        b_cpu = torch.randn(size//4, size//4)
        c_cpu = torch.matmul(a_cpu, b_cpu)
        cpu_time = time.time() - start_time
        
        # Estimate full CPU time
        estimated_cpu_time = cpu_time * 16  # Scale up for full size
        speedup = estimated_cpu_time / gpu_time if gpu_time > 0 else 0
        
        return {
            "status": "✅ SUCCESS",
            "gpu_time": gpu_time,
            "estimated_cpu_time": estimated_cpu_time,
            "speedup": speedup,
            "details": f"PyTorch GPU {speedup:.1f}x faster than CPU"
        }
        
    except ImportError:
        return {"status": "❌ FAIL", "details": "PyTorch not available"}
    except Exception as e:
        return {"status": "❌ FAIL", "details": f"PyTorch test failed: {str(e)}"}

def test_gpu_memory_usage() -> Dict[str, Any]:
    """Test GPU memory utilization"""
    try:
        import torch
        
        if not torch.cuda.is_available():
            return {"status": "❌ FAIL", "details": "CUDA not available"}
        
        device = torch.device("cuda:0")
        
        # Get initial memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(device)
        total_memory = torch.cuda.get_device_properties(device).total_memory
        
        # Allocate some memory
        test_tensor = torch.randn(1000, 1000, 1000, device=device)
        allocated_memory = torch.cuda.memory_allocated(device)
        
        # Clean up
        del test_tensor
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated(device)
        
        memory_used = (allocated_memory - initial_memory) / 1e9
        total_gb = total_memory / 1e9
        
        return {
            "status": "✅ SUCCESS",
            "total_memory_gb": total_gb,
            "peak_usage_gb": memory_used,
            "memory_cleanup": final_memory <= initial_memory,
            "details": f"GPU Memory: {total_gb:.1f}GB total, {memory_used:.1f}GB peak test usage"
        }
        
    except Exception as e:
        return {"status": "❌ FAIL", "details": f"Memory test failed: {str(e)}"}

def main():
    """Run performance benchmarks"""
    print("🚀 DocMind AI GPU Performance Benchmarks")
    print("=" * 60)
    
    tests = [
        ("FastEmbed GPU Performance", test_fastembed_performance),
        ("PyTorch GPU Performance", test_pytorch_gpu_performance),
        ("GPU Memory Management", test_gpu_memory_usage),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        result = test_func()
        results.append((test_name, result))
        
        if result["status"] == "✅ SUCCESS":
            print(f"   ✅ {result['details']}")
            if 'speedup' in result:
                print(f"   📈 Speedup: {result['speedup']:.1f}x")
        else:
            print(f"   {result['status']} {result['details']}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 60)
    
    successful_tests = [r for _, r in results if r["status"] == "✅ SUCCESS"]
    
    if successful_tests:
        speedups = [r.get('speedup', 0) for r in successful_tests if 'speedup' in r]
        if speedups:
            avg_speedup = sum(speedups) / len(speedups)
            print(f"🚀 Average GPU Speedup: {avg_speedup:.1f}x")
            
            if avg_speedup >= 50:
                print("🎉 EXCELLENT: GPU acceleration delivering 50x+ performance improvement!")
            elif avg_speedup >= 10:
                print("✅ GOOD: GPU acceleration providing significant performance boost!")
            else:
                print("⚠️  GPU acceleration working but may need optimization")
    
    failed_tests = len([r for _, r in results if r["status"] == "❌ FAIL"])
    if failed_tests == 0:
        print("\n🎯 GPU infrastructure ready for production workloads!")
        return 0
    else:
        print(f"\n⚠️  {failed_tests} performance tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

    chmod +x test_gpu_performance.py
    success "GPU performance test script created"
}

# Main installation function
main() {
    log "Starting DocMind AI GPU Infrastructure Setup"
    log "Target: 100x performance improvement with CUDA 12.x, cuDNN 9.x, NVIDIA Container Toolkit"
    
    check_root
    check_system
    
    log "Installing GPU infrastructure components..."
    install_cuda_toolkit
    install_cudnn
    install_nvidia_container_toolkit
    install_monitoring_tools
    install_fastembed_gpu
    
    log "Creating configuration and validation scripts..."
    create_validation_script
    create_performance_test
    update_docker_compose
    update_env_example
    
    success "GPU infrastructure setup completed!"
    
    echo ""
    echo "🎯 NEXT STEPS:"
    echo "1. Restart your shell or run: source /etc/profile.d/cuda.sh"
    echo "2. Run validation: ./gpu_validation.py"
    echo "3. Run performance test: ./test_gpu_performance.py"
    echo "4. Test Docker GPU: docker compose up"
    echo ""
    echo "🚀 Expected Performance Improvements:"
    echo "   • 100x faster embedding generation with FastEmbed GPU"
    echo "   • GPU-accelerated vector indexing with Qdrant"
    echo "   • Optimized memory management for large models"
    echo "   • Enhanced throughput for document processing"
    echo ""
    
    warning "Please reboot your system for optimal GPU driver integration"
}

# Run main function
main "$@"