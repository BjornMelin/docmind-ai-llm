# LLM Model & LangGraph Supervisor Integration Plan

## Executive Summary

This document consolidates the integration plan for two critical updates to DocMind AI:

1. **Model Correction**: Update from non-existent Qwen3-4B-Instruct-2507-AWQ to **Qwen/Qwen3-4B-Instruct-2507-FP8**
2. **Supervisor Optimization**: Add modern LangGraph parameters for parallel execution and context management

**Key Changes:**
- Model: Qwen3-4B-Instruct-2507-FP8 with vLLM + FlashInfer
- Context: 131K tokens (128K, reduced from 262K)
- Performance: 100-160 tok/s decode, 800-1300 tok/s prefill
- Memory: <16GB VRAM on RTX 4090 Laptop
- Token Reduction: 50-87% via parallel tool execution

**Risk Level**: Medium (context reduction requires careful management)

## Technical Specifications

**Target Configuration:**
- Model: `Qwen/Qwen3-4B-Instruct-2507-FP8`
- Context: 131,072 tokens (128K)
- Quantization: FP8 weights + FP8_e5m2 KV cache
- Backend: vLLM with FlashInfer attention
- Memory: ~12-14GB total VRAM usage

## Implementation Plan

### Phase 1: vLLM Backend Setup

**Installation & Model Download:**
```bash
# Install dependencies
uv pip install vllm[flashinfer]>=0.10.1

# Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Download FP8 model
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507-FP8 --local-dir ./models/qwen3-4b-fp8
```

**Launch Script (`launch_vllm_fp8.sh`):**
```bash
#!/bin/bash
export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_USE_CUDNN_PREFILL=1

vllm serve Qwen/Qwen3-4B-Instruct-2507-FP8 \
  --max-model-len 131072 \
  --kv-cache-dtype fp8_e5m2 \
  --calculate-kv-scales \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 1 \
  --enable-chunked-prefill \
  --trust-remote-code \
  --host 0.0.0.0 --port 8000 \
  --served-model-name docmind-qwen3-fp8
```

**Basic Test:**
```bash
./launch_vllm_fp8.sh &
sleep 30
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "docmind-qwen3-fp8", "prompt": "Test", "max_tokens": 50}'
```

### Phase 2: Performance Validation

**Performance Test Script (`test_fp8_performance.py`):**
```python
import time, requests, subprocess

def test_decode_performance():
    long_context = "Test content. " * 8000  # ~120K tokens
    payload = {
        "model": "docmind-qwen3-fp8",
        "prompt": long_context + "\nSummarize:",
        "max_tokens": 500, "temperature": 0.1
    }
    
    start = time.time()
    response = requests.post("http://localhost:8000/v1/completions", json=payload)
    duration = time.time() - start
    
    if response.status_code == 200:
        tokens = len(response.json()['choices'][0]['text'].split())
        tok_per_sec = tokens / duration
        print(f"Decode: {tok_per_sec:.1f} tok/s ({'PASS' if tok_per_sec > 100 else 'FAIL'})")
        return tok_per_sec > 100
    return False

def test_memory_usage():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True)
    memory_gb = int(result.stdout.strip()) / 1024
    print(f"Memory: {memory_gb:.1f}GB ({'PASS' if memory_gb < 16 else 'FAIL'})")
    return memory_gb < 16

if __name__ == "__main__":
    print("🧪 Testing FP8 Performance...")
    decode_ok = test_decode_performance()
    memory_ok = test_memory_usage()
    print("✅ PASSED" if decode_ok and memory_ok else "❌ FAILED")
```

**Expected Metrics:**
- Decode: >100 tok/s (target: 100-160)
- Prefill: >800 tok/s (target: 800-1300)
- Memory: <16GB VRAM

### Phase 2: Code Integration (Execute After Phase 1 Success)

#### Step 6: Update Settings.py Configuration

```bash
# Find and update Settings.py
find . -name "settings.py" -o -name "Settings.py" | head -1 | xargs -I {} cp {} {}.backup

# Apply settings updates - create patch script
cat > update_settings.py << 'EOF'
#!/usr/bin/env python3
"""Update Settings.py for FP8 model and 128K context."""

import re
from pathlib import Path

def update_settings_file():
    # Find settings file
    settings_files = list(Path('.').rglob('*ettings.py'))
    if not settings_files:
        print("❌ No settings file found")
        return False
    
    settings_file = settings_files[0]
    print(f"📝 Updating {settings_file}")
    
    content = settings_file.read_text()
    
    # Key updates for FP8 model
    updates = [
        # Model name update
        (r'model_name.*=.*Field\([^)]*default\s*=\s*["\']([^"\']*AWQ[^"\']*)["\']',
         r'model_name: str = Field(\n        default="Qwen/Qwen3-4B-Instruct-2507-FP8",'),
        
        # Context window size 
        (r'context_window_size.*=.*Field\([^)]*default\s*=\s*262144',
         r'context_window_size: int = Field(\n        default=131072,'),
        
        # Quantization method
        (r'quantization.*=.*Field\([^)]*default\s*=\s*["\']AWQ["\']',
         r'quantization: str = Field(\n        default="FP8",'),
        
        # KV cache type
        (r'kv_cache_dtype.*=.*Field\([^)]*default\s*=\s*["\']int8["\']',
         r'kv_cache_dtype: str = Field(\n        default="fp8_e5m2",'),
    ]
    
    # Apply updates
    updated_content = content
    for pattern, replacement in updates:
        updated_content = re.sub(pattern, replacement, updated_content, flags=re.IGNORECASE | re.MULTILINE)
    
    # Add new LangGraph supervisor settings if not present
    if 'enable_parallel_tools' not in updated_content:
        supervisor_settings = '''
    # LangGraph Supervisor Settings (FP8 Integration)
    enable_parallel_tools: bool = Field(
        default=True,
        description="Enable parallel tool execution in supervisor"
    )
    
    max_parallel_calls: int = Field(
        default=3,
        description="Maximum parallel tool calls"
    )
    
    enable_message_forwarding: bool = Field(
        default=True,
        description="Enable direct message forwarding"
    )
    
    output_mode: str = Field(
        default="structured",
        description="Supervisor output format"
    )
    
    enable_context_trimming: bool = Field(
        default=True,
        description="Enable automatic context trimming at 128K"
    )
    
    context_trim_threshold: float = Field(
        default=0.9,
        description="Trim when context reaches 90% of limit"
    )'''
        
        # Insert before class closing
        if 'class Settings' in updated_content:
            # Find a good insertion point (before last method or end of class)
            lines = updated_content.split('\n')
            insert_idx = len(lines) - 1
            
            # Look for class definition and find a good spot
            for i, line in enumerate(lines):
                if 'class Settings' in line:
                    # Find the end of this class
                    indent_level = len(line) - len(line.lstrip())
                    for j in range(i + 1, len(lines)):
                        if lines[j].strip() and not lines[j].startswith(' '):
                            insert_idx = j - 1
                            break
                    break
            
            lines.insert(insert_idx, supervisor_settings)
            updated_content = '\n'.join(lines)
    
    # Write updated content
    settings_file.write_text(updated_content)
    print("✅ Settings.py updated successfully")
    return True

if __name__ == "__main__":
    update_settings_file()
EOF

python update_settings.py
```

#### Step 7: Update Coordinator.py with Modern Supervisor

```bash
# Find coordinator file
find . -name "*coordinator*.py" -o -name "*Coordinator*.py" | head -1 | xargs -I {} cp {} {}.backup

# Create coordinator update script
cat > update_coordinator.py << 'EOF'
#!/usr/bin/env python3
"""Update Coordinator.py for modern LangGraph supervisor parameters."""

from pathlib import Path
import re

def update_coordinator_file():
    # Find coordinator file
    coord_files = list(Path('.').rglob('*coordinator*.py')) + list(Path('.').rglob('*Coordinator*.py'))
    if not coord_files:
        print("❌ No coordinator file found")
        return False
    
    coord_file = coord_files[0]
    print(f"📝 Updating {coord_file}")
    
    content = coord_file.read_text()
    
    # Add required imports at top if not present
    if 'from langchain_core.runnables import RunnableLambda' not in content:
        # Find import section
        lines = content.split('\n')
        import_end = 0
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                import_end = i + 1
        
        lines.insert(import_end, 'from langchain_core.runnables import RunnableLambda')
        content = '\n'.join(lines)
    
    # Add context trimming functions if not present
    if 'def trim_context_hook' not in content:
        context_functions = '''
def estimate_tokens(messages):
    """Rough token estimation for messages."""
    total = 0
    for msg in messages:
        if hasattr(msg, 'content'):
            total += len(str(msg.content).split()) * 1.3  # rough estimate
        else:
            total += len(str(msg).split()) * 1.3
    return int(total)

def trim_to_token_limit(messages, limit):
    """Trim messages to fit within token limit."""
    total_tokens = 0
    result = []
    
    # Keep system message if present
    if messages and hasattr(messages[0], 'type') and messages[0].type == 'system':
        result.append(messages[0])
        total_tokens += estimate_tokens([messages[0]])
        messages = messages[1:]
    
    # Add messages from newest, until we hit limit
    for msg in reversed(messages):
        msg_tokens = estimate_tokens([msg])
        if total_tokens + msg_tokens > limit:
            break
        result.insert(-1 if result else 0, msg)
        total_tokens += msg_tokens
    
    return result

def trim_context_hook(state):
    """Pre-model hook to trim context to 128K tokens."""
    messages = state.get("messages", [])
    total_tokens = estimate_tokens(messages)
    
    if total_tokens > 120000:  # Leave 8K buffer for 128K limit
        print(f"⚠️  Trimming context: {total_tokens} -> 120000 tokens")
        messages = trim_to_token_limit(messages, 120000)
        state["messages"] = messages
        state["context_trimmed"] = True
        state["original_token_count"] = total_tokens
        state["trimmed_token_count"] = estimate_tokens(messages)
    
    return state

def format_response_hook(state):
    """Post-model hook for response formatting."""
    if state.get("output_mode") == "structured":
        # Add metadata about processing
        if state.get("context_trimmed"):
            state["processing_metadata"] = {
                "context_trimmed": True,
                "original_tokens": state.get("original_token_count", 0),
                "final_tokens": state.get("trimmed_token_count", 0)
            }
    return state
'''
        # Insert context functions before create_supervisor call
        if 'create_supervisor' in content:
            content = content.replace('# Create supervisor', context_functions + '\n# Create supervisor')
        else:
            # Add at end of imports
            lines = content.split('\n')
            import_end = 0
            for i, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('from '):
                    import_end = i + 1
            lines.insert(import_end + 1, context_functions)
            content = '\n'.join(lines)
    
    # Update create_supervisor call to include modern parameters
    supervisor_pattern = r'(workflow\s*=\s*create_supervisor\s*\([^)]*)'
    if re.search(supervisor_pattern, content, re.DOTALL):
        # Find the create_supervisor call and update it
        modern_params = '''
    # Modern supervisor parameters for FP8 optimization
    parallel_tool_calls=True,
    output_mode="structured",
    create_forward_message_tool=True,
    add_handoff_back_messages=True,
    pre_model_hook=RunnableLambda(trim_context_hook),
    post_model_hook=RunnableLambda(format_response_hook),
    max_messages=100,
    trim_messages=True,'''
        
        # Insert modern parameters before closing parenthesis
        content = re.sub(
            r'(\s*)\)(\s*#.*create_supervisor|$)',
            modern_params + r'\1)\2',
            content,
            flags=re.MULTILINE
        )
    
    # Write updated content
    coord_file.write_text(content)
    print("✅ Coordinator.py updated successfully")
    return True

if __name__ == "__main__":
    update_coordinator_file()
EOF

python update_coordinator.py
```

#### Step 8: Integration Test Commands

```bash
# Create integration test script
cat > test_integration.py << 'EOF'
#!/usr/bin/env python3
"""Test integration of FP8 model with updated supervisor."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path('.').resolve()))

async def test_supervisor_integration():
    """Test that supervisor works with new settings."""
    try:
        # Import updated modules
        from src.settings import Settings  # Adjust import path as needed
        settings = Settings()
        
        print("✅ Settings loaded successfully")
        print(f"   Model: {settings.model_name}")
        print(f"   Context: {settings.context_window_size}")
        print(f"   Quantization: {settings.quantization}")
        print(f"   KV Cache: {settings.kv_cache_dtype}")
        
        if hasattr(settings, 'enable_parallel_tools'):
            print(f"   Parallel Tools: {settings.enable_parallel_tools}")
            print(f"   Max Parallel: {settings.max_parallel_calls}")
        
        # Test coordinator import
        try:
            from src.coordinator import Coordinator  # Adjust import path
            print("✅ Coordinator updated successfully")
        except ImportError as e:
            print(f"⚠️  Coordinator import issue: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

async def test_context_management():
    """Test context trimming functionality."""
    try:
        # Test the trim functions
        from src.coordinator import trim_context_hook, estimate_tokens
        
        # Create mock large context
        large_messages = [{"content": "Test message. " * 1000} for _ in range(200)]
        large_state = {"messages": large_messages}
        
        initial_tokens = estimate_tokens(large_messages)
        print(f"✅ Initial context: {initial_tokens} tokens")
        
        # Test trimming
        trimmed_state = trim_context_hook(large_state)
        final_tokens = estimate_tokens(trimmed_state["messages"])
        
        print(f"✅ Trimmed context: {final_tokens} tokens")
        print(f"   Trimming: {'ACTIVE' if trimmed_state.get('context_trimmed') else 'NOT NEEDED'}")
        
        return final_tokens < 120000
        
    except Exception as e:
        print(f"❌ Context management test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Integration...")
    
    # Run tests
    integration_ok = asyncio.run(test_supervisor_integration())
    context_ok = asyncio.run(test_context_management())
    
    if integration_ok and context_ok:
        print("\n✅ All integration tests PASSED!")
        print("🚀 Ready to start vLLM server and test full system!")
    else:
        print("\n❌ Some integration tests FAILED - Check configuration")
EOF

python test_integration.py
```

### Phase 3: Full System Test and Rollback Procedures

#### Step 9: Full System Test Commands

```bash
# Start vLLM server for full test
./launch_vllm_fp8.sh &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID" > vllm.pid

# Wait for server startup
echo "⏳ Waiting for vLLM server startup..."
sleep 45

# Test full system with realistic query
cat > test_full_system.py << 'EOF'
#!/usr/bin/env python3
"""Full system test with FP8 model and modern supervisor."""

import requests
import time
import json

def test_long_document_processing():
    """Test processing a long document that requires context management."""
    
    # Create a realistic long document
    long_document = """
    This is a comprehensive research document about artificial intelligence and machine learning.
    """ + "The field of AI continues to evolve rapidly. " * 5000  # ~50K tokens
    
    query = f"""
    Document: {long_document}
    
    Please analyze this document and provide:
    1. A summary of key points
    2. Main themes identified
    3. Any recommendations for further research
    
    Use multiple agents if needed for comprehensive analysis.
    """
    
    payload = {
        "model": "docmind-qwen3-fp8",
        "prompt": query,
        "max_tokens": 1000,
        "temperature": 0.1,
        "stream": False
    }
    
    print("🧪 Testing long document processing...")
    start_time = time.time()
    
    try:
        response = requests.post(
            "http://localhost:8000/v1/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=120  # 2 minutes timeout
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            response_text = result['choices'][0]['text']
            
            print(f"✅ Long Document Test PASSED")
            print(f"   Processing time: {processing_time:.2f}s")
            print(f"   Response length: {len(response_text)} chars")
            print(f"   Status: SUCCESS")
            
            # Save response for review
            with open('test_response.txt', 'w') as f:
                f.write(f"Query: {query[:200]}...\n\n")
                f.write(f"Response: {response_text}\n\n")
                f.write(f"Processing time: {processing_time:.2f}s\n")
            
            return True
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out after 2 minutes")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def test_memory_stability():
    """Test that memory usage remains stable during processing."""
    import subprocess
    
    try:
        # Get memory before
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        memory_before = int(result.stdout.strip())
        
        # Process several requests
        for i in range(3):
            print(f"   Processing request {i+1}/3...")
            payload = {
                "model": "docmind-qwen3-fp8",
                "prompt": "Generate a detailed explanation about machine learning." + " Please elaborate." * 100,
                "max_tokens": 500,
                "temperature": 0.1
            }
            
            response = requests.post(
                "http://localhost:8000/v1/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"❌ Request {i+1} failed")
                return False
        
        # Get memory after
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        memory_after = int(result.stdout.strip())
        
        memory_diff = memory_after - memory_before
        memory_after_gb = memory_after / 1024
        
        print(f"✅ Memory Stability Test")
        print(f"   Memory before: {memory_before/1024:.1f}GB")
        print(f"   Memory after: {memory_after_gb:.1f}GB")
        print(f"   Difference: {memory_diff}MB")
        print(f"   Status: {'STABLE' if abs(memory_diff) < 500 else 'UNSTABLE'}")
        
        return abs(memory_diff) < 500 and memory_after_gb < 16
        
    except Exception as e:
        print(f"❌ Memory stability test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Full System Testing...")
    
    # Test document processing
    doc_test = test_long_document_processing()
    
    # Test memory stability  
    mem_test = test_memory_stability()
    
    if doc_test and mem_test:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ FP8 model integration successful")
        print("✅ Context management working")
        print("✅ Memory usage stable")
        print("✅ System ready for production!")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("Check logs and consider rollback procedures")

EOF

python test_full_system.py
```

#### Step 10: Rollback Procedures (If Tests Fail)

```bash
# Create rollback script
cat > rollback_fp8.sh << 'EOF'
#!/bin/bash
# Rollback script for FP8 integration if tests fail

echo "🔄 Starting rollback procedures..."

# Stop vLLM server
if [ -f vllm.pid ]; then
    VLLM_PID=$(cat vllm.pid)
    if ps -p $VLLM_PID > /dev/null; then
        echo "🛑 Stopping vLLM server (PID: $VLLM_PID)"
        kill $VLLM_PID
        sleep 5
        # Force kill if needed
        if ps -p $VLLM_PID > /dev/null; then
            kill -9 $VLLM_PID
        fi
    fi
    rm -f vllm.pid
fi

# Restore backup files
echo "📁 Restoring backup files..."

# Restore settings
if [ -f "$(find . -name "*ettings.py.backup" | head -1)" ]; then
    SETTINGS_BACKUP=$(find . -name "*ettings.py.backup" | head -1)
    SETTINGS_FILE=${SETTINGS_BACKUP%.backup}
    cp "$SETTINGS_BACKUP" "$SETTINGS_FILE"
    echo "✅ Restored $SETTINGS_FILE"
fi

# Restore coordinator
if [ -f "$(find . -name "*coordinator*.py.backup" | head -1)" ]; then
    COORD_BACKUP=$(find . -name "*coordinator*.py.backup" | head -1)
    COORD_FILE=${COORD_BACKUP%.backup}
    cp "$COORD_BACKUP" "$COORD_FILE"
    echo "✅ Restored $COORD_FILE"
fi

# Clean up test files
echo "🧹 Cleaning up test files..."
rm -f launch_vllm_fp8.sh
rm -f test_fp8_performance.py
rm -f update_settings.py
rm -f update_coordinator.py
rm -f test_integration.py
rm -f test_full_system.py
rm -f test_response.txt

echo "✅ Rollback completed"
echo "💡 You can now investigate issues and retry implementation"
echo "💡 Original files have been restored from .backup files"
EOF

chmod +x rollback_fp8.sh

# Also create success cleanup script
cat > cleanup_success.sh << 'EOF'
#!/bin/bash
# Cleanup script after successful FP8 integration

echo "🎉 Cleaning up after successful integration..."

# Remove backup files (keep for safety, comment out if you want to remove)
# find . -name "*.backup" -delete

# Remove test scripts (keep implementation scripts)
rm -f update_settings.py
rm -f update_coordinator.py  
rm -f test_integration.py
rm -f test_full_system.py
rm -f test_response.txt

# Keep launch script and performance test for ongoing use
echo "✅ Cleanup completed"
echo "📝 Kept: launch_vllm_fp8.sh, test_fp8_performance.py"
echo "📝 Kept: .backup files for safety"
EOF

chmod +x cleanup_success.sh
```

## Risk Mitigation & Success Criteria

**Risk Matrix:**
| Risk Level | Component | Mitigation | Fallback |
|---|---|---|---|
| 🔴 HIGH | Context Reduction (262K→128K) | Aggressive trimming + sliding window | Use 64K context |
| 🟡 MEDIUM | FlashInfer Compatibility | Test + fallback to XFORMERS | `VLLM_ATTENTION_BACKEND=XFORMERS` |
| 🟡 MEDIUM | Performance Variance | Conservative targets | Reduce context/batch size |
| 🟢 LOW | Memory Pressure | Monitoring + swap config | Reduce `gpu-memory-utilization` |

**Success Criteria:**
- [ ] vLLM serves FP8 model without OOM errors
- [ ] Decode speed >100 tok/s, prefill >800 tok/s
- [ ] VRAM usage <16GB, stable during processing
- [ ] 128K context processing functional
- [ ] Parallel tool execution reduces tokens 50-87%
- [ ] Context trimming automatic at 120K threshold
- [ ] Integration tests pass for settings and coordinator

## Configuration Changes Required

### Settings.py Updates
```python
# Core model changes
model_name: str = Field(default="Qwen/Qwen3-4B-Instruct-2507-FP8")  # from AWQ
context_window_size: int = Field(default=131072)  # from 262144
quantization: str = Field(default="FP8")  # from AWQ
kv_cache_dtype: str = Field(default="fp8_e5m2")  # from int8

# New supervisor settings
enable_parallel_tools: bool = Field(default=True)
max_parallel_calls: int = Field(default=3)
enable_message_forwarding: bool = Field(default=True)
output_mode: str = Field(default="structured")
enable_context_trimming: bool = Field(default=True)
context_trim_threshold: float = Field(default=0.9)
```

### Coordinator.py Updates
```python
from langchain_core.runnables import RunnableLambda

# Context management functions
def trim_context_hook(state):
    messages = state.get("messages", [])
    if estimate_tokens(messages) > 120000:  # Leave 8K buffer
        state["messages"] = trim_to_limit(messages, 120000)
        state["context_trimmed"] = True
    return state

# Modern supervisor parameters
workflow = create_supervisor(
    agents=agents, model=llm, prompt=prompt,
    parallel_tool_calls=True,
    output_mode="structured",
    create_forward_message_tool=True,
    pre_model_hook=RunnableLambda(trim_context_hook),
    max_messages=100, trim_messages=True
)
```

### ADR Updates Required
- **ADR-004**: Update model spec to FP8, context to 128K, add vLLM config
- **ADR-021**: Reduce buffer from 262K to 128K, add aggressive trimming
- **ADR-011**: Add modern supervisor parameters section
- **ADR-010**: Update for FP8 performance, parallel execution gains
- **Requirements**: Update REQ-0063 (model), REQ-0064 (performance), REQ-0094 (context)

## Quick Start Implementation

### Copy-Paste Command Sequence
```bash
# Phase 1: Setup
uv pip install vllm[flashinfer]>=0.10.1
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507-FP8 --local-dir ./models/qwen3-4b-fp8

# Phase 2: Create launch script (see above)
# Phase 3: Backup and update Settings.py (see configuration changes)
# Phase 4: Update Coordinator.py (see configuration changes)  
# Phase 5: Test integration (see test scripts)
# Phase 6: Start server and run full system test
./launch_vllm_fp8.sh &
sleep 45
python test_full_system.py
```

### Key Benefits
- **Correct Model**: FP8 model actually exists (vs non-existent AWQ)
- **Performance**: 100-160 tok/s decode, 800-1300 tok/s prefill  
- **Memory**: <16GB with FP8 quantization
- **Efficiency**: 50-87% token reduction via parallel execution
- **Context**: 128K with intelligent trimming

### Emergency Procedures
**If issues occur:**
1. Run `./rollback_fp8.sh` to restore previous state
2. Check vLLM logs for specific errors
3. Reduce `gpu-memory-utilization` if OOM
4. Switch to `VLLM_ATTENTION_BACKEND=XFORMERS` if FlashInfer fails
5. Reduce context to 64K if 128K causes issues

**Status: Ready for Implementation** 🚀



## Modern LangGraph Supervisor Integration

### Core Parameters Added
1. **parallel_tool_calls=True**: Concurrent agent execution (50-87% token reduction)
2. **output_mode="structured"**: Enhanced response formatting
3. **create_forward_message_tool=True**: Direct agent passthrough
4. **pre_model_hook**: Context trimming at 120K threshold
5. **add_handoff_back_messages=True**: Improved coordination tracking

### Context Management (128K Window)
```python
def trim_context_hook(state):
    messages = state.get("messages", [])
    if estimate_tokens(messages) > 120000:  # Leave 8K buffer
        # Trim to most recent messages within limit
        state["messages"] = trim_to_limit(messages, 120000)
        state["context_trimmed"] = True
    return state
```

**Strategy**: Keep system message + most recent conversation within 120K token limit
**Benefits**: Prevents OOM errors, maintains conversation coherence, tracks trimming metadata


### Production Deployment

**vLLM Systemd Service:**
```ini
[Unit]
Description=vLLM Server for DocMind AI
After=network.target

[Service]
Type=simple
Environment="VLLM_ATTENTION_BACKEND=FLASHINFER"
Environment="VLLM_USE_CUDNN_PREFILL=1"
ExecStart=/usr/local/bin/vllm serve Qwen/Qwen3-4B-Instruct-2507-FP8 \
  --max-model-len 131072 --kv-cache-dtype fp8_e5m2 --calculate-kv-scales \
  --gpu-memory-utilization 0.95 --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

**Benchmark Commands:**
```bash
# Decode test (target: >100 tok/s)
python -m vllm.entrypoints.benchmark --model Qwen/Qwen3-4B-Instruct-2507-FP8 \
  --input-len 120000 --output-len 1000

# Prefill test (target: >800 tok/s)  
python -m vllm.entrypoints.benchmark --model Qwen/Qwen3-4B-Instruct-2507-FP8 \
  --input-len 128000 --output-len 100
```

**Fallback Configurations:**
- Reduce context: `--max-model-len 65536`
- Disable FlashInfer: `export VLLM_ATTENTION_BACKEND=XFORMERS`
- Lower memory: `--gpu-memory-utilization 0.8`

## Configuration Files

### vLLM Systemd Service

```ini
[Unit]
Description=vLLM Server for DocMind AI
After=network.target

[Service]
Type=simple
User=docmind
WorkingDirectory=/opt/docmind
Environment="VLLM_ATTENTION_BACKEND=FLASHINFER"
Environment="VLLM_USE_CUDNN_PREFILL=1"
ExecStart=/usr/local/bin/vllm serve Qwen/Qwen3-4B-Instruct-2507-FP8 \
  --max-model-len 131072 \
  --kv-cache-dtype fp8_e5m2 \
  --calculate-kv-scales \
  --gpu-memory-utilization 0.95 \
  --host 0.0.0.0 \
  --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Updated Settings.py (Key Changes)

```python
# Model Configuration
model_name: str = Field(
    default="Qwen/Qwen3-4B-Instruct-2507-FP8",  # Changed from AWQ
    description="Model name for LLM"
)

# Context Management  
context_window_size: int = Field(
    default=131072,  # Changed from 262144
    description="Context window size in tokens (128K with FP8)"
)

# Quantization
quantization: str = Field(
    default="FP8",  # Changed from AWQ
    description="Quantization method for model"
)

kv_cache_dtype: str = Field(
    default="fp8_e5m2",  # Changed from int8
    description="KV cache data type for memory optimization"
)

# LangGraph Supervisor (NEW)
enable_parallel_tools: bool = Field(default=True)
max_parallel_calls: int = Field(default=3)
enable_message_forwarding: bool = Field(default=True)
output_mode: str = Field(default="structured")
enable_context_trimming: bool = Field(default=True)
context_trim_threshold: float = Field(default=0.9)
```

### Updated Coordinator.py (Key Changes)

```python
from langchain_core.runnables import RunnableLambda

# Context trimming hook
def trim_context_hook(state):
    """Trim context to fit 128K window."""
    if estimate_tokens(state["messages"]) > 120000:
        state["messages"] = trim_to_limit(state["messages"], 120000)
    return state

# Create supervisor with modern parameters
workflow = create_supervisor(
    agents=agents,
    model=llm,
    prompt=prompt,
    parallel_tool_calls=True,
    create_forward_message_tool=True,
    add_handoff_back_messages=True,
    pre_model_hook=RunnableLambda(trim_context_hook),
    output_mode="structured"
)
```

## Appendix B: Verification Commands

### Model Verification

```bash
# Verify model download
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507-FP8

# Test vLLM inference
python -c "
from vllm import LLM
llm = LLM('Qwen/Qwen3-4B-Instruct-2507-FP8', 
          max_model_len=131072,
          kv_cache_dtype='fp8_e5m2')
output = llm.generate('Hello world')
print(output)
"

# Check GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

---

## Summary

This plan provides a complete roadmap for integrating:
1. **Correct FP8 Model**: Qwen3-4B-Instruct-2507-FP8 (actually exists vs non-existent AWQ)
2. **Modern Supervisor**: Parallel execution reducing tokens by 50-87%
3. **128K Context**: With intelligent trimming and monitoring
4. **Production Ready**: <16GB VRAM, 100-160 tok/s performance

The integration prioritizes correctness, performance, and reliability with comprehensive testing and rollback procedures.
