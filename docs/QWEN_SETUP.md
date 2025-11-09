# Qwen3-VL Setup Guide

This guide covers setting up Qwen3-VL-235B-A22B-Instruct for use with Hegarty.

## Overview

Qwen3-VL is a state-of-the-art multimodal vision-language model from Alibaba Cloud's Qwen team. The 235B-A22B variant uses a Mixture-of-Experts (MoE) architecture for efficient inference.

### Key Features

- **Visual Agent Capabilities**: Operates PC/mobile GUIs
- **Advanced Spatial Perception**: 2D/3D grounding, object positioning
- **Long Context**: Native 256K context, expandable to 1M
- **Video Understanding**: Handles hours-long video with frame-level indexing
- **Enhanced OCR**: Supports 32 languages with robust text recognition
- **Strong Reasoning**: Excels in STEM/Math tasks

[Model Card](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct)

## System Requirements

### Hardware Requirements

**Minimum (with quantization):**
- 1x A100 80GB or equivalent
- 128GB system RAM
- 500GB disk space

**Recommended (full precision):**
- 8x A100 80GB or H100 80GB
- 512GB+ system RAM
- 1TB disk space

**Optimal (production):**
- 8x H100 80GB SXM
- 1TB system RAM
- NVMe SSD storage

### Software Requirements

- Python 3.9+
- CUDA 11.8+ or 12.1+
- PyTorch 2.0+
- Transformers 4.57.0+
- Flash Attention 2 (optional but recommended)

## Installation

### 1. Install Dependencies

```bash
# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Transformers from source (for latest features)
pip install git+https://github.com/huggingface/transformers

# Or use released version (when 4.57.0+ is available)
# pip install transformers>=4.57.0

# Install additional dependencies
pip install accelerate sentencepiece pillow

# Optional: Install Flash Attention 2 for better performance
pip install flash-attn --no-build-isolation
```

### 2. Download Model

```bash
# Using Hugging Face CLI
pip install huggingface-hub

# Login (optional, for gated models)
huggingface-cli login

# Download model
huggingface-cli download Qwen/Qwen3-VL-235B-A22B-Instruct \
    --local-dir ./models/qwen3-vl-235b \
    --local-dir-use-symlinks False
```

### 3. Verify Installation

```python
from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor

# Test loading (this will take a while for first download)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-235B-A22B-Instruct")
print("Processor loaded successfully!")

# Note: Full model loading requires significant VRAM
# model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen3-VL-235B-A22B-Instruct",
#     device_map="auto",
#     dtype="auto"
# )
```

## Configuration Options

### Basic Usage

```python
from hegarty.mllm import QwenMLLM

# Standard configuration
mllm = QwenMLLM(
    model_name="Qwen/Qwen3-VL-235B-A22B-Instruct",
    device_map="auto",
    dtype="auto"
)
```

### Multi-GPU Setup

```python
# Automatic distribution across GPUs
mllm = QwenMLLM(
    model_name="Qwen/Qwen3-VL-235B-A22B-Instruct",
    device_map="auto",  # Distributes across available GPUs
    dtype="auto"
)

# Manual device mapping (advanced)
device_map = {
    "model.embed_tokens": 0,
    "model.layers.0-10": 0,
    "model.layers.11-20": 1,
    # ... customize as needed
    "lm_head": 7
}

mllm = QwenMLLM(
    model_name="Qwen/Qwen3-VL-235B-A22B-Instruct",
    device_map=device_map,
    dtype="bfloat16"
)
```

### Quantization (for limited VRAM)

```python
# 8-bit quantization
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

# Note: Pass this via custom model loading
# (requires modifying qwen.py to accept quantization_config)
```

### Flash Attention 2

```python
# Enable Flash Attention 2 for better performance
mllm = QwenMLLM(
    model_name="Qwen/Qwen3-VL-235B-A22B-Instruct",
    device_map="auto",
    dtype="bfloat16",
    attn_implementation="flash_attention_2"
)
```

## Performance Optimization

### Memory Optimization

1. **Use bfloat16**: Reduces memory by ~50% vs float32
2. **Enable Flash Attention 2**: Reduces memory for long sequences
3. **Gradient checkpointing**: If fine-tuning (not needed for inference)
4. **Smaller batch sizes**: Use batch_size=1 for large models

### Inference Optimization

1. **Use Flash Attention 2**: Faster attention computation
2. **Compile model**: Use `torch.compile()` for ~20% speedup
3. **vLLM or TGI**: For production serving with continuous batching

### Example: Production Setup

```python
import torch
from hegarty.mllm import QwenMLLM

# Optimized configuration
mllm = QwenMLLM(
    model_name="Qwen/Qwen3-VL-235B-A22B-Instruct",
    device_map="auto",
    dtype="bfloat16",
    attn_implementation="flash_attention_2",
    temperature=0.3,
    max_tokens=2000
)

# Optional: Compile for better performance (requires PyTorch 2.0+)
# mllm.model = torch.compile(mllm.model, mode="reduce-overhead")
```

## Integration with Hegarty

### Basic Integration

```python
from hegarty import HergartyAgent, Config
from hegarty.mllm import QwenMLLM

# Initialize Qwen provider
qwen = QwenMLLM(
    model_name="Qwen/Qwen3-VL-235B-A22B-Instruct",
    device_map="auto",
    attn_implementation="flash_attention_2"
)

# Configure agent
config = Config(
    frame_extraction_count=5,
    max_workers=6
)

# Create agent
agent = HergartyAgent(config=config)
agent.mllm = qwen

# Use agent
result = agent.process(
    image="path/to/image.jpg",
    question="What's the spatial relationship?",
    use_mental_rotation=False
)
```

### With Session Logging

```python
from pathlib import Path
from datetime import datetime

# Create session directory
session_dir = Path(f"temp/sessions/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
session_dir.mkdir(parents=True, exist_ok=True)

# Initialize with session logging
qwen = QwenMLLM(
    model_name="Qwen/Qwen3-VL-235B-A22B-Instruct",
    device_map="auto",
    session_dir=session_dir
)

# All MLLM calls will be logged to session_dir
agent = HergartyAgent(config=config, session_dir=session_dir)
agent.mllm = qwen
```

## Troubleshooting

### Out of Memory (OOM) Errors

**Problem**: CUDA out of memory

**Solutions**:
1. Reduce batch size to 1
2. Use quantization (8-bit or 4-bit)
3. Enable gradient checkpointing (if fine-tuning)
4. Use smaller model variant
5. Add more GPUs and use device_map="auto"

### Slow Inference

**Problem**: Generation is too slow

**Solutions**:
1. Enable Flash Attention 2
2. Use bfloat16 instead of float32
3. Compile model with torch.compile()
4. Use vLLM for production serving
5. Check GPU utilization with nvidia-smi

### Model Loading Issues

**Problem**: Cannot load model or processor

**Solutions**:
1. Update transformers: `pip install --upgrade transformers`
2. Install from source: `pip install git+https://github.com/huggingface/transformers`
3. Clear cache: `rm -rf ~/.cache/huggingface/hub`
4. Check internet connection for model download

### Flash Attention Issues

**Problem**: Flash Attention not working

**Solutions**:
1. Install properly: `pip install flash-attn --no-build-isolation`
2. Check CUDA version compatibility
3. Fall back to standard attention by removing `attn_implementation` parameter

## Alternative Models

If the 235B model is too large, consider these alternatives:

### Smaller Qwen Models

- **Qwen3-VL-72B**: Smaller, faster, still excellent performance
- **Qwen2.5-VL-72B**: Previous generation, well-tested
- **Qwen2-VL-7B**: Lightweight, runs on single GPU

```python
# Using smaller model
mllm = QwenMLLM(
    model_name="Qwen/Qwen3-VL-72B-Instruct",  # or other variants
    device_map="auto",
    dtype="bfloat16"
)
```

## Production Deployment

### Using vLLM (Recommended)

```bash
# Install vLLM
pip install vllm

# Serve model
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-235B-A22B-Instruct \
    --tensor-parallel-size 8 \
    --dtype bfloat16 \
    --max-model-len 8192
```

### Using Text Generation Inference (TGI)

```bash
# Using Docker
docker run --gpus all --shm-size 1g -p 8080:80 \
    -v $PWD/models:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id Qwen/Qwen3-VL-235B-A22B-Instruct \
    --num-shard 8 \
    --max-input-length 8192
```

## Resources

- [Qwen3-VL Model Card](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct)
- [Qwen GitHub](https://github.com/QwenLM/Qwen)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)
- [vLLM Documentation](https://docs.vllm.ai)

## Support

For issues specific to:
- **Qwen model**: [Qwen GitHub Issues](https://github.com/QwenLM/Qwen/issues)
- **Hegarty integration**: [Project Issues](https://github.com/yourusername/Hegarty-0.1/issues)
- **Transformers**: [Transformers Issues](https://github.com/huggingface/transformers/issues)

