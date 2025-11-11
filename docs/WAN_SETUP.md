# Wan2.2 Setup Guide

This guide helps you set up Wan2.2, an open-source video generation model, as a video model provider in Hegarty.

## Overview

Wan2.2 is an advanced video generation model that supports:
- Text-to-Video (T2V) generation
- Image-to-Video (I2V) generation
- Character animation (Wan2.2-Animate)
- Multiple model sizes: 5B, 14B, and Animate-14B
- High-quality video generation up to 1920x1080 resolution

## Prerequisites

### Hardware Requirements

**Minimum Requirements:**
- NVIDIA GPU with 16GB VRAM (for 5B model with optimizations)
- 32GB system RAM
- 100GB free disk space

**Recommended Requirements:**
- NVIDIA GPU with 24GB+ VRAM (RTX 3090/4090, A5000, etc.)
- 64GB system RAM
- 200GB free disk space
- NVMe SSD for faster model loading

**Multi-GPU Setup (for 14B models):**
- 2-8 GPUs with 24GB+ VRAM each
- CUDA 11.8 or higher
- NCCL for distributed training

### Software Requirements

- Python 3.8-3.10
- CUDA 11.8 or 12.1
- PyTorch 2.0+
- Linux (Ubuntu 20.04/22.04 recommended)

## Installation

### 1. Install Dependencies

First, activate your Hegarty virtual environment:

```bash
cd /path/to/Hegarty-0.1
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

### 2. Download Wan2.2 Repository

Clone the official Wan2.2 repository:

```bash
cd ~
git clone https://github.com/Wan-Video/Wan2.2.git
cd Wan2.2
```

### 3. Download Model Weights

Choose the model size based on your hardware:

#### Option A: 5B Model (Recommended for single GPU)

```bash
# Create model directory
mkdir -p ~/.cache/wan_models

# Download 5B model
cd ~/.cache/wan_models
wget https://huggingface.co/wan-video/Wan2.2-TI2V-5B/resolve/main/Wan2.2-TI2V-5B.tar.gz
tar -xzf Wan2.2-TI2V-5B.tar.gz
```

#### Option B: 14B Model (Requires high-end GPU or multi-GPU)

```bash
# Download 14B model
cd ~/.cache/wan_models
wget https://huggingface.co/wan-video/Wan2.2-14B/resolve/main/Wan2.2-14B.tar.gz
tar -xzf Wan2.2-14B.tar.gz
```

#### Option C: Animate-14B Model (For character animation)

```bash
# Download Animate model
cd ~/.cache/wan_models
wget https://huggingface.co/wan-video/Wan2.2-Animate-14B/resolve/main/Wan2.2-Animate-14B.tar.gz
tar -xzf Wan2.2-Animate-14B.tar.gz
```

### 4. Install Additional Dependencies

For TI2V models:
```bash
cd ~/Wan2.2
pip install -r requirements_t2v.txt
```

For Animate model:
```bash
cd ~/Wan2.2
pip install -r requirements_animate.txt
```

### 5. Set Environment Variables

Add to your `.bashrc` or `.zshrc`:

```bash
export WAN_REPO_PATH="$HOME/Wan2.2"
export WAN_MODEL_PATH="$HOME/.cache/wan_models"
```

## Usage in Hegarty

### Basic Text-to-Video Generation

```python
from hegarty.vm import WanVM

# Initialize Wan2.2
wan = WanVM(
    model_size="5B",
    device="cuda",
    offload_model=True,  # Save memory
    convert_model_dtype=True  # Use FP16 for faster inference
)

# Generate video from text
result = wan.generate_video(
    prompt="A beautiful sunset over the ocean with gentle waves",
    duration=5,  # seconds
    fps=24,
    resolution="1280x720"
)

print(f"Video saved to: {result['video_path']}")
```

### Image-to-Video Generation

```python
import base64

# Load and encode image
with open("input_image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode('utf-8')
    image_data = f"data:image/jpeg;base64,{image_base64}"

# Generate video from image
result = wan.generate_video(
    prompt="Camera slowly pans across the landscape",
    image=image_data,
    duration=5,
    fps=24,
    resolution="1280x720"
)
```

### Character Animation (Wan2.2-Animate)

```python
# Initialize Animate model
wan = WanVM(
    model_size="animate-14B",
    device="cuda",
    offload_model=True,
    use_prompt_extend=True  # Better results
)

# Animate character
result = wan.generate_video(
    prompt="The person is smiling and waving",
    image=character_image_base64,
    duration=3,
    fps=24,
    resolution="720x1280"  # Portrait for characters
)
```

### Multi-GPU Setup

For multiple GPUs, use FSDP and DeepSpeed Ulysses:

```python
wan = WanVM(
    model_size="14B",
    device="cuda",
    use_fsdp=True,
    ulysses_size=4,  # Number of GPUs
    offload_model=False
)
```

Run with:
```bash
python -m torch.distributed.run --nnodes 1 --nproc_per_node 4 your_script.py
```

## Performance Optimization

### Memory Optimization

1. **Model Offloading**: Move model to CPU when not in use
   ```python
   wan = WanVM(offload_model=True)
   ```

2. **Mixed Precision**: Use FP16/BF16
   ```python
   wan = WanVM(convert_model_dtype=True)
   ```

3. **CPU Offload for T5**: Offload text encoder
   ```python
   wan = WanVM(model_size="5B", device="cuda")
   # Add --t5_cpu flag in generate command
   ```

### Speed Optimization

1. **Flash Attention**: Automatically enabled on supported GPUs
2. **Compilation**: PyTorch 2.0 compilation for faster inference
3. **Batch Processing**: Process multiple videos in parallel

## Troubleshooting

### Out of Memory Errors

```
Error: CUDA out of memory
```

Solutions:
1. Enable model offloading: `offload_model=True`
2. Use smaller model (5B instead of 14B)
3. Reduce resolution or duration
4. Use gradient checkpointing

### Model Not Found

```
Error: Model not found at ~/.cache/wan_models/Wan2.2-TI2V-5B
```

Solutions:
1. Check model is downloaded to correct path
2. Set `model_dir` parameter explicitly
3. Verify extraction completed successfully

### Generation Quality Issues

1. **Blurry videos**: Increase inference steps
2. **Inconsistent motion**: Use prompt extension
3. **Poor composition**: Provide more detailed prompts

### CUDA/Driver Issues

```
Error: CUDA runtime error
```

Solutions:
1. Verify CUDA version: `nvidia-smi`
2. Check PyTorch CUDA compatibility
3. Update NVIDIA drivers

## Advanced Configuration

### Custom Model Paths

```python
wan = WanVM(
    model_size="5B",
    model_dir="/custom/path/to/model"
)
```

### Distributed Training Configuration

For large-scale deployment:

```python
# In your script
wan = WanVM(
    model_size="14B",
    use_fsdp=True,
    ulysses_size=8,
    dit_fsdp=True,
    t5_fsdp=True
)
```

Run with:
```bash
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 --master_addr=master_ip --master_port=12355 script.py
```

## Model Comparison

| Model | Parameters | VRAM Required | Speed (5s video) | Best For |
|-------|------------|---------------|------------------|-----------|
| 5B | 5B | 16GB | ~5 min | Single GPU, fast iteration |
| 14B | 14B active/27B total | 24GB+ | ~8 min | High quality, multi-GPU |
| Animate-14B | 14B | 24GB+ | ~10 min | Character animation |

## Example Outputs

See the `examples/wan_example.py` script for complete examples:

```bash
cd Hegarty-0.1
python examples/wan_example.py --example text2video
python examples/wan_example.py --example image2video
python examples/wan_example.py --example animate
```

## References

- [Wan2.2 GitHub Repository](https://github.com/Wan-Video/Wan2.2)
- [Wan2.2 Paper](https://arxiv.org/abs/2503.20314)
- [Model Weights on HuggingFace](https://huggingface.co/wan-video)
- [Official Demo](https://wan.video)

## Support

For issues specific to Wan2.2 integration in Hegarty:
1. Check this documentation
2. Review error logs in `temp/sessions/`
3. Create an issue in the Hegarty repository

For Wan2.2 model issues:
1. Check the [official repository](https://github.com/Wan-Video/Wan2.2)
2. Join their Discord or WeChat groups
3. Review existing issues on GitHub
