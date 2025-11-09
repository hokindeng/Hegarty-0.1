# Hegarty Submodules

## Overview

Hegarty uses git submodules to integrate external libraries and frameworks. This document describes how to work with submodules.

## Current Submodules

### neuronx-distributed-inference

**Location:** `external/neuronx-distributed-inference/`  
**Repository:** https://github.com/aws-neuron/neuronx-distributed-inference/  
**Purpose:** AWS Neuron-optimized inference for Llama 3.2 Multimodal models  
**Used by:** `MLlamaMLLM` MLLM provider

**Features:**
- Llama 3.2 11B/90B Multimodal inference
- Optimized for AWS Trainium (Trn1) and Inferentia (Inf2) instances
- Tensor parallelism support
- High-performance image-text understanding

## Working with Submodules

### Initial Clone (with submodules)

```bash
# Clone Hegarty with all submodules
git clone --recursive https://github.com/your-org/Hegarty-0.1.git

# Or if already cloned without submodules:
git submodule update --init --recursive
```

### Update Submodules

```bash
# Update all submodules to latest
git submodule update --remote

# Update specific submodule
git submodule update --remote external/neuronx-distributed-inference
```

### Add New Submodule

```bash
# Add a new submodule
git submodule add <repository-url> <local-path>

# Example:
git submodule add https://github.com/example/repo.git external/example-repo
```

### Remove Submodule

```bash
# Remove submodule
git submodule deinit -f external/submodule-name
rm -rf .git/modules/external/submodule-name
git rm -f external/submodule-name
```

## Installation

### Install neuronx-distributed-inference

```bash
# Navigate to submodule
cd external/neuronx-distributed-inference

# Install in development mode
pip install -e .

# Install additional requirements
pip install -r examples/requirements.txt
```

### Verify Installation

```python
# Test import
from neuronx_distributed_inference.models.mllama.modeling_mllama import (
    MLlamaForConditionalGeneration
)
print("✓ neuronx-distributed-inference installed successfully")
```

## Using Submodule Code

### Example: Using neuronx-distributed-inference

```python
from hegarty.mllm import MLlamaMLLM

# MLlamaMLLM provider uses neuronx-distributed-inference internally
mllm = MLlamaMLLM(
    model_path="/path/to/neuron/checkpoint",
    compiled_model_path="/path/to/compiled/model",
    model_size="11B",
    tp_degree=32
)
```

### Direct Import (if needed)

```python
# Import directly from submodule
from neuronx_distributed_inference.models.mllama.modeling_mllama import (
    MLlamaForConditionalGeneration
)

# Use model directly
model = MLlamaForConditionalGeneration.from_pretrained(
    "/path/to/checkpoint",
    tp_degree=32
)
```

## Best Practices

### 1. Keep Submodules Updated

Regularly update submodules to get latest features and bug fixes:

```bash
# Check for updates
git submodule status

# Update to latest
git submodule update --remote
```

### 2. Pin to Specific Versions

For production, pin submodules to specific commits:

```bash
cd external/neuronx-distributed-inference
git checkout v0.6.10598  # Specific version
cd ../..
git add external/neuronx-distributed-inference
git commit -m "Pin neuronx-distributed-inference to v0.6.10598"
```

### 3. Document Dependencies

Always document which Hegarty features depend on which submodules.

### 4. Test After Updates

After updating submodules, run tests to ensure compatibility:

```bash
python3 verify_refactoring.py
pytest tests/
```

## Troubleshooting

### Submodule not initialized

```bash
# Error: submodule directory is empty
# Solution:
git submodule update --init --recursive
```

### Submodule conflicts

```bash
# If submodule has local changes
cd external/neuronx-distributed-inference
git status
git stash  # or git reset --hard
cd ../..
git submodule update --remote
```

### Permission issues

```bash
# If SSH key issues with GitHub
# Use HTTPS instead:
git config submodule.external/neuronx-distributed-inference.url \
  https://github.com/aws-neuron/neuronx-distributed-inference.git
```

## Future Submodules

Potential submodules to add:

- **Runway SDK** - For Runway Gen-3 video generation
- **Anthropic SDK** - For Claude MLLM support
- **Custom inference engines** - For specialized hardware

## Resources

- [Git Submodules Documentation](https://git-scm.com/book/en/v2/Git-Tools-Submodules)
- [neuronx-distributed-inference Repo](https://github.com/aws-neuron/neuronx-distributed-inference/)
- [AWS Neuron Documentation](https://awsdocs-neuron.readthedocs-hosted.com/)

## Summary

✅ **Modular**: Submodules keep external code separate  
✅ **Version Control**: Pin to specific versions for stability  
✅ **Clean**: No vendor lock-in, easy to update  
✅ **Flexible**: Add/remove submodules as needed

