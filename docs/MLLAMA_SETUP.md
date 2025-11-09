# Llama 3.2 Multimodal (mllama) Setup Guide

## Overview

Hegarty now supports **Llama 3.2 Multimodal** (11B and 90B models) as an MLLM provider, running on AWS Neuron accelerators (Trainium/Inferentia). This integration uses the [neuronx-distributed-inference](https://github.com/aws-neuron/neuronx-distributed-inference/) library.

## Prerequisites

### 1. Hardware Requirements

- **AWS Trn1 instances** (Trainium) or **Inf2 instances** (Inferentia)
- Recommended: `trn1.32xlarge` for optimal performance
- Tensor parallelism degree: 32 (for trn1.32xlarge)

### 2. Software Requirements

```bash
# Install AWS Neuron SDK
pip install neuronx-cc==2.* torch-neuronx torchvision

# Install neuronx-distributed-inference (submodule)
cd external/neuronx-distributed-inference
pip install -e .

# Install additional requirements
pip install -r examples/requirements.txt
```

## Model Preparation

### Step 1: Download Llama 3.2 Multimodal Checkpoint

Download from [Meta's official website](https://www.llama.com/llama-downloads/):
- Llama 3.2 11B Instruct
- Llama 3.2 90B Instruct

```bash
# After downloading, you should have:
# /path/to/Llama-3.2-11B-Vision-Instruct/
# or
# /path/to/Llama-3.2-90B-Vision-Instruct/
```

### Step 2: Convert to Neuron Checkpoint

```bash
cd external/neuronx-distributed-inference/examples

python checkpoint_conversion_utils/convert_mllama_weights_to_neuron.py \
  --input-dir /path/to/Llama-3.2-11B-Vision-Instruct/ \
  --output-dir /path/to/neuron_checkpoint/Llama-3.2-11B/ \
  --instruct
```

### Step 3: Compile Model for Neuron

The model will be compiled on first use, or you can pre-compile:

```python
from neuronx_distributed_inference.models.mllama.modeling_mllama import (
    MLlamaForConditionalGeneration
)

model = MLlamaForConditionalGeneration.from_pretrained(
    "/path/to/neuron_checkpoint/Llama-3.2-11B/",
    tp_degree=32,
    batch_size=1,
    max_context_length=2048,
    seq_len=2176
)

# Save compiled model
model.save_compiled("/path/to/compiled_model/Llama-3.2-11B/")
```

## Usage in Hegarty

### Basic Usage

```python
from hegarty import HergartyAgent
from hegarty.mllm import MLlamaMLLM
from hegarty.vm import SoraVM

# Initialize mllama MLLM provider
mllm = MLlamaMLLM(
    model_path="/path/to/neuron_checkpoint/Llama-3.2-11B/",
    compiled_model_path="/path/to/compiled_model/Llama-3.2-11B/",
    model_size="11B",  # or "90B"
    tp_degree=32,      # Tensor parallelism degree
    batch_size=1,
    max_context_length=2048,
    seq_len=2176,
    temperature=0.3,
    max_tokens=2000
)

# Create agent with mllama
agent = HergartyAgent()
agent.mllm = mllm
agent.vm = SoraVM(api_key="sk-...")

# Process perspective-taking query
result = agent.process(
    image="data:image/jpeg;base64,...",
    question="If I rotate this object 90 degrees clockwise, what's on top?",
    use_mental_rotation=True
)

print(result['final_answer'])
print(f"Confidence: {result['confidence']:.2f}")
```

### Complete Example

```python
from hegarty import HergartyAgent, Config
from hegarty.mllm import MLlamaMLLM
from hegarty.vm import SoraVM
from pathlib import Path
import base64

# Load image
with open("test_image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()
    image_url = f"data:image/jpeg;base64,{image_data}"

# Configure
config = Config(
    sora_video_length=4,
    frame_extraction_count=5,
    frame_extraction_strategy="uniform",
    max_workers=6
)

# Initialize mllama on Neuron
mllm = MLlamaMLLM(
    model_path="/home/ubuntu/models/Llama-3.2-11B-neuron/",
    compiled_model_path="/home/ubuntu/compiled/Llama-3.2-11B/",
    model_size="11B",
    tp_degree=32,
    batch_size=1,
    temperature=0.3
)

# Initialize agent
agent = HergartyAgent(
    openai_client=None,  # Not needed with mllama
    sora_api_key="sk-...",
    config=config
)

# Set custom MLLM provider
agent.mllm = mllm

# Run perspective-taking task
result = agent.process(
    image=image_url,
    question="What would this scene look like from the opposite side?",
    use_mental_rotation=True,
    return_intermediate=True
)

# Results
print("Final Answer:", result['final_answer'])
print("Confidence:", result['confidence'])
print("Perspectives analyzed:", len(result['perspectives']))
```

## Model Specifications

### Llama 3.2 11B Multimodal

- **Parameters**: 11 billion
- **Vision Encoder**: High-resolution image understanding
- **Context Length**: Up to 128K tokens
- **Recommended Instance**: `trn1.32xlarge`
- **Memory**: ~40GB

### Llama 3.2 90B Multimodal

- **Parameters**: 90 billion
- **Vision Encoder**: Advanced spatial reasoning
- **Context Length**: Up to 128K tokens
- **Recommended Instance**: `trn1.32xlarge` (multiple instances for larger batch)
- **Memory**: ~200GB

## Performance Tuning

### Optimization Tips

1. **Tensor Parallelism**: Use `tp_degree=32` for trn1.32xlarge
2. **Batch Size**: Start with `batch_size=1`, increase based on memory
3. **Context Length**: Adjust `max_context_length` based on your use case
4. **Sequence Length**: Set `seq_len` slightly larger than max_context_length

### Example Configurations

**Low Latency (11B):**
```python
mllm = MLlamaMLLM(
    model_size="11B",
    tp_degree=32,
    batch_size=1,
    max_context_length=1024,
    seq_len=1152
)
```

**High Throughput (11B):**
```python
mllm = MLlamaMLLM(
    model_size="11B",
    tp_degree=32,
    batch_size=4,
    max_context_length=2048,
    seq_len=2176
)
```

**High Quality (90B):**
```python
mllm = MLlamaMLLM(
    model_size="90B",
    tp_degree=32,
    batch_size=1,
    max_context_length=4096,
    seq_len=4224
)
```

## Troubleshooting

### Common Issues

**1. Out of Memory**
- Reduce `batch_size`
- Reduce `max_context_length`
- Use 11B instead of 90B

**2. Compilation Time**
- First run takes 10-30 minutes (compilation)
- Subsequent runs use cached compiled model
- Pre-compile models for production

**3. Neuron Runtime Errors**
- Ensure correct Neuron SDK version
- Check instance type supports Neuron
- Verify `tp_degree` matches instance capacity

## Advantages of mllama on Neuron

### vs OpenAI GPT-4o

✅ **Cost**: ~10x cheaper per token  
✅ **Privacy**: On-premise, no data leaves your infrastructure  
✅ **Latency**: Lower latency for high throughput  
✅ **Customization**: Can fine-tune on your data  

❌ **Setup**: Requires model download and compilation  
❌ **Infrastructure**: Needs AWS Neuron instances  

### Use Cases

- **Production deployments** requiring cost efficiency
- **Privacy-sensitive** applications
- **High-throughput** batch processing
- **Custom fine-tuned** models

## References

- [AWS Neuron Documentation](https://awsdocs-neuron.readthedocs-hosted.com/)
- [neuronx-distributed-inference GitHub](https://github.com/aws-neuron/neuronx-distributed-inference/)
- [Llama 3.2 Model Card](https://www.llama.com/)
- [Meta Llama Downloads](https://www.llama.com/llama-downloads/)

## Support

For issues with:
- **Hegarty integration**: Open issue in Hegarty repo
- **Neuron compilation**: Check [AWS Neuron docs](https://awsdocs-neuron.readthedocs-hosted.com/)
- **Model access**: Contact Meta via official channels

## Future Enhancements

- [ ] Support for batch perspective analysis
- [ ] Multi-image input support
- [ ] Fine-tuning integration
- [ ] Quantization support for memory efficiency
- [ ] Automatic compilation caching

