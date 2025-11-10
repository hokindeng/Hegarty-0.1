# Hegarty: Perspective-Taking Artificial Intelligence

A modular framework for enhanced spatial reasoning with MLLM (Multimodal Large Language Model) and VM (Video Model).

## Quick Start

```bash
# Install
pip install -e .

# Set up environment
cp env_template.txt .env
# Add your API keys to .env

# Use it
python3 -c "from hegarty import HergartyClient; print('Ready!')"
```

## Basic Usage

```python
from hegarty import HergartyClient

client = HergartyClient()

response = client.chat.completions.create(
    model="hegarty-1.0",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Is the bag on the watermelon's left?"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ]
    }]
)

print(response.choices[0].message.content)
```

## Architecture

```
┌─────────────────────────────────────┐
│         HergartyClient              │  OpenAI-compatible API
└────────────┬────────────────────────┘
             │
┌────────────▼────────────────────────┐
│         HergartyAgent               │  Core orchestrator
└──┬─────────┬─────────┬──────────────┘
   │         │         │
   ▼         ▼         ▼
┌─────┐  ┌─────┐  ┌──────────┐
│MLLM │  │ VM  │  │  Frame   │
│     │  │     │  │Extractor │
└──┬──┘  └──┬──┘  └──────────┘
   │        │
   ▼        ▼
OpenAI    Sora-2
Claude    Runway
Gemini    Pika
mllama    ...
```

## Modular Providers

### MLLM Providers (Multimodal LLMs)

**Current:**
- `OpenAIMLLM` - GPT-4o, GPT-4o-mini
- `QwenMLLM` - Qwen2.5-VL-7B (Hugging Face Transformers)
- `MLlamaMLLM` - Llama 3.2 11B/90B (AWS Neuron)

**Add your own:**

```python
from hegarty.mllm import MLLMProvider

class YourMLLM(MLLMProvider):
    def detect_perspective(self, text): ...
    def rephrase_for_video(self, question, image): ...
    def analyze_perspective(self, ...): ...
    def synthesize_perspectives(self, ...): ...
```

### VM Providers (Video Models)

**Current:**
- `SoraVM` - Sora-2, Sora-2-Pro

**Add your own:**

```python
from hegarty.vm import VMProvider

class YourVM(VMProvider):
    def generate_video(self, prompt, image, ...):
        return {'video_path': '...', 'frames': [], 'metadata': {...}}
```

## Custom Configuration

```python
from hegarty import HergartyAgent, Config
from hegarty.mllm import OpenAIMLLM
from hegarty.vm import SoraVM

config = Config(
    gpt_model="gpt-4o",
    sora_video_length=4,
    frame_extraction_count=5,
    max_workers=6
)

agent = HergartyAgent(config=config)
agent.mllm = OpenAIMLLM(...)
agent.vm = SoraVM(...)

result = agent.process(
    image="data:image/jpeg;base64,...",
    question="What would this look like rotated 90 degrees?",
    use_mental_rotation=True
)
```

## Project Structure

```
hegarty/
├── mllm/              # MLLM providers
│   ├── base.py        # Base interface
│   ├── openai.py      # OpenAI implementation
│   ├── qwen.py        # Qwen3-VL implementation
│   └── mllama.py      # Llama 3.2 implementation
├── vm/                # Video model providers
│   ├── base.py        # Base interface
│   └── sora.py        # Sora implementation
├── agent.py           # Core orchestrator
├── client.py          # OpenAI-compatible client
├── synthesizer.py     # Perspective synthesis
├── frame_extractor.py # Frame extraction
└── config.py          # Configuration
```

## Documentation

- **`ARCHITECTURE.md`** - Detailed architecture and design principles
- **`docs/QWEN_SETUP.md`** - Setting up Qwen3-VL with Hugging Face Transformers
- **`docs/MLLAMA_SETUP.md`** - Setting up Llama 3.2 on AWS Neuron

## Web Interface

```bash
python hegarty_app.py
# Visit http://localhost:7860
```

## Key Features

- **Modular**: Swap MLLM/VM providers easily
- **Succinct**: Minimal code, no try-catch blocks
- **Parallel**: Concurrent perspective analysis
- **Debuggable**: Session management and logging
- **Extensible**: Add providers in <100 lines

## Examples

### Mix and Match Providers

```python
from hegarty.mllm import OpenAIMLLM, QwenMLLM, MLlamaMLLM
from hegarty.vm import SoraVM

# Use OpenAI + Sora
agent = HergartyAgent()
agent.mllm = OpenAIMLLM(...)
agent.vm = SoraVM(...)

# Or use Qwen3-VL (local deployment)
agent.mllm = QwenMLLM(
    model_name="Qwen/Qwen2.5-VL-7B-Instruct",
    device_map="auto",
    attn_implementation="flash_attention_2"  # optional, for better performance
)

# Or use Llama 3.2 + Sora (AWS Neuron)
agent.mllm = MLlamaMLLM(
    model_path="/path/to/checkpoint",
    model_size="11B",
    tp_degree=32
)
```

### Environment Variables

```bash
# .env
OPENAI_API_KEY=sk-...
SORA_API_KEY=sk-...
```

```python
from hegarty import HergartyClient

client = HergartyClient()  # Loads from .env
```

## Requirements

- Python 3.9+
- OpenAI API key (for OpenAI MLLM)
- Sora API key (optional, for video generation)
- PyTorch + Transformers (for Qwen3-VL, requires GPU with sufficient VRAM)
- AWS Neuron instances (optional, for Llama 3.2)

## License

MIT License - see LICENSE file

## Citation

```bibtex
@software{hegarty2024,
  title = {Hegarty: A Modular Perspective-Taking Agent},
  year = {2024},
  url = {https://github.com/yourusername/Hegarty-0.1}
}
```

Named after Mary Hegarty, pioneering researcher in spatial cognition.
