# Hegarty Quick Start Guide

## Overview

Hegarty is now a modular framework for perspective-taking AI with pluggable MLLM and VM providers.

## Installation

```bash
pip install -r requirements.txt
```

## Basic Usage

### Option 1: Simple Client (Recommended)

```python
from hegarty import HergartyClient

# Initialize client
client = HergartyClient(
    openai_api_key="sk-...",
    sora_api_key="sk-..."  # Optional
)

# Use like OpenAI client
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

### Option 2: Agent with Custom Providers

```python
from hegarty import HergartyAgent, Config
from hegarty.mllm import OpenAIMLLM
from hegarty.vm import SoraVM
from openai import OpenAI

# Configure
config = Config(
    gpt_model="gpt-4o",
    sora_video_length=4,
    frame_extraction_strategy="uniform"
)

# Create agent
openai_client = OpenAI(api_key="sk-...")
agent = HergartyAgent(
    openai_client=openai_client,
    sora_api_key="sk-...",
    config=config
)

# Process
result = agent.process(
    image="data:image/jpeg;base64,...",
    question="Rotate this object 90 degrees. What do you see?",
    use_mental_rotation=True,
    return_intermediate=False
)

print(result['final_answer'])
print(f"Confidence: {result['confidence']:.2f}")
```

## Adding New Providers

### Add a New MLLM Provider

1. Create `hegarty/mllm/your_provider.py`:

```python
from .base import MLLMProvider
from typing import Tuple, List, Dict, Any

class YourMLLM(MLLMProvider):
    def __init__(self, api_key: str, model: str = "default"):
        # Initialize your client
        self.api_key = api_key
        self.model = model
    
    def detect_perspective(self, text: str) -> Tuple[bool, float]:
        # Use your MLLM to detect perspective-taking tasks
        # Return (is_perspective_task, confidence)
        pass
    
    def rephrase_for_video(self, question: str, image: str) -> str:
        # Use your MLLM to rephrase question for video generation
        # Return video prompt string
        pass
    
    def analyze_perspective(
        self, image: str, question: str, perspective_label: str,
        context: List[Dict] = None, temperature: float = 0.3,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        # Use your MLLM to analyze image from this perspective
        # Return {'perspective': label, 'analysis': text, 'tokens_used': int}
        pass
    
    def synthesize_perspectives(
        self, perspectives: List[Dict], original_question: str,
        consistency_score: float, context: List[Dict] = None
    ) -> Tuple[str, float]:
        # Use your MLLM to synthesize multiple perspectives
        # Return (final_answer, confidence)
        pass
```

2. Export in `hegarty/mllm/__init__.py`:

```python
from .your_provider import YourMLLM

__all__ = [..., "YourMLLM"]
```

3. Use it:

```python
from hegarty.mllm import YourMLLM

mllm = YourMLLM(api_key="...", model="...")
agent = HergartyAgent()
agent.mllm = mllm
```

### Add a New VM Provider

1. Create `hegarty/vm/your_provider.py`:

```python
from .base import VMProvider
from typing import Dict, Any, Optional
from pathlib import Path

class YourVM(VMProvider):
    def __init__(self, api_key: str, model: str = "default"):
        self.api_key = api_key
        self.model = model
    
    def generate_video(
        self, prompt: str, image: str, duration: int = 4,
        fps: int = 10, resolution: str = "1280x720",
        session_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        # 1. Convert base64 image to your format
        # 2. Call your video generation API
        # 3. Download generated video
        # 4. Return in standard format:
        
        return {
            'video_path': '/path/to/generated_video.mp4',
            'frames': [],  # Leave empty, FrameExtractor fills this
            'metadata': {
                'video_id': 'your-job-id',
                'duration': duration,
                'prompt': prompt,
                # Add any provider-specific metadata
            }
        }
```

2. Export in `hegarty/vm/__init__.py`:

```python
from .your_provider import YourVM

__all__ = [..., "YourVM"]
```

3. Use it:

```python
from hegarty.vm import YourVM

vm = YourVM(api_key="...", model="...")
agent = HergartyAgent()
agent.vm = vm
```

## Mix and Match Providers

```python
from hegarty import HergartyAgent
from hegarty.mllm import OpenAIMLLM, MLlamaMLLM, ClaudeMLLM  # Future: ClaudeMLLM
from hegarty.vm import SoraVM, RunwayVM  # Future: RunwayVM

# Use OpenAI for MLLM, Sora for VM
agent = HergartyAgent()
agent.mllm = OpenAIMLLM(...)
agent.vm = SoraVM(...)

# Or use Llama 3.2 Multimodal (on AWS Neuron), Sora for VM
agent.mllm = MLlamaMLLM(
    model_path="/path/to/neuron/checkpoint",
    compiled_model_path="/path/to/compiled/model",
    model_size="11B",  # or "90B"
    tp_degree=32
)
agent.vm = SoraVM(...)
```

## Configuration

```python
from hegarty import Config

config = Config(
    # MLLM settings
    gpt_model="gpt-4o",           # or "gpt-4o-mini"
    temperature=0.3,
    max_tokens=2000,
    
    # VM settings
    sora_video_length=4,          # 4, 8, or 12 seconds
    sora_fps=10,
    sora_resolution="1280x720",   # or "720x1280"
    
    # Frame extraction
    frame_extraction_count=5,     # Number of frames to extract
    frame_extraction_window=30,   # Last N frames to consider
    frame_extraction_strategy="uniform",  # uniform, adaptive, or keyframe
    
    # Parallel processing
    max_workers=6,                # Parallel perspective analyses
    timeout=30
)
```

## Examples

### Example 1: Simple Perspective Question

```python
from hegarty import HergartyClient
import base64

# Load image
with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()
    image_url = f"data:image/jpeg;base64,{image_data}"

# Initialize client
client = HergartyClient(openai_api_key="sk-...")

# Ask perspective question
response = client.chat.completions.create(
    model="hegarty-1.0",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "If I rotate this 180 degrees, what's on top?"},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
    }]
)

print(response.choices[0].message.content)
```

### Example 2: Custom Configuration

```python
from hegarty import HergartyAgent, Config
from openai import OpenAI

# Custom config
config = Config(
    gpt_model="gpt-4o-mini",      # Use cheaper model
    sora_video_length=8,          # Longer video
    frame_extraction_count=10,    # More frames
    frame_extraction_strategy="adaptive",  # Adaptive extraction
    max_workers=8                 # More parallel workers
)

# Create agent
agent = HergartyAgent(
    openai_client=OpenAI(api_key="sk-..."),
    sora_api_key="sk-...",
    config=config
)

# Process with custom config
result = agent.process(
    image=image_url,
    question="What would this look like from the other side?",
    use_mental_rotation=True,
    return_intermediate=True  # Get frames and perspectives
)

# Access results
print(result['final_answer'])
print(f"Confidence: {result['confidence']:.2f}")
print(f"Rephrased prompt: {result['rephrased_prompt']}")
print(f"Number of frames: {len(result['frames'])}")
print(f"Number of perspectives: {len(result['perspectives'])}")
```

### Example 3: Without Video Generation

```python
# Use Hegarty for MLLM enhancement without video generation
agent = HergartyAgent(
    openai_client=OpenAI(api_key="sk-..."),
    sora_api_key=None  # No video generation
)

result = agent.process(
    image=image_url,
    question="Describe this from multiple angles",
    use_mental_rotation=False  # Skip video generation
)

print(result['final_answer'])
```

## Environment Variables

Create `.env` file:

```bash
OPENAI_API_KEY=sk-...
SORA_API_KEY=sk-...
```

Then use:

```python
from hegarty import HergartyClient

# Automatically loads from .env
client = HergartyClient()
```

## Debugging

Enable debug logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)

# Now see detailed logs
client = HergartyClient(...)
response = client.chat.completions.create(...)
```

Access session artifacts:

```python
from pathlib import Path

session_dir = Path("./debug_session")

result = agent.process(
    ...,
    session_dir=session_dir
)

# Check session_dir for:
# - Input images
# - Generated videos
# - Extracted frames
# - MLLM call logs
```

## Testing

Run verification:

```bash
python3 verify_refactoring.py
```

Expected output:
```
🎉 All verifications passed!

✅ Refactoring complete and verified!

New structure:
  - hegarty/mllm/    (MLLM providers)
  - hegarty/vm/      (VM providers)
```

## Architecture

See `ARCHITECTURE.md` for detailed architecture documentation.

## Summary

✅ **Modular**: Plug in any MLLM or VM provider  
✅ **Succinct**: 54% less code than before  
✅ **Simple**: No try-catch blocks, clean interfaces  
✅ **Extensible**: Easy to add new providers  
✅ **Fast**: Parallel perspective analysis  
✅ **Debuggable**: Session management and logging  

## Future Providers

Coming soon:
- Claude MLLM provider
- Gemini MLLM provider  
- Runway VM provider
- Pika VM provider

Contribute your own provider! See templates in:
- `hegarty/mllm/anthropic_example.py`
- `hegarty/vm/runway_example.py`

