# Hegarty Architecture

## Overview

Hegarty is a modular perspective-taking agent for enhanced spatial reasoning. The architecture is designed to easily swap MLLM (Multimodal Large Language Model) and VM (Video Model) providers.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     HergartyClient                          │
│                  (OpenAI-compatible API)                    │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                     HergartyAgent                           │
│                  (Core Orchestrator)                        │
└─────┬──────────────┬──────────────┬──────────────┬─────────┘
      │              │              │              │
      ▼              ▼              ▼              ▼
┌──────────┐  ┌───────────┐  ┌──────────┐  ┌──────────────┐
│   MLLM   │  │    VM     │  │  Frame   │  │ Perspective  │
│ Provider │  │ Provider  │  │Extractor │  │ Synthesizer  │
└────┬─────┘  └─────┬─────┘  └──────────┘  └──────────────┘
     │              │
     ▼              ▼
┌──────────┐  ┌──────────┐
│ OpenAI   │  │  Sora    │
│GPT-4o/4o │  │ Sora-2   │
│  -mini   │  │Sora-2-Pro│
└──────────┘  └──────────┘

Future:           Future:
┌──────────┐  ┌──────────┐
│ Claude   │  │ Runway   │
│ Gemini   │  │  Pika    │
└──────────┘  └──────────┘
```

## Directory Structure

```
hegarty/
├── mllm/                    # MLLM Provider Module
│   ├── __init__.py          # Export MLLMProvider, OpenAIMLLM, MLlamaMLLM
│   ├── base.py              # Base MLLMProvider interface
│   ├── openai.py            # OpenAI GPT-4o implementation
│   ├── mllama.py            # Llama 3.2 Multimodal implementation
│   └── [future providers]
│
├── vm/                      # Video Model Provider Module
│   ├── __init__.py          # Export VMProvider, SoraVM
│   ├── base.py              # Base VMProvider interface
│   ├── sora.py              # Sora-2 implementation
│   └── [future providers]
│
├── agent.py                 # Core orchestrator
├── client.py                # OpenAI-compatible client
├── synthesizer.py           # Perspective synthesis
├── frame_extractor.py       # Frame extraction
├── config.py                # Configuration
└── __init__.py              # Package exports
```

## Core Components

### 1. MLLM Providers (`hegarty/mllm/`)

**Base Interface:** `MLLMProvider`

All MLLM providers must implement:
- `detect_perspective(text: str) -> Tuple[bool, float]`
- `rephrase_for_video(question: str, image: str) -> str`
- `analyze_perspective(...) -> Dict[str, Any]`
- `synthesize_perspectives(...) -> Tuple[str, float]`

**Current Implementation:**
- `OpenAIMLLM` - GPT-4o, GPT-4o-mini
- `MLlamaMLLM` - Llama 3.2 Multimodal 11B/90B (AWS Neuron accelerators)

**Future Providers:**
- `AnthropicMLLM` - Claude 3.5 Sonnet/Opus
- `GoogleMLLM` - Gemini 1.5 Pro/Flash

### 2. VM Providers (`hegarty/vm/`)

**Base Interface:** `VMProvider`

All VM providers must implement:
- `generate_video(prompt, image, duration, fps, resolution, session_dir) -> Dict[str, Any]`

**Return format:**
```python
{
    'video_path': str,  # Path to generated video
    'frames': [],       # Empty (populated by FrameExtractor)
    'metadata': {       # Provider-specific metadata
        'video_id': str,
        'duration': int,
        'prompt': str,
        ...
    }
}
```

**Current Implementation:**
- `SoraVM` - Sora-2, Sora-2-Pro

**Future Providers:**
- `RunwayVM` - Runway Gen-3
- `PikaVM` - Pika 1.5
- `StabilityVM` - Stable Video Diffusion

### 3. Agent (`agent.py`)

Core orchestrator that:
1. Receives perspective-taking query
2. Rephrases using MLLM
3. Generates video using VM
4. Extracts frames
5. Analyzes perspectives in parallel
6. Synthesizes final answer

**Key Features:**
- Provider-agnostic (works with any MLLM/VM)
- Parallel perspective analysis
- ThreadPoolExecutor for concurrency
- Session management for debugging

### 4. Client (`client.py`)

OpenAI-compatible interface that:
- Mimics OpenAI API (`client.chat.completions.create()`)
- Auto-detects perspective-taking tasks
- Routes to Hegarty pipeline or standard GPT-4o
- Drop-in replacement for OpenAI client

### 5. Frame Extractor (`frame_extractor.py`)

Extracts key frames from videos using:
- **Uniform**: Equal spacing
- **Adaptive**: Based on visual change
- **Keyframe**: Maximum information content

### 6. Synthesizer (`synthesizer.py`)

Combines multiple perspective analyses:
- Calculates consistency across perspectives
- Uses MLLM to synthesize final answer
- Returns confidence score

## Data Flow

```
1. User Query (text + image)
        ↓
2. Perspective Detection (MLLM)
        ↓
3. Video Prompt Rephrasing (MLLM)
        ↓
4. Video Generation (VM)
        ↓
5. Frame Extraction
        ↓
6. Parallel Perspective Analysis (MLLM)
   - Original image
   - Frame 1, Frame 2, ..., Frame N
        ↓
7. Perspective Synthesis (MLLM)
        ↓
8. Final Answer + Confidence
```

## Key Design Principles

### 1. **Modularity**
- Easy to add new MLLM providers
- Easy to add new VM providers
- Components are loosely coupled

### 2. **Simplicity**
- No try-catch blocks (fail fast)
- Succinct implementations
- Clear interfaces

### 3. **Provider Agnostic**
- Agent doesn't know which MLLM/VM it's using
- Swap providers via configuration
- Standard interfaces ensure compatibility

### 4. **Extensibility**
- Base classes define contracts
- New providers implement interfaces
- No changes to core agent needed

## Adding New Providers

### Adding a New MLLM Provider

1. Create `hegarty/mllm/your_provider.py`:

```python
from .base import MLLMProvider

class YourMLLM(MLLMProvider):
    def __init__(self, api_key: str, model: str, ...):
        # Initialize your client
        pass
    
    def detect_perspective(self, text: str):
        # Implement using your MLLM
        pass
    
    def rephrase_for_video(self, question: str, image: str):
        # Implement using your MLLM
        pass
    
    def analyze_perspective(self, image, question, perspective_label, ...):
        # Implement using your MLLM
        pass
    
    def synthesize_perspectives(self, perspectives, original_question, ...):
        # Implement using your MLLM
        pass
```

2. Export in `hegarty/mllm/__init__.py`:

```python
from .your_provider import YourMLLM

__all__ = [..., "YourMLLM"]
```

3. Use in agent:

```python
from hegarty.mllm import YourMLLM

mllm = YourMLLM(api_key="...", model="...")
agent = HergartyAgent(mllm=mllm, ...)
```

### Adding a New VM Provider

1. Create `hegarty/vm/your_provider.py`:

```python
from .base import VMProvider

class YourVM(VMProvider):
    def __init__(self, api_key: str, model: str, ...):
        # Initialize your client
        pass
    
    def generate_video(self, prompt, image, duration, fps, resolution, session_dir):
        # Generate video
        # Download to session_dir
        # Return dict with video_path, frames, metadata
        return {
            'video_path': '/path/to/video.mp4',
            'frames': [],
            'metadata': {...}
        }
```

2. Export in `hegarty/vm/__init__.py`:

```python
from .your_provider import YourVM

__all__ = [..., "YourVM"]
```

3. Use in agent:

```python
from hegarty.vm import YourVM

vm = YourVM(api_key="...", model="...")
agent = HergartyAgent(vm=vm, ...)
```

## Configuration

### Config Object (`config.py`)

```python
@dataclass
class Config:
    # MLLM settings
    gpt_model: str = "gpt-4o"
    temperature: float = 0.3
    max_tokens: int = 2000
    
    # VM settings
    sora_video_length: int = 4
    sora_fps: int = 10
    sora_resolution: str = "1280x720"
    
    # Frame extraction
    frame_extraction_count: int = 5
    frame_extraction_window: int = 30
    frame_extraction_strategy: str = "uniform"
    
    # Parallel processing
    max_workers: int = 6
    timeout: int = 30
```

## Usage Examples

### Basic Usage

```python
from hegarty import HergartyClient

client = HergartyClient(
    openai_api_key="sk-...",
    sora_api_key="sk-..."
)

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

### Custom Providers

```python
from hegarty import HergartyAgent, Config
from hegarty.mllm import OpenAIMLLM
from hegarty.vm import SoraVM
from openai import OpenAI

config = Config(
    gpt_model="gpt-4o",
    sora_video_length=8,
    frame_extraction_strategy="adaptive"
)

openai_client = OpenAI(api_key="sk-...")

agent = HergartyAgent(
    openai_client=openai_client,
    sora_api_key="sk-...",
    config=config
)

result = agent.process(
    image="data:image/jpeg;base64,...",
    question="Rotate this 90 degrees clockwise. What do you see?",
    use_mental_rotation=True
)

print(result['final_answer'])
print(f"Confidence: {result['confidence']:.2f}")
```

### Mix and Match Providers

```python
from hegarty import HergartyAgent
from hegarty.mllm import OpenAIMLLM
from hegarty.vm import RunwayVM  # Future provider
from openai import OpenAI

# Use OpenAI for MLLM, Runway for VM
mllm = OpenAIMLLM(client=OpenAI(), model="gpt-4o")
vm = RunwayVM(api_key="...", model="gen3a_turbo")

agent = HergartyAgent()
agent.mllm = mllm
agent.vm = vm

# Process uses OpenAI + Runway
result = agent.process(...)
```

## Benefits of New Architecture

### 1. **No Try-Catch Blocks**
- Fail fast philosophy
- Errors propagate naturally
- Easier debugging
- Cleaner code

### 2. **Succinct Code**
- Removed ~500 lines of duplicate code
- Each file has single responsibility
- Clear interfaces
- Minimal boilerplate

### 3. **Modular Providers**
- Add new MLLM in <100 lines
- Add new VM in <150 lines
- No changes to core agent
- Test providers independently

### 4. **Easy Testing**
- Mock MLLM providers
- Mock VM providers
- Test components in isolation
- Integration tests with real providers

### 5. **Future-Proof**
- New models easy to integrate
- API changes isolated to provider
- Core logic remains stable
- Backward compatible

## Performance

### Parallel Processing

Agent uses ThreadPoolExecutor for parallel perspective analysis:

```python
# Analyzes all perspectives simultaneously
futures = [
    executor.submit(mllm.analyze_perspective, image=img, ...)
    for img in [original] + frames
]

perspectives = [future.result() for future in as_completed(futures)]
```

**Speedup:** N perspectives analyzed in ~1x time instead of Nx time

### Session Management

All artifacts saved to session directory:
```
temp/hegarty_session_20250108_123456/
├── sora_input_1234567890.png      # Input image
├── sora_video_1234567890.mp4      # Generated video
├── frames/                         # Extracted frames
│   ├── frame_000.png
│   ├── frame_001.png
│   └── ...
└── mllm_call_001_detect_perspective_*.json  # MLLM logs
    mllm_call_002_rephrase_for_video_*.json
    mllm_call_003_analyze_original_*.json
    ...
```

## Future Enhancements

### Near Term
1. Add Anthropic Claude support
2. Add Google Gemini support
3. Add Runway Gen-3 support
4. Add Pika support

### Medium Term
1. Caching layer for MLLM responses
2. Rate limiting and retry logic
3. Cost tracking per provider
4. Streaming support

### Long Term
1. Local VM support (CogVideoX)
2. Batch processing
3. Fine-tuning support

## Migration from Old Architecture

Old code continues to work:

```python
# Old imports still work
from hegarty import HergartyClient, HergartyAgent

# Old API still works
client = HergartyClient(openai_api_key="...")
response = client.chat.completions.create(...)
```

New capabilities available:

```python
# New modular providers
from hegarty.mllm import OpenAIMLLM
from hegarty.vm import SoraVM

# Mix and match as needed
mllm = OpenAIMLLM(...)
vm = SoraVM(...)
```

## Summary

The new Hegarty architecture is:
- ✅ **Modular**: Easy to add MLLM/VM providers
- ✅ **Succinct**: Removed 500+ lines of code
- ✅ **Simple**: No try-catch, clear interfaces
- ✅ **Extensible**: Base classes define contracts
- ✅ **Fast**: Parallel processing
- ✅ **Debuggable**: Session management and logging
- ✅ **Future-proof**: Ready for new models
- ✅ **Backward compatible**: Old code still works

