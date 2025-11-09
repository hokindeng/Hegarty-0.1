# MLLM Refactoring Summary

## Overview

Centralized all Multimodal Large Language Model (MLLM) calls into a single module `call_mllm.py`, eliminating duplicate code and creating a unified interface for all GPT-4o interactions.

## Changes Made

### 1. Created `hegarty/call_mllm.py` 

**New centralized MLLM interface with:**

- `MLLMCaller` class - Main class handling all MLLM interactions
- Built-in request/response logging for debugging
- Four main MLLM operations:
  1. `detect_perspective()` - Detect if question requires perspective-taking
  2. `rephrase_for_video()` - Convert question to video generation prompt
  3. `analyze_perspective()` - Analyze image from single perspective
  4. `synthesize_perspectives()` - Combine multiple perspective analyses

**Key features:**
- Automatic call logging to JSON files
- Base64 image truncation for readability
- Sequential call numbering
- Error handling and fallbacks
- Utility methods for image encoding

### 2. Refactored `hegarty/agent.py`

**Before:**
- Agent made direct `openai_client.chat.completions.create()` calls
- Had its own `_save_gpt_call()` and `_sanitize_messages_for_json()` methods
- Duplicated logging logic

**After:**
- Initializes `MLLMCaller` instance
- All MLLM calls go through `mllm_caller`
- Removed duplicate logging methods (70+ lines removed)
- Simplified methods:
  - `_rephrase_for_sora()` - Now 2 lines (was 30 lines)
  - `_analyze_single_perspective()` - Now 8 lines (was 40 lines)
  - `_fallback_completion()` - Uses `mllm_caller.call()` directly

**Code reduction:** ~140 lines removed, replaced with clean API calls

### 3. Refactored `hegarty/synthesizer.py`

**Before:**
- Synthesizer made direct OpenAI API calls
- Had its own `_save_gpt_call()` and `_sanitize_messages_for_json()` methods
- Managed its own call counter and session directory

**After:**
- Accepts `mllm_caller` parameter in `synthesize()` method
- Removed duplicate logging methods (70+ lines removed)
- Delegates synthesis to `mllm_caller.synthesize_perspectives()`
- Simplified initialization (no longer needs session tracking)

**Code reduction:** ~80 lines removed

### 4. Renamed `gpt_detector.py` → `call_mllm.py`

- Integrated detection logic into `MLLMCaller`
- Kept `GPT4OPerspectiveDetector` as backward compatibility wrapper
- Updated all imports in `client.py` and `__init__.py`

## Benefits

### 1. **Single Source of Truth**
- All MLLM interactions happen in one place
- Consistent logging format across all calls
- Easy to add new MLLM providers (Claude, Gemini, etc.)

### 2. **Code Reduction**
- Removed ~220 lines of duplicate code
- Simplified agent and synthesizer
- Easier to maintain and test

### 3. **Better Debugging**
- All MLLM calls logged to `mllm_call_XXX_*.json` files
- Sequential numbering shows call order
- Consistent format makes debugging easier

### 4. **Flexibility**
- Easy to swap MLLM providers
- Can add model-specific optimizations in one place
- Simple to add new MLLM operations

### 5. **Cleaner Architecture**
```
Before:
agent.py ──┐
           ├──> OpenAI API
synthesizer.py ─┘

After:
agent.py ──┐
           ├──> MLLMCaller ──> OpenAI API
synthesizer.py ─┘
```

## File Structure

```
hegarty/
├── call_mllm.py          # NEW: Centralized MLLM interface
├── agent.py              # REFACTORED: Uses MLLMCaller
├── synthesizer.py        # REFACTORED: Uses MLLMCaller  
├── client.py             # UPDATED: Import from call_mllm
├── __init__.py           # UPDATED: Export MLLMCaller
└── gpt_detector.py       # DELETED: Merged into call_mllm
```

## Logging Output

All MLLM calls now saved as:
```
temp/hegarty_debug/session_YYYYMMDD_HHMMSS/
├── mllm_call_001_detect_perspective_*.json
├── mllm_call_002_rephrase_for_video_*.json
├── mllm_call_003_analyze_original_*.json
├── mllm_call_004_analyze_perspective_1_*.json
├── ...
└── mllm_call_008_synthesize_*.json
```

## API Usage Examples

### Using MLLMCaller Directly

```python
from hegarty.call_mllm import MLLMCaller
from openai import OpenAI
from pathlib import Path

client = OpenAI()
mllm = MLLMCaller(
    openai_client=client,
    model="gpt-4o",
    session_dir=Path("./logs")
)

# Detect perspective task
is_perspective, confidence = mllm.detect_perspective(
    "Is the bag on the watermelon's left?"
)

# Rephrase for video
video_prompt = mllm.rephrase_for_video(
    question="Rotate the object 90 degrees",
    image=base64_image
)

# Analyze perspective
analysis = mllm.analyze_perspective(
    image=base64_image,
    question="What do you see from this angle?",
    perspective_label="front_view"
)

# Synthesize multiple perspectives
final_answer, confidence = mllm.synthesize_perspectives(
    perspectives=[...],
    original_question="...",
    consistency_score=0.85
)
```

### Backward Compatibility

Old code still works:
```python
from hegarty import GPT4OPerspectiveDetector

detector = GPT4OPerspectiveDetector(use_mini=True)
is_perspective, conf = detector.analyze("Is this a rotation task?")
```

## Future Extensions

With centralized MLLM interface, easy to add:

1. **Multiple Model Support**
```python
mllm = MLLMCaller(provider="anthropic", model="claude-3-opus")
mllm = MLLMCaller(provider="google", model="gemini-1.5-pro")
```

2. **Caching Layer**
```python
mllm = MLLMCaller(cache=True, cache_ttl=3600)
```

3. **Rate Limiting**
```python
mllm = MLLMCaller(rate_limit=10, rate_limit_window=60)
```

4. **Cost Tracking**
```python
total_cost = mllm.get_total_cost()
cost_by_call = mllm.get_cost_breakdown()
```

## Testing

Run tests to verify refactoring:
```bash
# Test imports
python3 -c "from hegarty import MLLMCaller, GPT4OPerspectiveDetector; print('✓ Imports work')"

# Test backward compatibility
python3 -c "from hegarty.client import HergartyClient; print('✓ Client works')"

# Test agent
python3 -c "from hegarty.agent import HergartyAgent; print('✓ Agent works')"
```

## Migration Guide

No code changes needed for existing users! The refactoring is backward compatible.

If you were using internal methods (not recommended), update as follows:

**Old:**
```python
agent._rephrase_for_sora(question, image)
```

**New:**
```python
agent.mllm_caller.rephrase_for_video(question, image)
```

