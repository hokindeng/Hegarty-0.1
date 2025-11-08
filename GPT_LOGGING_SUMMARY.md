# GPT-4o API Call Logging - Implementation Summary

## Overview
Added comprehensive logging of all GPT-4o API calls to help debug issues like the "I'm sorry, I can't assist with that" problem.

## What Was Added

### 1. **Agent (`hegarty/agent.py`)**

Added logging infrastructure:
- `session_dir` - stores the current session directory
- `gpt_call_counter` - tracks the number of GPT calls in a session
- `_save_gpt_call()` - saves request/response to JSON file
- `_sanitize_messages_for_json()` - truncates base64 images for readability

Logging points added:
1. **`_rephrase_for_sora()`** - Saves the rephrasing request that was causing the "I'm sorry" issue
2. **`_analyze_single_perspective()`** - Saves each perspective analysis call

### 2. **Synthesizer (`hegarty/synthesizer.py`)**

Added same logging infrastructure:
- `session_dir`, `gpt_call_counter`, `_save_gpt_call()`, `_sanitize_messages_for_json()`

Logging points added:
3. **`synthesize()`** - Saves the final synthesis call

## Output Format

Each GPT call is saved as: `gpt_call_XXX_<call_name>_<timestamp>.json`

Example filenames:
- `gpt_call_001_rephrase_for_sora_20251108_214819.json`
- `gpt_call_002_analyze_original_20251108_214820.json`
- `gpt_call_003_analyze_perspective_1_20251108_214821.json`
- `gpt_call_008_synthesize_20251108_214825.json`

### JSON Structure

```json
{
  "call_number": 1,
  "call_name": "rephrase_for_sora",
  "timestamp": "20251108_214819",
  "model": "gpt-4o",
  "request": {
    "messages": [
      {
        "role": "system",
        "content": "You are an expert at creating video generation prompts."
      },
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "You are helping to create a video prompt..."
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQA... [base64 image truncated, total length: 45678]"
            }
          }
        ]
      }
    ]
  },
  "response": {
    "content": "I'm sorry, I can't assist with that.",
    "model": "gpt-4o-2024-11-20",
    "usage": {
      "completion_tokens": 10,
      "prompt_tokens": 1234,
      "total_tokens": 1244
    }
  }
}
```

## Where Files Are Saved

All JSON files are saved to the session debug directory:
- `temp/hegarty_debug/session_YYYYMMDD_HHMMSS/gpt_call_*.json`

This is the same directory where `debug_info.txt` and `original_image.jpg` are saved.

## How It Works

1. When `HergartyAgent.process()` is called with a `session_dir`, it stores the directory
2. Every time a GPT-4o API call is made:
   - The request messages are captured
   - The response is captured  
   - Both are saved to a numbered JSON file
   - Base64 images are truncated to first 100 chars for readability
3. The counter increments so files are numbered sequentially
4. When the synthesizer is called, it inherits the counter and continues the sequence

## Benefits

1. **Full transparency** - See exactly what's being sent to GPT-4o
2. **Easy debugging** - Identify when/why GPT refuses to help
3. **Token tracking** - See how many tokens each call uses
4. **Temporal ordering** - Files are numbered in the order calls were made
5. **Readable format** - JSON with truncated images is easy to inspect

## Example Debug Workflow

When you see "I'm sorry, I can't assist with that":

1. Go to the session folder (shown in `debug_info.txt`)
2. Open `gpt_call_001_rephrase_for_sora_*.json`
3. Look at the `request.messages` to see what was sent
4. Look at the `response.content` to see what GPT returned
5. Identify the problematic content (e.g., "waterlesson" typo)

## Next Steps

Now that all GPT calls are logged, you can:
- Analyze why certain prompts fail
- Optimize prompts based on actual requests
- Track token usage patterns
- Debug perspective analysis quality
- Improve synthesis logic

