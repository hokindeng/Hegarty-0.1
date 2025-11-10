"""Example: Using Qwen3-VL with Hegarty Agent"""

from pathlib import Path
from hegarty import HergartyAgent, Config
from hegarty.mllm import QwenMLLM
from hegarty.vm import SoraVM

# Initialize Qwen3-VL MLLM provider
# Note: This requires a GPU with sufficient VRAM (e.g., 80GB for full model)
# For smaller GPUs, consider using a quantized version or smaller model variant
qwen_mllm = QwenMLLM(
    model_name="Qwen/Qwen2.5-VL-7B-Instruct",
    device_map="auto",  # Automatically distributes model across available devices
    dtype="auto",  # Automatically selects appropriate dtype
    attn_implementation="flash_attention_2",  # Optional: for better performance
    temperature=0.3,
    max_tokens=2000
)

# Configure agent
config = Config(
    frame_extraction_count=5,
    max_workers=6
)

# Create agent with Qwen provider
agent = HergartyAgent(config=config)
agent.mllm = qwen_mllm

# Optional: Add video model if you have Sora access
# agent.vm = SoraVM(api_key="your-sora-key")

# Example 1: Simple perspective question
print("Example 1: Simple perspective-taking question")
print("-" * 50)

image_path = "path/to/your/image.jpg"
question = "Is the bag on the watermelon's left?"

result = agent.process(
    image=image_path,
    question=question,
    use_mental_rotation=False  # Set to True if you have video model configured
)

print(f"Question: {question}")
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.2f}")
print()

# Example 2: Mental rotation task (requires video model)
print("Example 2: Mental rotation task (requires video model)")
print("-" * 50)

rotation_question = "What would this scene look like from behind the person?"

try:
    result = agent.process(
        image=image_path,
        question=rotation_question,
        use_mental_rotation=True
    )
    
    print(f"Question: {rotation_question}")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Video generated: {result.get('video_path', 'N/A')}")
except Exception as e:
    print(f"Note: Mental rotation requires video model. Error: {e}")

print()

# Example 3: Using with OpenAI-compatible client
print("Example 3: Using with HergartyClient")
print("-" * 50)

from hegarty import HergartyClient

# Create client with custom agent
client = HergartyClient()
client.agent.mllm = qwen_mllm

# Use OpenAI-compatible API
response = client.chat.completions.create(
    model="hegarty-1.0",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's the spatial relationship between objects?"},
            {"type": "image_url", "image_url": {"url": f"file://{image_path}"}}
        ]
    }]
)

print(f"Response: {response.choices[0].message.content}")
print()

# Tips for using Qwen3-VL
print("Tips for using Qwen3-VL:")
print("-" * 50)
print("1. Model Size: 7B runs on a single high-memory GPU; use quantization for smaller GPUs")
print("2. Consider quantized versions for smaller GPUs (int8, int4)")
print("3. Use flash_attention_2 for better memory efficiency")
print("4. Batch size of 1 is recommended for large models")
print("5. Multi-GPU setup: Use device_map='auto' for automatic distribution")
print("6. For production: Consider serving via vLLM or TGI for better throughput")

