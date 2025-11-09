"""
Basic usage example for Hegarty package

Demonstrates the perspective-taking pipeline:
1. Input: Image + Question
2. [Perspective Detection] GPT-4o analyzes if mental rotation is needed
3. [Mental Rotation Generation] Sora-2 creates video of the transformation
4. [Multi-Perspective Analysis] Extract 5 frames from last 30 frames, 6 parallel GPT-4o calls
5. [Synthesis] GPT-4o synthesizes all perspectives
6. Output: Comprehensive Answer
"""

import os
from hegarty import HergartyClient
import base64
from pathlib import Path


def encode_image(image_path: str) -> str:
    """Encode an image file to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def main():
    """
    Demonstrate the Hegarty perspective-taking pipeline.
    """
    print("Hegarty Perspective-Taking Pipeline")
    print("=" * 60)
    
    # Initialize client
    client = HergartyClient(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        # Sora API key is optional - will use simulation if not provided
        sora_api_key=os.getenv("SORA_API_KEY")
    )
    
    print("✓ Client initialized")
    print("  - Using GPT-4o for detection and analysis")
    print("  - Using Sora-2 simulation for video generation")
    
    # Example 1: Perspective-taking question (triggers full pipeline)
    print("\n" + "=" * 60)
    print("Example 1: Mental Rotation Task (Full Pipeline)")
    print("-" * 60)
    
    question = "If I rotate this cube 90 degrees clockwise around the vertical axis, what face will be visible?"
    
    # Create a simple base64 encoded image (placeholder)
    # In real usage, you would encode an actual image
    placeholder_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    
    response = client.chat.completions.create(
        model="hegarty-1.0",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": placeholder_image}}
                ]
            }
        ]
    )
    
    print(f"Question: {question}")
    print(f"\nAnswer: {response.choices[0].message.content}")
    
    # Example 2: Non-perspective question (routes to standard GPT-4o)
    print("\n" + "=" * 60)
    print("Example 2: Standard Question (Bypasses Pipeline)")
    print("-" * 60)
    
    response = client.chat.completions.create(
        model="hegarty-1.0",
        messages=[
            {
                "role": "user",
                "content": "What is the capital of France?"
            }
        ]
    )
    
    print(f"Question: What is the capital of France?")
    print(f"Answer: {response.choices[0].message.content}")
    
    # Example 3: Perspective Detection
    print("\n" + "=" * 60)
    print("Example 3: GPT-4o Perspective Detection")
    print("-" * 60)
    
    test_questions = [
        "What is the weather today?",
        "Rotate this shape 180 degrees and describe it",
        "How does photosynthesis work?",
        "What would this look like from the other side?",
        "Calculate the area of a triangle",
        "If I turn this object upside down, which face is visible?"
    ]
    
    print("Testing perspective detection:\n")
    for q in test_questions:
        is_perspective, confidence = client.agent.mllm.detect_perspective(q)
        status = "✓ Perspective" if is_perspective else "✗ Standard"
        print(f"{status} ({confidence:.2f}): {q}")
    
    print("\n" + "=" * 60)
    print("Pipeline demonstration complete!")
    print("\nThe Hegarty pipeline automatically:")
    print("  1. Detects perspective-taking tasks with GPT-4o")
    print("  2. Generates mental rotation videos with Sora-2")
    print("  3. Extracts 5 frames from the last 30 frames")
    print("  4. Analyzes original + 5 frames in parallel (6 total)")
    print("  5. Synthesizes perspectives into final answer")


if __name__ == "__main__":
    main()
