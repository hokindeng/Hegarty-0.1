"""
Basic usage example for Hegarty package
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
    Demonstrate basic usage of Hegarty client.
    """
    print("Hegarty Basic Usage Example")
    print("=" * 50)
    
    # Initialize client
    client = HergartyClient(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        # Sora API key is optional - will use simulation if not provided
        sora_api_key=os.getenv("SORA_API_KEY")
    )
    
    print("✓ Client initialized")
    
    # Example 1: Text-only perspective question (without image)
    print("\n" + "=" * 50)
    print("Example 1: Mental Rotation Question")
    print("-" * 50)
    
    response = client.chat.completions.create(
        model="hegarty-1.0",
        messages=[
            {
                "role": "user",
                "content": "If I have a cube with the number 1 on top, 2 facing me, and 3 on my right, what number would be on top if I rotate it 90 degrees clockwise around the vertical axis?"
            }
        ]
    )
    
    print(f"Question: Mental rotation of a numbered cube")
    print(f"Answer: {response.choices[0].message.content}")
    
    # Example 2: Perspective detection
    print("\n" + "=" * 50)
    print("Example 2: Perspective Detection")
    print("-" * 50)
    
    from hegarty import PerspectiveDetector
    
    detector = PerspectiveDetector()
    
    questions = [
        "What is the capital of France?",
        "Rotate this shape 180 degrees and describe it",
        "How does photosynthesis work?",
        "What would this look like from the other side?",
        "Explain quantum mechanics",
        "If I turn this object upside down, which face is visible?"
    ]
    
    for question in questions:
        is_perspective, confidence = detector.analyze(question)
        status = "✓ Perspective task" if is_perspective else "✗ Not perspective"
        print(f"{status} (conf: {confidence:.2f}): {question[:50]}")
    
    # Example 3: Using standard GPT-4o mode
    print("\n" + "=" * 50)
    print("Example 3: Standard GPT-4o Mode")
    print("-" * 50)
    
    response = client.chat.completions.create(
        model="gpt-4o",  # Use standard GPT-4o without Hegarty pipeline
        messages=[
            {
                "role": "user",
                "content": "What is machine learning?"
            }
        ]
    )
    
    print("Question: What is machine learning?")
    print(f"Answer: {response.choices[0].message.content[:200]}...")
    
    print("\n" + "=" * 50)
    print("Examples completed successfully!")


if __name__ == "__main__":
    main()
