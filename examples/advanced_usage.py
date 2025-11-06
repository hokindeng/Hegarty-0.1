"""
Advanced usage examples for Hegarty package
"""

import os
import time
import asyncio
from typing import List, Dict, Any
from hegarty import HergartyClient, HergartyAgent, Config
from openai import OpenAI
import base64
import numpy as np
from PIL import Image
import io


def create_test_image() -> str:
    """Create a simple test image with geometric shapes."""
    # Create a simple image with shapes
    img = Image.new('RGB', (512, 512), color='white')
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    
    # Draw a square
    draw.rectangle([100, 100, 200, 200], fill='red', outline='black')
    
    # Draw a circle
    draw.ellipse([250, 100, 350, 200], fill='blue', outline='black')
    
    # Draw a triangle
    draw.polygon([(200, 250), (150, 350), (250, 350)], fill='green', outline='black')
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return f"data:image/png;base64,{img_base64}"


def example_custom_config():
    """
    Example: Using custom configuration
    """
    print("\n" + "=" * 50)
    print("Custom Configuration Example")
    print("-" * 50)
    
    # Create custom configuration
    config = Config(
        gpt_model="gpt-4o",
        temperature=0.2,  # Lower temperature for more consistent results
        max_tokens=1500,
        sora_video_length=2,  # Shorter videos
        sora_fps=5,  # Lower FPS for faster processing
        frame_extraction_count=3,  # Fewer frames for speed
        frame_extraction_strategy="adaptive",  # Use adaptive extraction
        max_workers=4,  # Parallel workers
        enable_cache=True,
        cache_ttl=7200  # 2 hour cache
    )
    
    # Initialize client with custom config
    client = HergartyClient(config=config)
    
    print("✓ Client initialized with custom configuration")
    print(f"  - Model: {config.gpt_model}")
    print(f"  - Temperature: {config.temperature}")
    print(f"  - Frame extraction: {config.frame_extraction_strategy}")
    print(f"  - Cache enabled: {config.enable_cache}")
    
    # Use the configured client
    test_image = create_test_image()
    
    response = client.chat.completions.create(
        model="hegarty-1.0",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "If I rotate this image 90 degrees counterclockwise, describe the new positions of the shapes."},
                    {"type": "image_url", "image_url": {"url": test_image}}
                ]
            }
        ]
    )
    
    print("\nResponse:", response.choices[0].message.content[:300] + "...")


def example_direct_agent():
    """
    Example: Using HergartyAgent directly for more control
    """
    print("\n" + "=" * 50)
    print("Direct Agent Usage Example")
    print("-" * 50)
    
    # Initialize OpenAI client
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create agent
    agent = HergartyAgent(
        openai_client=openai_client,
        sora_api_key=os.getenv("SORA_API_KEY")  # Optional
    )
    
    print("✓ Agent initialized")
    
    # Process with detailed results
    test_image = create_test_image()
    
    result = agent.process(
        image=test_image,
        question="What would these shapes look like if viewed from directly above after a 45-degree rotation?",
        use_mental_rotation=True,
        num_perspectives=4,
        return_intermediate=True  # Get intermediate results
    )
    
    print(f"\nFinal Answer: {result['final_answer'][:300]}...")
    print(f"Confidence: {result['confidence']:.2f}")
    
    if 'perspectives' in result:
        print(f"Number of perspectives analyzed: {len(result['perspectives'])}")
        for i, p in enumerate(result['perspectives'][:2]):  # Show first 2
            print(f"\n  Perspective {i+1} ({p['perspective']}):")
            print(f"    {p['analysis'][:150]}...")
    
    if 'rephrased_prompt' in result:
        print(f"\nSora prompt: {result['rephrased_prompt']}")


def example_batch_processing():
    """
    Example: Batch processing multiple queries
    """
    print("\n" + "=" * 50)
    print("Batch Processing Example")
    print("-" * 50)
    
    client = HergartyClient()
    
    # Create test image
    test_image = create_test_image()
    
    # Multiple queries
    queries = [
        {
            "image": test_image,
            "question": "Describe the spatial arrangement of the shapes."
        },
        {
            "image": test_image,
            "question": "If I flip this image horizontally, what changes?"
        },
        {
            "image": test_image,
            "question": "Rotate 180 degrees and describe the result."
        }
    ]
    
    print(f"Processing {len(queries)} queries...")
    
    start_time = time.time()
    results = client.batch_process(queries)
    elapsed = time.time() - start_time
    
    print(f"✓ Processed in {elapsed:.2f} seconds")
    
    for i, (query, result) in enumerate(zip(queries, results)):
        print(f"\nQuery {i+1}: {query['question']}")
        print(f"Answer: {result.choices[0].message.content[:200]}...")


def example_perspective_detection_analysis():
    """
    Example: Detailed perspective detection analysis
    """
    print("\n" + "=" * 50)
    print("Perspective Detection Analysis")
    print("-" * 50)
    
    from hegarty import PerspectiveDetector
    
    detector = PerspectiveDetector()
    
    test_questions = [
        "Mentally rotate this object 90 degrees clockwise.",
        "What color is the sky?",
        "Imagine viewing this from behind.",
        "Calculate the area of a circle.",
        "If you were standing on the other side, what would you see?",
        "How does gravity work?",
        "Flip this shape vertically and horizontally.",
        "What is the meaning of life?"
    ]
    
    print("Analyzing questions for perspective-taking requirements:\n")
    
    for question in test_questions:
        result = detector.detailed_analysis(question)
        
        icon = "🔄" if result.is_perspective_task else "📝"
        print(f"{icon} {question}")
        print(f"   Perspective task: {result.is_perspective_task}")
        print(f"   Confidence: {result.confidence:.2f}")
        if result.detected_keywords:
            print(f"   Keywords: {', '.join(result.detected_keywords[:3])}")
        print(f"   Reasoning: {result.reasoning}")
        print()


def example_caching():
    """
    Example: Demonstrate caching behavior
    """
    print("\n" + "=" * 50)
    print("Caching Example")
    print("-" * 50)
    
    # Enable caching
    config = Config(enable_cache=True, cache_ttl=60)
    client = HergartyClient(config=config)
    
    test_image = create_test_image()
    question = "Describe what you see in this image."
    
    print("First call (not cached):")
    start = time.time()
    response1 = client.chat.completions.create(
        model="hegarty-1.0",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": test_image}}
            ]
        }]
    )
    time1 = time.time() - start
    print(f"  Time: {time1:.2f}s")
    
    print("\nSecond call (should be cached):")
    start = time.time()
    response2 = client.chat.completions.create(
        model="hegarty-1.0",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": test_image}}
            ]
        }]
    )
    time2 = time.time() - start
    print(f"  Time: {time2:.2f}s")
    
    if time2 < time1 * 0.5:
        print("  ✓ Caching is working!")
    else:
        print("  ⚠ Cache might not be working (or was a cache miss)")


def main():
    """
    Run all advanced examples.
    """
    print("Hegarty Advanced Usage Examples")
    print("=" * 50)
    
    examples = [
        ("Custom Configuration", example_custom_config),
        ("Direct Agent Usage", example_direct_agent),
        ("Batch Processing", example_batch_processing),
        ("Perspective Detection", example_perspective_detection_analysis),
        # ("Caching", example_caching),  # Uncomment if caching is implemented
    ]
    
    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n⚠ Error in {name}: {str(e)}")
            print("  (This might be due to missing API keys or dependencies)")
    
    print("\n" + "=" * 50)
    print("Advanced examples completed!")


if __name__ == "__main__":
    main()
