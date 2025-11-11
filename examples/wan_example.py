#!/usr/bin/env python3
"""
Wan2.2 Video Generation Example

This example demonstrates how to use the Wan2.2 video model provider
for generating videos from text prompts and/or images.
"""

import os
import sys
import base64
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from hegarty.vm import WanVM
from hegarty.config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string"""
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/png;base64,{encoded}"


def example_text_to_video():
    """Example: Generate video from text prompt only"""
    logger.info("=== Text-to-Video Generation Example ===")
    
    # Initialize Wan2.2 with 5B model for faster inference
    wan = WanVM(
        model_size="5B",
        device="cuda",  # Use "cpu" if no GPU available
        offload_model=True,  # Offload model to save memory
        convert_model_dtype=True  # Use lower precision for faster inference
    )
    
    # Text prompt for video generation
    prompt = "A serene lake surrounded by mountains at sunset, with gentle ripples on the water"
    
    # Generate video
    result = wan.generate_video(
        prompt=prompt,
        duration=5,  # 5 second video
        fps=24,
        resolution="1280x720"
    )
    
    logger.info(f"Video generated: {result['video_path']}")
    logger.info(f"Metadata: {result['metadata']}")
    
    return result


def example_image_to_video():
    """Example: Generate video from image and text prompt"""
    logger.info("=== Image-to-Video Generation Example ===")
    
    # Initialize Wan2.2
    wan = WanVM(
        model_size="5B",
        device="cuda",
        offload_model=True
    )
    
    # Example: Use a sample image (you'll need to provide your own)
    image_path = Path(__file__).parent / "sample_image.jpg"
    
    if not image_path.exists():
        logger.warning(f"Sample image not found at {image_path}")
        logger.info("Creating a simple test image...")
        
        # Create a simple test image
        from PIL import Image
        import numpy as np
        
        # Create a gradient image
        width, height = 1280, 720
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create a gradient
        for y in range(height):
            for x in range(width):
                img_array[y, x] = [
                    int(255 * x / width),  # Red gradient
                    int(255 * y / height),  # Green gradient
                    128  # Constant blue
                ]
        
        img = Image.fromarray(img_array)
        img.save(image_path)
        logger.info(f"Test image saved to {image_path}")
    
    # Encode image to base64
    image_base64 = encode_image_to_base64(str(image_path))
    
    # Text prompt to guide video generation
    prompt = "Camera slowly zooms out revealing a vast landscape"
    
    # Generate video
    result = wan.generate_video(
        prompt=prompt,
        image=image_base64,
        duration=5,
        fps=24,
        resolution="1280x720"
    )
    
    logger.info(f"Video generated: {result['video_path']}")
    
    return result


def example_animate_model():
    """Example: Using Wan2.2-Animate for character animation"""
    logger.info("=== Wan2.2-Animate Example ===")
    
    # Initialize Animate model (requires more resources)
    wan = WanVM(
        model_size="animate-14B",
        device="cuda",
        offload_model=True,
        convert_model_dtype=True,
        use_prompt_extend=True  # Better results with prompt extension
    )
    
    # For animate model, you need a character image
    character_image_path = Path(__file__).parent / "character.jpg"
    
    if not character_image_path.exists():
        logger.error("Animate model requires a character image!")
        logger.info("Please provide a character image at: " + str(character_image_path))
        return None
    
    # Encode character image
    image_base64 = encode_image_to_base64(str(character_image_path))
    
    # Animation prompt
    prompt = "The character is smiling and waving their hand"
    
    # Generate animated video
    result = wan.generate_video(
        prompt=prompt,
        image=image_base64,
        duration=3,  # Shorter duration for animation
        fps=24,
        resolution="720x1280"  # Portrait orientation
    )
    
    logger.info(f"Animated video generated: {result['video_path']}")
    
    return result


def example_multi_gpu():
    """Example: Using multiple GPUs with FSDP"""
    logger.info("=== Multi-GPU Generation Example ===")
    
    # Initialize with multi-GPU support
    wan = WanVM(
        model_size="14B",  # Larger model benefits from multi-GPU
        device="cuda",
        use_fsdp=True,  # Enable FSDP
        ulysses_size=4,  # Number of GPUs for Ulysses parallelism
        offload_model=False  # Don't offload when using multiple GPUs
    )
    
    prompt = "A futuristic city with flying cars and neon lights"
    
    # Generate high-resolution video
    result = wan.generate_video(
        prompt=prompt,
        duration=10,  # Longer video
        fps=24,
        resolution="1920x1080"  # Full HD
    )
    
    logger.info(f"HD video generated: {result['video_path']}")
    
    return result


def main():
    """Run examples based on command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Wan2.2 Video Generation Examples")
    parser.add_argument(
        "--example",
        choices=["text2video", "image2video", "animate", "multigpu", "all"],
        default="text2video",
        help="Which example to run"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        help="Path to Wan2.2 model directory"
    )
    
    args = parser.parse_args()
    
    # Set model directory if provided
    if args.model_dir:
        os.environ["WAN_MODEL_DIR"] = args.model_dir
    
    # Create output directory
    output_dir = Path(__file__).parent / "wan_outputs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(output_dir)
    
    logger.info(f"Output directory: {output_dir}")
    
    # Run selected example
    if args.example == "text2video" or args.example == "all":
        example_text_to_video()
    
    if args.example == "image2video" or args.example == "all":
        example_image_to_video()
    
    if args.example == "animate" or args.example == "all":
        example_animate_model()
    
    if args.example == "multigpu":
        if torch.cuda.device_count() > 1:
            example_multi_gpu()
        else:
            logger.warning("Multi-GPU example requires multiple GPUs!")


if __name__ == "__main__":
    import torch
    
    # Check PyTorch and CUDA
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        logger.info(f"GPU name: {torch.cuda.get_device_name(0)}")
    
    main()
