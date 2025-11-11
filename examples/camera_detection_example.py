"""Example: Using PerspectiveFields camera detection with Hegarty"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from PIL import Image
from hegarty import HergartyAgent, Config
from hegarty.mllm import QwenMLLM
from hegarty.vm import SoraVM
from hegarty.camera_detector import CameraDetector
import json

def demo_camera_detection():
    """Demonstrate camera detection from an image"""
    print("=== Camera Detection Demo ===")
    print()
    
    # Initialize camera detector
    detector = CameraDetector(model_version='Paramnet-360Cities-edina-centered')
    
    # Example image path (replace with your own)
    image_path = "path/to/your/image.jpg"
    
    if not os.path.exists(image_path):
        print(f"Please provide a valid image path. Current path not found: {image_path}")
        return
    
    # Load and detect camera parameters
    print(f"Loading image: {image_path}")
    image = Image.open(image_path)
    
    print("Detecting camera parameters...")
    params = detector.detect_from_image(image)
    
    # Display results
    print("\nDetected Camera Parameters:")
    print(f"  Roll: {params.get('roll', 0.0):.1f}°")
    print(f"  Pitch (Elevation): {params.get('pitch', 0.0):.1f}°")
    print(f"  Vertical FOV: {params.get('vfov', 60.0):.1f}°")
    print(f"  Confidence: {params.get('confidence', 0.0):.2f}")
    
    # Generate trajectory for video
    trajectory = detector.get_camera_trajectory(params, target_azimuth=180.0)
    
    print("\nSuggested Camera Trajectory:")
    print(f"  Start: {trajectory['start_elevation']:.1f}° elevation, {trajectory['start_azimuth']:.1f}° azimuth")
    print(f"  End: {trajectory['end_elevation']:.1f}° elevation, {trajectory['end_azimuth']:.1f}° azimuth")
    print(f"  Duration: {trajectory['duration']:.1f} seconds")
    
    return params, trajectory


def demo_hegarty_with_camera_detection():
    """Demonstrate full Hegarty pipeline with camera detection"""
    print("\n=== Hegarty with Camera Detection Demo ===")
    print()
    
    # Configure agent with camera detection enabled
    config = Config(
        use_camera_detection=True,
        camera_detector_model="Paramnet-360Cities-edina-centered",
        camera_detection_confidence_threshold=0.5,
        frame_extraction_count=1,
        max_workers=6
    )
    
    # Initialize agent
    agent = HergartyAgent(config=config)
    
    # Use Qwen MLLM provider
    agent.mllm = QwenMLLM(
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        temperature=0.3,
        max_tokens=2000
    )
    
    # Add Sora VM if available
    sora_key = os.getenv("SORA_API_KEY")
    if sora_key:
        agent.vm = SoraVM(api_key=sora_key)
        print("Sora VM initialized")
    else:
        print("No SORA_API_KEY found - running without video generation")
    
    # Process a perspective-taking question
    image_path = "path/to/your/image.jpg"
    question = "What would this scene look like from the teddy bear's perspective?"
    
    print(f"Question: {question}")
    print("Processing with camera-aware prompt generation...")
    
    # Create session directory
    session_dir = Path("temp/demo_session")
    session_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and encode image
    if os.path.exists(image_path):
        image = Image.open(image_path)
        import base64
        from io import BytesIO
        
        buffer = BytesIO()
        image.save(buffer, format='JPEG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        image_data = f"data:image/jpeg;base64,{image_base64}"
        
        # Process with agent
        result = agent.process(
            image=image_data,
            question=question,
            use_mental_rotation=True,
            return_intermediate=True,
            session_dir=session_dir
        )
        
        # Show results
        print("\nResults:")
        print(f"  Final Answer: {result.get('final_answer', 'N/A')}")
        print(f"  Confidence: {result.get('confidence', 0.0):.2f}")
        
        if 'camera_params' in result:
            print(f"\n  Detected Camera Parameters:")
            cam = result['camera_params']
            print(f"    Pitch: {cam.get('pitch', 0.0):.1f}°")
            print(f"    Roll: {cam.get('roll', 0.0):.1f}°")
            print(f"    vFOV: {cam.get('vfov', 60.0):.1f}°")
        
        if 'rephrased_prompt' in result:
            print(f"\n  Generated Sora Prompt:")
            print(f"    {result['rephrased_prompt']}")
        
        # Check if prompt was saved
        prompt_file = session_dir / "sora_prompt.json"
        if prompt_file.exists():
            with open(prompt_file, 'r') as f:
                prompt_data = json.load(f)
            print(f"\n  Saved prompt file: {prompt_file}")
            print(f"    Content: {prompt_data['prompt']}")
    else:
        print(f"Image not found: {image_path}")


if __name__ == "__main__":
    # Run camera detection demo
    params, trajectory = demo_camera_detection()
    
    # Run full Hegarty demo with camera detection
    demo_hegarty_with_camera_detection()
