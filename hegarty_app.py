#!/usr/bin/env python3
"""
Hegarty Gradio Web App
Simple image + question interface for perspective-taking AI
"""

import os
import base64
import logging
import json
from pathlib import Path
from typing import Optional, Tuple, List
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from PIL import Image
import cv2
import numpy as np
from hegarty import HergartyClient, HergartyAgent, Config
from hegarty.mllm import QwenMLLM
from hegarty.vm import SoraVM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global client
hegarty_client: Optional[HergartyClient] = None
current_provider: str = "qwen"  # Default provider

# Create temp directory structure for all session files
TEMP_DIR = Path.cwd() / "temp"
SESSIONS_DIR = TEMP_DIR / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"All session files will be saved to: {SESSIONS_DIR}")


def encode_image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    import io
    if image.mode != 'RGB':
        image = image.convert('RGB')
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=85)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"


def extract_frames_from_video(video_path: Path, output_dir: Path, num_frames: int = 5) -> List[Path]:
    """Extract frames from video file using OpenCV."""
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return []
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    logger.info(f"Video: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s")
    
    # Calculate frame indices (focus on last 30 frames as per original logic)
    window_size = min(30, total_frames)
    start_frame = max(0, total_frames - window_size)
    frame_indices = np.linspace(start_frame, total_frames - 1, num_frames, dtype=int)
    
    extracted_frames = []
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            frame_path = output_dir / f"frame_{i:03d}_idx{frame_idx}.jpg"
            cv2.imwrite(str(frame_path), frame)
            extracted_frames.append(frame_path)
            logger.info(f"Extracted frame {i+1}/{num_frames} from index {frame_idx}")
    
    cap.release()
    return extracted_frames


def create_session_folder() -> Path:
    """Create a timestamped folder for this session."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = SESSIONS_DIR / f"session_{timestamp}"
    session_dir.mkdir(exist_ok=True)
    
    return session_dir


def initialize_hegarty(provider: str = "qwen", session_dir: Optional[Path] = None):
    """
    Initialize Hegarty client with specified MLLM provider.
    
    Args:
        provider: MLLM provider to use ("qwen")
        session_dir: Optional session directory for logging
    """
    global hegarty_client, current_provider
    
    openai_key = os.getenv("OPENAI_API_KEY")
    sora_key = os.getenv("SORA_API_KEY")
    
    # Check env variable for provider preference
    env_provider = os.getenv("MLLM_PROVIDER", provider).lower()
    provider = "qwen"
    current_provider = provider
    
    config = Config(
        temperature=0.3, 
        max_tokens=1000, 
        sora_video_length=4, 
        frame_extraction_count=5, 
        max_workers=6
    )
    
    logger.info("Initializing with Qwen3-VL MLLM provider")
    
    # Get Qwen configuration from environment
    qwen_model = os.getenv("QWEN_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
    qwen_device = os.getenv("QWEN_DEVICE_MAP", "auto")
    qwen_dtype = os.getenv("QWEN_DTYPE", "auto")
    qwen_attn = os.getenv("QWEN_ATTENTION", None)  # e.g., "flash_attention_2"
    
    # Create agent with Qwen provider
    agent = HergartyAgent(config=config)
    agent.mllm = QwenMLLM(
        model_name=qwen_model,
        device_map=qwen_device,
        dtype=qwen_dtype,
        attn_implementation=qwen_attn,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        session_dir=session_dir
    )
    
    # Add Sora VM if available
    if sora_key:
        from hegarty.vm import SoraVM
        agent.vm = SoraVM(api_key=sora_key)
    
    # Create client wrapper
    hegarty_client = HergartyClient(config=config)
    hegarty_client.agent = agent
    
    logger.info(f"Qwen3-VL initialized with model: {qwen_model}")
    
    logger.info(f"Hegarty initialized with provider: {provider}")


def process_image_question(image: Optional[Image.Image], question: str, provider: str = "qwen") -> Tuple[str, Optional[str], List[str], str]:
    """
    Process image and question through Hegarty with full debugging info.
    
    Args:
        image: Input image
        question: Question about the image
        provider: MLLM provider to use ("qwen")
    
    Returns:
        Tuple of (answer, video_path, frame_paths, debug_info)
        - answer: The AI's response
        - video_path: Path to downloaded Sora video (None if not available)
        - frame_paths: List of paths to extracted frames
        - debug_info: Debug information string
    """
    global hegarty_client, current_provider
    
    if not image:
        return "Please upload an image first.", None, [], ""
    
    # Create session folder for this request
    session_dir = create_session_folder()
    
    # Initialize or reinitialize if provider changed
    if not hegarty_client or current_provider != provider:
        logger.info(f"Initializing with provider: {provider}")
        initialize_hegarty(provider=provider, session_dir=session_dir)
    
    # Save original image
    original_image_path = session_dir / "original_image.jpg"
    image.save(original_image_path)
    
    image_base64 = encode_image_to_base64(image)
    
    # Get intermediate results from Hegarty
    debug_info = [f"Session folder: {session_dir}"]
    debug_info.append(f"MLLM Provider: {current_provider.upper()}")
    debug_info.append(f"Question: {question}")
    debug_info.append(f"Original image saved: {original_image_path}")
    
    # Check if this will trigger perspective detection
    is_perspective = False
    confidence = 0.0
    if hegarty_client.agent.mllm:
        is_perspective, confidence = hegarty_client.agent.mllm.detect_perspective(question)
        debug_info.append(f"Perspective detection: {is_perspective} (confidence: {confidence:.2f})")
    
    # Call agent with session directory for organizing files
    result = hegarty_client.agent.process(
        image=image_base64,
        question=question,
        return_intermediate=True,
        session_dir=session_dir
    )
    
    answer = result['final_answer']
    final_confidence = result.get('confidence', 0.0)
    debug_info.append(f"Final confidence: {final_confidence:.2f}")
    
    # Save debug data
    debug_data = {
        'question': question,
        'is_perspective_task': is_perspective,
        'detection_confidence': confidence,
        'final_answer': answer,
        'final_confidence': final_confidence,
        'result': result
    }
    
    debug_file = session_dir / "debug_data.json"
    with open(debug_file, 'w') as f:
        json.dump(debug_data, f, indent=2, default=str)
    
    video_path = None
    frame_paths = []
    
    if 'rephrased_prompt' in result:
        debug_info.append(f"Rephrased prompt: {result['rephrased_prompt']}")
    
    # Video path is saved by SoraVM in session_dir
    video_files = list(session_dir.glob("sora_*.mp4"))
    if video_files:
        video_path = str(video_files[0])
        debug_info.append(f"Video found: {video_path}")
        
        # Frames are saved by FrameExtractor in session_dir/frames
        frames_dir = session_dir / "frames"
        if frames_dir.exists():
            frame_files = sorted(frames_dir.glob("*.png"))
            frame_paths = [str(f) for f in frame_files]
            debug_info.append(f"Found {len(frame_paths)} frames")
    else:
        debug_info.append("No video generated")
    
    debug_info_str = "\n".join(debug_info)
    
    # Save debug info
    with open(session_dir / "debug_info.txt", 'w') as f:
        f.write(debug_info_str)
    
    return answer, video_path, frame_paths, debug_info_str


def create_interface():
    """Create debugging interface: image + question = answer + video + frames + debug."""
    with gr.Blocks(title="Hegarty AI - Debug Mode", theme=gr.themes.Soft()) as interface:
        gr.HTML("<h1 style='text-align: center;'>🧠 Hegarty AI - Debug Mode</h1>")
        gr.HTML("<p style='text-align: center;'>Enhanced spatial reasoning with multiple MLLM providers</p>")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("<h3>📤 Input</h3>")
                image_input = gr.Image(label="Upload Image", type="pil")
                question_input = gr.Textbox(
                    label="Ask a question about the image", 
                    placeholder="What would this look like if rotated 90 degrees?",
                    lines=2
                )
                
                # Provider selection
                provider_dropdown = gr.Dropdown(
                    label="MLLM Provider",
                    choices=["qwen"],
                    value="qwen",
                    info="Select the multimodal language model provider"
                )
                
                submit_btn = gr.Button("🚀 Process with Hegarty", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                gr.HTML("<h3>💬 Answer</h3>")
                answer_output = gr.Textbox(label="Final Answer", lines=6, interactive=False)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("<h3>🎬 Sora Video</h3>")
                video_output = gr.Video(label="Generated Video", interactive=False, value=None)
                
            with gr.Column(scale=1):
                gr.HTML("<h3>🖼️ Extracted Frames</h3>")
                frames_gallery = gr.Gallery(
                    label="Frames Used for Analysis",
                    show_label=True,
                    elem_id="gallery",
                    columns=3,
                    rows=2,
                    height="auto",
                    value=[]
                )
        
        with gr.Row():
            gr.HTML("<h3>🔍 Debug Information</h3>")
            debug_output = gr.Textbox(
                label="Processing Details",
                lines=10,
                interactive=False,
                max_lines=20
            )
        
        # Add examples
        gr.HTML("<h3>💡 Example Questions</h3>")
        examples = [
            "What would this look like if rotated 90 degrees clockwise?",
            "If I flip this object upside down, what face would be visible?",
            "What would I see if I looked at this from the other side?",
            "How would this appear if rotated 180 degrees?",
            "What's the view from the back of this object?"
        ]
        
        example_buttons = []
        for example in examples:
            btn = gr.Button(example, size="sm")
            example_buttons.append(btn)
            btn.click(
                fn=lambda ex=example: ex,
                inputs=[],
                outputs=[question_input]
            )
        
        submit_btn.click(
            fn=process_image_question,
            inputs=[image_input, question_input, provider_dropdown],
            outputs=[answer_output, video_output, frames_gallery, debug_output]
        )
    
    return interface


def main():
    """Launch the simple interface."""
    print("🧠 Starting Hegarty AI...")
    print(f"📁 All session files will be saved to: {SESSIONS_DIR}")
    
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        # Allow Gradio to serve files from temp directory
        allowed_paths=[str(TEMP_DIR)]
    )


if __name__ == "__main__":
    main()