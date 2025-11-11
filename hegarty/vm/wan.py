"""Wan2.2 video generation provider - open-source local model"""

import time
import logging
import base64
import json
import subprocess
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from io import BytesIO
import tempfile

import torch
from PIL import Image
import numpy as np

from .base import VMProvider

logger = logging.getLogger(__name__)


class WanVM(VMProvider):
    """Wan2.2 video generation provider for local inference"""
    
    def __init__(
        self,
        model_size: str = "5B",
        model_dir: Optional[str] = None,
        device: Optional[str] = None,
        offload_model: bool = True,
        convert_model_dtype: bool = True,
        use_fsdp: bool = False,
        ulysses_size: Optional[int] = None,
        use_prompt_extend: bool = False
    ):
        """
        Initialize Wan2.2 video model.
        
        Args:
            model_size: Model size - "5B", "14B", or "animate-14B"
            model_dir: Path to model weights directory
            device: Device to run on (cuda/cpu)
            offload_model: Whether to offload model to save memory
            convert_model_dtype: Convert model to lower precision
            use_fsdp: Use FSDP for multi-GPU
            ulysses_size: DeepSpeed Ulysses parallelism size
            use_prompt_extend: Use prompt extension for better results
        """
        self.model_size = model_size
        self.model_name = self._get_model_name(model_size)
        
        # Set model directory
        if model_dir is None:
            home = Path.home()
            model_dir = home / ".cache" / "wan_models" / self.model_name
        self.model_dir = Path(model_dir)
        
        # Device setup
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Model configuration
        self.offload_model = offload_model
        self.convert_model_dtype = convert_model_dtype
        self.use_fsdp = use_fsdp
        self.ulysses_size = ulysses_size
        self.use_prompt_extend = use_prompt_extend
        
        # Model constraints
        self.constraints = self._get_constraints(model_size)
        
        logger.info(f"Wan VM initialized with {model_size} on {self.device}")
        
        # Model will be loaded on first use
        self.model_loaded = False
        self.generate_script = None
        
    def _get_model_name(self, model_size: str) -> str:
        """Map model size to model name"""
        model_map = {
            "5B": "Wan2.2-TI2V-5B",
            "14B": "Wan2.2-14B",
            "animate-14B": "Wan2.2-Animate-14B"
        }
        if model_size not in model_map:
            raise ValueError(f"Unknown model size: {model_size}. Choose from: {list(model_map.keys())}")
        return model_map[model_size]
    
    def _get_constraints(self, model_size: str) -> Dict[str, Any]:
        """Get model constraints based on size"""
        # All models support similar durations and resolutions
        return {
            "min_duration": 1,
            "max_duration": 20,  # seconds
            "supported_fps": [8, 16, 24],
            "supported_resolutions": [
                "480x640",    # 480p portrait
                "640x480",    # 480p landscape
                "720x1280",   # 720p portrait  
                "1280x720",   # 720p landscape
                "1080x1920",  # 1080p portrait
                "1920x1080",  # 1080p landscape
            ],
            "default_duration": 5,
            "default_fps": 24,
            "default_resolution": "1280x720"
        }
    
    def _setup_model(self):
        """Download and setup the model if needed"""
        if not self.model_dir.exists():
            logger.info(f"Model not found at {self.model_dir}. Please download it first.")
            self._download_instructions()
            raise RuntimeError(f"Model not found. Please download {self.model_name} first.")
        
        # Find generate.py script
        wan_repo = self.model_dir.parent.parent  # Assuming structure like ~/.cache/wan_models/
        generate_script = wan_repo / "generate.py"
        
        if not generate_script.exists():
            # Try to find it in the current working directory or a known location
            possible_paths = [
                Path.cwd() / "Wan2.2" / "generate.py",
                Path("/opt/wan2.2/generate.py"),
                Path.home() / "Wan2.2" / "generate.py"
            ]
            
            for path in possible_paths:
                if path.exists():
                    generate_script = path
                    break
            else:
                raise RuntimeError(
                    "Could not find Wan2.2 generate.py script. "
                    "Please clone the Wan2.2 repository and set up the environment."
                )
        
        self.generate_script = generate_script
        self.model_loaded = True
        logger.info(f"Model setup complete. Using generate.py at: {generate_script}")
    
    def _download_instructions(self):
        """Print instructions for downloading the model"""
        instructions = f"""
To use Wan2.2, you need to:

1. Clone the repository:
   git clone https://github.com/Wan-Video/Wan2.2.git
   cd Wan2.2

2. Download model weights:
   # For {self.model_name}:
   wget https://huggingface.co/hpwang/Wan2.2/resolve/main/{self.model_name}.tar.gz
   tar -xzf {self.model_name}.tar.gz -C {self.model_dir.parent}

3. Install dependencies:
   pip install -r requirements.txt
   
4. For TI2V-5B model also install:
   pip install -r requirements_t2v.txt
   
5. For Animate model also install:
   pip install -r requirements_animate.txt
"""
        logger.info(instructions)
    
    def generate_video(
        self,
        prompt: str,
        image: Optional[str] = None,
        duration: int = 5,
        fps: int = 24,
        resolution: str = "1280x720",
        session_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Generate video using Wan2.2.
        
        Args:
            prompt: Text prompt for video generation
            image: Base64 encoded image (optional for TI2V models, required for animate)
            duration: Video duration in seconds
            fps: Frames per second
            resolution: Output resolution (WxH)
            session_dir: Session directory for organizing files
        
        Returns:
            Dict with video_path, frames, and metadata
        """
        logger.info(f"Generating video with Wan2.2: {prompt[:50]}...")
        
        # Ensure model is set up
        if not self.model_loaded:
            self._setup_model()
        
        # Validate parameters
        if fps not in self.constraints["supported_fps"]:
            logger.warning(f"FPS {fps} not in supported list, using default")
            fps = self.constraints["default_fps"]
            
        if resolution not in self.constraints["supported_resolutions"]:
            logger.warning(f"Resolution {resolution} not supported, using default")
            resolution = self.constraints["default_resolution"]
        
        # Create output directory
        if session_dir is None:
            session_dir = Path.cwd() / "temp" / f"wan_session_{int(time.time())}"
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Save generation parameters
        params = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "duration": duration,
            "fps": fps,
            "resolution": resolution,
            "model": self.model_name,
            "has_image": image is not None
        }
        
        param_file = session_dir / "wan_params.json"
        with open(param_file, 'w') as f:
            json.dump(params, f, indent=2)
        
        # Handle input image if provided
        input_image_path = None
        if image:
            input_image_path = self._save_input_image(image, session_dir)
        
        # Generate video
        video_path = self._run_generation(
            prompt=prompt,
            image_path=input_image_path,
            duration=duration,
            fps=fps,
            resolution=resolution,
            output_dir=session_dir
        )
        
        # Extract frames if needed
        frames = []
        if video_path and Path(video_path).exists():
            frames_dir = session_dir / "frames"
            frames_dir.mkdir(exist_ok=True)
            # Could extract frames here if needed
        
        return {
            'video_path': video_path,
            'frames': frames,
            'metadata': {
                'model': self.model_name,
                'duration': duration,
                'fps': fps,
                'resolution': resolution,
                'prompt': prompt,
                'session_dir': str(session_dir)
            }
        }
    
    def _save_input_image(self, image_data: str, output_dir: Path) -> str:
        """Save base64 image to file"""
        if image_data.startswith('data:'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image_path = output_dir / "input_image.png"
        
        # Save and potentially resize image
        img = Image.open(BytesIO(image_bytes))
        img = img.convert("RGB")
        img.save(image_path, "PNG")
        
        return str(image_path)
    
    def _run_generation(
        self,
        prompt: str,
        image_path: Optional[str],
        duration: int,
        fps: int,
        resolution: str,
        output_dir: Path
    ) -> str:
        """Run the actual video generation"""
        
        # Prepare command
        cmd = ["python", str(self.generate_script)]
        
        # Add task based on model
        if "animate" in self.model_size.lower():
            cmd.extend(["--task", "animate-14B"])
            if not image_path:
                raise ValueError("Animate model requires an input image")
        else:
            cmd.extend(["--task", f"i2v-{self.model_size}"])
        
        # Add model directory
        cmd.extend(["--ckpt_dir", str(self.model_dir)])
        
        # Add input image if provided
        if image_path:
            cmd.extend(["--src_video_path", image_path])
        
        # Add prompt
        cmd.extend(["--prompt", prompt])
        
        # Add video parameters
        width, height = resolution.split('x')
        cmd.extend([
            "--resolution", f"{width},{height}",
            "--duration", str(duration),
            "--fps", str(fps)
        ])
        
        # Add model configuration
        if self.offload_model:
            cmd.append("--offload_model")
        if self.convert_model_dtype:
            cmd.append("--convert_model_dtype")
        if self.use_prompt_extend:
            cmd.append("--use_prompt_extend")
            
        # Add distributed training args if using multiple GPUs
        if self.use_fsdp:
            cmd.append("--dit_fsdp")
            cmd.append("--t5_fsdp")
        if self.ulysses_size:
            cmd.extend(["--ulysses_size", str(self.ulysses_size)])
        
        # Set output path
        output_path = output_dir / f"wan_video_{int(time.time())}.mp4"
        cmd.extend(["--output", str(output_path)])
        
        # Log the command
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Execute the generation
        env = os.environ.copy()
        if self.device == "cuda":
            # Set CUDA visible devices if needed
            if "CUDA_VISIBLE_DEVICES" not in env:
                env["CUDA_VISIBLE_DEVICES"] = "0"
        
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            cwd=str(self.generate_script.parent)
        )
        
        if result.returncode != 0:
            logger.error(f"Generation failed: {result.stderr}")
            raise RuntimeError(f"Video generation failed: {result.stderr}")
        
        if not output_path.exists():
            # Try to find the output video
            for video_file in output_dir.glob("*.mp4"):
                output_path = video_file
                break
            else:
                raise RuntimeError("Generated video not found")
        
        logger.info(f"Video generated successfully: {output_path}")
        return str(output_path)
