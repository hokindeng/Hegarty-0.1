"""
SoraInterface: Interface for Sora-2 video generation using real OpenAI API
"""

import os
import time
import asyncio
import logging
import tempfile
from typing import Optional, Dict, Any, Union, Tuple
from pathlib import Path
from io import BytesIO
from PIL import Image
import base64
import httpx

logger = logging.getLogger(__name__)

try:
    import cv2
except ImportError as e:
    raise ImportError("Please `pip install opencv-python httpx pillow`") from e


class SoraInterface:
    """
    Real Sora-2 video generation interface using OpenAI API.
    """

    def __init__(
        self,
        api_key: str,
        config: Optional[Any] = None,
        model: str = "sora-2",
        base_url: str = "https://api.openai.com/v1"
    ):
        """
        Initialize Sora interface with real API.
        
        Args:
            api_key: OpenAI API key with Sora access
            config: Configuration object
            model: Sora model to use ("sora-2" or "sora-2-pro")
            base_url: API base URL
        """
        if not api_key:
            raise ValueError("Sora API key is required")
        
        self.api_key = api_key
        self.config = config
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.request_timeout_sec = 300.0
        
        # Store last job result for debugging
        self.last_job_result = None
        
        # Model constraints
        self.model_constraints = self._get_model_constraints(model)
        self.upload_field_name = "input_reference"
        self.allowed_mimes = {"image/jpeg", "image/png", "image/webp"}
        
        logger.info(f"Sora interface initialized with {model}")

    def _get_model_constraints(self, model: str) -> Dict[str, Any]:
        constraints = {
            "sora-2": {
                "durations": ["4", "8", "12"],
                "sizes": ["1280x720", "720x1280"],
                "description": "OpenAI Sora-2 - High-quality video generation",
            },
            "sora-2-pro": {
                "durations": ["4", "8", "12"],
                "sizes": ["1280x720", "720x1280", "1024x1792", "1792x1024"],
                "description": "OpenAI Sora-2-Pro - Enhanced model with more resolution options",
            },
        }
        if model not in constraints:
            raise ValueError(f"Unknown Sora model: {model}. Available: {list(constraints.keys())}")
        return constraints[model]

    def generate_video(
        self,
        prompt: str,
        image: str,
        duration: int = 4,
        fps: int = 10,
        resolution: str = "1280x720"
    ) -> Dict[str, Any]:
        """
        Generate a video showing mental rotation or perspective change.
        
        Args:
            prompt: Text prompt describing the transformation
            image: Base64 encoded starting image
            duration: Video duration in seconds (4, 8, or 12)
            fps: Frames per second (ignored, Sora uses its own FPS)
            resolution: Output resolution
        
        Returns:
            Dictionary containing video data and metadata
        """
        logger.info(f"Generating video: {prompt[:50]}...")
        
        # Convert base64 image to temp file
        temp_image_path = self._base64_to_temp_file(image)
        
        try:
            # Run async generation
            result = asyncio.run(self._generate_video_async(
                prompt=prompt,
                image_path=temp_image_path,
                duration=str(duration),
                size=resolution
            ))
            
            return result
            
        finally:
            # Clean up temp file
            if temp_image_path and Path(temp_image_path).exists():
                Path(temp_image_path).unlink()

    def _base64_to_temp_file(self, image_data: str) -> str:
        """Convert base64 image to temporary file."""
        # Remove data URL prefix if present
        if image_data.startswith('data:'):
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        
        # Create temp file in project temp directory
        temp_dir = Path.cwd() / "temp"
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / f"sora_input_{int(time.time())}.png"
        
        # Save image
        with open(temp_path, "wb") as f:
            f.write(image_bytes)
        
        return str(temp_path)

    async def _generate_video_async(
        self,
        prompt: str,
        image_path: str,
        duration: str = "4",
        size: str = "1280x720"
    ) -> Dict[str, Any]:
        """Async video generation using real Sora API."""
        
        # Validate duration and size
        if duration not in self.model_constraints["durations"]:
            raise ValueError(f"Duration {duration} not supported. Allowed: {self.model_constraints['durations']}")
        
        if size not in self.model_constraints["sizes"]:
            raise ValueError(f"Size {size} not supported. Allowed: {self.model_constraints['sizes']}")
        
        # Prepare image for upload
        filename, file_bytes, mime = await self._prepare_image_for_upload(image_path, size, auto_pad=True)
        
        # Create video job
        video_id = await self._create_video_job(
            prompt=prompt,
            image_file=(filename, file_bytes, mime),
            duration_str=duration,
            size=size
        )
        
        logger.info(f"Sora video generation started. Job ID: {video_id}")
        
        # Poll for completion
        job_result = await self._poll_video_job(video_id)
        
        # Store result for debugging access
        self.last_job_result = job_result
        
        status = job_result.get("status")
        if status not in {"completed", "succeeded"}:
            raise Exception(f"Video generation failed: {job_result}")
        
        # Download the actual video file
        video_path = await self._download_video(job_result, video_id)
        
        # Create result in expected format
        return {
            'video_path': video_path,  # Path to the downloaded video file
            'frames': [],  # Will be populated by frame extractor
            'metadata': {
                'video_id': video_id,
                'duration': duration,
                'size': size,
                'prompt': prompt,
                'status': status,
                'sora_result': job_result
            }
        }

    async def _prepare_image_for_upload(
        self,
        image_path: str,
        target_size: str,
        auto_pad: bool = True
    ) -> Tuple[str, BytesIO, str]:
        """Prepare image for Sora API upload."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        target_w, target_h = map(int, target_size.split("x"))

        with Image.open(path) as img:
            img = img.convert("RGB")
            current_w, current_h = img.size

            if (current_w, current_h) != (target_w, target_h) and auto_pad:
                # Resize with padding
                padded = Image.new("RGB", (target_w, target_h), color=(128, 128, 128))
                scale = min(target_w / current_w, target_h / current_h)
                new_w = max(1, int(current_w * scale))
                new_h = max(1, int(current_h * scale))
                resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                x_offset = (target_w - new_w) // 2
                y_offset = (target_h - new_h) // 2
                padded.paste(resized, (x_offset, y_offset))
                img = padded

        # Determine format
        ext = path.suffix.lower()
        if ext in {".jpg", ".jpeg"}:
            mime = "image/jpeg"
            pil_format = "JPEG"
            filename = path.stem + ".jpg"
        elif ext == ".webp":
            mime = "image/webp"
            pil_format = "WEBP" 
            filename = path.stem + ".webp"
        else:
            mime = "image/png"
            pil_format = "PNG"
            filename = path.stem + ".png"

        bio = BytesIO()
        img.save(bio, format=pil_format)
        bio.seek(0)
        return filename, bio, mime

    def _auth_headers(self, idempotency_key: Optional[str] = None) -> Dict[str, str]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return headers

    async def _create_video_job(
        self,
        prompt: str,
        image_file: Tuple[str, BytesIO, str],
        duration_str: str,
        size: str,
        idempotency_key: Optional[str] = None
    ) -> str:
        """Create Sora video generation job."""
        url = f"{self.base_url}/videos"
        
        headers = self._auth_headers(
            idempotency_key=idempotency_key or f"sora-{int(time.time()*1000)}"
        )

        filename, file_bytes, mime = image_file

        data = {
            "model": self.model,
            "prompt": prompt,
            "size": size,
            "seconds": duration_str
        }

        files = {
            self.upload_field_name: (filename, file_bytes, mime)
        }

        async with httpx.AsyncClient() as client:
            resp = await client.request(
                "POST", url, 
                headers=headers, 
                data=data, 
                files=files,
                timeout=self.request_timeout_sec
            )

        if resp.status_code != 200:
            raise Exception(f"Failed to create video job: {resp.status_code} {resp.text}")

        job = resp.json()
        video_id = job.get("id")
        if not video_id:
            raise Exception(f"Invalid create response (no id): {job}")
        return video_id

    async def _poll_video_job(self, video_id: str, max_wait_time: int = 14400) -> Dict[str, Any]:
        """Poll video job until completion."""
        url = f"{self.base_url}/videos/{video_id}"
        headers = self._auth_headers()
        terminal = {"completed", "succeeded", "failed", "cancelled", "rejected"}

        start = time.time()
        interval = 2.0

        async with httpx.AsyncClient() as client:
            while time.time() - start < max_wait_time:
                resp = await client.request(
                    "GET", url, 
                    headers=headers,
                    timeout=self.request_timeout_sec
                )
                
                if resp.status_code != 200:
                    logger.warning(f"Poll {video_id} -> {resp.status_code}")
                    await asyncio.sleep(interval)
                    continue

                job = resp.json()
                status = job.get("status", "unknown")
                progress = job.get("progress")

                logger.info(f"[{video_id}] status={status} progress={progress}")

                if status in terminal:
                    if status not in {"completed", "succeeded"}:
                        err = job.get("error")
                        raise Exception(f"Video generation failed: status={status} error={err}")
                    return job

                await asyncio.sleep(interval)
                interval = min(10.0, interval * 1.25)

        raise TimeoutError(f"Video generation timed out after {max_wait_time} seconds")

    async def _download_video(self, job_result: Dict[str, Any], video_id: str) -> str:
        """Download the generated video file using direct content endpoint."""
        # Use direct content endpoint - no need to search for URLs in job result
        content_url = f"{self.base_url}/videos/{video_id}/content"
        headers = self._auth_headers()
        
        logger.info(f"Downloading video from content endpoint: {content_url}")
        
        # Create temp directory path
        temp_dir = Path.cwd() / "temp"
        temp_dir.mkdir(exist_ok=True)
        
        # Download video
        video_filename = f"sora_video_{video_id}_{int(time.time())}.mp4"
        video_path = temp_dir / video_filename
        
        async with httpx.AsyncClient() as client:
            async with client.stream("GET", content_url, headers=headers, timeout=self.request_timeout_sec) as resp:
                if resp.status_code != 200:
                    error_text = await resp.aread() if hasattr(resp, 'aread') else 'Unknown error'
                    raise Exception(f"Failed to download video: {resp.status_code} {error_text}")
                
                with open(video_path, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
        
        logger.info(f"Video saved to: {video_path}")
        return str(video_path)

