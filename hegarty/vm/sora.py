"""Sora video generation provider"""

import time
import asyncio
import logging
import base64
import json
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from io import BytesIO

import httpx
from PIL import Image

from .base import VMProvider

logger = logging.getLogger(__name__)


class SoraVM(VMProvider):
    """Sora video generation provider"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "sora-2",
        base_url: str = "https://api.openai.com/v1"
    ):
        if not api_key:
            raise ValueError("Sora API key required")
        
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = 300.0
        
        self.constraints = self._get_constraints(model)
        logger.info(f"Sora VM initialized with {model}")
    
    def _get_constraints(self, model: str) -> Dict[str, Any]:
        constraints = {
            "sora-2": {
                "durations": ["4", "8", "12"],
                "sizes": ["1280x720", "720x1280"],
            },
            "sora-2-pro": {
                "durations": ["4", "8", "12"],
                "sizes": ["1280x720", "720x1280", "1024x1792", "1792x1024"],
            },
        }
        if model not in constraints:
            raise ValueError(f"Unknown Sora model: {model}")
        return constraints[model]
    
    def generate_video(
        self,
        prompt: str,
        image: str,
        duration: int = 4,
        fps: int = 10,
        resolution: str = "1280x720",
        session_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        logger.info(f"Generating video: {prompt[:50]}...")
        
        # Save the Sora prompt to JSON
        if session_dir:
            prompt_data = {
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "duration": duration,
                "fps": fps,
                "resolution": resolution
            }
            
            prompt_file = session_dir / "sora_prompt.json"
            with open(prompt_file, 'w') as f:
                json.dump(prompt_data, f, indent=2)
            logger.info(f"Saved Sora prompt to: {prompt_file}")
        
        temp_image = self._save_temp_image(image, session_dir)
        
        result = asyncio.run(self._generate_async(
            prompt=prompt,
            image_path=temp_image,
            duration=str(duration),
            size=resolution,
            session_dir=session_dir
        ))
        
        if temp_image and Path(temp_image).exists():
            Path(temp_image).unlink()
        
        return result
    
    def _save_temp_image(self, image_data: str, session_dir: Optional[Path]) -> str:
        if image_data.startswith('data:'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        
        temp_dir = session_dir if session_dir else Path.cwd() / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"sora_input_{int(time.time())}.png"
        
        with open(temp_path, "wb") as f:
            f.write(image_bytes)
        
        return str(temp_path)
    
    async def _generate_async(
        self,
        prompt: str,
        image_path: str,
        duration: str,
        size: str,
        session_dir: Optional[Path]
    ) -> Dict[str, Any]:
        if duration not in self.constraints["durations"]:
            raise ValueError(f"Duration {duration} not supported")
        if size not in self.constraints["sizes"]:
            raise ValueError(f"Size {size} not supported")
        
        filename, file_bytes, mime = await self._prepare_image(image_path, size)
        video_id = await self._create_job(prompt, (filename, file_bytes, mime), duration, size)
        
        logger.info(f"Sora job started: {video_id}")
        
        job_result = await self._poll_job(video_id)
        status = job_result.get("status")
        
        if status not in {"completed", "succeeded"}:
            # This should not happen as _poll_job already raises on failure
            logger.error(f"Unexpected job status: {job_result}")
            raise Exception(f"Video generation failed with status {status}: {job_result}")
        
        video_path = await self._download_video(video_id, session_dir)
        
        return {
            'video_path': video_path,
            'frames': [],
            'metadata': {
                'video_id': video_id,
                'duration': duration,
                'size': size,
                'prompt': prompt,
                'status': status
            }
        }
    
    async def _prepare_image(self, image_path: str, target_size: str) -> Tuple[str, BytesIO, str]:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        target_w, target_h = map(int, target_size.split("x"))
        
        with Image.open(path) as img:
            img = img.convert("RGB")
            current_w, current_h = img.size
            
            if (current_w, current_h) != (target_w, target_h):
                padded = Image.new("RGB", (target_w, target_h), color=(128, 128, 128))
                scale = min(target_w / current_w, target_h / current_h)
                new_w = max(1, int(current_w * scale))
                new_h = max(1, int(current_h * scale))
                resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                x_offset = (target_w - new_w) // 2
                y_offset = (target_h - new_h) // 2
                padded.paste(resized, (x_offset, y_offset))
                img = padded
        
        ext = path.suffix.lower()
        if ext in {".jpg", ".jpeg"}:
            mime, pil_format, filename = "image/jpeg", "JPEG", path.stem + ".jpg"
        elif ext == ".webp":
            mime, pil_format, filename = "image/webp", "WEBP", path.stem + ".webp"
        else:
            mime, pil_format, filename = "image/png", "PNG", path.stem + ".png"
        
        bio = BytesIO()
        img.save(bio, format=pil_format)
        bio.seek(0)
        return filename, bio, mime
    
    async def _create_job(
        self,
        prompt: str,
        image_file: Tuple[str, BytesIO, str],
        duration: str,
        size: str
    ) -> str:
        url = f"{self.base_url}/videos"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Idempotency-Key": f"sora-{int(time.time()*1000)}"
        }
        
        filename, file_bytes, mime = image_file
        data = {"model": self.model, "prompt": prompt, "size": size, "seconds": duration}
        files = {"input_reference": (filename, file_bytes, mime)}
        
        # Log request details for debugging
        logger.info(f"Creating Sora job - model: {self.model}, size: {size}, duration: {duration}s")
        logger.debug(f"Prompt: {prompt}")
        
        async with httpx.AsyncClient() as client:
            resp = await client.request("POST", url, headers=headers, data=data, files=files, timeout=self.timeout)
        
        if resp.status_code != 200:
            error_details = ""
            if resp.headers.get("content-type", "").startswith("application/json"):
                error_json = resp.json()
                error_details = f", error: {error_json}"
            raise Exception(f"Failed to create job: {resp.status_code} {resp.text}{error_details}")
        
        job = resp.json()
        video_id = job.get("id")
        if not video_id:
            raise Exception(f"Invalid response (no id): {job}")
        return video_id
    
    async def _poll_job(self, video_id: str, max_wait: int = 14400) -> Dict[str, Any]:
        url = f"{self.base_url}/videos/{video_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        terminal = {"completed", "succeeded", "failed", "cancelled", "rejected"}
        
        start = time.time()
        interval = 2.0
        
        async with httpx.AsyncClient() as client:
            while time.time() - start < max_wait:
                resp = await client.request("GET", url, headers=headers, timeout=self.timeout)
                
                if resp.status_code != 200:
                    logger.warning(f"Poll request failed: {resp.status_code} {resp.text}")
                    await asyncio.sleep(interval)
                    continue
                
                job = resp.json()
                status = job.get("status", "unknown")
                
                logger.info(f"[{video_id}] status={status}")
                
                if status in terminal:
                    if status not in {"completed", "succeeded"}:
                        # Extract error details from the response
                        error_msg = job.get("error", {}).get("message", "No error details provided")
                        error_type = job.get("error", {}).get("type", "Unknown error type")
                        error_code = job.get("error", {}).get("code", "No error code")
                        
                        # Log the full response for debugging
                        logger.error(f"Generation failed - Full response: {job}")
                        
                        # Raise with detailed error information
                        raise Exception(
                            f"Generation failed: status={status}, "
                            f"error_type={error_type}, "
                            f"error_code={error_code}, "
                            f"message={error_msg}, "
                            f"video_id={video_id}"
                        )
                    return job
                
                await asyncio.sleep(interval)
                interval = min(10.0, interval * 1.25)
        
        raise TimeoutError(f"Timed out after {max_wait}s")
    
    async def _download_video(self, video_id: str, session_dir: Optional[Path]) -> str:
        url = f"{self.base_url}/videos/{video_id}/content"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        logger.info(f"Downloading video: {url}")
        
        temp_dir = session_dir if session_dir else Path.cwd() / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        video_path = temp_dir / f"sora_{video_id}_{int(time.time())}.mp4"
        
        async with httpx.AsyncClient() as client:
            async with client.stream("GET", url, headers=headers, timeout=self.timeout) as resp:
                if resp.status_code != 200:
                    raise Exception(f"Download failed: {resp.status_code}")
                
                with open(video_path, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
        
        logger.info(f"Video saved: {video_path}")
        return str(video_path)

