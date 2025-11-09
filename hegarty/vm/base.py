"""Base interface for Video Model providers"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path


class VMProvider(ABC):
    """Base interface for video generation providers"""
    
    @abstractmethod
    def generate_video(
        self,
        prompt: str,
        image: str,
        duration: int = 4,
        fps: int = 10,
        resolution: str = "1280x720",
        session_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Generate video from prompt and image.
        
        Args:
            prompt: Text prompt for video generation
            image: Base64 encoded starting image
            duration: Video duration in seconds
            fps: Frames per second
            resolution: Output resolution
            session_dir: Session directory for organizing files
        
        Returns:
            Dict with video_path, frames, and metadata
        """
        pass

