"""
Configuration management for Hegarty
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """
    Configuration for Hegarty agent and components.
    """
    # GPT-4o settings
    gpt_model: str = "gpt-4o"
    temperature: float = 0.3
    max_tokens: int = 2000
    
    # Sora-2 settings
    sora_video_length: int = 4  # seconds - cost-efficient at $0.40 per video
    sora_fps: int = 10
    sora_resolution: str = "1280x720"  # Sora-2 supported resolution
    
    # Frame extraction settings
    frame_extraction_count: int = 5
    frame_extraction_window: int = 30  # Last N frames to consider
    frame_extraction_strategy: str = "uniform"  # uniform, adaptive, keyframe
    
    # Parallel processing
    max_workers: int = 6
    timeout: int = 30
    
    # API settings
    max_image_size: int = 10485760  # 10MB
    supported_formats: List[str] = field(default_factory=lambda: [
        "jpeg", "jpg", "png", "gif", "webp"
    ])
