"""Configuration management"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """Configuration for Hegarty agent and components"""
    
    # MLLM settings
    gpt_model: str = "gpt-4o"
    temperature: float = 0.3
    max_tokens: int = 2000
    
    # VM settings
    sora_video_length: int = 4  # seconds
    sora_fps: int = 10
    sora_resolution: str = "1280x720"
    
    # Camera control settings
    camera_elevation_default: float = 20.0  # degrees, slightly above to avoid occlusion
    camera_azimuth_start: float = 0.0  # degrees
    camera_azimuth_end: float = 180.0  # degrees for opposite perspective
    
    # Camera detection settings
    use_camera_detection: bool = True  # Use PerspectiveFields to detect camera params
    camera_detector_model: str = "Paramnet-360Cities-edina-centered"  # PerspectiveFields model
    camera_detection_confidence_threshold: float = 0.5  # Min confidence to use detected params
    
    # Frame extraction
    frame_extraction_count: int = 1  # Extract only the last frame
    frame_extraction_window: int = 30
    frame_extraction_strategy: str = "uniform"  # uniform, adaptive, keyframe
    
    # Parallel processing
    max_workers: int = 6
    timeout: int = 30
    
    # API settings
    max_image_size: int = 10485760  # 10MB
    supported_formats: List[str] = field(default_factory=lambda: [
        "jpeg", "jpg", "png", "gif", "webp"
    ])
