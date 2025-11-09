"""Frame extraction from videos"""

import logging
from typing import List, Optional
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

logger = logging.getLogger(__name__)


class FrameExtractor:
    """Extract key frames from video"""
    
    def __init__(self, strategy: str = "uniform"):
        self.strategy = strategy
        logger.info(f"FrameExtractor: {strategy} strategy")
    
    def extract_frames(
        self,
        video_data: dict,
        num_frames: int = 5,
        window_size: int = 30,
        session_dir: Optional[Path] = None
    ) -> List[np.ndarray]:
        video_path = video_data.get('video_path')
        if not video_path or not Path(video_path).exists():
            logger.error("No valid video path")
            return []
        
        logger.info(f"Extracting {num_frames} frames from {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Failed to open video")
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Video: {total_frames} frames @ {fps:.2f} FPS")
        
        if total_frames == 0:
            cap.release()
            return []
        
        # Calculate which frames to extract (from last window_size frames)
        start_frame = max(0, total_frames - window_size)
        frame_count = min(window_size, total_frames)
        indices = self._calculate_indices(frame_count, num_frames, self.strategy)
        indices = [start_frame + i for i in indices]
        
        # Extract only needed frames
        extracted = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                extracted.append(frame_rgb)
        
        cap.release()
        self._save_frames(extracted, video_path, session_dir)
        
        logger.info(f"Extracted {len(extracted)} frames")
        return extracted
    
    def _calculate_indices(self, frame_count: int, num_frames: int, strategy: str) -> List[int]:
        """Calculate frame indices to extract"""
        if frame_count <= num_frames:
            return list(range(frame_count))
        
        step = (frame_count - 1) / (num_frames - 1)
        return [int(i * step) for i in range(num_frames)]
    
    def _save_frames(self, frames: List[np.ndarray], video_path: str, session_dir: Optional[Path]):
        temp_dir = session_dir / "frames" if session_dir else Path.cwd() / "temp" / "frames"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        video_name = Path(video_path).stem
        for i, frame in enumerate(frames):
            frame_path = temp_dir / f"{video_name}_frame_{i:03d}.png"
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            Image.fromarray(frame).save(frame_path)
        
        logger.info(f"Saved {len(frames)} frames to {temp_dir}")
