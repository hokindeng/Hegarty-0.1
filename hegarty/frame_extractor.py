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
        
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            all_frames.append(frame_rgb)
        
        cap.release()
        
        if not all_frames:
            return []
        
        if len(all_frames) > window_size:
            all_frames = all_frames[-window_size:]
        
        extracted = self._extract_by_strategy(all_frames, num_frames)
        self._save_frames(extracted, video_path, session_dir)
        
        logger.info(f"Extracted {len(extracted)} frames")
        return extracted
    
    def _extract_by_strategy(self, frames: List[np.ndarray], num_frames: int) -> List[np.ndarray]:
        if len(frames) <= num_frames:
            return frames
        
        if self.strategy == "adaptive":
            return self._adaptive_extraction(frames, num_frames)
        elif self.strategy == "keyframe":
            return self._keyframe_extraction(frames, num_frames)
        else:
            return self._uniform_extraction(frames, num_frames)
    
    def _uniform_extraction(self, frames: List[np.ndarray], num_frames: int) -> List[np.ndarray]:
        step = (len(frames) - 1) / (num_frames - 1)
        indices = [int(i * step) for i in range(num_frames)]
        return [frames[i] for i in indices]
    
    def _adaptive_extraction(self, frames: List[np.ndarray], num_frames: int) -> List[np.ndarray]:
        differences = [(i, np.mean((frames[i-1].astype(float) - frames[i].astype(float)) ** 2)) 
                      for i in range(1, len(frames))]
        differences.sort(key=lambda x: x[1], reverse=True)
        
        selected = [0] + [idx for idx, _ in differences[:num_frames-2]] + [len(frames)-1]
        selected = sorted(list(set(selected)))[:num_frames]
        return [frames[i] for i in selected]
    
    def _keyframe_extraction(self, frames: List[np.ndarray], num_frames: int) -> List[np.ndarray]:
        scores = [(i, self._info_score(frame)) for i, frame in enumerate(frames)]
        scores.sort(key=lambda x: x[1], reverse=True)
        selected = sorted([idx for idx, _ in scores[:num_frames]])
        return [frames[i] for i in selected]
    
    def _info_score(self, frame: np.ndarray) -> float:
        gray = np.mean(frame, axis=2) if len(frame.shape) == 3 else frame
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        hist = hist / (hist.sum() + 1e-10)
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return entropy
    
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
