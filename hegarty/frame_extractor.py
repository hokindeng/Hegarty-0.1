"""
FrameExtractor: Extract key frames from video for analysis
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
from pathlib import Path
import cv2

logger = logging.getLogger(__name__)


class FrameExtractor:
    """
    Extracts key frames from video for parallel perspective analysis.
    
    Implements multiple extraction strategies:
    - Uniform: Equal spacing between frames
    - Adaptive: Based on visual change detection
    - Keyframe: Extract frames with maximum information
    """
    
    def __init__(self, config: Optional[Any] = None):
        """
        Initialize frame extractor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.strategy = config.frame_extraction_strategy if config else "uniform"
        
        logger.info(f"FrameExtractor initialized with {self.strategy} strategy")
    
    def extract_frames(
        self,
        video_data: Dict[str, Any],
        num_frames: int = 5,
        window_size: int = 30
    ) -> List[np.ndarray]:
        """
        Extract key frames from video data.
        
        Args:
            video_data: Video data dictionary from Sora (contains video_path)
            num_frames: Number of frames to extract
            window_size: Size of window to consider (last N frames)
        
        Returns:
            List of extracted frames as numpy arrays
        """
        # Check for video path first (new method)
        video_path = video_data.get('video_path')
        if video_path:
            return self._extract_frames_from_video(video_path, num_frames, window_size)
        
        # Fallback to old method for backwards compatibility
        frames = video_data.get('frames', [])
        
        if not frames:
            logger.error("No video path or frames in video data")
            return []
        
        # Get last window_size frames
        if len(frames) > window_size:
            frames = frames[-window_size:]
        
        logger.info(f"Extracting {num_frames} frames from {len(frames)} available")
        
        # Apply extraction strategy
        if self.strategy == "uniform":
            extracted = self._uniform_extraction(frames, num_frames)
        elif self.strategy == "adaptive":
            extracted = self._adaptive_extraction(frames, num_frames)
        elif self.strategy == "keyframe":
            extracted = self._keyframe_extraction(frames, num_frames)
        else:
            logger.warning(f"Unknown strategy {self.strategy}, using uniform")
            extracted = self._uniform_extraction(frames, num_frames)
        
        logger.info(f"Extracted {len(extracted)} frames")
        
        return extracted
    
    def _uniform_extraction(
        self,
        frames: List[np.ndarray],
        num_frames: int
    ) -> List[np.ndarray]:
        """
        Extract frames with uniform spacing.
        
        This is the default strategy that samples frames at regular intervals.
        """
        if len(frames) <= num_frames:
            return frames
        
        # Calculate indices for uniform sampling
        # We want to include first and last frame
        indices = []
        step = (len(frames) - 1) / (num_frames - 1)
        
        for i in range(num_frames):
            idx = int(i * step)
            indices.append(idx)
        
        # Ensure unique indices
        indices = list(dict.fromkeys(indices))
        
        extracted = [frames[i] for i in indices]
        
        logger.debug(f"Uniform extraction indices: {indices}")
        
        return extracted
    
    def _adaptive_extraction(
        self,
        frames: List[np.ndarray],
        num_frames: int
    ) -> List[np.ndarray]:
        """
        Extract frames based on visual change detection.
        
        Selects frames that show maximum visual difference.
        """
        if len(frames) <= num_frames:
            return frames
        
        # Calculate frame differences
        differences = []
        for i in range(1, len(frames)):
            diff = self._calculate_frame_difference(frames[i-1], frames[i])
            differences.append((i, diff))
        
        # Sort by difference (descending)
        differences.sort(key=lambda x: x[1], reverse=True)
        
        # Select frames with highest differences
        selected_indices = [0]  # Always include first frame
        for idx, diff in differences[:num_frames-2]:
            selected_indices.append(idx)
        selected_indices.append(len(frames) - 1)  # Always include last frame
        
        # Sort indices and remove duplicates
        selected_indices = sorted(list(set(selected_indices)))[:num_frames]
        
        extracted = [frames[i] for i in selected_indices]
        
        logger.debug(f"Adaptive extraction indices: {selected_indices}")
        
        return extracted
    
    def _keyframe_extraction(
        self,
        frames: List[np.ndarray],
        num_frames: int
    ) -> List[np.ndarray]:
        """
        Extract frames based on information content.
        
        Selects frames that contain maximum visual information.
        """
        if len(frames) <= num_frames:
            return frames
        
        # Calculate information content for each frame
        info_scores = []
        for i, frame in enumerate(frames):
            score = self._calculate_information_score(frame)
            info_scores.append((i, score))
        
        # Sort by information score (descending)
        info_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top frames by information content
        selected_indices = [idx for idx, score in info_scores[:num_frames]]
        
        # Sort indices for temporal order
        selected_indices.sort()
        
        extracted = [frames[i] for i in selected_indices]
        
        logger.debug(f"Keyframe extraction indices: {selected_indices}")
        
        return extracted
    
    def _calculate_frame_difference(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray
    ) -> float:
        """
        Calculate visual difference between two frames.
        
        Uses mean squared error for simplicity.
        """
        if frame1.shape != frame2.shape:
            # Resize if needed
            frame2 = self._resize_frame(frame2, frame1.shape)
        
        # Calculate MSE
        diff = np.mean((frame1.astype(float) - frame2.astype(float)) ** 2)
        
        return diff
    
    def _calculate_information_score(self, frame: np.ndarray) -> float:
        """
        Calculate information content score for a frame.
        
        Uses entropy as a measure of information content.
        """
        # Convert to grayscale if color
        if len(frame.shape) == 3:
            gray = np.mean(frame, axis=2)
        else:
            gray = frame
        
        # Calculate histogram
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        
        # Normalize histogram
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist = hist / hist_sum
        else:
            hist = hist + 1e-10  # Avoid division by zero
        
        # Calculate entropy
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Calculate edge strength as additional measure
        edges = self._detect_edges(gray)
        edge_score = np.mean(edges)
        
        # Combine entropy and edge score
        info_score = entropy * 0.7 + edge_score * 0.3
        
        return info_score
    
    def _detect_edges(self, gray_frame: np.ndarray) -> np.ndarray:
        """
        Simple edge detection using Sobel filters.
        """
        # Simple Sobel edge detection
        # In production, use cv2.Sobel or similar
        h, w = gray_frame.shape
        edges = np.zeros_like(gray_frame)
        
        # Simplified edge detection
        for i in range(1, h-1):
            for j in range(1, w-1):
                # Horizontal gradient
                gx = gray_frame[i+1, j] - gray_frame[i-1, j]
                # Vertical gradient
                gy = gray_frame[i, j+1] - gray_frame[i, j-1]
                # Edge magnitude
                edges[i, j] = np.sqrt(gx**2 + gy**2)
        
        return edges
    
    def _resize_frame(
        self,
        frame: np.ndarray,
        target_shape: tuple
    ) -> np.ndarray:
        """
        Resize frame to target shape.
        """
        # Convert to PIL for resizing
        pil_image = Image.fromarray(frame.astype(np.uint8))
        
        # Resize
        target_size = (target_shape[1], target_shape[0])  # PIL uses (width, height)
        resized = pil_image.resize(target_size, Image.Resampling.LANCZOS)
        
        return np.array(resized)
    
    def _extract_frames_from_video(
        self,
        video_path: str,
        num_frames: int = 5,
        window_size: int = 30
    ) -> List[np.ndarray]:
        """
        Extract frames directly from video file using OpenCV.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            window_size: Size of window to consider (last N frames)
        
        Returns:
            List of extracted frames as numpy arrays
        """
        if not Path(video_path).exists():
            logger.error(f"Video file not found: {video_path}")
            return []
        
        logger.info(f"Extracting frames from video: {video_path}")
        
        try:
            # Open video file
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                logger.error(f"Failed to open video file: {video_path}")
                return []
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            logger.info(f"Video properties: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s")
            
            if total_frames == 0:
                logger.error("Video has no frames")
                cap.release()
                return []
            
            # Read all frames first
            all_frames = []
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB for consistency with PIL
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                all_frames.append(frame_rgb)
                frame_idx += 1
            
            cap.release()
            
            if not all_frames:
                logger.error("No frames could be read from video")
                return []
            
            logger.info(f"Read {len(all_frames)} frames from video")
            
            # Apply window_size limit (get last N frames)
            if len(all_frames) > window_size:
                all_frames = all_frames[-window_size:]
                logger.info(f"Using last {window_size} frames")
            
            # Apply extraction strategy
            if self.strategy == "uniform":
                extracted = self._uniform_extraction(all_frames, num_frames)
            elif self.strategy == "adaptive":
                extracted = self._adaptive_extraction(all_frames, num_frames)
            elif self.strategy == "keyframe":
                extracted = self._keyframe_extraction(all_frames, num_frames)
            else:
                logger.warning(f"Unknown strategy {self.strategy}, using uniform")
                extracted = self._uniform_extraction(all_frames, num_frames)
            
            # Save extracted frames to temp directory for debugging
            self._save_frames_to_temp(extracted, video_path)
            
            logger.info(f"Successfully extracted {len(extracted)} frames using {self.strategy} strategy")
            
            return extracted
        
        except Exception as e:
            logger.error(f"Error extracting frames from video {video_path}: {e}")
            return []
    
    def _save_frames_to_temp(self, frames: List[np.ndarray], video_path: str) -> None:
        """
        Save extracted frames to temp directory for debugging and intermediate storage.
        
        Args:
            frames: List of extracted frames
            video_path: Original video path (used for naming)
        """
        try:
            # Create temp directory for frames
            temp_dir = Path.cwd() / "temp" / "frames"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Get video filename for naming
            video_name = Path(video_path).stem
            
            # Save each frame
            for i, frame in enumerate(frames):
                frame_filename = f"{video_name}_frame_{i:03d}.png"
                frame_path = temp_dir / frame_filename
                
                # Convert numpy array to PIL Image and save
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                
                image = Image.fromarray(frame)
                image.save(frame_path)
            
            logger.info(f"Saved {len(frames)} extracted frames to {temp_dir}")
        
        except Exception as e:
            logger.warning(f"Failed to save frames to temp directory: {e}")

