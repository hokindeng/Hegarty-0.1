"""Camera parameter detection using PerspectiveFields"""

import sys
import os
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2
from PIL import Image
import torch

# Add PerspectiveFields to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'external', 'PerspectiveFields'))

try:
    from perspective2d import PerspectiveFields
except ImportError:
    raise ImportError("PerspectiveFields not found. Please ensure the submodule is initialized: git submodule update --init --recursive")

logger = logging.getLogger(__name__)


class CameraDetector:
    """Detects camera parameters from images using PerspectiveFields"""
    
    def __init__(self, model_version: str = 'Paramnet-360Cities-edina-centered'):
        """
        Initialize camera detector with PerspectiveFields model.
        
        Args:
            model_version: PerspectiveFields model version
                - 'Paramnet-360Cities-edina-centered': For uncropped images
                - 'Paramnet-360Cities-edina-uncentered': For cropped images  
                - 'PersNet_paramnet-GSV-centered': For street view images
        """
        self.model_version = model_version
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize model
        try:
            self.model = PerspectiveFields(model_version).eval()
            if self.device == 'cuda':
                self.model = self.model.cuda()
            logger.info(f"CameraDetector initialized with {model_version} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize PerspectiveFields model: {e}")
            raise
    
    def detect_from_image(self, image: Any) -> Dict[str, Any]:
        """
        Detect camera parameters from input image.
        
        Args:
            image: Input image as numpy array (BGR), PIL Image, or path string
            
        Returns:
            Dict containing:
                - roll: Camera roll in degrees
                - pitch: Camera pitch in degrees  
                - vfov: Vertical field of view in degrees
                - up: Up vector field (H, W, 2)
                - latitude: Latitude field in radians (H, W)
                - confidence: Detection confidence score
        """
        # Convert input to BGR numpy array
        if isinstance(image, str):
            img_bgr = cv2.imread(image)
        elif isinstance(image, Image.Image):
            img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            if image.shape[-1] == 3 and image.ndim == 3:
                img_bgr = image
            else:
                raise ValueError(f"Invalid image shape: {image.shape}")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
            
        if img_bgr is None:
            raise ValueError("Failed to load image")
            
        # Run inference
        try:
            with torch.no_grad():
                predictions = self.model.inference(img_bgr=img_bgr)
        except Exception as e:
            logger.error(f"PerspectiveFields inference failed: {e}")
            raise
            
        # Extract results based on model version
        result = {
            'up': predictions['pred_gravity_original'],
            'latitude': predictions['pred_latitude_original'],
        }
        
        # Model-specific outputs
        if 'pred_roll' in predictions:
            result['roll'] = float(predictions['pred_roll'])
        else:
            result['roll'] = 0.0
            
        if 'pred_pitch' in predictions:
            result['pitch'] = float(predictions['pred_pitch'])
        else:
            result['pitch'] = 0.0
            
        if 'pred_vfov' in predictions:
            result['vfov'] = float(predictions['pred_vfov'])
        else:
            result['vfov'] = 60.0  # default
            
        # Principal point if available (for uncentered models)
        if 'pred_cx' in predictions and 'pred_cy' in predictions:
            result['cx'] = float(predictions['pred_cx'])
            result['cy'] = float(predictions['pred_cy'])
            
        # Calculate confidence from perspective field consistency
        result['confidence'] = self._calculate_confidence(result['up'], result['latitude'])
        
        logger.info(f"Detected camera params - Roll: {result['roll']:.1f}°, "
                   f"Pitch: {result['pitch']:.1f}°, vFOV: {result['vfov']:.1f}°, "
                   f"Confidence: {result['confidence']:.2f}")
        
        return result
    
    def _calculate_confidence(self, up_field: np.ndarray, latitude_field: np.ndarray) -> float:
        """
        Calculate detection confidence based on field consistency.
        
        Args:
            up_field: Up vector field (H, W, 2)
            latitude_field: Latitude field (H, W)
            
        Returns:
            Confidence score between 0 and 1
        """
        # Check up vector consistency (should be similar across image)
        up_std = np.std(up_field.reshape(-1, 2), axis=0).mean()
        up_confidence = max(0, 1 - up_std * 2)  # Lower std = higher confidence
        
        # Check latitude smoothness
        lat_grad = np.gradient(latitude_field)
        lat_smoothness = 1 / (1 + np.std(lat_grad))
        
        # Combined confidence
        confidence = (up_confidence + lat_smoothness) / 2
        return float(np.clip(confidence, 0, 1))
    
    def get_camera_trajectory(
        self, 
        current_params: Dict[str, float],
        target_azimuth: float = 180.0,
        maintain_elevation: bool = True
    ) -> Dict[str, Any]:
        """
        Generate camera trajectory parameters for video generation.
        
        Args:
            current_params: Current camera parameters from detection
            target_azimuth: Target azimuth angle in degrees  
            maintain_elevation: Whether to maintain elevation during rotation
            
        Returns:
            Dict with trajectory parameters
        """
        # Use detected pitch as elevation
        elevation = current_params.get('pitch', 0.0)
        
        # Adjust elevation to be slightly above if too low
        if abs(elevation) < 10:
            elevation = 20.0  # Default to 20° for better perspective
            
        trajectory = {
            'start_elevation': elevation,
            'start_azimuth': 0.0,
            'end_elevation': elevation if maintain_elevation else elevation,
            'end_azimuth': target_azimuth,
            'vfov': current_params.get('vfov', 60.0),
            'duration': 4.0  # seconds
        }
        
        return trajectory
