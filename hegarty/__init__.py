"""
Hegarty: A Perspective-Taking Agent for Enhanced Spatial Reasoning
"""

__version__ = "0.1.0"
__author__ = "Hegarty Research Team"

from .client import HergartyClient
from .agent import HergartyAgent
from .detector import PerspectiveDetector
from .config import Config

__all__ = [
    "HergartyClient",
    "HergartyAgent", 
    "PerspectiveDetector",
    "Config"
]