"""
Hegarty: A Perspective-Taking Agent for Enhanced Spatial Reasoning
"""

__version__ = "0.1.0"
__author__ = "Hegarty Research Team"

from .client import HergartyClient
from .agent import HergartyAgent
from .gpt_detector import GPT4OPerspectiveDetector
from .config import Config

__all__ = [
    "HergartyClient",
    "HergartyAgent", 
    "GPT4OPerspectiveDetector",
    "Config"
]