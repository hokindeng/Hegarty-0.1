"""Hegarty: Perspective-Taking Agent for Enhanced Spatial Reasoning"""

__version__ = "0.1.0"
__author__ = "Hegarty Research Team"

from .client import HergartyClient
from .agent import HergartyAgent
from .config import Config
from .mllm import MLLMProvider, QwenMLLM
from .vm import VMProvider, SoraVM

__all__ = [
    "HergartyClient",
    "HergartyAgent",
    "Config",
    "MLLMProvider",
    "QwenMLLM",
    "VMProvider",
    "SoraVM"
]
