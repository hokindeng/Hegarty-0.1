"""Base interface for MLLM providers"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple


class MLLMProvider(ABC):
    """Base interface for multimodal LLM providers"""
    
    @abstractmethod
    def detect_perspective(self, text: str) -> Tuple[bool, float]:
        """Detect if text describes perspective-taking task"""
        pass
    
    @abstractmethod
    def rephrase_for_video(self, question: str, image: str) -> str:
        """Rephrase question for video generation"""
        pass
    
    @abstractmethod
    def analyze_perspective(
        self, 
        image: str, 
        question: str, 
        perspective_label: str,
        context: List[Dict] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """Analyze single perspective"""
        pass
    
    @abstractmethod
    def synthesize_perspectives(
        self,
        perspectives: List[Dict],
        original_question: str,
        consistency_score: float,
        context: List[Dict] = None
    ) -> Tuple[str, float]:
        """Synthesize multiple perspectives into final answer"""
        pass

