"""
GPT4OPerspectiveDetector: Uses GPT-4o for perspective-taking detection
"""

import logging
import json
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class GPTDetectionResult:
    """Result of GPT-4o perspective detection analysis"""
    is_perspective_task: bool
    confidence: float
    reasoning: str
    detected_aspects: List[str]  # e.g., ["rotation", "viewpoint change", "spatial transformation"]


class GPT4OPerspectiveDetector:
    """
    Advanced perspective detector using GPT-4o for nuanced detection.
    
    This detector uses GPT-4o to understand context and semantics beyond
    simple keyword matching, providing more accurate detection for complex cases.
    """
    
    SYSTEM_PROMPT = """You are an expert at detecting perspective-taking and mental rotation tasks in questions.

Perspective-taking tasks include:
- Mental rotation (rotating objects in mind)
- Viewpoint changes (looking from different angles/positions)
- Spatial transformations (flipping, reflecting, transforming)
- Perspective shifts (viewing from another person/object's perspective)
- 3D visualization from 2D images

Analyze the given text and respond with a JSON object:
{
    "is_perspective_task": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation",
    "detected_aspects": ["list", "of", "perspective", "aspects"]
}

Be accurate - not all spatial questions require perspective-taking. For example, "measure the distance" is spatial but not perspective-taking."""
    
    def __init__(
        self, 
        openai_client: Optional[OpenAI] = None,
        model: str = "gpt-4o",
        temperature: float = 0.1,
        use_mini: bool = False  # Option to use gpt-4o-mini for cost savings
    ):
        """
        Initialize GPT-4o perspective detector.
        
        Args:
            openai_client: OpenAI client instance
            model: Model to use for detection
            temperature: Temperature for GPT-4o (lower = more consistent)
            use_mini: Use gpt-4o-mini for faster/cheaper detection
        """
        self.client = openai_client or OpenAI()
        self.model = "gpt-4o-mini" if use_mini else model
        self.temperature = temperature
        
        logger.info(f"GPT4OPerspectiveDetector initialized with {self.model}")
    
    def analyze(self, text: str) -> Tuple[bool, float]:
        """
        Analyze text using GPT-4o to determine if it's a perspective-taking task.
        
        Args:
            text: The question or prompt to analyze
        
        Returns:
            Tuple of (is_perspective_task, confidence_score)
        """
        result = self.detailed_analysis(text)
        return result.is_perspective_task, result.confidence
    
    def detailed_analysis(self, text: str) -> GPTDetectionResult:
        """
        Perform detailed GPT-4o analysis of the text.
        
        Args:
            text: The question or prompt to analyze
        
        Returns:
            GPTDetectionResult with detailed information
        """
        if not text:
            return GPTDetectionResult(False, 0.0, "No text provided", [])
        
        # Call GPT-4o for analysis
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze this text:\n\n{text}"}
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},  # Ensure JSON response
            max_tokens=200  # Limited tokens needed for detection
        )
        
        # Parse response
        content = response.choices[0].message.content
        result_data = json.loads(content)
        
        # Extract fields with defaults
        is_perspective = bool(result_data.get("is_perspective_task", False))
        confidence = float(result_data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
        reasoning = result_data.get("reasoning", "No reasoning provided")
        aspects = result_data.get("detected_aspects", [])
        
        logger.debug(f"GPT-4o detection: {is_perspective} (conf: {confidence:.2f}) - {reasoning}")
        
        return GPTDetectionResult(
            is_perspective_task=is_perspective,
            confidence=confidence,
            reasoning=reasoning,
            detected_aspects=aspects
        )
    
    def _fallback_detection(self, text: str) -> GPTDetectionResult:
        """
        Simple fallback detection when GPT-4o fails.
        """
        text_lower = text.lower()
        
        # Basic keywords
        keywords = ["rotate", "turn", "flip", "perspective", "angle", "view", "mental rotation"]
        found = [k for k in keywords if k in text_lower]
        
        is_perspective = len(found) > 0
        confidence = min(len(found) * 0.3, 0.8)
        
        return GPTDetectionResult(
            is_perspective_task=is_perspective,
            confidence=confidence,
            reasoning=f"Fallback detection found keywords: {', '.join(found)}" if found else "No keywords found",
            detected_aspects=found
        )

