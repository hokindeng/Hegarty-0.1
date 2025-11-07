"""
PerspectiveDetector: Detects if a question requires perspective-taking or mental rotation
"""

import re
import logging
from typing import Tuple, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Result of perspective detection analysis"""
    is_perspective_task: bool
    confidence: float
    detected_keywords: List[str]
    reasoning: str


class PerspectiveDetector:
    """
    Detects whether a question involves perspective-taking or mental rotation.
    
    Uses keyword matching and pattern recognition to identify spatial reasoning tasks.
    """
    
    # Keywords that strongly indicate perspective-taking
    STRONG_KEYWORDS = [
        'rotate', 'rotation', 'turn', 'flip', 'spin',
        'perspective', 'viewpoint', 'angle', 'view',
        'mental rotation', 'mentally rotate',
        'from above', 'from below', 'from behind', 'from the side',
        'other side', 'opposite side', 'reverse side',
        'clockwise', 'counterclockwise', 'degrees'
    ]
    
    # Keywords that moderately indicate perspective-taking
    MODERATE_KEYWORDS = [
        'look like', 'appear', 'see',
        'behind', 'in front', 'left of', 'right of',
        'orientation', 'position', 'spatial',
        'transform', 'transformation',
        'imagine', 'visualize'
    ]
    
    # Patterns that indicate perspective questions
    PERSPECTIVE_PATTERNS = [
        r'what (would|will|does).*look like.*from',
        r'if.*rotate.*what',
        r'from.*perspective',
        r'how.*appear.*to',
        r'what.*see.*from',
        r'(rotate|turn|flip).*\d+.*degrees?',
        r'mental(ly)?\s+rotat'
    ]
    
    def __init__(self, config: Optional[Any] = None):
        """
        Initialize the perspective detector.
        
        Args:
            config: Optional configuration object
        """
        self.config = config
        self.confidence_threshold = 0.7 if not config else config.perspective_confidence_threshold
        
        # Compile regex patterns
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in self.PERSPECTIVE_PATTERNS
        ]
        
        logger.info("PerspectiveDetector initialized")
    
    def analyze(self, text: str) -> Tuple[bool, float]:
        """
        Analyze text to determine if it's a perspective-taking task.
        
        Args:
            text: The question or prompt to analyze
        
        Returns:
            Tuple of (is_perspective_task, confidence_score)
        """
        result = self.detailed_analysis(text)
        return result.is_perspective_task, result.confidence
    
    def detailed_analysis(self, text: str) -> DetectionResult:
        """
        Perform detailed analysis of the text.
        
        Args:
            text: The question or prompt to analyze
        
        Returns:
            DetectionResult with detailed information
        """
        if not text:
            return DetectionResult(False, 0.0, [], "No text provided")
        
        text_lower = text.lower()
        detected_keywords = []
        score = 0.0
        
        # Check for strong keywords
        strong_matches = []
        for keyword in self.STRONG_KEYWORDS:
            if keyword in text_lower:
                strong_matches.append(keyword)
                detected_keywords.append(keyword)
                score += 0.3  # Each strong keyword adds 0.3 to confidence
        
        # Check for moderate keywords
        moderate_matches = []
        for keyword in self.MODERATE_KEYWORDS:
            if keyword in text_lower:
                moderate_matches.append(keyword)
                detected_keywords.append(keyword)
                score += 0.15  # Each moderate keyword adds 0.15 to confidence
        
        # Check for patterns
        pattern_matches = []
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                pattern_matches.append(pattern.pattern)
                score += 0.4  # Pattern matches add significant confidence
        
        # Cap confidence at 1.0
        confidence = min(score, 1.0)
        
        # Determine if it's a perspective task
        is_perspective_task = confidence >= self.confidence_threshold
        
        # Build reasoning
        reasoning_parts = []
        if strong_matches:
            reasoning_parts.append(f"Strong keywords: {', '.join(strong_matches)}")
        if moderate_matches:
            reasoning_parts.append(f"Moderate keywords: {', '.join(moderate_matches)}")
        if pattern_matches:
            reasoning_parts.append(f"Pattern matches: {len(pattern_matches)} patterns")
        
        if not reasoning_parts:
            reasoning = "No perspective-taking indicators found"
        else:
            reasoning = "; ".join(reasoning_parts)
        
        logger.debug(f"Detection result: {is_perspective_task} (confidence: {confidence:.2f})")
        logger.debug(f"Reasoning: {reasoning}")
        
        return DetectionResult(
            is_perspective_task=is_perspective_task,
            confidence=confidence,
            detected_keywords=detected_keywords,
            reasoning=reasoning
        )
    
    def batch_analyze(self, texts: List[str]) -> List[DetectionResult]:
        """
        Analyze multiple texts in batch.
        
        Args:
            texts: List of questions or prompts
        
        Returns:
            List of DetectionResult objects
        """
        results = []
        for text in texts:
            results.append(self.detailed_analysis(text))
        
        return results
