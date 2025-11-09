"""Perspective synthesis"""

import logging
import re
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


class PerspectiveSynthesizer:
    """Synthesize multiple perspective analyses"""
    
    def __init__(self):
        logger.info("PerspectiveSynthesizer initialized")
    
    def synthesize(
        self,
        perspectives: List[Dict[str, Any]],
        original_question: str,
        context: Optional[List[Dict]] = None,
        mllm_provider = None
    ) -> Tuple[str, float]:
        if not perspectives:
            return "Unable to analyze perspectives.", 0.0
        
        logger.info(f"Synthesizing {len(perspectives)} perspectives")
        
        consistency = self._calculate_consistency(perspectives)
        
        if mllm_provider:
            answer, confidence = mllm_provider.synthesize_perspectives(
                perspectives=perspectives,
                original_question=original_question,
                consistency_score=consistency,
                context=context
            )
        else:
            answer = perspectives[0].get('analysis', 'Unable to synthesize.')
            confidence = consistency
        
        logger.info(f"Synthesis complete. Confidence: {confidence:.2f}")
        return answer, confidence
    
    def _calculate_consistency(self, perspectives: List[Dict[str, Any]]) -> float:
        if len(perspectives) < 2:
            return 1.0
        
        terms_by_perspective = [self._extract_spatial_terms(p.get('analysis', '').lower()) 
                               for p in perspectives]
        
        total_comparisons = 0
        total_overlap = 0
        
        for i in range(len(terms_by_perspective)):
            for j in range(i + 1, len(terms_by_perspective)):
                terms1, terms2 = terms_by_perspective[i], terms_by_perspective[j]
                if terms1 and terms2:
                    overlap = len(terms1.intersection(terms2))
                    union = len(terms1.union(terms2))
                    if union > 0:
                        total_overlap += overlap / union
                        total_comparisons += 1
        
        return total_overlap / total_comparisons if total_comparisons > 0 else 0.5
    
    def _extract_spatial_terms(self, text: str) -> set:
        keywords = {
            'left', 'right', 'top', 'bottom', 'front', 'back',
            'above', 'below', 'behind', 'beside', 'between',
            'north', 'south', 'east', 'west',
            'clockwise', 'counterclockwise',
            'vertical', 'horizontal', 'diagonal',
            'center', 'edge', 'corner',
            'visible', 'hidden', 'obscured'
        }
        
        words = set(re.findall(r'\b\w+\b', text.lower()))
        found = words.intersection(keywords)
        numbers = re.findall(r'\b\d+\b', text)
        found.update(numbers[:5])
        
        return found
