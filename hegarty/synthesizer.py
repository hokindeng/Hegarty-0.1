"""
PerspectiveSynthesizer: Synthesize multiple perspective analyses into a final answer
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import re

from openai import OpenAI

logger = logging.getLogger(__name__)


class PerspectiveSynthesizer:
    """
    Synthesizes multiple perspective analyses into a coherent final answer.
    
    This component takes the parallel analyses from different viewpoints
    and combines them into a comprehensive response.
    """
    
    def __init__(self, openai_client: OpenAI, config: Optional[Any] = None):
        """
        Initialize the synthesizer.
        
        Args:
            openai_client: OpenAI client for GPT-4o calls
            config: Configuration object
        """
        self.openai_client = openai_client
        self.config = config
        
        logger.info("PerspectiveSynthesizer initialized")
    
    def synthesize(
        self,
        perspectives: List[Dict[str, Any]],
        original_question: str,
        context: Optional[List[Dict]] = None
    ) -> Tuple[str, float]:
        """
        Synthesize multiple perspective analyses into a final answer.
        
        Args:
            perspectives: List of perspective analysis results
            original_question: The original user question
            context: Conversation context
        
        Returns:
            Tuple of (final_answer, confidence_score)
        """
        if not perspectives:
            logger.error("No perspectives to synthesize")
            return "Unable to analyze perspectives.", 0.0
        
        logger.info(f"Synthesizing {len(perspectives)} perspectives")
        
        # Extract and format perspective analyses
        formatted_perspectives = self._format_perspectives(perspectives)
        
        # Calculate initial confidence based on consistency
        consistency_score = self._calculate_consistency(perspectives)
        
        # Generate synthesis prompt
        synthesis_prompt = self._create_synthesis_prompt(
            original_question,
            formatted_perspectives,
            consistency_score
        )
        
        # Call GPT-4o for final synthesis
        try:
            messages = context or []
            messages.append({
                "role": "system",
                "content": "You are an expert at spatial reasoning and perspective analysis. Synthesize multiple viewpoint analyses into a comprehensive answer."
            })
            messages.append({
                "role": "user",
                "content": synthesis_prompt
            })
            
            response = self.openai_client.chat.completions.create(
                model=self.config.gpt_model if self.config else "gpt-4o",
                messages=messages,
                temperature=0.2,  # Lower temperature for synthesis
                max_tokens=self.config.max_tokens if self.config else 2000
            )
            
            final_answer = response.choices[0].message.content
            
            # Calculate final confidence
            confidence = self._calculate_final_confidence(
                consistency_score,
                perspectives,
                final_answer
            )
            
            logger.info(f"Synthesis complete. Confidence: {confidence:.2f}")
            
            return final_answer, confidence
            
        except Exception as e:
            logger.error(f"Error in synthesis: {str(e)}")
            # Fallback to best single perspective
            best_perspective = self._select_best_perspective(perspectives)
            return best_perspective.get('analysis', 'Unable to synthesize perspectives.'), 0.3
    
    def _format_perspectives(
        self,
        perspectives: List[Dict[str, Any]]
    ) -> str:
        """
        Format perspective analyses for synthesis prompt.
        """
        formatted = []
        
        for p in perspectives:
            perspective_text = f"**{p.get('perspective', 'Unknown')}:**\n"
            perspective_text += p.get('analysis', 'No analysis available')
            formatted.append(perspective_text)
        
        return "\n\n".join(formatted)
    
    def _create_synthesis_prompt(
        self,
        question: str,
        formatted_perspectives: str,
        consistency_score: float
    ) -> str:
        """
        Create the synthesis prompt for GPT-4o.
        """
        consistency_note = ""
        if consistency_score < 0.5:
            consistency_note = """
Note: The perspectives show some inconsistencies. Please identify the most reliable information and note any uncertainties.
"""
        elif consistency_score > 0.8:
            consistency_note = """
Note: The perspectives show high consistency, indicating reliable spatial understanding.
"""
        
        prompt = f"""You have analyzed an object/scene from multiple perspectives to answer a spatial reasoning question.

Original Question: {question}

Perspective Analyses:
{formatted_perspectives}

{consistency_note}

Task: Synthesize these multiple perspective analyses into a single, comprehensive answer that:
1. Directly answers the original question
2. Integrates insights from all perspectives
3. Resolves any contradictions by identifying the most reliable information
4. Provides spatial clarity and confidence in the answer

Final Synthesized Answer:"""
        
        return prompt
    
    def _calculate_consistency(
        self,
        perspectives: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate consistency score across perspectives.
        
        Higher score indicates more agreement between perspectives.
        """
        if len(perspectives) < 2:
            return 1.0
        
        # Extract key spatial terms from each analysis
        spatial_terms_by_perspective = []
        for p in perspectives:
            analysis = p.get('analysis', '').lower()
            terms = self._extract_spatial_terms(analysis)
            spatial_terms_by_perspective.append(terms)
        
        # Calculate overlap between perspectives
        total_comparisons = 0
        total_overlap = 0
        
        for i in range(len(spatial_terms_by_perspective)):
            for j in range(i + 1, len(spatial_terms_by_perspective)):
                terms1 = spatial_terms_by_perspective[i]
                terms2 = spatial_terms_by_perspective[j]
                
                if terms1 and terms2:
                    overlap = len(terms1.intersection(terms2))
                    union = len(terms1.union(terms2))
                    
                    if union > 0:
                        similarity = overlap / union
                        total_overlap += similarity
                        total_comparisons += 1
        
        if total_comparisons == 0:
            return 0.5  # Default middle confidence
        
        consistency = total_overlap / total_comparisons
        
        return consistency
    
    def _extract_spatial_terms(self, text: str) -> set:
        """
        Extract spatial and directional terms from text.
        """
        spatial_keywords = {
            'left', 'right', 'top', 'bottom', 'front', 'back',
            'above', 'below', 'behind', 'beside', 'between',
            'north', 'south', 'east', 'west',
            'clockwise', 'counterclockwise',
            'vertical', 'horizontal', 'diagonal',
            'center', 'edge', 'corner',
            'visible', 'hidden', 'obscured'
        }
        
        # Extract words from text
        words = set(re.findall(r'\b\w+\b', text.lower()))
        
        # Find spatial terms
        found_terms = words.intersection(spatial_keywords)
        
        # Also extract numbers (angles, distances)
        numbers = re.findall(r'\b\d+\b', text)
        found_terms.update(numbers[:5])  # Limit to first 5 numbers
        
        return found_terms
    
    def _calculate_final_confidence(
        self,
        consistency_score: float,
        perspectives: List[Dict[str, Any]],
        final_answer: str
    ) -> float:
        """
        Calculate final confidence score for the synthesis.
        """
        # Start with consistency score
        confidence = consistency_score * 0.5
        
        # Add confidence based on number of perspectives
        perspective_bonus = min(len(perspectives) / 10, 0.3)
        confidence += perspective_bonus
        
        # Check if final answer contains uncertainty markers
        uncertainty_markers = [
            'uncertain', 'unclear', 'might', 'possibly',
            'probably', 'seems', 'appears', 'likely'
        ]
        
        answer_lower = final_answer.lower()
        uncertainty_count = sum(1 for marker in uncertainty_markers if marker in answer_lower)
        
        # Reduce confidence for uncertainty
        uncertainty_penalty = min(uncertainty_count * 0.05, 0.2)
        confidence -= uncertainty_penalty
        
        # Ensure confidence is in [0, 1]
        confidence = max(0.1, min(1.0, confidence))
        
        return confidence
    
    def _select_best_perspective(
        self,
        perspectives: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Select the best single perspective as fallback.
        
        Prefers the original perspective or the most detailed analysis.
        """
        # Try to find original perspective
        for p in perspectives:
            if 'original' in p.get('perspective', '').lower():
                return p
        
        # Otherwise select longest analysis (most detailed)
        best = max(perspectives, key=lambda p: len(p.get('analysis', '')))
        
        return best
    
    def batch_synthesize(
        self,
        batch_perspectives: List[List[Dict[str, Any]]],
        questions: List[str],
        contexts: Optional[List[List[Dict]]] = None
    ) -> List[Tuple[str, float]]:
        """
        Synthesize multiple sets of perspectives in batch.
        
        Args:
            batch_perspectives: List of perspective sets
            questions: List of original questions
            contexts: Optional list of contexts
        
        Returns:
            List of (answer, confidence) tuples
        """
        results = []
        
        for i, perspectives in enumerate(batch_perspectives):
            question = questions[i] if i < len(questions) else ""
            context = contexts[i] if contexts and i < len(contexts) else None
            
            answer, confidence = self.synthesize(perspectives, question, context)
            results.append((answer, confidence))
        
        return results
