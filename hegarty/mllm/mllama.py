"""Llama 3.2 Multimodal MLLM provider using NeuronX Distributed Inference"""

import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

from .base import MLLMProvider

logger = logging.getLogger(__name__)


class MLlamaMLLM(MLLMProvider):
    """Llama 3.2 Multimodal (11B/90B) provider using AWS Neuron accelerators"""
    
    DETECTION_PROMPT = """Analyze if this text describes a perspective-taking or mental rotation task.

Perspective-taking tasks include:
- Mental rotation (rotating objects in mind)
- Viewpoint changes (looking from different angles/positions)
- Spatial transformations (flipping, reflecting, transforming)
- Perspective shifts (viewing from another person/object's perspective)
- 3D visualization from 2D images

Respond with JSON:
{
    "is_perspective_task": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation",
    "detected_aspects": ["list", "of", "aspects"]
}"""
    
    def __init__(
        self,
        model_path: str,
        compiled_model_path: str,
        model_size: str = "11B",  # "11B" or "90B"
        tp_degree: int = 32,
        batch_size: int = 1,
        max_context_length: int = 2048,
        seq_len: int = 2176,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        session_dir: Optional[Path] = None
    ):
        """
        Initialize Llama 3.2 Multimodal MLLM provider.
        
        Args:
            model_path: Path to neuron checkpoint
            compiled_model_path: Path to compiled model
            model_size: "11B" or "90B"
            tp_degree: Tensor parallelism degree (32 for Trn1.32xlarge)
            batch_size: Batch size for inference
            max_context_length: Maximum context length
            seq_len: Sequence length
            temperature: Default temperature
            max_tokens: Default max tokens
            session_dir: Session directory for logging
        """
        self.model_path = Path(model_path)
        self.compiled_model_path = Path(compiled_model_path)
        self.model_size = model_size
        self.tp_degree = tp_degree
        self.batch_size = batch_size
        self.max_context_length = max_context_length
        self.seq_len = seq_len
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.session_dir = session_dir
        self.call_counter = 0
        
        # Initialize model (lazy loading)
        self.model = None
        self._initialized = False
        
        logger.info(f"MLlamaMLLM initialized: {model_size}, tp_degree={tp_degree}")
    
    def _init_model(self):
        """Lazy initialization of the mllama model"""
        if self._initialized:
            return
        
        # Import neuronx_distributed_inference
        from neuronx_distributed_inference.models.mllama.modeling_mllama import (
            MLlamaForConditionalGeneration
        )
        
        logger.info(f"Loading mllama model from {self.model_path}")
        
        # Load model configuration
        config = {
            'model_path': str(self.model_path),
            'compiled_model_path': str(self.compiled_model_path),
            'tp_degree': self.tp_degree,
            'batch_size': self.batch_size,
            'max_context_length': self.max_context_length,
            'seq_len': self.seq_len,
        }
        
        # Initialize model
        self.model = MLlamaForConditionalGeneration.from_pretrained(
            str(self.model_path),
            **config
        )
        
        self._initialized = True
        logger.info("mllama model loaded successfully")
    
    def detect_perspective(self, text: str) -> Tuple[bool, float]:
        """Detect if text describes perspective-taking task using mllama"""
        if not text:
            return False, 0.0
        
        self._init_model()
        
        prompt = f"{self.DETECTION_PROMPT}\n\nAnalyze this text:\n\n{text}"
        
        # Generate response using mllama
        response = self._generate_text(prompt, max_tokens=200, temperature=0.1)
        
        # Parse JSON response
        result_data = json.loads(response)
        is_perspective = bool(result_data.get("is_perspective_task", False))
        confidence = max(0.0, min(1.0, float(result_data.get("confidence", 0.5))))
        
        logger.debug(f"Perspective: {is_perspective} (conf: {confidence:.2f})")
        return is_perspective, confidence
    
    def rephrase_for_video(self, question: str, image: str) -> str:
        """Rephrase question for video generation using mllama"""
        self._init_model()
        
        prompt = f"""Given this question about perspective-taking, create a simple video prompt.

Question: {question}

Extract who or what perspective is being asked about, then format as:
"Please rotate the scene to the [person/object/viewpoint] perspective"

Video prompt:"""
        
        response = self._generate_with_image(prompt, image, max_tokens=100, temperature=0.3)
        result = response.strip()
        
        # Check for refusal
        refusal_phrases = ["I'm sorry", "I cannot", "I can't", "unable to"]
        if any(phrase.lower() in result.lower() for phrase in refusal_phrases):
            raise ValueError(f"Model refused to generate video prompt: {result}")
        
        return result
    
    def analyze_perspective(
        self,
        image: str,
        question: str,
        perspective_label: str,
        context: List[Dict] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """Analyze perspective using mllama vision-language model"""
        self._init_model()
        
        prompt = f"""Analyze this image viewing {perspective_label}.

Question: {question}

Focus on:
1. What is visible from this view
2. Spatial relationships you can determine
3. How this perspective helps answer the question

Analysis:"""
        
        analysis = self._generate_with_image(
            prompt, 
            image, 
            max_tokens=max_tokens, 
            temperature=temperature
        )
        
        return {
            'perspective': perspective_label,
            'analysis': analysis,
            'tokens_used': len(analysis.split())  # Approximate
        }
    
    def synthesize_perspectives(
        self,
        perspectives: List[Dict],
        original_question: str,
        consistency_score: float,
        context: List[Dict] = None
    ) -> Tuple[str, float]:
        """Synthesize multiple perspectives using mllama"""
        self._init_model()
        
        formatted = "\n\n".join([
            f"**{p.get('perspective', f'Perspective {i}')}**:\n{p.get('analysis', '')}"
            for i, p in enumerate(perspectives, 1)
        ])
        
        consistency_note = ""
        if consistency_score < 0.5:
            consistency_note = "\nNote: Perspectives show inconsistencies. Identify most reliable information."
        elif consistency_score > 0.8:
            consistency_note = "\nNote: Perspectives show high consistency."
        
        prompt = f"""Synthesize multiple perspective analyses into one answer.

Original Question: {original_question}

Perspective Analyses:
{formatted}
{consistency_note}

Task: Create comprehensive answer that:
1. Directly answers the original question
2. Integrates insights from all perspectives
3. Resolves contradictions
4. Provides spatial clarity

Final Answer:"""
        
        final_answer = self._generate_text(prompt, max_tokens=self.max_tokens, temperature=0.2)
        
        # Calculate confidence
        confidence = consistency_score * 0.7
        answer_lower = final_answer.lower()
        if any(w in answer_lower for w in ["yes", "no", "definitely", "clearly"]):
            confidence += 0.2
        elif any(w in answer_lower for w in ["uncertain", "unclear", "possibly", "maybe"]):
            confidence -= 0.2
        
        confidence = max(0.0, min(1.0, confidence))
        return final_answer, confidence
    
    def _generate_text(self, prompt: str, max_tokens: int = None, temperature: float = None) -> str:
        """Generate text-only response using mllama"""
        # Prepare inputs
        inputs = self._prepare_text_inputs(prompt)
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens or self.max_tokens,
            temperature=temperature or self.temperature,
            do_sample=temperature > 0,
            top_k=1 if temperature == 0 else 50
        )
        
        # Decode output
        response = self._decode_outputs(outputs)
        self.call_counter += 1
        
        return response
    
    def _generate_with_image(
        self, 
        prompt: str, 
        image: str, 
        max_tokens: int = None, 
        temperature: float = None
    ) -> str:
        """Generate response with image input using mllama"""
        # Prepare multimodal inputs
        inputs = self._prepare_multimodal_inputs(prompt, image)
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens or self.max_tokens,
            temperature=temperature or self.temperature,
            do_sample=temperature > 0,
            top_k=1 if temperature == 0 else 50
        )
        
        # Decode output
        response = self._decode_outputs(outputs)
        self.call_counter += 1
        
        return response
    
    def _prepare_text_inputs(self, prompt: str) -> Dict[str, Any]:
        """Prepare text-only inputs for mllama"""
        raise NotImplementedError(
            "MLlamaMLLM is a stub implementation. "
            "Full implementation requires neuronx-distributed-inference setup. "
            "See docs/MLLAMA_SETUP.md for instructions."
        )
    
    def _prepare_multimodal_inputs(self, prompt: str, image: str) -> Dict[str, Any]:
        """Prepare multimodal inputs (text + image) for mllama"""
        raise NotImplementedError(
            "MLlamaMLLM is a stub implementation. "
            "Full implementation requires neuronx-distributed-inference setup. "
            "See docs/MLLAMA_SETUP.md for instructions."
        )
    
    def _decode_outputs(self, outputs) -> str:
        """Decode model outputs to text"""
        raise NotImplementedError(
            "MLlamaMLLM is a stub implementation. "
            "Full implementation requires neuronx-distributed-inference setup. "
            "See docs/MLLAMA_SETUP.md for instructions."
        )

