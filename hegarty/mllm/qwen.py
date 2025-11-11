"""Qwen3-VL MLLM provider using Hugging Face Transformers"""

import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from datetime import datetime
import base64
from io import BytesIO

from PIL import Image
import requests
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

from .base import MLLMProvider

logger = logging.getLogger(__name__)


class QwenMLLM(MLLMProvider):
    """Qwen3-VL-235B-A22B-Instruct multimodal provider"""
    
    DETECTION_PROMPT = """You are an expert at detecting perspective-taking and mental rotation tasks.

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
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device_map: str = "auto",
        dtype: str = "auto",
        attn_implementation: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        session_dir: Optional[Path] = None
    ):
        """
        Initialize Qwen3-VL MLLM provider.
        
        Args:
            model_name: Hugging Face model identifier
            device_map: Device mapping strategy ("auto", "cuda", "cpu", etc.)
            dtype: Data type ("auto", torch.bfloat16, etc.)
            attn_implementation: Attention implementation ("flash_attention_2" for better performance)
            temperature: Default sampling temperature
            max_tokens: Default max tokens to generate
            session_dir: Session directory for logging
        """
        self.model_name = model_name
        self.device_map = device_map
        self.dtype_str = dtype
        self.attn_implementation = attn_implementation
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.session_dir = session_dir
        self.call_counter = 0
        
        # Lazy loading
        self.model = None
        self.processor = None
        self._initialized = False
        
        logger.info(f"QwenMLLM initialized: {model_name}")
    
    def _init_model(self):
        """Lazy initialization of the Qwen3-VL model"""
        if self._initialized:
            return
        
        logger.info(f"Loading Qwen VL model: {self.model_name}")
        
        # Prepare model loading kwargs
        kwargs = {
            "device_map": self.device_map
        }
        
        # Handle dtype
        if self.dtype_str == "auto":
            kwargs["torch_dtype"] = "auto"
        else:
            kwargs["torch_dtype"] = torch.bfloat16 if self.dtype_str == "bfloat16" else torch.float16
        
        # Handle attention implementation
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation
        
        # Load model
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            **kwargs
        )
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        
        self._initialized = True
        logger.info("Qwen VL model loaded successfully")
    
    def detect_perspective(self, text: str) -> Tuple[bool, float]:
        """Detect if text describes perspective-taking task using Qwen3-VL"""
        if not text:
            return False, 0.0
        
        self._init_model()
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.DETECTION_PROMPT}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": f"Analyze this text:\n\n{text}"}]
            }
        ]
        
        response = self._generate(messages, "detect_perspective", temperature=0.1, max_tokens=200)
        content = response.strip()
        
        # Strip markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        
        try:
            result_data = json.loads(content)
            is_perspective = bool(result_data.get("is_perspective_task", False))
            confidence = max(0.0, min(1.0, float(result_data.get("confidence", 0.5))))
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON response: {content}")
            # Fallback: check for keywords
            is_perspective = any(keyword in text.lower() for keyword in [
                "rotate", "perspective", "view", "angle", "flip", "transform"
            ])
            confidence = 0.5
        
        logger.debug(f"Perspective: {is_perspective} (conf: {confidence:.2f})")
        return is_perspective, confidence
    
    def rephrase_for_video(self, question: str, image: str, camera_params: Dict[str, Any] = None) -> str:
        """Rephrase question for video generation using Qwen3-VL"""
        self._init_model()
        
        # Use detected camera parameters or defaults
        if camera_params and camera_params.get('confidence', 0) > 0.5:
            elevation = camera_params.get('pitch', 20.0)
            # Ensure elevation is reasonable for viewing
            if abs(elevation) < 10:
                elevation = 20.0
            logger.info(f"Using detected camera elevation: {elevation}°")
        else:
            elevation = 20.0  # Default elevation
            logger.info("Using default camera elevation: 20°")
        
        prompt = f"""Given this question about perspective-taking, create a camera control prompt for video generation.

Question: {question}

Current camera parameters detected:
- Elevation (pitch): {elevation:.1f}°
- Camera should maintain this viewing angle

Generate a technical camera prompt with these parameters:
1. Identify the target object/person for perspective shift
2. Use the detected elevation of {elevation:.0f}°
3. Start at azimuth 0°, end at azimuth 180° for opposite perspective
4. Maintain smooth horizontal rotation

Output EXACTLY in this format, replacing only the [object] placeholder:
"First frame: Your camera is tilted at {elevation:.0f}° elevation, viewing from 0° azimuth.
Final frame: Your camera remains at {elevation:.0f}° elevation, but rotates horizontally to 180° azimuth.
Create a smooth video showing the camera's horizontal rotation around the [object], and try to maintain the tilted viewing angle throughout."

Video prompt:"""
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an expert at technical camera control prompts for video generation. Always output precise camera angles in degrees."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image}
                ]
            }
        ]
        
        response = self._generate(messages, "rephrase_for_video", temperature=0.3, max_tokens=100)
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
        """Analyze perspective using Qwen3-VL vision-language model"""
        self._init_model()
        
        prompt = f"""Analyze this image viewing {perspective_label}.

Question: {question}

Focus on:
1. What is visible from this view
2. Spatial relationships you can determine
3. How this perspective helps answer the question

Analysis:"""
        
        messages = context or []
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image}
            ]
        })
        
        response = self._generate(messages, f"analyze_{perspective_label}", temperature, max_tokens)
        
        # Estimate tokens (rough approximation)
        tokens_used = len(response.split()) * 1.3
        
        return {
            'perspective': perspective_label,
            'analysis': response,
            'tokens_used': int(tokens_used)
        }
    
    def synthesize_perspectives(
        self,
        perspectives: List[Dict],
        original_question: str,
        consistency_score: float,
        context: List[Dict] = None
    ) -> Tuple[str, float]:
        """Synthesize multiple perspectives using Qwen3-VL"""
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
        
        messages = context or []
        messages.extend([
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an expert at spatial reasoning and perspective analysis."}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ])
        
        final_answer = self._generate(messages, "synthesize", temperature=0.2, max_tokens=self.max_tokens)
        
        # Calculate confidence
        confidence = consistency_score * 0.7
        answer_lower = final_answer.lower()
        if any(w in answer_lower for w in ["yes", "no", "definitely", "clearly"]):
            confidence += 0.2
        elif any(w in answer_lower for w in ["uncertain", "unclear", "possibly", "maybe"]):
            confidence -= 0.2
        
        confidence = max(0.0, min(1.0, confidence))
        return final_answer, confidence
    
    def _generate(
        self,
        messages: List[Dict],
        call_name: str,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """Generate response using Qwen3-VL model"""
        temp = temperature if temperature is not None else self.temperature
        max_new_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        # Prepare inputs using processor
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # Move to device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temp,
                do_sample=temp > 0,
                top_k=50 if temp > 0 else 1
            )
        
        # Decode output (only the new tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        response = output_text[0] if output_text else ""
        
        # Save call logs
        self._save_call(call_name, messages, response)
        
        return response
    
    def _save_call(self, call_name: str, messages: List[Dict], response: str):
        """Save call logs to session directory"""
        if not self.session_dir:
            return
        
        self.call_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mllm_call_{self.call_counter:03d}_{call_name}_{timestamp}.json"
        filepath = self.session_dir / filename
        
        data = {
            "call_number": self.call_counter,
            "call_name": call_name,
            "timestamp": timestamp,
            "model": self.model_name,
            "request": {"messages": self._sanitize_messages(messages)},
            "response": {
                "content": response,
                "model": self.model_name
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _sanitize_messages(self, messages: List[Dict]) -> List[Dict]:
        """Sanitize messages for logging (truncate base64 images)"""
        sanitized = []
        for msg in messages:
            sanitized_msg = {"role": msg["role"]}
            
            if isinstance(msg["content"], str):
                sanitized_msg["content"] = msg["content"]
            elif isinstance(msg["content"], list):
                sanitized_content = []
                for item in msg["content"]:
                    if item["type"] == "text":
                        sanitized_content.append(item)
                    elif item["type"] == "image":
                        image_data = item["image"]
                        if isinstance(image_data, str):
                            if image_data.startswith("data:") and len(image_data) > 100:
                                truncated = image_data[:100] + f"... [base64 truncated, length: {len(image_data)}]"
                                sanitized_content.append({"type": "image", "image": truncated})
                            else:
                                sanitized_content.append(item)
                        else:
                            sanitized_content.append({"type": "image", "image": "[PIL Image object]"})
                sanitized_msg["content"] = sanitized_content
            
            sanitized.append(sanitized_msg)
        return sanitized

