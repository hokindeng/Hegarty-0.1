"""OpenAI GPT-4o MLLM provider"""

import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from datetime import datetime

from openai import OpenAI
from .base import MLLMProvider, DetectionResult

logger = logging.getLogger(__name__)


class OpenAIMLLM(MLLMProvider):
    """OpenAI GPT-4o multimodal provider"""
    
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
        client: OpenAI,
        model: str = "gpt-4o",
        temperature: float = 0.3,
        max_tokens: int = 2000,
        session_dir: Optional[Path] = None
    ):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.session_dir = session_dir
        self.call_counter = 0
    
    def detect_perspective(self, text: str) -> Tuple[bool, float]:
        if not text:
            return False, 0.0
        
        messages = [
            {"role": "system", "content": self.DETECTION_PROMPT},
            {"role": "user", "content": f"Analyze this text:\n\n{text}"}
        ]
        
        response = self._call(messages, "detect_perspective", temperature=0.1, max_tokens=200, json_mode=True)
        result_data = json.loads(response.choices[0].message.content)
        
        is_perspective = bool(result_data.get("is_perspective_task", False))
        confidence = max(0.0, min(1.0, float(result_data.get("confidence", 0.5))))
        
        logger.debug(f"Perspective: {is_perspective} (conf: {confidence:.2f})")
        return is_perspective, confidence
    
    def rephrase_for_video(self, question: str, image: str) -> str:
        prompt = f"""Create a concise video prompt for Sora to visualize this spatial transformation.

Question: {question}

Describe:
1. Starting state (what's in the image)
2. Transformation needed
3. Ending state

Keep under 50 words, focus on visual transformation.

Video prompt:"""
        
        messages = [
            {"role": "system", "content": "You are an expert at video generation prompts."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image}}
            ]}
        ]
        
        response = self._call(messages, "rephrase_for_video", temperature=0.3, max_tokens=100)
        result = response.choices[0].message.content.strip()
        
        refusal_phrases = ["I'm sorry", "I cannot", "I can't", "unable to"]
        if any(phrase.lower() in result.lower() for phrase in refusal_phrases):
            return "Camera rotating 360 degrees around scene to show all perspectives"
        
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
                {"type": "image_url", "image_url": {"url": image}}
            ]
        })
        
        response = self._call(messages, f"analyze_{perspective_label}", temperature, max_tokens)
        
        return {
            'perspective': perspective_label,
            'analysis': response.choices[0].message.content,
            'tokens_used': response.usage.total_tokens if hasattr(response, 'usage') else 0
        }
    
    def synthesize_perspectives(
        self,
        perspectives: List[Dict],
        original_question: str,
        consistency_score: float,
        context: List[Dict] = None
    ) -> Tuple[str, float]:
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
            {"role": "system", "content": "You are an expert at spatial reasoning and perspective analysis."},
            {"role": "user", "content": prompt}
        ])
        
        response = self._call(messages, "synthesize", temperature=0.2, max_tokens=self.max_tokens)
        final_answer = response.choices[0].message.content
        
        confidence = consistency_score * 0.7
        answer_lower = final_answer.lower()
        if any(w in answer_lower for w in ["yes", "no", "definitely", "clearly"]):
            confidence += 0.2
        elif any(w in answer_lower for w in ["uncertain", "unclear", "possibly", "maybe"]):
            confidence -= 0.2
        
        confidence = max(0.0, min(1.0, confidence))
        return final_answer, confidence
    
    def _call(
        self,
        messages: List[Dict],
        call_name: str,
        temperature: float = None,
        max_tokens: int = None,
        json_mode: bool = False
    ):
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens
        }
        
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        
        response = self.client.chat.completions.create(**kwargs)
        self._save_call(call_name, messages, response)
        return response
    
    def _save_call(self, call_name: str, messages: List[Dict], response: Any):
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
            "model": self.model,
            "request": {"messages": self._sanitize_messages(messages)},
            "response": {
                "content": response.choices[0].message.content if hasattr(response, 'choices') else str(response),
                "model": response.model if hasattr(response, 'model') else self.model,
                "usage": response.usage.dict() if hasattr(response, 'usage') and response.usage else None
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _sanitize_messages(self, messages: List[Dict]) -> List[Dict]:
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
                    elif item["type"] == "image_url":
                        image_url = item["image_url"]["url"]
                        if "base64" in image_url and len(image_url) > 100:
                            truncated = image_url[:100] + f"... [base64 truncated, length: {len(image_url)}]"
                            sanitized_content.append({"type": "image_url", "image_url": {"url": truncated}})
                        else:
                            sanitized_content.append(item)
                sanitized_msg["content"] = sanitized_content
            
            sanitized.append(sanitized_msg)
        return sanitized

