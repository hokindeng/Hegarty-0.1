"""OpenAI-compatible client interface"""

import os
import logging
from typing import Optional, List, Dict, Any
import time

from dotenv import load_dotenv
load_dotenv()

from .agent import HergartyAgent
from .config import Config

logger = logging.getLogger(__name__)


class CompletionResponse:
    """Wrapper for chat completion responses"""
    
    def __init__(self, content: str, model: str = "hegarty-1.0", usage: Optional[Dict] = None):
        self.id = f"chatcmpl-{int(time.time())}"
        self.object = "chat.completion"
        self.created = int(time.time())
        self.model = model
        self.choices = [type('Choice', (), {
            'index': 0,
            'message': type('Message', (), {
                'role': 'assistant',
                'content': content
            })(),
            'finish_reason': 'stop'
        })()]
        self.usage = usage or {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}


class HergartyClient:
    """OpenAI-compatible client for Hegarty perspective-taking agent"""
    
    def __init__(
        self,
        sora_api_key: Optional[str] = None,
        config: Optional[Config] = None,
        **kwargs
    ):
        self.sora_api_key = sora_api_key or os.getenv("SORA_API_KEY")
        
        self.config = config or Config(**kwargs)
        self.openai_api_key = None
        self.openai_client = None
        
        self.agent = HergartyAgent(
            openai_client=None,
            sora_api_key=self.sora_api_key,
            config=self.config
        )
        
        self.chat = type('Chat', (), {'completions': self})()
        logger.info("HergartyClient initialized")
    
    def create(
        self,
        model: str = "hegarty-1.0",
        messages: List[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> CompletionResponse:
        if not messages:
            raise ValueError("messages parameter required")
        
        last_message = messages[-1]
        
        if not isinstance(last_message.get('content'), (str, list)):
            raise ValueError("Message content must be string or list")
        
        text_content = None
        image_content = None
        
        if isinstance(last_message['content'], str):
            text_content = last_message['content']
        else:
            for item in last_message['content']:
                if item.get('type') == 'text':
                    text_content = item.get('text')
                elif item.get('type') == 'image_url':
                    image_content = item.get('image_url', {}).get('url')
        
        is_perspective_task = False
        if text_content and self.agent.mllm:
            is_perspective_task, confidence = self.agent.mllm.detect_perspective(text_content)
            logger.info(f"Perspective: {is_perspective_task} (confidence: {confidence})")
        
        if model == "hegarty-1.0" and is_perspective_task and image_content:
            logger.info("Routing to Hegarty pipeline")
            response_content = self._handle_perspective_task(
                text=text_content,
                image=image_content,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
        else:
            logger.info("OpenAI provider removed; returning message")
            response_content = "OpenAI provider removed. Provide an image and perspective task to use the Hegarty pipeline."
        
        return CompletionResponse(content=response_content, model=model)
    
    def _handle_perspective_task(
        self,
        text: str,
        image: str,
        messages: List[Dict],
        temperature: Optional[float],
        max_tokens: Optional[int]
    ) -> str:
        result = self.agent.process(
            image=image,
            question=text,
            context_messages=messages[:-1],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return result['final_answer']
