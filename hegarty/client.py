"""
HergartyClient: OpenAI-compatible client interface for the Hegarty agent
"""

import os
import logging
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
import time

from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from .agent import HergartyAgent
from .detector import PerspectiveDetector
from .config import Config

logger = logging.getLogger(__name__)


class CompletionResponse:
    """Wrapper for chat completion responses"""
    def __init__(self, content: str, model: str = "hegarty-1.0", usage: Optional[Dict] = None):
        self.id = f"chatcmpl-{int(time.time())}"
        self.object = "chat.completion"
        self.created = int(time.time())
        self.model = model
        self.choices = [
            type('Choice', (), {
                'index': 0,
                'message': type('Message', (), {
                    'role': 'assistant',
                    'content': content
                })(),
                'finish_reason': 'stop'
            })()
        ]
        self.usage = usage or {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }


class HergartyClient:
    """
    OpenAI-compatible client for the Hegarty perspective-taking agent.
    
    This client provides the same interface as the OpenAI Python client,
    but routes perspective-taking tasks through the Hegarty pipeline.
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        sora_api_key: Optional[str] = None,
        config: Optional[Config] = None,
        **kwargs
    ):
        """
        Initialize the Hegarty client.
        
        Args:
            openai_api_key: OpenAI API key for GPT-4o access
            sora_api_key: Sora API key (optional, uses simulation if not provided)
            config: Configuration object
            **kwargs: Additional configuration parameters
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.sora_api_key = sora_api_key or os.getenv("SORA_API_KEY")
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY or pass openai_api_key")
        
        # Initialize configuration
        self.config = config or Config(**kwargs)
        
        # Initialize OpenAI client for GPT-4o calls
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        
        # Initialize Hegarty agent
        self.agent = HergartyAgent(
            openai_client=self.openai_client,
            sora_api_key=self.sora_api_key,
            config=self.config
        )
        
        # Initialize perspective detector
        self.detector = PerspectiveDetector(self.config)
        
        # Create chat.completions interface
        self.chat = type('Chat', (), {'completions': self})()
        
        logger.info("HergartyClient initialized successfully")
    
    def create(
        self,
        model: str = "hegarty-1.0",
        messages: List[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> CompletionResponse:
        """
        Create a chat completion, using Hegarty pipeline for perspective tasks.
        
        This method mimics the OpenAI chat.completions.create interface.
        """
        if not messages:
            raise ValueError("messages parameter is required")
        
        # Extract the last message (current query)
        last_message = messages[-1]
        
        if not isinstance(last_message.get('content'), (str, list)):
            raise ValueError("Message content must be string or list")
        
        # Handle different content formats
        text_content = None
        image_content = None
        
        if isinstance(last_message['content'], str):
            text_content = last_message['content']
        else:
            # Parse multimodal content
            for item in last_message['content']:
                if item.get('type') == 'text':
                    text_content = item.get('text')
                elif item.get('type') == 'image_url':
                    image_content = item.get('image_url', {}).get('url')
        
        # Check if this is a perspective-taking task
        is_perspective_task = False
        if text_content:
            is_perspective_task, confidence = self.detector.analyze(text_content)
            logger.info(f"Perspective detection: {is_perspective_task} (confidence: {confidence})")
        
        # Route to appropriate handler
        if model == "hegarty-1.0" and is_perspective_task and image_content:
            # Use Hegarty pipeline for perspective-taking
            logger.info("Routing to Hegarty perspective-taking pipeline")
            response_content = self._handle_perspective_task(
                text=text_content,
                image=image_content,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
        else:
            # Use standard GPT-4o
            logger.info("Routing to standard GPT-4o")
            response_content = self._handle_standard_completion(
                messages=messages,
                temperature=temperature or self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens,
                model=self.config.gpt_model
            )
        
        return CompletionResponse(content=response_content, model=model)
    
    def _handle_perspective_task(
        self,
        text: str,
        image: str,
        messages: List[Dict],
        temperature: Optional[float],
        max_tokens: Optional[int]
    ) -> str:
        """
        Handle perspective-taking task using the Hegarty pipeline.
        """
        # Process through Hegarty agent
        result = self.agent.process(
            image=image,
            question=text,
            context_messages=messages[:-1],  # Previous conversation context
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return result['final_answer']
    
    def _handle_standard_completion(
        self,
        messages: List[Dict],
        temperature: float,
        max_tokens: int,
        model: str
    ) -> str:
        """
        Handle standard completion using GPT-4o directly.
        """
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    def batch_process(
        self,
        queries: List[Dict[str, Any]],
        max_workers: Optional[int] = None
    ) -> List[CompletionResponse]:
        """
        Process multiple queries in batch.
        
        Args:
            queries: List of query dictionaries with 'image' and 'question' keys
            max_workers: Maximum number of parallel workers
        
        Returns:
            List of completion responses
        """
        results = []
        for query in queries:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": query.get('question')},
                    {"type": "image_url", "image_url": {"url": query.get('image')}}
                ] if query.get('image') else query.get('question')
            }]
            
            response = self.create(messages=messages)
            results.append(response)
        
        return results
