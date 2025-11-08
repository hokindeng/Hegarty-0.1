"""
HergartyAgent: Core orchestration agent for perspective-taking pipeline
"""

import logging
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
import io

from openai import OpenAI
import numpy as np
from PIL import Image

from .sora_interface import SoraInterface
from .frame_extractor import FrameExtractor
from .synthesizer import PerspectiveSynthesizer
from .config import Config

logger = logging.getLogger(__name__)


class HergartyAgent:
    """
    Core agent that orchestrates the perspective-taking pipeline.
    
    This agent coordinates:
    1. Question rephrasing for Sora-2
    2. Video generation via Sora-2
    3. Frame extraction
    4. Parallel GPT-4o analysis
    5. Synthesis of multiple perspectives
    """
    
    def __init__(
        self,
        openai_client: OpenAI,
        sora_api_key: Optional[str] = None,
        config: Optional[Config] = None
    ):
        """
        Initialize the Hegarty agent.
        
        Args:
            openai_client: OpenAI client instance
            sora_api_key: Sora API key (optional)
            config: Configuration object
        """
        self.openai_client = openai_client
        self.config = config or Config()
        
        # Initialize components
        self.sora = SoraInterface(api_key=sora_api_key, config=config)
        self.frame_extractor = FrameExtractor(config=config)
        self.synthesizer = PerspectiveSynthesizer(openai_client, config)
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        
        logger.info("HergartyAgent initialized")
    
    def process(
        self,
        image: str,
        question: str,
        context_messages: Optional[List[Dict]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_mental_rotation: bool = True,
        num_perspectives: int = 6,
        return_intermediate: bool = False
    ) -> Dict[str, Any]:
        """
        Process a perspective-taking query through the full pipeline.
        
        Args:
            image: Base64 encoded image or URL
            question: The perspective-taking question
            context_messages: Previous conversation context
            temperature: Temperature for GPT-4o calls
            max_tokens: Max tokens for responses
            use_mental_rotation: Whether to use Sora-2 mental rotation
            num_perspectives: Number of parallel perspective analyses
            return_intermediate: Whether to return intermediate results
        
        Returns:
            Dictionary with final answer and optionally intermediate results
        """
        logger.info(f"Processing perspective-taking query: {question[:100]}...")
        
        result = {
            'final_answer': None,
            'confidence': 0.0
        }
        
        # Step 1: Rephrase question for Sora-2
        rephrased_prompt = self._rephrase_for_sora(question, image)
        logger.info(f"Rephrased for Sora: {rephrased_prompt}")
        
        if return_intermediate:
            result['rephrased_prompt'] = rephrased_prompt
        
        # Step 2: Generate mental rotation video with Sora-2
        if use_mental_rotation:
            video_data = self.sora.generate_video(
                prompt=rephrased_prompt,
                image=image,
                duration=self.config.sora_video_length,
                fps=self.config.sora_fps
            )
            logger.info("Video generation complete")
            
            # Step 3: Extract frames
            frames = self.frame_extractor.extract_frames(
                video_data,
                num_frames=self.config.frame_extraction_count,
                window_size=self.config.frame_extraction_window
            )
            logger.info(f"Extracted {len(frames)} frames")
            
            if return_intermediate:
                result['frames'] = [self._encode_frame(f) for f in frames]
        else:
            # Use only original image if mental rotation disabled
            frames = []
        
        # Step 4: Parallel perspective analysis
        perspectives = self._analyze_perspectives(
            original_image=image,
            frames=frames,
            question=question,
            context=context_messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        logger.info(f"Analyzed {len(perspectives)} perspectives")
        
        if return_intermediate:
            result['perspectives'] = perspectives
        
        # Step 5: Synthesize final answer
        final_answer, confidence = self.synthesizer.synthesize(
            perspectives=perspectives,
            original_question=question,
            context=context_messages
        )
        
        result['final_answer'] = final_answer
        result['confidence'] = confidence
        
        logger.info(f"Pipeline complete. Confidence: {confidence:.2f}")
        
        return result
    
    def _rephrase_for_sora(self, question: str, image: str) -> str:
        """
        Rephrase the question to create a prompt for Sora-2 video generation.
        """
        prompt = f"""You are helping to create a video prompt for Sora-2 to visualize a mental rotation or perspective change.

Original question: {question}

Create a concise video generation prompt that will help visualize the transformation or perspective change needed to answer this question. The prompt should describe:
1. The starting state (what's in the image)
2. The transformation or rotation needed
3. The ending state or perspective

Keep it under 50 words and focus on visual transformation.

Video prompt:"""
        
        response = self.openai_client.chat.completions.create(
            model=self.config.gpt_model,
            messages=[
                {"role": "system", "content": "You are an expert at creating video generation prompts."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image}}
                ]}
            ],
            temperature=0.3,
            max_tokens=100
        )
        
        return response.choices[0].message.content.strip()
    
    def _analyze_perspectives(
        self,
        original_image: str,
        frames: List[np.ndarray],
        question: str,
        context: Optional[List[Dict]],
        temperature: Optional[float],
        max_tokens: Optional[int]
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple perspectives in parallel using GPT-4o.
        """
        perspectives = []
        futures = []
        
        # Analyze original image
        futures.append(
            self.executor.submit(
                self._analyze_single_perspective,
                image=original_image,
                question=question,
                perspective_label="original",
                context=context,
                temperature=temperature,
                max_tokens=max_tokens
            )
        )
        
        # Analyze each frame
        for i, frame in enumerate(frames):
            frame_base64 = self._encode_frame(frame)
            futures.append(
                self.executor.submit(
                    self._analyze_single_perspective,
                    image=frame_base64,
                    question=question,
                    perspective_label=f"perspective_{i+1}",
                    context=context,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            )
        
        # Collect results
        for future in as_completed(futures):
            result = future.result(timeout=self.config.timeout)
            perspectives.append(result)
        
        return perspectives
    
    def _analyze_single_perspective(
        self,
        image: str,
        question: str,
        perspective_label: str,
        context: Optional[List[Dict]],
        temperature: Optional[float],
        max_tokens: Optional[int]
    ) -> Dict[str, Any]:
        """
        Analyze a single perspective using GPT-4o.
        """
        prompt = f"""Analyze this image to help answer the following question. You are viewing {perspective_label}.

Question: {question}

Provide a detailed analysis of what you observe from this specific perspective/angle. Focus on:
1. What is visible in this view
2. What spatial relationships you can determine
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
        
        response = self.openai_client.chat.completions.create(
            model=self.config.gpt_model,
            messages=messages,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens
        )
        
        return {
            'perspective': perspective_label,
            'analysis': response.choices[0].message.content,
            'tokens_used': response.usage.total_tokens if hasattr(response, 'usage') else 0
        }
    
    def _encode_frame(self, frame: np.ndarray) -> str:
        """
        Encode a frame as base64 string.
        """
        if isinstance(frame, np.ndarray):
            # Convert numpy array to PIL Image
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            
            image = Image.fromarray(frame)
            
            # Convert to base64
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            base64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return f"data:image/png;base64,{base64_string}"
        
        return frame  # Assume already encoded
    
    def _fallback_completion(
        self,
        image: str,
        question: str,
        context: Optional[List[Dict]],
        temperature: Optional[float],
        max_tokens: Optional[int]
    ) -> str:
        """
        Fallback to standard GPT-4o completion when pipeline fails.
        """
        messages = context or []
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": image}}
            ]
        })
        
        response = self.openai_client.chat.completions.create(
            model=self.config.gpt_model,
            messages=messages,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens
        )
        
        return response.choices[0].message.content
    
    def cleanup(self):
        """
        Cleanup resources.
        """
        self.executor.shutdown(wait=True)
        logger.info("HergartyAgent cleanup complete")
