"""Core orchestration agent"""

import logging
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import base64
import io
import time
import shutil

from openai import OpenAI
import numpy as np
from PIL import Image

from .mllm import OpenAIMLLM
from .vm import SoraVM
from .frame_extractor import FrameExtractor
from .synthesizer import PerspectiveSynthesizer
from .config import Config

logger = logging.getLogger(__name__)


class HergartyAgent:
    """Core agent orchestrating perspective-taking pipeline"""
    
    def __init__(
        self,
        openai_client: Optional[OpenAI] = None,
        sora_api_key: Optional[str] = None,
        config: Optional[Config] = None
    ):
        self.config = config or Config()
        
        self.mllm = OpenAIMLLM(
            client=openai_client,
            model=self.config.gpt_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        ) if openai_client else None
        
        self.vm = SoraVM(api_key=sora_api_key) if sora_api_key else None
        self.frame_extractor = FrameExtractor(strategy=self.config.frame_extraction_strategy)
        self.synthesizer = PerspectiveSynthesizer()
        
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
        return_intermediate: bool = False,
        session_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        self.mllm.session_dir = session_dir
        self.mllm.call_counter = 0
        
        logger.info(f"Processing: {question[:100]}...")
        
        result = {'final_answer': None, 'confidence': 0.0}
        
        # Step 1: Rephrase for video
        rephrased = self.mllm.rephrase_for_video(question, image)
        logger.info(f"Rephrased: {rephrased}")
        
        if return_intermediate:
            result['rephrased_prompt'] = rephrased
        
        # Step 2: Generate video
        frames = []
        if use_mental_rotation and self.vm:
            video_data = self.vm.generate_video(
                prompt=rephrased,
                image=image,
                duration=self.config.sora_video_length,
                fps=self.config.sora_fps,
                session_dir=session_dir
            )
            logger.info("Video generated")
            
            frames = self.frame_extractor.extract_frames(
                video_data,
                num_frames=self.config.frame_extraction_count,
                window_size=self.config.frame_extraction_window,
                session_dir=session_dir
            )
            logger.info(f"Extracted {len(frames)} frames")
            
            if return_intermediate:
                result['frames'] = [self._encode_frame(f) for f in frames]
        
        # Step 3: Parallel analysis
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
        
        # Step 4: Synthesize
        final_answer, confidence = self.synthesizer.synthesize(
            perspectives=perspectives,
            original_question=question,
            context=context_messages,
            mllm_provider=self.mllm
        )
        
        result['final_answer'] = final_answer
        result['confidence'] = confidence
        
        logger.info(f"Complete. Confidence: {confidence:.2f}")
        return result
    
    def _analyze_perspectives(
        self,
        original_image: str,
        frames: List[np.ndarray],
        question: str,
        context: Optional[List[Dict]],
        temperature: Optional[float],
        max_tokens: Optional[int]
    ) -> List[Dict[str, Any]]:
        perspectives = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []
            
            futures.append(executor.submit(
                self.mllm.analyze_perspective,
                image=original_image,
                question=question,
                perspective_label="original",
                context=context,
                temperature=temperature,
                max_tokens=max_tokens
            ))
            
            for i, frame in enumerate(frames):
                frame_base64 = self._encode_frame(frame)
                futures.append(executor.submit(
                    self.mllm.analyze_perspective,
                    image=frame_base64,
                    question=question,
                    perspective_label=f"perspective_{i+1}",
                    context=context,
                    temperature=temperature,
                    max_tokens=max_tokens
                ))
            
            for future in as_completed(futures):
                perspectives.append(future.result(timeout=self.config.timeout))
        
        return perspectives
    
    def _encode_frame(self, frame: np.ndarray) -> str:
        if isinstance(frame, np.ndarray):
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            image = Image.fromarray(frame)
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            base64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return f"data:image/png;base64,{base64_string}"
        return frame
    
    @staticmethod
    def cleanup_old_sessions(temp_dir: Path = Path.cwd() / "temp", max_age_hours: int = 24):
        """Remove session directories older than max_age_hours"""
        if not temp_dir.exists():
            return
        cutoff = time.time() - (max_age_hours * 3600)
        for session_dir in temp_dir.iterdir():
            if session_dir.is_dir() and session_dir.stat().st_mtime < cutoff:
                shutil.rmtree(session_dir)
                logger.info(f"Removed old session: {session_dir.name}")
