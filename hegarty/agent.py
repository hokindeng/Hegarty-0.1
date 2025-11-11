"""Core orchestration agent"""

import logging
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import base64
import io
import time
import shutil

import numpy as np
from PIL import Image

from .mllm import MLLMProvider
from .vm import SoraVM
from .frame_extractor import FrameExtractor
from .synthesizer import PerspectiveSynthesizer
from .config import Config
from .camera_detector import CameraDetector

logger = logging.getLogger(__name__)


class HergartyAgent:
    """Core agent orchestrating perspective-taking pipeline"""
    
    def __init__(
        self,
        openai_client: Optional[object] = None,
        sora_api_key: Optional[str] = None,
        config: Optional[Config] = None
    ):
        self.config = config or Config()
        self.openai_client = None
        
        self.mllm: Optional[MLLMProvider] = None
        
        self.vm = SoraVM(api_key=sora_api_key) if sora_api_key else None
        self.frame_extractor = FrameExtractor(strategy=self.config.frame_extraction_strategy)
        self.synthesizer = PerspectiveSynthesizer()
        
        # Initialize camera detector if enabled
        self.camera_detector = None
        if self.config.use_camera_detection:
            try:
                self.camera_detector = CameraDetector(model_version=self.config.camera_detector_model)
                logger.info(f"Camera detector initialized with {self.config.camera_detector_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize camera detector: {e}")
                self.camera_detector = None
        
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
        logger.info(f"Processing: {question[:100]}...")
        
        # Use injected MLLM provider (e.g., Qwen)
        mllm = self.mllm
        
        result = {'final_answer': None, 'confidence': 0.0}
        
        # Step 1a: Detect camera parameters if enabled
        camera_params = None
        if self.camera_detector and use_mental_rotation:
            try:
                # Decode base64 image for camera detection
                if image.startswith('data:'):
                    image_data = image.split(',')[1]
                else:
                    image_data = image
                image_bytes = base64.b64decode(image_data)
                pil_image = Image.open(io.BytesIO(image_bytes))
                
                # Detect camera parameters
                camera_params = self.camera_detector.detect_from_image(pil_image)
                
                # Save camera detection results
                if session_dir and camera_params:
                    import json
                    camera_file = session_dir / "camera_params.json"
                    with open(camera_file, 'w') as f:
                        json.dump({
                            'roll': camera_params.get('roll', 0.0),
                            'pitch': camera_params.get('pitch', 0.0),
                            'vfov': camera_params.get('vfov', 60.0),
                            'confidence': camera_params.get('confidence', 0.0)
                        }, f, indent=2)
                    logger.info(f"Saved camera params to: {camera_file}")
                
                if return_intermediate:
                    result['camera_params'] = camera_params
                    
            except Exception as e:
                logger.warning(f"Camera detection failed: {e}")
                camera_params = None
        
        # Step 1b: Rephrase for video with camera parameters
        rephrased = mllm.rephrase_for_video(question, image, camera_params)
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
            mllm=mllm,
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
            mllm_provider=mllm
        )
        
        result['final_answer'] = final_answer
        result['confidence'] = confidence
        
        logger.info(f"Complete. Confidence: {confidence:.2f}")
        return result
    
    def _analyze_perspectives(
        self,
        mllm: MLLMProvider,
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
                mllm.analyze_perspective,
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
                    mllm.analyze_perspective,
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
