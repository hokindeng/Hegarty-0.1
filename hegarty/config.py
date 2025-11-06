"""
Configuration management for Hegarty
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List
import yaml
from pathlib import Path


@dataclass
class Config:
    """
    Configuration for Hegarty agent and components.
    """
    # GPT-4o settings
    gpt_model: str = "gpt-4o"
    temperature: float = 0.3
    max_tokens: int = 2000
    top_p: float = 0.9
    
    # Sora-2 settings
    sora_video_length: int = 3  # seconds
    sora_fps: int = 10
    sora_resolution: str = "1024x1024"
    sora_quality: str = "high"
    
    # Frame extraction settings
    frame_extraction_count: int = 5
    frame_extraction_window: int = 30  # Last N frames to consider
    frame_extraction_strategy: str = "uniform"  # uniform, adaptive, keyframe
    
    # Parallel processing
    max_workers: int = 6
    timeout: int = 30
    retry_attempts: int = 3
    
    # Perspective detection
    perspective_confidence_threshold: float = 0.7
    perspective_keywords: List[str] = field(default_factory=lambda: [
        "rotate", "turn", "flip", "perspective", "viewpoint",
        "from above", "from below", "other side"
    ])
    
    # API settings
    api_rate_limit: int = 100  # requests per minute
    max_image_size: int = 10485760  # 10MB
    supported_formats: List[str] = field(default_factory=lambda: [
        "jpeg", "jpg", "png", "gif", "webp"
    ])
    
    # Caching
    enable_cache: bool = True
    cache_ttl: int = 3600  # seconds
    cache_max_entries: int = 1000
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """
        Load configuration from YAML file.
        
        Args:
            path: Path to YAML configuration file
        
        Returns:
            Config object
        """
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Extract hegarty config section
        hegarty_config = data.get('hegarty', {})
        
        # Flatten nested configuration
        config_dict = {}
        
        # GPT-4o settings
        if 'gpt4o' in hegarty_config:
            for key, value in hegarty_config['gpt4o'].items():
                config_dict[f"gpt_{key}" if key != 'model' else 'gpt_model'] = value
        
        # Sora-2 settings
        if 'sora2' in hegarty_config:
            for key, value in hegarty_config['sora2'].items():
                config_dict[f"sora_{key}"] = value
        
        # Frame extraction settings
        if 'frame_extraction' in hegarty_config:
            for key, value in hegarty_config['frame_extraction'].items():
                config_dict[f"frame_extraction_{key}"] = value
        
        # Parallel processing settings
        if 'parallel_processing' in hegarty_config:
            for key, value in hegarty_config['parallel_processing'].items():
                config_dict[key] = value
        
        # Perspective detection settings
        if 'perspective_detection' in hegarty_config:
            pd = hegarty_config['perspective_detection']
            if 'confidence_threshold' in pd:
                config_dict['perspective_confidence_threshold'] = pd['confidence_threshold']
            if 'keywords' in pd:
                config_dict['perspective_keywords'] = pd['keywords']
        
        # API settings
        if 'api' in hegarty_config:
            for key, value in hegarty_config['api'].items():
                config_dict[f"api_{key}" if key != 'rate_limit' else 'api_rate_limit'] = value
        
        # Cache settings
        if 'cache' in hegarty_config:
            cache = hegarty_config['cache']
            if 'enabled' in cache:
                config_dict['enable_cache'] = cache['enabled']
            if 'ttl' in cache:
                config_dict['cache_ttl'] = cache['ttl']
            if 'max_entries' in cache:
                config_dict['cache_max_entries'] = cache['max_entries']
        
        return cls(**config_dict)
    
    @classmethod
    def from_env(cls) -> "Config":
        """
        Load configuration from environment variables.
        
        Returns:
            Config object
        """
        config_dict = {}
        
        # Map environment variables to config fields
        env_mappings = {
            'HEGARTY_GPT_MODEL': 'gpt_model',
            'HEGARTY_TEMPERATURE': ('temperature', float),
            'HEGARTY_MAX_TOKENS': ('max_tokens', int),
            'HEGARTY_TOP_P': ('top_p', float),
            'HEGARTY_SORA_VIDEO_LENGTH': ('sora_video_length', int),
            'HEGARTY_SORA_FPS': ('sora_fps', int),
            'HEGARTY_SORA_RESOLUTION': 'sora_resolution',
            'HEGARTY_MAX_WORKERS': ('max_workers', int),
            'HEGARTY_TIMEOUT': ('timeout', int),
            'HEGARTY_CACHE_ENABLED': ('enable_cache', lambda x: x.lower() == 'true'),
            'HEGARTY_CACHE_TTL': ('cache_ttl', int),
        }
        
        for env_var, mapping in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                if isinstance(mapping, tuple):
                    field_name, converter = mapping
                    config_dict[field_name] = converter(value)
                else:
                    config_dict[mapping] = value
        
        return cls(**config_dict)
    
    def to_dict(self) -> dict:
        """
        Convert config to dictionary.
        
        Returns:
            Dictionary representation of config
        """
        return {
            key: getattr(self, key)
            for key in self.__dataclass_fields__.keys()
        }
    
    def save_yaml(self, path: str):
        """
        Save configuration to YAML file.
        
        Args:
            path: Path to save YAML file
        """
        # Structure config for YAML
        structured_config = {
            'hegarty': {
                'model_version': '1.0',
                'gpt4o': {
                    'model': self.gpt_model,
                    'temperature': self.temperature,
                    'max_tokens': self.max_tokens,
                    'top_p': self.top_p,
                },
                'sora2': {
                    'video_length': self.sora_video_length,
                    'fps': self.sora_fps,
                    'resolution': self.sora_resolution,
                    'quality': self.sora_quality,
                },
                'frame_extraction': {
                    'count': self.frame_extraction_count,
                    'window': self.frame_extraction_window,
                    'strategy': self.frame_extraction_strategy,
                },
                'parallel_processing': {
                    'max_workers': self.max_workers,
                    'timeout': self.timeout,
                    'retry_attempts': self.retry_attempts,
                },
                'perspective_detection': {
                    'confidence_threshold': self.perspective_confidence_threshold,
                    'keywords': self.perspective_keywords,
                },
                'api': {
                    'rate_limit': self.api_rate_limit,
                    'max_image_size': self.max_image_size,
                    'supported_formats': self.supported_formats,
                },
                'cache': {
                    'enabled': self.enable_cache,
                    'ttl': self.cache_ttl,
                    'max_entries': self.cache_max_entries,
                }
            }
        }
        
        with open(path, 'w') as f:
            yaml.dump(structured_config, f, default_flow_style=False, sort_keys=False)


def load_config(path: Optional[str] = None) -> Config:
    """
    Load configuration from file or environment.
    
    Args:
        path: Optional path to configuration file
    
    Returns:
        Config object
    """
    # Check for config file in standard locations
    if not path:
        for potential_path in ['config.yaml', 'hegarty.yaml', '.hegarty.yaml']:
            if Path(potential_path).exists():
                path = potential_path
                break
    
    # Load from file if exists
    if path and Path(path).exists():
        return Config.from_yaml(path)
    
    # Otherwise load from environment with defaults
    return Config.from_env()
