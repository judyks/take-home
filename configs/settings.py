"""
Configuration settings for the Video Generation API
"""
import os
from typing import Optional

class Settings:
    """app settings w/environment variable support"""
    
    # API config
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_WORKERS: int = int(os.getenv("API_WORKERS", "1"))
    
    # ai model
    MODEL_NAME: str = os.getenv("MODEL_NAME", "Lightricks/LTX-Video-0.9.7-distilled")
    MODEL_CACHE_DIR: Optional[str] = os.getenv("MODEL_CACHE_DIR")
    MODEL_DEVICE: str = os.getenv("MODEL_DEVICE", "auto")  # auto, cuda, cpu
    
    # video generation 
    MAX_VIDEO_DURATION: int = int(os.getenv("MAX_VIDEO_DURATION", "10"))
    DEFAULT_VIDEO_DURATION: int = int(os.getenv("DEFAULT_VIDEO_DURATION", "3"))
    MAX_RESOLUTION: int = int(os.getenv("MAX_RESOLUTION", "512"))
    DEFAULT_FPS: int = int(os.getenv("DEFAULT_FPS", "8"))
    
    # storage 
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "./outputs")
    MAX_STORAGE_GB: float = float(os.getenv("MAX_STORAGE_GB", "100"))
    CLEANUP_AFTER_DAYS: int = int(os.getenv("CLEANUP_AFTER_DAYS", "7"))
    
    # security 
    ENABLE_RATE_LIMIT: bool = os.getenv("ENABLE_RATE_LIMIT", "false").lower() == "true"
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
    API_KEY_REQUIRED: bool = os.getenv("API_KEY_REQUIRED", "false").lower() == "true"
    API_KEY: Optional[str] = os.getenv("API_KEY")
    
    # logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: Optional[str] = os.getenv("LOG_FILE")
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # dev settings
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    RELOAD: bool = os.getenv("RELOAD", "false").lower() == "true"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return os.getenv("ENVIRONMENT", "development").lower() == "production"
    
    @property
    def cuda_available(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def get_device(self) -> str:
        """Get the appropriate device for model inference"""
        if self.MODEL_DEVICE == "auto":
            return "cuda" if self.cuda_available else "cpu"
        return self.MODEL_DEVICE

settings = Settings()
