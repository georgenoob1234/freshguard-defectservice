"""
Configuration module for DefectDetector service.
"""

import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Service settings
    SERVICE_NAME: str = "DefectDetector"
    SERVICE_HOST: str = "0.0.0.0"
    SERVICE_PORT: int = 8400
    DEBUG: bool = False
    
    # Model settings
    MODEL_PATH: str = os.environ.get("MODEL_PATH", "models/defect_seg.pt")
    CONFIDENCE_THRESHOLD: float = 0.3
    
    # Inference settings
    INFERENCE_DEVICE: str = "cpu"  # "cpu", "cuda", or "cuda:0"
    IMAGE_SIZE: int = 640  # YOLO default inference size
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

