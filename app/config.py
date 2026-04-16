"""
Configuration module for ClassificationService.
Defect classification using YOLO classification models.
"""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )
    
    # Service settings
    SERVICE_NAME: str = "ClassificationService"
    SERVICE_HOST: str = "0.0.0.0"
    SERVICE_PORT: int = 8400
    DEBUG: bool = False
    
    # Model settings
    MODEL_PATH: str = os.environ.get("MODEL_PATH", "models/defect_cls.pt")
    
    # Classification settings
    DEFECT_MIN_CONF: float = 0.5  # Minimum confidence to report defect
    DEFECT_CLASS_NAME: str = "defect"  # Class name that indicates defect
    
    # Inference settings
    INFERENCE_DEVICE: str = "cpu"  # "cpu", "cuda", or "cuda:0"
    IMAGE_SIZE: int = 224  # YOLO default inference size


settings = Settings()

