"""
Inference module for ClassificationService.
Handles YOLO classification model loading and defect classification.
"""

import time
import numpy as np
from typing import List, Optional
from pathlib import Path

from app.config import settings
from app.logging_config import logger
from app.models import DefectResult


class ClassificationInferenceEngine:
    """
    YOLO Classification based defect detection engine.
    Loads model once at startup and provides inference method.
    """
    
    def __init__(self):
        """Initialize the inference engine."""
        self.model = None
        self.is_loaded = False
        self._load_model()
    
    def _load_model(self) -> None:
        """
        Load the YOLO classification model at startup.
        Model is loaded once and reused for all requests.
        """
        try:
            from ultralytics import YOLO
            
            model_path = Path(settings.MODEL_PATH)
            
            if not model_path.exists():
                logger.warning(
                    f"Model not found at {model_path}. "
                    "Service will return empty defects until model is provided."
                )
                self.is_loaded = False
                return
            
            logger.info(f"Loading YOLO classification model from {model_path}")
            self.model = YOLO(str(model_path))
            
            # Set device
            if settings.INFERENCE_DEVICE != "cpu":
                self.model.to(settings.INFERENCE_DEVICE)
            
            self.is_loaded = True
            logger.info(f"Model loaded successfully on {settings.INFERENCE_DEVICE}")
            logger.info(f"Model classes: {self.model.names}")
            
        except ImportError:
            logger.error("ultralytics package not installed")
            self.is_loaded = False
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            self.is_loaded = False
    
    def classify(
        self,
        image: np.ndarray,
        image_id: str = "",
        fruit_id: str = ""
    ) -> List[DefectResult]:
        """
        Run defect classification on an image.
        
        Args:
            image: RGB uint8 numpy array (HxWx3) from PIL; converted to BGR for YOLO.
            image_id: For logging purposes
            fruit_id: For logging purposes
            
        Returns:
            List of DefectResult objects (empty if no defect, one item if defect detected)
        """
        if not self.is_loaded or self.model is None:
            logger.warning("Model not loaded, returning empty defects")
            return []
        
        try:
            # PIL gives RGB. Ultralytics preprocess treats numpy HxWx3 as BGR and
            # applies BGR→RGB before the model; pass BGR so the net sees correct RGB.
            yolo_input = (
                image[:, :, ::-1].copy()
                if image.ndim == 3 and image.shape[2] == 3
                else image
            )

            start_time = time.time()
            results = self.model.predict(
                source=yolo_input,
                imgsz=settings.IMAGE_SIZE,
                verbose=False
            )
            
            inference_time_ms = (time.time() - start_time) * 1000
            
            # Extract classification results
            if not results or len(results) == 0:
                logger.warning(f"No results from model for fruit_id={fruit_id}")
                return []
            
            result = results[0]
            
            # Get top1 prediction
            probs = result.probs
            if probs is None:
                logger.warning(f"No classification probs for fruit_id={fruit_id}")
                return []
            
            top1_idx = int(probs.top1)
            top1_conf = float(probs.top1conf)
            
            # Get class name - try both result.names and model.names
            if hasattr(result, 'names') and result.names:
                class_name = result.names[top1_idx]
            elif hasattr(self.model, 'names') and self.model.names:
                class_name = self.model.names[top1_idx]
            else:
                class_name = str(top1_idx)
            
            # Log inference results
            logger.info(
                f"Classification: image_id={image_id}, fruit_id={fruit_id}, "
                f"predicted_class={class_name}, confidence={top1_conf:.4f}, "
                f"inference_time={inference_time_ms:.1f}ms"
            )
            
            # Check if it's a defect
            is_defect = (
                class_name.lower() == settings.DEFECT_CLASS_NAME.lower() and
                top1_conf >= settings.DEFECT_MIN_CONF
            )
            
            if is_defect:
                # Return defect with segmentation=null as per spec
                defect = DefectResult(
                    type="defect",
                    confidence=round(top1_conf, 4),
                    segmentation=None
                )
                return [defect]
            else:
                # No defect - return empty list
                return []
            
        except Exception as e:
            logger.error(f"Inference error for fruit_id={fruit_id}: {str(e)}")
            raise


# Global inference engine instance (loaded once at module import)
inference_engine: Optional[ClassificationInferenceEngine] = None


def get_inference_engine() -> ClassificationInferenceEngine:
    """
    Get or create the global inference engine.
    Ensures model is loaded only once.
    """
    global inference_engine
    if inference_engine is None:
        inference_engine = ClassificationInferenceEngine()
    return inference_engine


def run_inference(
    image: np.ndarray,
    image_id: str = "",
    fruit_id: str = ""
) -> List[DefectResult]:
    """
    Convenience function to run defect classification.
    
    Args:
        image: RGB numpy array
        image_id: For logging
        fruit_id: For logging
        
    Returns:
        List of DefectResult objects
    """
    engine = get_inference_engine()
    return engine.classify(image, image_id, fruit_id)
