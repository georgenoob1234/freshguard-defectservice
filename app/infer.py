"""
Inference module for DefectDetector service.
Handles YOLO-Seg model loading and defect detection.
"""

import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path

from app.config import settings
from app.logging_config import logger
from app.models import DefectResult, SegmentationData


class DefectInferenceEngine:
    """
    YOLO-Seg based defect detection engine.
    Loads model once at startup and provides inference method.
    """
    
    def __init__(self):
        """Initialize the inference engine."""
        self.model = None
        self.is_loaded = False
        self._load_model()
    
    def _load_model(self) -> None:
        """
        Load the YOLO segmentation model at startup.
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
            
            logger.info(f"Loading YOLO-Seg model from {model_path}")
            self.model = YOLO(str(model_path))
            
            # Set device
            if settings.INFERENCE_DEVICE != "cpu":
                self.model.to(settings.INFERENCE_DEVICE)
            
            self.is_loaded = True
            logger.info(f"Model loaded successfully on {settings.INFERENCE_DEVICE}")
            
        except ImportError:
            logger.error("ultralytics package not installed")
            self.is_loaded = False
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            self.is_loaded = False
    
    def detect(self, bgr_image: np.ndarray) -> List[DefectResult]:
        """
        Run defect detection on a BGR image.
        
        Args:
            bgr_image: BGR numpy array (mandatory format)
            
        Returns:
            List of DefectResult objects (empty if no defects)
        """
        if not self.is_loaded or self.model is None:
            logger.warning("Model not loaded, returning empty defects")
            return []
        
        try:
            # Run inference with YOLO-Seg
            results = self.model(
                bgr_image,
                imgsz=settings.IMAGE_SIZE,
                conf=settings.CONFIDENCE_THRESHOLD,
                verbose=False
            )
            
            defects = []
            
            for result in results:
                # Check if segmentation masks are available
                if result.masks is None:
                    continue
                
                # Process each detection
                masks = result.masks
                boxes = result.boxes
                
                if masks is None or boxes is None:
                    continue
                
                for i, (mask, box) in enumerate(zip(masks.data, boxes)):
                    confidence = float(box.conf[0])
                    
                    # Skip low confidence detections
                    if confidence < settings.CONFIDENCE_THRESHOLD:
                        continue
                    
                    # Extract polygon from mask
                    polygon = self._mask_to_polygon(
                        mask.cpu().numpy(),
                        bgr_image.shape[:2]
                    )
                    
                    segmentation = None
                    if polygon is not None and len(polygon) >= 3:
                        segmentation = SegmentationData(polygon=polygon)
                    
                    defect = DefectResult(
                        type="defect",
                        confidence=round(confidence, 4),
                        segmentation=segmentation
                    )
                    defects.append(defect)
                    
                    # Only return first defect (binary: defect/no defect)
                    break
                
                if defects:
                    break
            
            return defects
            
        except Exception as e:
            logger.error(f"Inference error: {str(e)}")
            return []
    
    def _mask_to_polygon(
        self,
        mask: np.ndarray,
        original_size: Tuple[int, int]
    ) -> Optional[List[List[float]]]:
        """
        Convert a binary mask to polygon coordinates.
        
        Args:
            mask: Binary mask from YOLO output
            original_size: (height, width) of original image
            
        Returns:
            List of [x, y] coordinate pairs in crop pixel space
        """
        try:
            import cv2
            
            # Resize mask to original image size if needed
            if mask.shape[:2] != original_size:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (original_size[1], original_size[0]),
                    interpolation=cv2.INTER_LINEAR
                )
            
            # Ensure binary mask
            mask = (mask > 0.5).astype(np.uint8) * 255
            
            # Find contours
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                return None
            
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Simplify polygon
            epsilon = 0.01 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            # Convert to list of [x, y] pairs
            polygon = [[float(pt[0][0]), float(pt[0][1])] for pt in approx]
            
            return polygon if len(polygon) >= 3 else None
            
        except Exception as e:
            logger.error(f"Mask to polygon conversion error: {str(e)}")
            return None


# Global inference engine instance (loaded once at module import)
inference_engine: Optional[DefectInferenceEngine] = None


def get_inference_engine() -> DefectInferenceEngine:
    """
    Get or create the global inference engine.
    Ensures model is loaded only once.
    """
    global inference_engine
    if inference_engine is None:
        inference_engine = DefectInferenceEngine()
    return inference_engine


def run_inference(bgr_image: np.ndarray) -> List[DefectResult]:
    """
    Convenience function to run defect detection.
    
    Args:
        bgr_image: BGR numpy array
        
    Returns:
        List of DefectResult objects
    """
    engine = get_inference_engine()
    return engine.detect(bgr_image)

