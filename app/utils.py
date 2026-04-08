"""
Utility functions for ClassificationService.
Handles image preprocessing and RGB conversion.
"""

import io
import numpy as np
from PIL import Image
from typing import Tuple


def load_image_as_rgb(image_bytes: bytes) -> np.ndarray:
    """
    Load image from bytes and convert to RGB numpy array.
    
    This is the MANDATORY preprocessing step before inference.
    Flow: bytes → PIL.Image → RGB numpy
    
    Args:
        image_bytes: Raw image bytes (JPEG or PNG)
        
    Returns:
        RGB numpy array ready for classification inference
        
    Raises:
        ValueError: If image cannot be decoded
    """
    try:
        # Load image via PIL
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary (handles grayscale, RGBA, etc.)
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        
        # Convert to numpy array (RGB order)
        rgb_array = np.array(pil_image)
        
        return rgb_array
        
    except Exception as e:
        raise ValueError(f"Failed to decode image: {str(e)}")


def load_image_as_bgr(image_bytes: bytes) -> np.ndarray:
    """
    Load image from bytes and convert to BGR numpy array.
    Kept for backwards compatibility.
    
    Args:
        image_bytes: Raw image bytes (JPEG or PNG)
        
    Returns:
        BGR numpy array
        
    Raises:
        ValueError: If image cannot be decoded
    """
    rgb_array = load_image_as_rgb(image_bytes)
    # Convert RGB to BGR
    bgr_array = rgb_array[:, :, ::-1].copy()
    return bgr_array


def get_image_dimensions(array: np.ndarray) -> Tuple[int, int]:
    """
    Get image dimensions from numpy array.
    
    Args:
        array: RGB or BGR numpy array
        
    Returns:
        Tuple of (height, width)
    """
    return array.shape[:2]


def validate_image_bytes(image_bytes: bytes) -> bool:
    """
    Validate that bytes represent a valid image.
    
    Args:
        image_bytes: Raw bytes to validate
        
    Returns:
        True if valid image, False otherwise
    """
    if not image_bytes or len(image_bytes) == 0:
        return False
    
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
        return True
    except Exception:
        return False
