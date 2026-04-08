"""
API endpoints for ClassificationService.
Implements the /detect-defects endpoint per Brain contract.

This is a drop-in replacement for the DefectDetector service,
using YOLO classification instead of segmentation.
"""

from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from app.models import DefectDetectionResponse, DefectResult
from app.utils import load_image_as_rgb, validate_image_bytes
from app.infer import run_inference
from app.logging_config import logger


router = APIRouter()


@router.post(
    "/detect-defects",
    response_model=DefectDetectionResponse,
    responses={
        200: {"description": "Successful detection response"},
        400: {"description": "Invalid input"},
        500: {"description": "Internal server error"}
    }
)
async def detect_defects(
    image: UploadFile = File(..., description="Binary crop of one fruit"),
    image_id: str = Form(..., description="ID of the original full image"),
    fruit_id: str = Form(..., description="ID of the fruit assigned by Brain")
) -> DefectDetectionResponse:
    """
    Classify defects in a cropped fruit image.
    
    Receives a single cropped fruit image and returns defect classification results.
    Uses YOLO classification model to determine if fruit has defect.
    
    - **image**: Binary image file (JPEG or PNG)
    - **image_id**: ID echoed back from request
    - **fruit_id**: ID echoed back from request
    
    Returns:
        DefectDetectionResponse with defects list (empty if no defect detected)
    """
    logger.info(f"Processing classification: image_id={image_id}, fruit_id={fruit_id}")
    
    # Validate input fields
    if not image_id or not fruit_id:
        logger.warning("Missing required fields")
        raise HTTPException(
            status_code=400,
            detail="Missing required fields: image_id and fruit_id are required"
        )
    
    try:
        # Read image bytes
        image_bytes = await image.read()
        
        # Validate image
        if not validate_image_bytes(image_bytes):
            logger.warning(f"Invalid image data for fruit_id={fruit_id}")
            raise HTTPException(
                status_code=400,
                detail="Invalid image data: could not decode image"
            )
        
        # Load image as RGB numpy array
        rgb_image = load_image_as_rgb(image_bytes)
        logger.debug(f"Image loaded: shape={rgb_image.shape}")
        
        # Run classification inference
        defects = run_inference(rgb_image, image_id, fruit_id)
        
        # Build response
        response = DefectDetectionResponse(
            image_id=image_id,
            fruit_id=fruit_id,
            defects=defects
        )
        
        defect_count = len(defects)
        if defect_count > 0:
            logger.info(
                f"Defect detected: fruit_id={fruit_id}, "
                f"confidence={defects[0].confidence}"
            )
        else:
            logger.info(f"No defects detected: fruit_id={fruit_id}")
        
        return response
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Value error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Inference failure -> 500 with clear error message
        logger.error(f"Inference error for fruit_id={fruit_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "ClassificationService"}
