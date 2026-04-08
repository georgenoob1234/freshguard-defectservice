"""
Pydantic models for ClassificationService.
Defines request/response schemas matching Brain's expected format.

NOTE: These schemas are identical to the old DefectDetector service
to maintain full compatibility with Brain integration.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class SegmentationData(BaseModel):
    """Segmentation polygon data in crop pixel coordinates."""
    polygon: List[List[float]] = Field(
        ...,
        description="List of [x, y] coordinate pairs forming the defect polygon"
    )


class DefectResult(BaseModel):
    """Single defect detection result."""
    type: str = Field(
        default="defect",
        description="Defect type - always 'defect' for binary classification"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Detection confidence score"
    )
    segmentation: Optional[SegmentationData] = Field(
        default=None,
        description="Segmentation polygon in crop pixel space"
    )


class DefectDetectionResponse(BaseModel):
    """Response schema for /detect-defects endpoint."""
    image_id: str = Field(
        ...,
        description="ID of the original full image (echoed from request)"
    )
    fruit_id: str = Field(
        ...,
        description="ID of the fruit assigned by Brain (echoed from request)"
    )
    defects: List[DefectResult] = Field(
        default_factory=list,
        description="List of detected defects (empty if none found)"
    )


class ErrorResponse(BaseModel):
    """Error response schema."""
    detail: str = Field(..., description="Error message")
    image_id: Optional[str] = None
    fruit_id: Optional[str] = None

