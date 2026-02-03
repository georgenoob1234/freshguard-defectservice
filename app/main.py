"""
Main FastAPI application for DefectDetector service.

This is a passive microservice that:
- Receives cropped fruit images from Brain
- Detects defects using YOLO-Seg model
- Returns defect results with segmentation polygons

All requests originate from Brain - this service never initiates requests.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.logging_config import logger
from app.api import router
from app.infer import get_inference_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Loads model at startup, cleans up on shutdown.
    """
    # Startup: Load model once
    logger.info(f"Starting {settings.SERVICE_NAME} service...")
    engine = get_inference_engine()
    
    if engine.is_loaded:
        logger.info("Model loaded and ready for inference")
    else:
        logger.warning("Model not loaded - service will return empty defects")
    
    yield
    
    # Shutdown
    logger.info(f"Shutting down {settings.SERVICE_NAME} service...")


# Create FastAPI application
app = FastAPI(
    title=settings.SERVICE_NAME,
    description=(
        "Defect detection microservice for fruit analysis system. "
        "Receives cropped fruit images and returns defect segmentation results."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": settings.SERVICE_NAME,
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "detect_defects": "POST /detect-defects",
            "health": "GET /health"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.SERVICE_HOST,
        port=settings.SERVICE_PORT,
        reload=settings.DEBUG
    )

