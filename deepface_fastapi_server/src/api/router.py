from fastapi import APIRouter
from api.endpoints import processing, blacklist, processed_images, detection

api_router = APIRouter()

# Include endpoint routers
api_router.include_router(processing.router, prefix="/process", tags=["Image Processing"])
api_router.include_router(blacklist.router, prefix="/blacklist", tags=["Blacklist Management"])
api_router.include_router(processed_images.router, prefix="/processed-images", tags=["Processed Image Results"])
api_router.include_router(detection.router, prefix="/detect", tags=["Face Detection"]) 