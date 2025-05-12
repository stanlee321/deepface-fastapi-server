from fastapi import APIRouter
from api.endpoints import ( 
                           blacklist, 
                           detection,
                           entry)

api_router = APIRouter()


# Router
api_router.include_router(entry.router, prefix="/core", tags=["Route request based on type"]) 

# Include endpoint routers
api_router.include_router(blacklist.router, prefix="/face/blacklist", tags=["Blacklist Management"])
api_router.include_router(detection.router, prefix="/face/detect", tags=["Face Detection"]) 

