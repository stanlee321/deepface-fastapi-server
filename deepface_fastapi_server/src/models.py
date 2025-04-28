from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime # Import datetime

class FacialArea(BaseModel):
    x: int
    y: int
    w: int
    h: int
    left_eye: Optional[Tuple[int, int]] = None
    right_eye: Optional[Tuple[int, int]] = None
    # Add other landmarks if needed (e.g., from retinaface)
    nose: Optional[Tuple[int, int]] = None
    mouth_left: Optional[Tuple[int, int]] = None
    mouth_right: Optional[Tuple[int, int]] = None

class BlacklistMatch(BaseModel):
    # Fields directly from DeepFace.find results (dictionary keys)
    identity: str # Path to matched image in blacklist DB
    # hash: Optional[str] = None # Internal use for deepface, maybe not needed in API response
    target_x: int
    target_y: int
    target_w: int
    target_h: int
    source_x: int
    source_y: int
    source_w: int
    source_h: int
    threshold: float
    distance: float

    class Config:
        # Allow creation from dicts without exact field match (extra fields ignored)
        extra = 'ignore'

class DetectedFaceResult(BaseModel):
    face_index: int
    facial_area: FacialArea
    confidence: float
    # Adapt based on the output structure of the processing logic
    # If using DeepFace.find directly, it returns matches per face
    blacklist_matches: List[BlacklistMatch] = []

class ImageProcessingResult(BaseModel):
    image_path_or_identifier: str # Use an identifier if input is not a path
    faces: List[DetectedFaceResult]
    error: Optional[str] = None

class ProcessImagesRequest(BaseModel):
    # Allow providing image paths, base64 strings, or URLs
    # Example: List of strings, validation happens in the endpoint
    images: List[str] = Field(..., description="List of image paths, public URLs, or base64 encoded strings.")
    # Optional parameters to override config
    detector_backend: Optional[str] = None
    model_name: Optional[str] = None
    distance_metric: Optional[str] = None
    threshold: Optional[float] = None

# --- Blacklist CRUD Models ---
class BlacklistBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Unique name or identifier for the blacklist entry.")
    reason: Optional[str] = Field(None, max_length=255, description="Reason for adding to the blacklist.")

class BlacklistCreate(BlacklistBase):
    # In a real scenario, you might accept image uploads/paths here
    # to associate with the blacklist entry.
    # image_paths: Optional[List[str]] = None
    pass

class BlacklistRecord(BlacklistBase):
    id: int
    added_date: datetime # Use datetime for type hint
    reference_image_dir: Optional[str] = None # Add optional field for image dir path

    class Config:
        from_attributes = True # Renamed from orm_mode in Pydantic v2 