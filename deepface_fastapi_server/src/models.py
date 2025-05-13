from pydantic import BaseModel, Field
from typing import List, Optional, Any, Tuple, Union
from datetime import datetime # Import datetime

class FacialArea(BaseModel):
    # Allow float coordinates for AWS, DeepFace might return int
    x: Optional[Union[int, float]] = None
    y: Optional[Union[int, float]] = None
    w: Optional[Union[int, float]] = None
    h: Optional[Union[int, float]] = None
    left_eye: Optional[Tuple[Union[int, float], Union[int, float]]] = None
    right_eye: Optional[Tuple[Union[int, float], Union[int, float]]] = None
    # Add other landmarks if needed (e.g., from retinaface)
    nose: Optional[Tuple[Union[int, float], Union[int, float]]] = None
    mouth_left: Optional[Tuple[Union[int, float], Union[int, float]]] = None
    mouth_right: Optional[Tuple[Union[int, float], Union[int, float]]] = None

    # Add validator if needed to ensure coords are present if object is created
    # or convert floats to int if required downstream?

class WeaponArea(BaseModel):
    # Allow float coordinates for AWS, Custom model might return int
    x: Optional[Union[int, float]] = None
    y: Optional[Union[int, float]] = None
    w: Optional[Union[int, float]] = None
    h: Optional[Union[int, float]] = None

class BlacklistMatch(BaseModel):
    # Fields directly from DeepFace.find results or AWS mapped results
    identity: str # Path (DeepFace) or aws_rekognition_external_id:id (AWS)
    # hash: Optional[str] = None # Internal use for deepface, maybe not needed in API response
    
    # Allow None for target coords as AWS doesn't provide them
    target_x: Optional[int] = None 
    target_y: Optional[int] = None
    target_w: Optional[int] = None
    target_h: Optional[int] = None
    
    # Allow float/int/None for source coords
    source_x: Optional[Union[int, float]] = None
    source_y: Optional[Union[int, float]] = None
    source_w: Optional[Union[int, float]] = None
    source_h: Optional[Union[int, float]] = None
    
    threshold: float
    distance: float
    
    # Add AWS specific fields as optional if needed in response
    aws_similarity: Optional[float] = None
    aws_face_id: Optional[str] = None
    
    # Add DeepFace specific fields (or keep generic)
    model: Optional[str] = None
    detector_backend: Optional[str] = None
    similarity_metric: Optional[str] = None

    class Config:
        # Allow creation from dicts without exact field match (extra fields ignored)
        extra = 'ignore'

class DetectedFaceResult(BaseModel):
    face_index: int
    # Make facial_area optional as it might not be available reliably from matches
    facial_area: Optional[FacialArea] = None 
    # Make confidence optional as it might not be available from matches
    confidence: Optional[float] = None 
    blacklist_matches: List[BlacklistMatch] = []

class FaceImageProcessingResult(BaseModel):
    image_path_or_identifier: str # Use an identifier if input is not a path
    faces: List[DetectedFaceResult]
    error: Optional[str] = None
    # Add fields to match ProcessedImageRecord for consistency if needed
    db_id: Optional[int] = None # Optional as it's set after DB save
    code: Optional[str] = None
    app_type: Optional[str] = None
    saved_image_path: Optional[str] = None # Path where image copy was saved
    processing_timestamp: Optional[Any] = None # Keep Any or use datetime if validated
    has_blacklist_match: Optional[bool] = None # Flag indicating if any match occurred
    cropped_face_path: Optional[str] = None # Path to the saved cropped face image

class ProcessImagesRequest(BaseModel):
    # Allow providing image paths, base64 strings, or URLs
    # Example: List of strings, validation happens in the endpoint
    images: List[str] = Field(..., description="List of image paths, public URLs, or base64 encoded strings.")
    # Optional parameters to override config
    code: str = Field(..., description="Internal code for the request.")
    app_type: str = Field(..., description="Type of application for the request.")
    
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
        
        
# Define a model for the list response to include pagination info
class ProcessedImageRecord(FaceImageProcessingResult):
    # Inherits: image_path_or_identifier, faces, error, saved_image_path, has_blacklist_match, cropped_face_path
    # Add the DB specific fields that are always present when retrieved
    db_id: int 
    processing_timestamp: Any # Keep Any for flexibility from DB
    
    # Ensure inherited fields are included during validation/serialization if needed.
    # Might need a custom validator or model_dump setting if base model fields
    # aren't automatically picked up correctly from the stored JSON.


# --- Face Detection Endpoint Models ---

class DetectFaceRequest(BaseModel):
    image: str = Field(..., description="Single image input: path, public URL, or base64 encoded string.")
    # Optional override for detector, otherwise uses config default
    detector_backend: Optional[str] = None
    
    # Internal Code
    code: str = Field(..., description="Internal code for the request.")
    app_type: str = Field(..., description="Type of application for the request.")

class DetectFaceResponseItem(BaseModel):
    # Corresponds roughly to one item from DeepFace.extract_faces
    facial_area: FacialArea # Reuse existing model, already allows optional/float coords
    confidence: Optional[float] = None # Confidence score from the detector

# The response will be a list of these items
# No need for a separate wrapper model if just returning List[DetectFaceResponseItem]

class DetectWeaponsRequest(BaseModel):
    image: str = Field(..., description="Single image input: path, public URL, or base64 encoded string.")
    # Optional override for detector, otherwise uses config default
    detector_backend: Optional[str] = None

    # Internal Code
    code: str = Field(..., description="Internal code for the request.")
    app_type: str = Field(..., description="Type of application for the request.")

class DetectWeaponsResponseItem(BaseModel):
    # Corresponds roughly to one item from DeepFace.extract_faces
    weapon_area: Optional[WeaponArea] = None # Reuse existing model, already allows optional/float coords
    confidence: Optional[float] = None # Confidence score from the detector

class WeaponImageProcessingResult(BaseModel):
    image_path_or_identifier: str # Use an identifier if input is not a path
    weapons: List[DetectWeaponsResponseItem]
    error: Optional[str] = None
    code: Optional[str] = None
    app_type: Optional[str] = None
    saved_image_path: Optional[str] = None # Path where image copy was saved
    cropped_weapon_path: Optional[str] = None # Path to the saved cropped weapon image

# The response will be a list of these items
# No need for a separate wrapper model if just returning List[DetectWeaponsResponseItem]


class PaginatedProcessedImagesResponse(BaseModel): 
    total_items: int
    items: List[Union[FaceImageProcessingResult, WeaponImageProcessingResult]]
    limit: int
    offset: int