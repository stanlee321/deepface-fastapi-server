from fastapi import APIRouter, HTTPException, Body, status
from typing import List, Optional
import logging

# Import models and CRUD functions
from models import DetectFaceRequest, DetectFaceResponseItem, FacialArea
from crud import face_crud
from config import settings # To get default detector

router = APIRouter()
log = logging.getLogger(__name__)

@router.post("/detect-face", response_model=List[DetectFaceResponseItem])
async def detect_face_endpoint(request: DetectFaceRequest = Body(...)):
    """
    Detects faces in a single image using the configured DeepFace detector backend.
    Returns a list of detected faces with coordinates and confidence.
    """
    log.info(f"Received request for face detection. Detector: {request.detector_backend or settings.DETECTOR_BACKEND}")

    try:
        # Use the extract_faces function which primarily performs detection
        # Use detector from request if provided, otherwise fallback to config
        detected_faces_data = face_crud.extract_faces_from_image(
            img_data=request.image,
            detector_backend=request.detector_backend or settings.DETECTOR_BACKEND,
            align=False, # Alignment not typically needed just for detection boxes
            enforce_detection=False # Return empty list if no face found
        )

        if not detected_faces_data:
            log.info("No faces detected in the provided image.")
            return []

        # Map the raw results to the response model
        response_items: List[DetectFaceResponseItem] = []
        for i, face_data in enumerate(detected_faces_data):
            try:
                facial_area_data = face_data.get('facial_area')
                confidence_score = face_data.get('confidence')

                # Validate/create FacialArea model
                face_area_obj = None
                if isinstance(facial_area_data, dict):
                    try:
                        face_area_obj = FacialArea.model_validate(facial_area_data)
                    except Exception as area_val_err:
                        #log.warning(f"Could not validate facial_area for face {i}: {area_val_err}. Area data: {facial_area_data}")
                        pass
                else:
                    log.warning(f"Facial area data missing or invalid type for face {i}")

                # Only add if we have a valid facial area
                if face_area_obj:
                    response_items.append(
                        DetectFaceResponseItem(
                            facial_area=face_area_obj,
                            confidence=confidence_score
                        )
                    )
                else:
                    log.warning(f"Skipping face {i} in response due to missing/invalid facial_area.")

            except Exception as item_err:
                # log.error(f"Error processing detected face data item {i}: {item_err}. Data: {face_data}")
                # Optionally skip this item or raise a more specific error?
                pass

        log.info(f"Successfully detected {len(response_items)} faces.")
        return response_items

    except ValueError as ve:
         # Catch potential errors from resolve_image_input if input is invalid
         log.error(f"Invalid image input provided: {ve[:120]}")
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid image input: {ve[:120]}")
    except Exception as e:
        log.exception(f"An unexpected error occurred during face detection: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to perform face detection.") 