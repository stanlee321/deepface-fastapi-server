import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np # DeepFace might need numpy

# Import the settings instance
from config import settings

# Import CRUD modules for both backends
from crud import face_crud # DeepFace backend
from crud import aws_rekognition_crud # AWS Rekognition backend
from crud.face_crud import resolve_image_input # Reuse helper for input handling

log = logging.getLogger(__name__)

async def find_blacklist_matches(
    img_data: Union[str, np.ndarray],
    threshold: Optional[float] = None # Threshold interpretation depends on backend
) -> List[Dict[str, Any]]:
    """Finds blacklist matches for an image using the configured backend.

    Args:
        img_data: Input image (path, URL, base64 string, or NumPy array).
        threshold: Optional threshold override. For DeepFace, it's distance;
                   for AWS, it will be converted from distance back to similarity.

    Returns:
        A list of dictionaries, each representing a match, consistent with the
        original DeepFace.find output format.
    """
    log.info(f"Finding blacklist matches using backend: {settings.FACE_PROCESSING_BACKEND}")

    if settings.FACE_PROCESSING_BACKEND == 'deepface':
        # Use existing DeepFace logic
        # Note: face_crud.find_matches_in_blacklist handles input resolution
        try:
            # DeepFace threshold is distance (lower is better)
            # Pass threshold directly if provided
            matches = face_crud.find_matches_in_blacklist(
                img_data=img_data,
                threshold=threshold,
                # Using defaults from config implicitly inside face_crud
                model_name=settings.MODEL_NAME,
                distance_metric=settings.DISTANCE_METRIC,
                detector_backend=settings.DETECTOR_BACKEND
            )
            # DeepFace returns list of DataFrames; convert if needed (check original return type)
            # Assuming face_crud.find_matches_in_blacklist already returns List[List[Dict]] or similar
            # If it returned list[pd.DataFrame], conversion is needed here.
            # Based on face_crud.py, it returns List[List[Dict]] which is close, needs flattening?
            # Let's assume for now it returns the desired List[Dict] for the first face, or handle batches later
            
            # Correction: DeepFace.find with batched=False (default in face_crud) returns List[pd.DataFrame]
            # We need to convert this to List[Dict]
            processed_matches = []
            if matches and isinstance(matches, list) and len(matches) > 0:
                # Assuming we process the first face found in the image
                df = matches[0]
                # Ensure df is a DataFrame and not empty before converting
                if not df.empty:
                     # Convert DataFrame rows to dictionaries
                     processed_matches = df.to_dict('records')
                     log.info(f"DeepFace found {len(processed_matches)} potential matches.")
                else:
                    log.info("DeepFace: Input processed, but no matches found above threshold.")
            else:
                 log.info("DeepFace: No faces detected or no matches returned.")
            return processed_matches
            
        except Exception as e:
            log.exception(f"Error during DeepFace blacklist matching: {e}")
            return [] # Return empty list on error

    elif settings.FACE_PROCESSING_BACKEND == 'aws_rekognition':
        # Use AWS Rekognition logic
        try:
            # 1. Resolve input image to bytes (AWS search_faces_by_image needs bytes)
            # We need a way to get bytes from path/URL/base64/ndarray
            img_bytes = None
            resolved_input = resolve_image_input(img_data) # Use helper from face_crud
            
            if isinstance(resolved_input, str):
                # If it's base64
                if face_crud.is_base64(resolved_input):
                    try:
                        header, encoded = resolved_input.split(',', 1)
                        img_bytes = face_crud.base64.b64decode(encoded)
                    except ValueError:
                         img_bytes = face_crud.base64.b64decode(resolved_input)
                # If it's a file path
                elif face_crud.os.path.exists(resolved_input):
                    with open(resolved_input, 'rb') as f:
                         img_bytes = f.read()
            elif isinstance(resolved_input, np.ndarray):
                 # Convert numpy array (BGR) to bytes (e.g., JPEG format)
                 is_success, buffer = face_crud.cv2.imencode(".jpg", resolved_input)
                 if is_success:
                     img_bytes = buffer.tobytes()
                 else:
                    log.error("Failed to encode numpy array to JPG bytes for AWS.")
                    return []

            if img_bytes is None:
                 log.error(f"Could not resolve image input to bytes for AWS: {type(img_data)}")
                 return []
                 
            # 2. Convert distance threshold (lower is better, 0-1ish) back to AWS Similarity threshold (higher is better, 0-100)
            # Use default 90 if no threshold provided
            aws_threshold = settings.AWS_THRESHOLD # Default AWS similarity threshold
            if threshold is not None:
                 # Inverse mapping: similarity = 100 - (distance * 100)
                 # Clamp between 0 and 100
                 aws_threshold = max(0.0, min(100.0, 100.0 - (threshold * 100.0)))
                 log.info(f"Using AWS Similarity threshold: {aws_threshold:.2f} (converted from input distance threshold: {threshold:.4f})" )
            else:
                 log.info(f"Using default AWS Similarity threshold: {aws_threshold:.2f}")
                 
            # 3. Call AWS search function
            matches = await aws_rekognition_crud.search_face_aws(
                collection_id=settings.AWS_REKOGNITION_COLLECTION_ID,
                image_bytes=img_bytes,
                threshold=aws_threshold
                # max_matches can be added if needed
            )
            # aws_rekognition_crud.search_face_aws already maps the output
            # to the desired List[Dict] format.
            log.info(f"AWS Rekognition found {len(matches)} matches.")
            return matches

        except Exception as e:
            log.exception(f"Error during AWS Rekognition blacklist matching: {e}")
            return [] # Return empty list on error

    else:
        log.error(f"Invalid FACE_PROCESSING_BACKEND configured: {settings.FACE_PROCESSING_BACKEND}")
        raise ValueError(f"Unsupported face processing backend: {settings.FACE_PROCESSING_BACKEND}")

# We might need other service functions later, e.g., for extracting face info
# if the API needs it separately from blacklist searching.
# async def extract_faces_service(...):
#     if FACE_PROCESSING_BACKEND == 'deepface': ...
#     elif FACE_PROCESSING_BACKEND == 'aws_rekognition':
#         # AWS detect_faces could be used here, map output
#         pass 