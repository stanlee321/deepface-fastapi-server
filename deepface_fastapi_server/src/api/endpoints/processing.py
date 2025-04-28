from fastapi import APIRouter, HTTPException, Body, Depends, status
from typing import List, Optional
import asyncio # For potential parallel processing
import logging

from src.models import (ProcessImagesRequest, ImageProcessingResult,
                      DetectedFaceResult, FacialArea, BlacklistMatch)
from src.crud import face_crud
from src.crud import processed_image_crud # Import the new CRUD module
from src.config import (
    BLACKLIST_DB_PATH, DETECTOR_BACKEND, MODEL_NAME,
    DISTANCE_METRIC
)

router = APIRouter()

async def process_single_image(img_input: str, request_params: ProcessImagesRequest) -> ImageProcessingResult:
    """Helper function to process a single image (path, url, or base64)."""
    image_faces: List[DetectedFaceResult] = []
    error_msg: Optional[str] = None
    saved_image_path: Optional[str] = None # Variable to store path of saved image

    try:
        # --- A. Save a copy of the incoming image --- 
        saved_image_path = await face_crud.save_incoming_image(img_input)
        if not saved_image_path:
            # If saving failed, create an error result and skip DeepFace processing
            error_msg = "Failed to save or process input image."
            result_obj = ImageProcessingResult(
                 image_path_or_identifier=img_input[:100] + ("..." if len(img_input) > 100 else ""),
                 faces=[],
                 error=error_msg
            )
            # Optionally log to DB even on failure?
            # await processed_image_crud.add_processed_image(img_input, "SAVE_FAILED", result_obj)
            return result_obj # Return error result immediately

        # --- B. Process with DeepFace (using original input identifier) ---
        # 1. Extract faces (provides coordinates and face arrays)
        # Pass detector from params/config
        # Note: face_obj["face"] is a numpy array (RGB float 0-1)
        #       face_obj["facial_area"] has coords
        extracted_faces = face_crud.extract_faces_from_image(
            img_input,
            detector_backend=request_params.detector_backend or DETECTOR_BACKEND
        )

        # 2. Find matches in blacklist using DeepFace.find
        # Pass the *original* img_input (path/url/base64) as DeepFace.find handles it,
        # and pass the relevant parameters.
        blacklist_matches_list = face_crud.find_matches_in_blacklist(
            img_input,
            db_path=BLACKLIST_DB_PATH,
            model_name=request_params.model_name or MODEL_NAME,
            distance_metric=request_params.distance_metric or DISTANCE_METRIC,
            detector_backend=request_params.detector_backend or DETECTOR_BACKEND,
            threshold=request_params.threshold,
            refresh_database=False # Keep False for API endpoint performance
        )

        # 3. Correlate results and build response model
        if len(extracted_faces) != len(blacklist_matches_list):
            logging.warning(
                f"Face count mismatch for image '{img_input[:50]}...'. "
                f"Extracted: {len(extracted_faces)}, Find Results: {len(blacklist_matches_list)}. "
                f"This might happen if face detection settings differ or 'find' fails internally."
            )
            # Fallback: return extracted faces without blacklist info
            for i, face_obj in enumerate(extracted_faces):
                try:
                    facial_area_model = FacialArea(**face_obj["facial_area"])
                    image_faces.append(DetectedFaceResult(
                        face_index=i,
                        facial_area=facial_area_model,
                        confidence=face_obj["confidence"],
                        blacklist_matches=[] # Indicate missing/mismatched data
                    ))
                except Exception as model_error:
                    logging.error(f"Error creating FacialArea model for face {i}: {model_error}")
                    error_msg = f"Error processing face {i} data."
        else:
            # Iterate through extracted faces and corresponding find results
            for i, face_obj in enumerate(extracted_faces):
                try:
                    facial_area_model = FacialArea(**face_obj["facial_area"])
                    
                    # Process the DataFrame for the i-th face
                    current_matches_df = blacklist_matches_list[i]
                    current_matches: List[BlacklistMatch] = []
                    if not current_matches_df.empty:
                        # Convert DataFrame rows to list of dicts, then validate with Pydantic
                        match_dicts = current_matches_df.to_dict('records')
                        current_matches = [
                            BlacklistMatch.model_validate(match_dict)
                            for match_dict in match_dicts
                        ]

                    image_faces.append(DetectedFaceResult(
                        face_index=i,
                        facial_area=facial_area_model,
                        confidence=face_obj["confidence"],
                        blacklist_matches=current_matches
                    ))
                except Exception as model_error:
                     logging.error(f"Error creating Pydantic models for face {i}: {model_error}")
                     # Append partial result or set overall error?
                     error_msg = f"Error processing result data for face {i}."
                     # Add face with empty matches to indicate partial failure
                     try: # Safeguard model creation even in error path
                         facial_area_model = FacialArea(**face_obj["facial_area"])
                         image_faces.append(DetectedFaceResult(
                            face_index=i,
                            facial_area=facial_area_model,
                            confidence=face_obj.get("confidence", 0.0),
                            blacklist_matches=[]
                         ))
                     except: pass # Ignore if facial area itself is bad

    except Exception as e:
        # Use logging.exception to include the full traceback
        logging.exception(f"Unhandled error processing image '{img_input[:50]}...': {e}") # Log full traceback
        error_msg = f"An unexpected error occurred processing this image."

    # --- C. Construct Final Result Object --- 
    result_obj = ImageProcessingResult(
        image_path_or_identifier=img_input[:100] + ("..." if len(img_input) > 100 else ""), # Truncate long base64
        faces=image_faces,
        error=error_msg
    )
    
    # --- D. Log Result to Database --- 
    if saved_image_path: # Only log if image was successfully saved
        await processed_image_crud.add_processed_image(
            input_identifier=img_input[:1024], # Original identifier (truncated)
            saved_image_path=saved_image_path, # Path where copy was saved
            result=result_obj # The full result object
        )
    else:
         log.warning(f"Skipping DB log for '{img_input[:50]}...' because image saving failed earlier.")

    return result_obj

@router.post("/process-images", response_model=List[ImageProcessingResult])
async def process_images_endpoint(request: ProcessImagesRequest = Body(...)):
    """
    Processes a list of images (paths, URLs, or base64 strings).
    For each image, extracts faces, gets coordinates, and checks against the blacklist.
    Returns a list of results, one for each input image.
    """
    if not request.images:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No images provided.")

    # --- Option 1: Sequential Processing (Simpler, blocks worker for longer) ---
    # final_results: List[ImageProcessingResult] = []
    # for img_input in request.images:
    #     result = await process_single_image(img_input, request)
    #     final_results.append(result)

    # --- Option 2: Concurrent Processing using asyncio.gather (More complex, better for I/O) ---
    # Note: DeepFace calls are CPU-bound. Running them concurrently on a single
    #       worker via asyncio won't speed up the CPU part, but can help if
    #       there's I/O involved (like URL downloads in resolve_image_input).
    #       True CPU parallelism requires multiple processes (e.g., multiprocessing).
    tasks = [process_single_image(img_input, request) for img_input in request.images]
    try:
        final_results = await asyncio.gather(*tasks)
    except Exception as e:
        # If one task fails uncaught, gather might raise it.
        logging.error(f"Error during concurrent image processing: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during batch processing.")

    return final_results

# --- Example: Optional endpoint for single image processing ---
# @router.post("/process-single-image", response_model=ImageProcessingResult)
# async def process_single_image_endpoint(request: ProcessImagesRequest = Body(...)):
#     """ Processes a single image provided in a list format. """
#     if not request.images or len(request.images) != 1:
#         raise HTTPException(status_code=400, detail="Please provide exactly one image in the 'images' list.")
#     result = await process_single_image(request.images[0], request)
#     return result 