from fastapi import APIRouter, HTTPException, Body, status
from typing import List, Optional
# import asyncio # For potential parallel processing
import logging
from tqdm import tqdm

from models import (ProcessImagesRequest, ImageProcessingResult,
                      DetectedFaceResult, FacialArea, BlacklistMatch,
                      ProcessedImageRecord, PaginatedProcessedImagesResponse)
                      
# Import the new service layer
from services import face_processing_service
# Keep face_crud for helper functions like save_incoming_image
from crud import face_crud 
from crud import processed_image_crud # Import the new CRUD module
# Keep config imports if needed, although service layer might handle them
from config import (
    settings # Import settings instead
)
# Import the specific model for type hinting
from models import FacialArea 


# --- Endpoint to fetch processed results (Added based on README update) --- 
from models import PaginatedProcessedImagesResponse # Import the pagination model

router = APIRouter()
log = logging.getLogger(__name__)

async def process_single_image(img_input: str, request_params: ProcessImagesRequest) -> ImageProcessingResult:
    """Helper function to process a single image (path, url, or base64)."""
    image_faces_results: List[DetectedFaceResult] = [] # Renamed for clarity
    error_msg: Optional[str] = None
    saved_image_path: Optional[str] = None
    first_match_face_area: Optional[FacialArea] = None # Initialize here

    try:
        
        # First check if face exists in the image
        log.info(f"Extracting faces from image: {img_input[:10]}... with detector: {request_params.detector_backend or settings.DETECTOR_BACKEND}")
        detected_faces_data = face_crud.extract_faces_from_image(
            img_data=img_input,
            detector_backend=request_params.detector_backend or settings.DETECTOR_BACKEND,
            align=False, # Alignment not typically needed just for detection boxes
            enforce_detection=False # Return empty list if no face found
        )

        if not detected_faces_data:
            log.info("No faces detected in the provided image.")
            return []
        
        faces = [face['facial_area'] for face in detected_faces_data]
        
        log.info(f"Detected {len(detected_faces_data)} with facials {faces} faces in image: {img_input[:10]}...")
        # Map the raw results to the response model
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
                        log.warning(f"Could not validate facial_area for face {i}: {area_val_err}. Area data: {facial_area_data}")
                        pass
                else:
                    log.warning(f"Facial area data missing or invalid type for face {i}")

                log.info(f"Face {i} confidence score: {confidence_score}, result: {face_area_obj}")
                # Only add if we have a valid facial area
                if face_area_obj and confidence_score and confidence_score > settings.DETECTION_CONFIDENCE_THRESHOLD:
                    # image_faces_results.append(
                    #     DetectedFaceResult(
                    #         face_index=i,
                    #         facial_area=face_area_obj,
                    #         confidence=confidence_score
                    #     )
                    # )
                    break
                else:
                    log.warning(f"Skipping face {i} in response due to missing/invalid facial_area or confidence score.")
                    result_obj = ImageProcessingResult(
                        image_path_or_identifier=img_input[:100] + ("..." if len(img_input) > 100 else ""),
                        faces=[], # Empty faces list
                        error=error_msg,
                                saved_image_path=None, # Indicate save failed
                                has_blacklist_match=False # Default
                            )
                    return result_obj

            except Exception as item_err:
                log.error(f"Error processing detected face data item {i}: {item_err}. Data: {face_data}")
                # Optionally skip this item or raise a more specific error?
                
        # --- A. Save a copy of the incoming image --- 
        log.info(f"Saving image: {img_input[:10]}...")
        saved_image_path = await face_crud.save_incoming_image(img_input)
        if not saved_image_path:
            error_msg = "Failed to save or process input image."
            result_obj = ImageProcessingResult(
                 image_path_or_identifier=img_input[:100] + ("..." if len(img_input) > 100 else ""),
                 faces=[], # Empty faces list
                 error=error_msg,
                 saved_image_path=None, # Indicate save failed
                 has_blacklist_match=False # Default
            )
            # Skipping DB logging if save failed
            return result_obj

        # --- B. Process using the Service Layer --- 
        logging.info(f"Finding blacklist matches via service layer for image: {img_input[:50]}...")
        
        # Call the service layer function to find matches
        # It handles backend switching and returns a consistent List[Dict] format
        matches = await face_processing_service.find_blacklist_matches(
            img_data=img_input, # Pass original identifier
            threshold=request_params.threshold  or settings.BLACKLIST_CONFIDENCE_THRESHOLD # Pass optional threshold override
        )

        # --- C. Construct Response from Matches --- 
        # The service function returns matches directly. We need to structure them
        # into the DetectedFaceResult format. This might require adjustment
        # as find_matches doesn't inherently group by detected face index like
        # the previous logic combining extract_faces and find.
        
        # Simplification: Assume the API contract requires *matches* found in the image,
        # not necessarily a breakdown per *detected* face if multiple faces exist.
        # The current model `ImageProcessingResult` has a `faces: List[DetectedFaceResult]`
        # field. Let's adapt: Create *one* DetectedFaceResult if matches are found,
        # containing all matches. This is a potential change to the exact response meaning.
        # If *strict* per-face results are needed, the service layer needs enhancement.
        
        has_match_flag = False
        if matches:
            # Create BlacklistMatch models from the dictionaries returned by the service
            validated_matches: List[BlacklistMatch] = []
            for match_dict in matches:
                try:
                     validated_matches.append(BlacklistMatch.model_validate(match_dict))
                     has_match_flag = True # Mark if any valid match exists
                except Exception as model_error:
                     log.error(f"Error validating BlacklistMatch model for match data: {match_dict}. Error: {model_error}")
                     # Optionally include partial results or skip this match
                     
            # Extract bounding box from the first match (assuming it represents the main matched face)
            if validated_matches: # Check if we have at least one validated match
                first_match_data = matches[0] # Get raw dict again for coords
                try:
                    # Attempt to create FacialArea from the first match's source coordinates
                    x_coord = first_match_data.get('source_x')
                    y_coord = first_match_data.get('source_y')
                    w_coord = first_match_data.get('source_w')
                    h_coord = first_match_data.get('source_h')
                    
                    # Only create FacialArea if all coordinates are present (not None)
                    if all(coord is not None for coord in [x_coord, y_coord, w_coord, h_coord]):
                         first_match_face_area = FacialArea(
                             x=x_coord,
                             y=y_coord,
                             w=w_coord,
                             h=h_coord,
                         )
                    else:
                        log.warning("Could not create FacialArea from first match data as coordinates were missing.")
                except Exception as area_error:
                    log.error(f"Error creating FacialArea from match data {first_match_data}: {area_error}")
                    
            # Create a single DetectedFaceResult encompassing all matches found
            # Confidence is not directly available from search results, set to None
            detected_face = DetectedFaceResult(
                 face_index=0, 
                 facial_area=first_match_face_area, 
                 confidence=None, # Confidence isn't directly available here
                 blacklist_matches=validated_matches
            )
            image_faces_results.append(detected_face)
        else:
             log.info(f"No blacklist matches found for image: {img_input[:50]}...")
             # Keep image_faces_results empty
             
    except Exception as e:
        logging.exception(f"Unhandled error processing image '{img_input[:50]}...': {e}")
        error_msg = f"An unexpected error occurred processing this image."

    # --- D. Construct Final Result Object --- 
    final_cropped_face_path = None # Initialize path variable
    log.info(f"image_faces_results: {image_faces_results}")
    # --- Attempt Cropping --- 
    if saved_image_path and image_faces_results and image_faces_results[0].facial_area:
        face_area_to_crop = image_faces_results[0].facial_area
        if isinstance(face_area_to_crop, FacialArea):
            log.info(f"Attempting to crop face from {saved_image_path}")
            try:
                # Call cropping utility
                final_cropped_face_path = face_crud.crop_and_save_face(
                    original_image_path=saved_image_path,
                    face_coords=face_area_to_crop,
                    output_dir=settings.CROPPED_FACES_OUTPUT_DIR
                )
            except Exception as crop_err:
                log.error(f"Failed to execute cropping function: {crop_err}")
                # Leave final_cropped_face_path as None
        else:
             log.warning("Cannot crop face: facial area data is not available or invalid.")
             
    # If cropping wasn't attempted or failed, use the original saved path as per requirement
    if final_cropped_face_path is None:
         final_cropped_face_path = saved_image_path 
         
    # --- Attempt to Draw Bounding Box if Enabled --- 
    if settings.DRAW_BOUNDING_BOXES and saved_image_path and first_match_face_area:
        # Check if first_match_face_area is actually a FacialArea instance
        if isinstance(first_match_face_area, FacialArea):
                log.info(f"Attempting to draw bounding box on {saved_image_path}")
                try:
                    # Call drawing function (fire and forget, or await if made async)
                    # Making it async requires changes in face_crud and here
                    # For now, run synchronously within the async endpoint
                    face_crud.draw_bounding_box_on_image(
                        image_path=saved_image_path,
                        box_coords=first_match_face_area,
                        match_status=has_match_flag
                    )
                except Exception as draw_err:
                    # Use the module-level logger
                    log.error(f"Failed to execute drawing function: {draw_err}")
        else:
                log.warning("Cannot draw bounding box: facial area data is not available or invalid.")
    elif settings.DRAW_BOUNDING_BOXES:
            log.info("Skipping bounding box drawing: Flag enabled but no matches or facial area found.")
         
    result_obj = ImageProcessingResult(
        image_path_or_identifier=img_input[:100] + ("..." if len(img_input) > 100 else ""),
        faces=image_faces_results, # Use the potentially single result
        error=error_msg,
        saved_image_path=saved_image_path, # Include path to saved image
        has_blacklist_match=has_match_flag, # Set flag based on if matches were found
        cropped_face_path=final_cropped_face_path # Add the determined path
    )
    
    # --- E. Log Result to Database --- 
    if saved_image_path: 
        await processed_image_crud.add_processed_image(
            input_identifier=img_input[:100] + ("..." if len(img_input) > 100 else ""), # Original identifier (truncated)
            saved_image_path=saved_image_path, # Path where copy was saved
            result=result_obj, # The full result object (includes cropped path now)
            cropped_face_path=final_cropped_face_path # Pass path explicitly to DB function
        )
    else:
        # This case is handled earlier, but included defensively
        logging.warning(f"Skipping DB log for '{img_input[:50]}...' because image saving failed.")

    return result_obj

@router.post("/process-images", response_model=List[ImageProcessingResult])
async def process_images_endpoint(request: ProcessImagesRequest = Body(...)):
    """
    Processes a list of images (paths, URLs, or base64 strings) using the configured backend.
    For each image, finds blacklist matches and returns results.
    """
    if not request.images:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No images provided.")

    final_results: List[ImageProcessingResult] = []
    # Using sequential processing for simplicity now.
    # Consider asyncio.gather or background tasks for production.
    log.info(f"Processing {len(request.images)} images sequentially...")
    for img_input in tqdm(request.images, desc="Processing Images"):
        result = await process_single_image(img_input, request)
        final_results.append(result)
    log.info(f"Finished processing {len(request.images)} images.")
    
    return final_results


@router.get("/processed-images/", response_model=PaginatedProcessedImagesResponse)
async def get_processed_images_results(
    offset: int = 0,
    limit: int = 10 # Default limit consistent with Gradio app?
):
    """Retrieves previously processed image results from the database with pagination."""
    if offset < 0:
        raise HTTPException(status_code=400, detail="Offset cannot be negative")
    if limit < 1:
        raise HTTPException(status_code=400, detail="Limit must be at least 1")
        
    results = await processed_image_crud.get_processed_images(offset=offset, limit=limit)
    total = await processed_image_crud.count_processed_images()
    
    # Validate results against the Pydantic model used for DB storage/retrieval
    # Assuming get_processed_images returns list of DB records (RowProxy/dict)
    # Each 'item' has fields like id, processing_timestamp, result_data (JSON string)
    validated_items = []
    for item in results:
        if item.result_data:
            try:
                # Reconstruct the original ImageProcessingResult from the stored JSON
                # Use model_validate_json as item.result_data is likely a string
                result_obj = ImageProcessingResult.model_validate_json(item.result_data)

                # Create the final ProcessedImageRecord for the response
                # Combine data from the DB row and the reconstructed result_obj
                record = ProcessedImageRecord(
                    db_id=item.id, # Get ID from the raw DB row
                    processing_timestamp=item.processing_timestamp, # Get timestamp from raw row
                    # Fields from the reconstructed result_obj
                    image_path_or_identifier=result_obj.image_path_or_identifier,
                    faces=result_obj.faces,
                    error=result_obj.error,
                    saved_image_path=result_obj.saved_image_path, 
                    has_blacklist_match=result_obj.has_blacklist_match,
                    # Use the value directly from the DB column for the response
                    cropped_face_path=item.cropped_face_path 
                )
                validated_items.append(record)
            except Exception as e:
                log.error(f"Error validating result_data for DB record ID {item.id}: {e}")
                # Optionally skip or add placeholder

    # Construct the paginated response
    return PaginatedProcessedImagesResponse(
        total_items=total,
        items=validated_items, # This is now List[ProcessedImageRecord]
        offset=offset,
        limit=limit
    ) 