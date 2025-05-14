import logging

from typing import List

# Import models and CRUD functions
from models import (ProcessImagesRequest, 
                    DetectWeaponsResponseItem, 
                    WeaponArea, 
                    WeaponImageProcessingResult
                    )
from crud import weapons_crud
from crud import common
from crud import processed_image_crud

from config import settings


log = logging.getLogger(__name__)

async def process_single_weapons_image(img_input: str, request_params: ProcessImagesRequest) -> WeaponImageProcessingResult:
    """
    Detects weapons in a single image using the configured DeepFace detector backend.
    Returns a list of detected weapons with coordinates and confidence.
    """
    # log.info(f"Received request for weapons detection. Detector: {request_params }")

    try:

        # --- A. Save a copy of the incoming image --- 
        saved_image_path = await common.save_incoming_image(img_input)
        if not saved_image_path:
            error_msg = "Failed to save or process input image."
            result_obj = WeaponImageProcessingResult(
                 image_path_or_identifier=img_input[:100] + ("..." if len(img_input) > 100 else ""),
                 weapons=[], # Empty faces list
                 error=error_msg,

            )
            # Skipping DB logging if save failed
            return result_obj

    
    except Exception as e:
        log.error(f"Error processing weapons detection: {e}")
        return []
      
    # Use the extract_faces function which primarily performs detection
    # Use detector from request if provided, otherwise fallback to config
    detected_weapons_data = weapons_crud.extract_weapons_from_image(
        img_data=saved_image_path
    )

    if len(detected_weapons_data) == 0:
        log.info("No weapons detected in the provided image.")
        
        return WeaponImageProcessingResult(
            image_path_or_identifier=img_input[:100] + ("..." if len(img_input) > 100 else ""),
            weapons=[],
            error="No weapons detected in the provided image."
        )

    # Map the raw results to the response model
    response_items: List[DetectWeaponsResponseItem] = []
    
    for i, weapon_data in enumerate(detected_weapons_data):
        try:
            weapon = weapon_data.get('objects')[0]
            print(f"weapon: {weapon}")
            weapon_area_data = weapon.get('weapon_area')
            confidence_score = weapon.get('confidence')

            # Validate/create WeaponArea model
            weapon_area_obj = None
            if isinstance(weapon_area_data, dict):
                weapon_area_obj = WeaponArea.model_validate(weapon_area_data)
            else:
                log.warning(f"Weapon area data missing or invalid type for weapon {i}")

            # Only add if we have a valid weapon area
            if weapon_area_obj and confidence_score and confidence_score >= settings.WEAPON_DETECTION_CONFIDENCE_THRESHOLD:
                
                response_items.append(
                    DetectWeaponsResponseItem(
                        weapon_area=weapon_area_obj,
                        confidence=confidence_score
                    )
                )
            else:
                log.warning(f"Skipping face {i} in response due to missing/invalid weapon_area.")

        except Exception as item_err:
            log.error(f"Error processing detected face data item {i}: {item_err}. Data: {weapon_data}")
            # Optionally skip this item or raise a more specific error?

    log.info(f"Successfully detected {len(response_items)} faces.")
        
      
    # --- D. Construct Final Result Object --- 
    final_cropped_weapon_path = None # Initialize path variable
    # log.info(f"image_weapons_results: {response_items}")
    
    # --- Attempt Cropping --- 
    if saved_image_path and response_items and response_items[0].weapon_area:
        weapon_area_to_crop = response_items[0].weapon_area
        if isinstance(weapon_area_to_crop, WeaponArea):
            log.info(f"Attempting to crop face from {saved_image_path}")
            try:
                # Call cropping utility
                final_cropped_weapon_path = common.crop_and_save_object(
                    original_image_path=saved_image_path,
                    object_coords=weapon_area_to_crop,
                    output_dir=settings.CROPPED_WEAPONS_OUTPUT_DIR
                )
            except Exception as crop_err:
                log.error(f"Failed to execute cropping function: {crop_err}")
                # Leave final_cropped_face_path as None
        else:
             log.warning("Cannot crop face: facial area data is not available or invalid.")
             
    # If cropping wasn't attempted or failed, use the original saved path as per requirement
    if final_cropped_weapon_path is None:
         final_cropped_weapon_path = saved_image_path 
         
    # --- Attempt to Draw Bounding Box if Enabled --- 
    if settings.DRAW_BOUNDING_BOXES and saved_image_path and len(response_items) > 0:
        print(f"drawing bounding box: {response_items}")
        # Check if first_match_face_area is actually a FacialArea instance
        if isinstance(response_items[0].weapon_area, WeaponArea):
            log.info(f"Attempting to draw bounding box on {saved_image_path}")
            try:
                # Call drawing function (fire and forget, or await if made async)
                # Making it async requires changes in face_crud and here
                # For now, run synchronously within the async endpoint
                common.draw_bounding_box_on_image(
                    image_path=saved_image_path,
                    box_coords=response_items[0].weapon_area,
                    match_status=False
                )
            except Exception as draw_err:
                # Use the module-level logger
                log.error(f"Failed to execute drawing function: {draw_err}")
    elif settings.DRAW_BOUNDING_BOXES:
        log.info("Skipping bounding box drawing: Flag enabled but no matches or facial area found.")
         
    result_obj = WeaponImageProcessingResult(
        image_path_or_identifier=img_input[:25] + ("..." if len(img_input) > 100 else ""),
        weapons=response_items, # Use the potentially single result
        error=None,
        saved_image_path=saved_image_path, # Include path to saved image
        cropped_weapon_path=final_cropped_weapon_path # Add the determined path
    )
    
    # --- E. Log Result to Database --- 
    if saved_image_path:
        await processed_image_crud.add_processed_image(
            input_identifier=img_input[:25] + ("..." if len(img_input) > 100 else ""), # Original identifier (truncated)
            saved_image_path=saved_image_path, # Path where copy was saved
            code=request_params.code,
            app_type=request_params.app_type,
            result=result_obj, # The full result object (includes cropped path now)
            cropped_path=final_cropped_weapon_path # Pass path explicitly to DB function
        )
    else:
        # This case is handled earlier, but included defensively
        logging.warning(f"Skipping DB log for '{img_input[:25]}...' because image saving failed.")

    return result_obj
