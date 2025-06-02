"""
Plate Detection Module using AlpAPI Integration

This module integrates with the Plate Recognizer API (AlpAPI) to detect and recognize license plates in images.

Setup Requirements:
1. Add the following environment variables to your .env file:
   - ALP_API_KEY: Your Plate Recognizer API key
   - ALP_BASE_URL: Base URL for the API (default: https://api.platerecognizer.com)
   - PLATE_DETECTION_CONFIDENCE_THRESHOLD: Minimum confidence threshold (default: 0.70)
   - CROPPED_PLATES_OUTPUT_DIR: Directory to save cropped plate images

2. The API supports region specification for better accuracy:
   - Include 'regions' parameter in your request with region codes like ["mx", "us-ca"]

3. Response includes:
   - Detected plate coordinates (x, y, width, height)
   - Confidence score
   - Recognized plate text/number

Example usage:
POST /process-images
{
    "images": ["path/to/image.jpg"],
    "code": "test",
    "app_type": "plate",
    "regions": ["mx", "us-ca"]
}
"""

import logging
import requests
from typing import List, Dict, Tuple, Optional

# Import models and CRUD functions
from models import (ProcessImagesRequest, 
                    DetectPlatesResponseItem, 
                    PlateArea, 
                    PlateImageProcessingResult
                    )
from crud import common
from crud import processed_image_crud

from config import settings


log = logging.getLogger(__name__)


class AlpAPI:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
    
    def process_plate_recognition(self, image_data: bytes, regions: list = None) -> Optional[Tuple[str, Dict[str, int], float]]:
        """
        Process image through the plate recognition API and return plate number, coordinates and score.
        
        Args:
            image_data: Bytes of the image file
            regions: Optional list of region codes (e.g., ["mx", "us-ca"])
            
        Returns:
            Tuple containing (plate_number, coordinates_dict, score) or None if no plate was found
        """
        try:
            data = {}
            if regions:
                data['regions'] = regions
                
            response = requests.post(
                f'{self.base_url}/v1/plate-reader/',
                data=data,
                files={'upload': image_data},
                headers={'Authorization': f'Token {self.api_key}'}
            )
            
            response.raise_for_status()
            result = response.json()
            
            log.info(f"AlpAPI response: {result}")
            if result['results'] and len(result['results']) > 0:
                first_result = result['results'][0]
                plate_number = first_result['plate']
                coordinates = first_result['box']
                score = first_result['score']
                
                return plate_number, coordinates, score
                
            return None
            
        except Exception as e:
            log.error(f"Error processing plate recognition: {str(e)}")
            return None


async def process_single_plate_image(img_input: str, request_params: ProcessImagesRequest) -> PlateImageProcessingResult:
    """
    Detects plates in a single image using the AlpAPI plate recognition service.
    Returns a list of detected plates with coordinates and confidence.
    """
    log.info(f"Received request for plate detection. Parameters: {request_params}")

    try:
        # --- A. Save a copy of the incoming image --- 
        saved_image_path = await common.save_incoming_image(img_input)
        if not saved_image_path:
            error_msg = "Failed to save or process input image."
            result_obj = PlateImageProcessingResult(
                 image_path_or_identifier=img_input[:100] + ("..." if len(img_input) > 100 else ""),
                 plates=[], # Empty plates list
                 error=error_msg,
                 saved_image_path=None,
                 cropped_plate_path=None
            )
            # Skipping DB logging if save failed
            return result_obj

    except Exception as e:
        log.error(f"Error processing plate detection: {e}")
        result_obj = PlateImageProcessingResult(
            image_path_or_identifier=img_input[:100] + ("..." if len(img_input) > 100 else ""),
            plates=[],
            error=f"Error processing image: {str(e)}",
            saved_image_path=None,
            cropped_plate_path=None
        )
        return result_obj
      
    # Initialize AlpAPI with configuration
    alp_api = AlpAPI(
        api_key=settings.ALP_API_KEY,  # Add this to your settings
        base_url=settings.ALP_BASE_URL  # Add this to your settings
    )
    
    # Read the saved image file as bytes
    try:
        with open(saved_image_path, 'rb') as image_file:
            image_data = image_file.read()
    except Exception as e:
        log.error(f"Error reading saved image file: {e}")
        return PlateImageProcessingResult(
            image_path_or_identifier=img_input[:100] + ("..." if len(img_input) > 100 else ""),
            plates=[],
            saved_image_path=saved_image_path,
            cropped_plate_path=None,
            error=f"Error reading image file: {str(e)}"
        )

    # Process plate recognition
    detection_regions = getattr(request_params, 'regions', ["mx", "us-ca"])  # Default regions
    plate_result = alp_api.process_plate_recognition(
        image_data=image_data,
        regions=detection_regions
    )

    if not plate_result:
        log.info("No plates detected in the provided image.")
        
        return PlateImageProcessingResult(
            image_path_or_identifier=img_input[:100] + ("..." if len(img_input) > 100 else ""),
            plates=[],
            saved_image_path=saved_image_path,
            cropped_plate_path=None,
            error="No plates detected in the provided image."
        )

    # Extract plate data from API result
    plate_number, coordinates, confidence_score = plate_result
    
    # Map the raw results to the response model
    response_items: List[DetectPlatesResponseItem] = []
    
    try:
        # Convert AlpAPI coordinates to PlateArea format
        # AlpAPI returns coordinates as {xmin, ymin, xmax, ymax}
        plate_area_obj = PlateArea(
            x=coordinates['xmin'],
            y=coordinates['ymin'], 
            w=coordinates['xmax'] - coordinates['xmin'],
            h=coordinates['ymax'] - coordinates['ymin']
        )

        # Only add if confidence meets threshold
        if confidence_score and confidence_score >= settings.PLATE_DETECTION_CONFIDENCE_THRESHOLD:
            response_items.append(
                DetectPlatesResponseItem(
                    plate_area=plate_area_obj,
                    confidence=confidence_score,
                    plate_text=plate_number
                )
            )
        else:
            log.warning(f"Skipping plate due to low confidence: {confidence_score}")

    except Exception as item_err:
        log.error(f"Error processing detected plate data: {item_err}. Data: {plate_result}")

    log.info(f"Successfully detected {len(response_items)} plates.")
        
      
    # --- D. Construct Final Result Object --- 
    final_cropped_plate_path = None # Initialize path variable
    
    # --- Attempt Cropping --- 
    if saved_image_path and response_items and response_items[0].plate_area:
        plate_area_to_crop = response_items[0].plate_area
        if isinstance(plate_area_to_crop, PlateArea):
            log.info(f"Attempting to crop plate from {saved_image_path}")
            try:
                # Call cropping utility
                final_cropped_plate_path = common.crop_and_save_object(
                    original_image_path=saved_image_path,
                    object_coords=plate_area_to_crop,
                    output_dir=settings.CROPPED_PLATES_OUTPUT_DIR
                )
            except Exception as crop_err:
                log.error(f"Failed to execute cropping function: {crop_err}")
                # Leave final_cropped_plate_path as None
        else:
             log.warning("Cannot crop plate: plate area data is not available or invalid.")
             
    # If cropping wasn't attempted or failed, use the original saved path as per requirement
    if final_cropped_plate_path is None:
         final_cropped_plate_path = saved_image_path 
         
    # --- Attempt to Draw Bounding Box if Enabled --- 
    if settings.DRAW_BOUNDING_BOXES and saved_image_path and len(response_items) > 0:
        log.info(f"Drawing bounding box: {response_items}")
        if isinstance(response_items[0].plate_area, PlateArea):
            log.info(f"Attempting to draw bounding box on {saved_image_path}")
            try:
                # Call drawing function
                common.draw_bounding_box_on_image(
                    image_path=saved_image_path,
                    box_coords=response_items[0].plate_area,
                    match_status=False
                )
            except Exception as draw_err:
                log.error(f"Failed to execute drawing function: {draw_err}")
    elif settings.DRAW_BOUNDING_BOXES:
        log.info("Skipping bounding box drawing: Flag enabled but no matches or plate area found.")
         
    result_obj = PlateImageProcessingResult(
        image_path_or_identifier=img_input[:25] + ("..." if len(img_input) > 100 else ""),
        plates=response_items,
        error=None,
        saved_image_path=saved_image_path,
        cropped_plate_path=final_cropped_plate_path
    )
    
    # --- E. Log Result to Database --- 
    if saved_image_path:
        await processed_image_crud.add_processed_image(
            input_identifier=img_input[:25] + ("..." if len(img_input) > 100 else ""),
            saved_image_path=saved_image_path,
            code=request_params.code,
            app_type=request_params.app_type,
            result=result_obj,
            cropped_path=final_cropped_plate_path
        )
    else:
        log.warning(f"Skipping DB log for '{img_input[:25]}...' because image saving failed.")

    return result_obj
