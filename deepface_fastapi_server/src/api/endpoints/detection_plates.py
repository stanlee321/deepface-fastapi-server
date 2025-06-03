"""
Plate Detection Module using AlpAPI Integration with Batch Processing, Global Deduplication, and Smart Early Exit

This module integrates with the Plate Recognizer API (AlpAPI) to detect and recognize license plates in images.
After collecting enough detections, it sends only the highest confidence detection to the parking management API.

KEY FEATURES: 
1. Batch Processing with Best Selection
2. Global Plate Deduplication (prevents same plate from being sent multiple times EVER)
3. Smart Early Exit (skips expensive processing when code already completed)

PROCESSING FLOW WITH OPTIMIZATIONS:
1. CHECK if this code already has a plate sent to parking API
   - IF YES: Skip AlpAPI processing, save image with "skipped" status, return immediately
   - IF NO: Continue with normal processing
2. Detect plate using AlpAPI (only if not already processed)
3. SAVE detection to database (parking_api_sent=false initially)
4. COUNT total detections for this 'code'
5. IF we have >= BATCH_DETECTION_THRESHOLD detections:
   - FIND the highest confidence detection for this code
   - CHECK if this plate was ever sent to parking API globally (across ALL codes)
   - IF never sent globally: SEND it to parking API
   - IF already sent globally: BLOCK (prevent duplicate)
   - MARK it as parking_api_sent=true
6. IF we don't have enough detections yet:
   - WAIT for more detections
   - NO sending to parking API yet

PERFORMANCE OPTIMIZATIONS:
- **Early Exit**: Once a code has a sent plate, all subsequent images for that code skip AlpAPI processing
- **Cost Savings**: Reduces expensive AlpAPI calls by up to 75% (if processing 20 images but first 4 already found best plate)
- **Speed**: Much faster processing of remaining images (just save, no detection)
- **Global Deduplication**: Fast indexed lookups on dedicated sent_plates table

GLOBAL DEDUPLICATION:
- Once a plate (e.g., "2843ATS") is sent to parking API, it will NEVER be sent again
- This applies across ALL processing sessions and codes
- Prevents true duplicates in the parking system

Setup Requirements:
1. Add the following environment variables to your .env file:
   - ALP_API_KEY: Your Plate Recognizer API key
   - ALP_BASE_URL: Base URL for the API (default: https://api.platerecognizer.com)
   - PLATE_DETECTION_CONFIDENCE_THRESHOLD: Minimum confidence threshold (default: 0.85)
   - CROPPED_PLATES_OUTPUT_DIR: Directory to save cropped plate images

2. Parking API Integration (optional):
   - SEND_TO_PARKING_API: Enable/disable parking API calls (default: true)
   - PARKING_API_URL: Parking API endpoint (default: https://backend-vialika.vercel.app/api/v1/parkings)
   - PARKING_SOURCE: Source identifier (default: "camera")
   - PARKING_LATITUDE: Location latitude (default: -17.393398)
   - PARKING_LONGITUDE: Location longitude (default: -66.248857)

3. Batch Processing Configuration:
   - BATCH_DETECTION_THRESHOLD: Number of detections needed before processing (default: 4)

4. The API supports region specification for better accuracy:
   - Include 'regions' parameter in your request with region codes like ["mx", "us-ca"]

5. Response includes:
   - Detected plate coordinates (x, y, width, height)
   - Confidence score
   - Recognized plate text/number
   - parking_api_sent: Whether this detection was sent to parking API
   - superseded_by_better_detection: Whether this detection was superseded by a better one

6. Database Tracking:
   - All detections are saved with complete audit trail
   - parking_api_sent tracks which detections were actually sent
   - superseded_by_better_detection tracks which were replaced by better ones
   - sent_plates table tracks all plates sent to parking API for fast duplicate checking

Example Optimized Workflow:
Session - Code: "W4FQ394J_3" (20 images)
- Images 1-4: Process with AlpAPI → Find "2843ATS" conf=1.0 → Send to parking API ✅
- Images 5-20: Skip AlpAPI processing ⚡ → Save with "skipped" status
- Result: 75% reduction in AlpAPI calls, much faster processing!

Result: Maximum efficiency with zero duplicates and minimal API usage.
"""

import logging
import requests
import time
from typing import List, Dict, Tuple, Optional

# Import models and CRUD functions
from models import (ProcessImagesRequest, 
                    DetectPlatesResponseItem, 
                    PlateArea, 
                    PlateImageProcessingResult
                    )
from crud import common
from crud import processed_image_crud
from crud import sent_plates_crud

from config import settings

# FastAPI endpoint to view sent plates (for debugging/monitoring)
from fastapi import APIRouter
from typing import List

router = APIRouter()

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
            
            time.sleep(2)
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


class ParkingAPI:
    def __init__(self, api_url: str, source: str, latitude: float, longitude: float):
        self.api_url = api_url
        self.source = source
        self.latitude = latitude
        self.longitude = longitude
    
    def send_plate_to_parking(self, license_plate: str, confidence: float, internal_code: str) -> bool:
        """
        Send detected license plate information to the parking API.
        
        Args:
            license_plate: The detected license plate text
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            payload = {
                "licensePlate": license_plate,
                "source": self.source,
                "location": {
                    "latitude": self.latitude,
                    "longitude": self.longitude
                },
                "confidence": confidence,
                "internal_code": internal_code
            }
            
            log.info(f"Sending plate to parking API: {payload}")
            
            response = requests.post(
                self.api_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=10  # 10 second timeout
            )
            
            response.raise_for_status()
            log.info(f"Successfully sent plate {license_plate} to parking API. Response: {response.status_code}")
            return True
            
        except requests.exceptions.Timeout:
            log.error(f"Timeout sending plate {license_plate} to parking API")
            return False
        except requests.exceptions.RequestException as e:
            log.error(f"Error sending plate {license_plate} to parking API: {str(e)}")
            return False
        except Exception as e:
            log.error(f"Unexpected error sending plate {license_plate} to parking API: {str(e)}")
            return False


async def process_single_plate_image(img_input: str, request_params: ProcessImagesRequest) -> PlateImageProcessingResult:
    """
    Detects plates in a single image using the AlpAPI plate recognition service.
    Returns a list of detected plates with coordinates and confidence.
    """
    # log.info(f"Received request for plate detection. Parameters: {request_params}")

    # --- EARLY EXIT: Check if this code already has a plate sent to parking API ---
    code_already_processed = await sent_plates_crud.check_code_already_processed(request_params.code)
    if code_already_processed:
        log.info(f"Skipping AlpAPI processing for code '{request_params.code}' - already sent a plate to parking API")
        
        # Still save the image but skip expensive AlpAPI processing
        try:
            saved_image_path = await common.save_incoming_image(img_input)
            if saved_image_path:
                # Save a record indicating this image was skipped due to code already processed
                await processed_image_crud.add_processed_image(
                    input_identifier=img_input[:25] + ("..." if len(img_input) > 100 else ""),
                    saved_image_path=saved_image_path,
                    code=request_params.code,
                    app_type=request_params.app_type,
                    result=PlateImageProcessingResult(
                        image_path_or_identifier=img_input[:25] + ("..." if len(img_input) > 100 else ""),
                        plates=[],
                        error="Skipped: Code already processed and sent to parking API",
                        saved_image_path=saved_image_path,
                        cropped_plate_path=None,
                        parking_api_sent=None,
                        superseded_by_better_detection=None
                    ),
                    cropped_path=None
                )
                
                return PlateImageProcessingResult(
                    image_path_or_identifier=img_input[:25] + ("..." if len(img_input) > 100 else ""),
                    plates=[],
                    error="Skipped: Code already processed and sent to parking API",
                    saved_image_path=saved_image_path,
                    cropped_plate_path=None,
                    parking_api_sent=None,
                    superseded_by_better_detection=None
                )
        except Exception as e:
            log.error(f"Error saving skipped image: {e}")
            
        # Fallback result if saving fails
        return PlateImageProcessingResult(
            image_path_or_identifier=img_input[:25] + ("..." if len(img_input) > 100 else ""),
            plates=[],
            error="Skipped: Code already processed and sent to parking API",
            saved_image_path=None,
            cropped_plate_path=None,
            parking_api_sent=None,
            superseded_by_better_detection=None
        )

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
                 cropped_plate_path=None,
                 parking_api_sent=None,
                 superseded_by_better_detection=None
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
            cropped_plate_path=None,
            parking_api_sent=None,
            superseded_by_better_detection=None
        )
        return result_obj
      
    # Initialize AlpAPI with configuration
    alp_api = AlpAPI(
        api_key=settings.ALP_API_KEY,  # Add this to your settings
        base_url=settings.ALP_BASE_URL  # Add this to your settings
    )
    
    # Initialize ParkingAPI with configuration
    parking_api = None
    if settings.SEND_TO_PARKING_API:
        parking_api = ParkingAPI(
            api_url=settings.PARKING_API_URL,
            source=settings.PARKING_SOURCE,
            latitude=settings.PARKING_LATITUDE,
            longitude=settings.PARKING_LONGITUDE
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
            parking_api_sent=None,
            superseded_by_better_detection=None,
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
            parking_api_sent=None,
            superseded_by_better_detection=None,
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
        
    # --- D. Construct Final Result Object and Handle Cropping --- 
    final_cropped_plate_path = None # Initialize path variable early
    
    # --- Attempt Cropping --- 
    if saved_image_path and response_items and len(response_items) > 0 and response_items[0].plate_area:
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

    # --- Send Plate to Parking API if Enabled and Successful Detection ---
    parking_api_sent = False
    
    # --- E. Log Result to Database FIRST (before parking API logic) --- 
    if saved_image_path:
        current_record_id = await processed_image_crud.add_processed_image(
            input_identifier=img_input[:25] + ("..." if len(img_input) > 100 else ""),
            saved_image_path=saved_image_path,
            code=request_params.code,
            app_type=request_params.app_type,
            result=PlateImageProcessingResult(
                image_path_or_identifier=img_input[:25] + ("..." if len(img_input) > 100 else ""),
                plates=response_items,
                error=None,
                saved_image_path=saved_image_path,
                cropped_plate_path=final_cropped_plate_path,
                parking_api_sent=False,  # Will be updated later
                superseded_by_better_detection=None
            ),
            cropped_path=final_cropped_plate_path
        )
    else:
        log.warning(f"Skipping DB log for '{img_input[:25]}...' because image saving failed.")
        current_record_id = None

    # --- NOW check batch processing logic and send to parking API ---
    if parking_api and response_items and response_items[0].plate_text:
        detected_plate_text = response_items[0].plate_text
        log.info(f"Processing plate '{detected_plate_text}' for batch logic")
        
        # Check if we should process this code for sending (batch-based)
        await process_batch_detection_for_parking_api(
            code=request_params.code,
            current_record_id=current_record_id,
            parking_api=parking_api
        )
        
        # The parking_api_sent status will be updated by the batch processing function
        # Let's check if our current record was marked as sent
        if current_record_id:
            updated_record = await processed_image_crud.get_processed_image_by_id(current_record_id)
            if updated_record:
                import json
                try:
                    result_data = json.loads(updated_record.result_json)
                    parking_api_sent = result_data.get('parking_api_sent', False)
                except (json.JSONDecodeError, KeyError):
                    parking_api_sent = False

    result_obj = PlateImageProcessingResult(
        image_path_or_identifier=img_input[:25] + ("..." if len(img_input) > 100 else ""),
        plates=response_items,
        error=None,
        saved_image_path=saved_image_path,
        cropped_plate_path=final_cropped_plate_path,
        parking_api_sent=parking_api_sent,
        superseded_by_better_detection=None
    )
    
    return result_obj


async def should_send_plate_to_parking_api(plate_text: str, confidence: float, code: str, current_record_id: int = None) -> tuple[bool, bool]:
    """
    Determines if a detected plate should be sent to the parking API based on CODE+PLATE deduplication logic.
    
    CRITICAL IMPROVEMENT: Never send the same plate twice for the same code, even if confidence improves.
    
    LOGIC:
    1. Check if this exact plate has EVER been sent for this code before
    2. If YES → never send again (prevents duplicates)
    3. If NO → send it (first time seeing this plate for this code)
    
    This prevents scenarios like:
    - Code ABC123: "6709UBS" conf=0.977 → send ✅
    - Code ABC123: "6709UBS" conf=0.997 → DON'T send ❌ (already sent this plate)
    - Code ABC123: "6709UBS" conf=0.998 → DON'T send ❌ (already sent this plate)
    - Code ABC123: "DIFFERENT" conf=0.999 → send ✅ (different plate)
    
    Args:
        plate_text: The detected plate text
        confidence: The confidence score of current detection
        code: The internal code for this processing session
        current_record_id: The ID of the current record being processed (to exclude from comparison)
        
    Returns:
        Tuple of (should_send, is_best_so_far) where:
        - should_send: True if this plate should be sent to parking API
        - is_best_so_far: True if this is the best detection for this code so far
    """
    try:
        # Get all previous plate detections for this code
        previous_records = await processed_image_crud.get_processed_images_by_code_and_app_type(
            code=code, 
            app_type="plate"
        )
        
        current_is_best = True
        plate_already_sent_for_code = False  # Track if THIS SPECIFIC PLATE was ever sent for this code
        
        # Check all previous detections for this code
        for record in previous_records:
            # Skip the current record to avoid comparing against itself
            if current_record_id and record.id == current_record_id:
                continue
                
            try:
                import json
                result_data = json.loads(record.result_json)
                
                if result_data.get('plates') and len(result_data['plates']) > 0:
                    previous_plate = result_data['plates'][0]
                    previous_plate_text = previous_plate.get('plate_text', '').upper()
                    previous_confidence = previous_plate.get('confidence', 0.0)
                    previous_sent_to_api = result_data.get('parking_api_sent', False)
                    
                    # CRITICAL CHECK: Has this exact plate ever been sent for this code?
                    if previous_sent_to_api and previous_plate_text == plate_text.upper():
                        plate_already_sent_for_code = True
                        log.info(f"Plate '{plate_text}' was already sent for code '{code}' with confidence {previous_confidence}")
                    
                    # Check if current is still the best overall detection for this code
                    if previous_confidence > confidence:
                        current_is_best = False
                        
            except (json.JSONDecodeError, KeyError) as e:
                log.warning(f"Error parsing previous record result_json: {e}")
                continue
        
        # DECISION LOGIC: Only send if this specific plate has NEVER been sent for this code
        should_send = not plate_already_sent_for_code
        
        if plate_already_sent_for_code:
            log.info(f"BLOCKING duplicate: plate '{plate_text}' for code '{code}' "
                    f"(already sent this plate before, current conf: {confidence})")
        else:
            log.info(f"ALLOWING first send: plate '{plate_text}' for code '{code}' "
                    f"(never sent this plate before, conf: {confidence})")
        
        log.info(f"PLATE+CODE deduplication check: plate='{plate_text}', confidence={confidence}, "
                f"code='{code}', should_send={should_send}, is_best={current_is_best}, "
                f"plate_already_sent={plate_already_sent_for_code}")
        
        return should_send, current_is_best
        
    except Exception as e:
        log.error(f"Error in plate+code deduplication logic: {e}")
        # If there's an error, default to not sending (fail-safe to prevent duplicates)
        return False, True


async def update_previous_parking_api_status(code: str, best_plate_text: str, best_confidence: float, current_saved_path: str):
    """
    Update previous records for the same code to mark them as superseded by better detection.
    
    NOTE: With the new PLATE+CODE deduplication logic, this function is less critical since
    we prevent sending the same plate twice. But we still track superseding for audit purposes.
    
    Args:
        code: The internal code for this processing session
        best_plate_text: The plate text of the best detection
        best_confidence: The confidence of the best detection
        current_saved_path: The path to the current saved image
    """
    try:
        # Get all previous plate detections for this code
        previous_records = await processed_image_crud.get_processed_images_by_code_and_app_type(
            code=code, 
            app_type="plate"
        )
        
        for record in previous_records:
            # Skip the current record (don't mark it as superseded)
            if record.saved_image_path == current_saved_path:
                continue
                
            try:
                import json
                result_data = json.loads(record.result_json)
                
                if result_data.get('plates') and len(result_data['plates']) > 0:
                    previous_plate = result_data['plates'][0]
                    previous_plate_text = previous_plate.get('plate_text', '').upper()
                    previous_confidence = previous_plate.get('confidence', 0.0)
                    
                    # Mark as superseded if this record has lower confidence than current best
                    # (for audit purposes, even though we prevent sending duplicates)
                    if previous_confidence < best_confidence:
                        
                        # Don't change parking_api_sent status - keep original value for audit
                        result_data['superseded_by_better_detection'] = True
                        
                        # Update the record in database
                        await processed_image_crud.update_processed_image_result(
                            record_id=record.id,
                            new_result_json=json.dumps(result_data)
                        )
                        
                        log.info(f"Updated record {record.id}: marked as superseded by better detection "
                                f"(plate '{previous_plate_text}' conf={previous_confidence} < "
                                f"new plate '{best_plate_text}' conf={best_confidence})")
                        
            except (json.JSONDecodeError, KeyError) as e:
                log.warning(f"Error updating previous record result_json: {e}")
                continue
                
    except Exception as e:
        log.error(f"Error updating previous parking API status: {e}")


async def update_current_record_parking_status(code: str, saved_image_path: str, parking_api_sent: bool):
    """
    Update the current record to reflect the parking API status.
    
    Args:
        code: The internal code for this processing session
        saved_image_path: The path to the current saved image
        parking_api_sent: Whether the current detection was sent to parking API
    """
    try:
        # Get the current record for this saved image path
        current_record = await processed_image_crud.get_processed_image_by_saved_path(saved_image_path)
        
        if current_record:
            import json
            # Parse the current result_json
            result_data = json.loads(current_record.result_json)
            
            # Update the parking API status
            result_data['parking_api_sent'] = parking_api_sent
            
            # Update the record in database
            await processed_image_crud.update_processed_image_result(
                record_id=current_record.id,
                new_result_json=json.dumps(result_data)
            )
            
            log.info(f"Updated record {current_record.id}: marked parking_api_sent={parking_api_sent}")
        else:
            log.warning(f"Could not find record with saved_image_path: {saved_image_path}")
            
    except Exception as e:
        log.error(f"Error updating current parking API status: {e}")


async def process_batch_detection_for_parking_api(code: str, current_record_id: int, parking_api: ParkingAPI):
    """
    Process batch detections for a given code and send the highest confidence detection to the parking API
    only when we have enough detections and haven't sent anything for this code yet.
    
    BATCH LOGIC:
    1. Count total detections for this code
    2. If we have >= BATCH_DETECTION_THRESHOLD detections:
       - Check if we already sent something for this code
       - If not sent yet, find the highest confidence detection
       - Send it to parking API
       - Mark it as sent and mark others as not sent
    3. If we don't have enough detections yet, wait for more
    
    Args:
        code: The internal code for this processing session
        current_record_id: The ID of the current record being processed
        parking_api: The ParkingAPI instance for sending detections to the parking API
    """
    try:
        # Get all plate detections for this code
        all_records = await processed_image_crud.get_processed_images_by_code_and_app_type(
            code=code, 
            app_type="plate"
        )
        
        # Count valid detections and check if anything was already sent
        valid_detections = []
        already_sent_for_code = False
        
        for record in all_records:
            try:
                import json
                result_data = json.loads(record.result_json)
                
                if result_data.get('plates') and len(result_data['plates']) > 0:
                    plate = result_data['plates'][0]
                    plate_text = plate.get('plate_text', '').upper()
                    confidence = plate.get('confidence', 0.0)
                    sent_to_api = result_data.get('parking_api_sent', False)
                    
                    # Track if anything was already sent for this code
                    if sent_to_api:
                        already_sent_for_code = True
                        log.info(f"Already sent plate '{plate_text}' for code '{code}' - skipping batch processing")
                        return
                    
                    # Collect valid detections (only if plate is long enough)
                    if len(plate_text) > 5:
                        valid_detections.append({
                            'record_id': record.id,
                            'plate_text': plate_text,
                            'confidence': confidence,
                            'saved_image_path': record.saved_image_path
                        })
                        
            except (json.JSONDecodeError, KeyError) as e:
                log.warning(f"Error parsing record result_json: {e}")
                continue
        
        total_detections = len(valid_detections)
        log.info(f"Code '{code}': Found {total_detections} valid detections (threshold: {settings.BATCH_DETECTION_THRESHOLD})")
        
        # Check if we have enough detections to process the batch
        if total_detections >= settings.BATCH_DETECTION_THRESHOLD:
            log.info(f"Processing batch for code '{code}' ({total_detections} detections)")
            
            # Find the highest confidence detection
            best_detection = max(valid_detections, key=lambda x: x['confidence'])
            best_plate_text = best_detection['plate_text']
            best_confidence = best_detection['confidence']
            best_record_id = best_detection['record_id']
            best_saved_path = best_detection['saved_image_path']
            
            log.info(f"Best detection for code '{code}': plate='{best_plate_text}', confidence={best_confidence}")
            
            # GLOBAL CHECK: Has this plate ever been sent to parking API (across ALL codes)?
            global_already_sent = await sent_plates_crud.check_plate_sent_globally(best_plate_text)
            if global_already_sent:
                log.info(f"GLOBAL BLOCK: Plate '{best_plate_text}' was already sent to parking API in a previous session - skipping")
                return
            
            # Send the highest confidence detection to the parking API
            try:
                parking_success = parking_api.send_plate_to_parking(best_plate_text, best_confidence, code)
                
                if parking_success:
                    log.info(f"Successfully sent plate '{best_plate_text}' to parking API for code '{code}'")
                    
                    # Record the sent plate in our dedicated table
                    await sent_plates_crud.add_sent_plate(
                        plate_text=best_plate_text,
                        code=code,
                        confidence=best_confidence,
                        internal_code=code,
                        parking_api_response="201"  # Assuming success means 201
                    )
                    
                    # Update the best record to indicate it was sent to parking API
                    await update_record_parking_status_by_id(best_record_id, True)
                    
                    # Mark all other records for this code as not sent (for audit purposes)
                    for detection in valid_detections:
                        if detection['record_id'] != best_record_id:
                            await update_record_parking_status_by_id(detection['record_id'], False)
                            await mark_record_as_superseded(detection['record_id'], best_plate_text, best_confidence)
                    
                else:
                    log.warning(f"Failed to send plate '{best_plate_text}' to parking API for code '{code}'")
                    
            except Exception as api_error:
                log.error(f"Error sending plate '{best_plate_text}' to parking API: {api_error}")
                
        else:
            log.info(f"Code '{code}': Waiting for more detections ({total_detections}/{settings.BATCH_DETECTION_THRESHOLD})")
            
    except Exception as e:
        log.error(f"Error processing batch detection for parking API: {e}")


async def update_record_parking_status_by_id(record_id: int, parking_api_sent: bool):
    """Update a specific record's parking API status by record ID."""
    try:
        # Get the record by ID
        record = await processed_image_crud.get_processed_image_by_id(record_id)
        
        if record:
            import json
            result_data = json.loads(record.result_json)
            result_data['parking_api_sent'] = parking_api_sent
            
            # Update the record in database
            await processed_image_crud.update_processed_image_result(
                record_id=record_id,
                new_result_json=json.dumps(result_data)
            )
            
            log.info(f"Updated record {record_id}: parking_api_sent={parking_api_sent}")
        else:
            log.warning(f"Could not find record with ID: {record_id}")
            
    except Exception as e:
        log.error(f"Error updating parking API status for record {record_id}: {e}")


async def mark_record_as_superseded(record_id: int, best_plate_text: str, best_confidence: float):
    """Mark a specific record as superseded by a better detection."""
    try:
        # Get the record by ID
        record = await processed_image_crud.get_processed_image_by_id(record_id)
        
        if record:
            import json
            result_data = json.loads(record.result_json)
            result_data['superseded_by_better_detection'] = True
            
            # Update the record in database
            await processed_image_crud.update_processed_image_result(
                record_id=record_id,
                new_result_json=json.dumps(result_data)
            )
            
            log.info(f"Updated record {record_id}: marked as superseded by '{best_plate_text}' (conf={best_confidence})")
        else:
            log.warning(f"Could not find record with ID: {record_id}")
            
    except Exception as e:
        log.error(f"Error marking record {record_id} as superseded: {e}")



@router.get("/sent-plates")
async def get_sent_plates(limit: int = 50, offset: int = 0):
    """
    Get all plates that have been sent to the parking API.
    
    This endpoint is useful for debugging and monitoring which plates
    have been sent to prevent duplicates.
    """
    try:
        sent_plates = await sent_plates_crud.get_all_sent_plates(limit=limit, offset=offset)
        
        # Convert to list of dictionaries for JSON response
        result = []
        for plate in sent_plates:
            result.append({
                "id": plate.id,
                "plate_text": plate.plate_text,
                "code": plate.code,
                "confidence": plate.confidence,
                "sent_timestamp": plate.sent_timestamp.isoformat() if plate.sent_timestamp else None,
                "parking_api_response": plate.parking_api_response,
                "internal_code": plate.internal_code
            })
        
        return {
            "sent_plates": result,
            "total_shown": len(result),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        log.error(f"Error retrieving sent plates: {e}")
        return {"error": "Failed to retrieve sent plates", "details": str(e)}

@router.get("/sent-plates/{plate_text}")
async def check_plate_sent(plate_text: str):
    """
    Check if a specific plate has been sent to the parking API.
    
    Args:
        plate_text: The plate text to check
        
    Returns:
        Information about whether the plate was sent and when
    """
    try:
        was_sent = await sent_plates_crud.check_plate_sent_globally(plate_text)
        
        if was_sent:
            # Get details about when it was sent
            all_records = await sent_plates_crud.get_all_sent_plates(limit=1000)
            matching_records = [r for r in all_records if r.plate_text.upper() == plate_text.upper()]
            
            return {
                "plate_text": plate_text.upper(),
                "was_sent": True,
                "records": [
                    {
                        "code": r.code,
                        "confidence": r.confidence,
                        "sent_timestamp": r.sent_timestamp.isoformat() if r.sent_timestamp else None,
                        "parking_api_response": r.parking_api_response
                    }
                    for r in matching_records
                ]
            }
        else:
            return {
                "plate_text": plate_text.upper(),
                "was_sent": False,
                "records": []
            }
            
    except Exception as e:
        log.error(f"Error checking plate {plate_text}: {e}")
        return {"error": f"Failed to check plate {plate_text}", "details": str(e)}
