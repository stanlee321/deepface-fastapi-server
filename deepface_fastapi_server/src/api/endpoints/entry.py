
import json

from fastapi import APIRouter, Body, HTTPException, status, Query
from tqdm import tqdm
from typing import List, Union

from models import ProcessImagesRequest, FaceImageProcessingResult, WeaponImageProcessingResult
from api.endpoints.processing_face import process_single_face_image
from api.endpoints.detection_weapons import process_single_weapons_image
from crud import processed_image_crud
from models import PaginatedProcessedImagesResponse, ProcessedImageRecord
from services.queue import get_mqtt_client
from config import settings

import logging
log = logging.getLogger(__name__)

router = APIRouter()


@router.post("/process-images", response_model=List[Union[FaceImageProcessingResult, WeaponImageProcessingResult]])
async def route_request(request: ProcessImagesRequest = Body(...)):
    """
    Processes a list of images (paths, URLs, or base64 strings) using the configured backend.
    For each image, finds blacklist matches and returns results.
    """
    
    if not request.images:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No images provided.")
    final_results: List[Union[FaceImageProcessingResult, WeaponImageProcessingResult]] = []
    
    
    code =  request.code
    app_type = request.app_type
    image_url = final_results[0].saved_image_path if len(final_results) > 0 else ""
    
    print("request" , request)
    crop_url = ""
    if request.app_type == "face":
        # Using sequential processing for simplicity now.
        # Consider asyncio.gather or background tasks for production.
        log.info(f"Processing {len(request.images)} images for face detection sequentially...")
        for img_input in tqdm(request.images, desc="Processing Images"):
            result = await process_single_face_image(img_input, request)
            final_results.append(result)
            crop_url = result.cropped_face_path


    elif request.app_type == "weapons":
        # Using sequential processing for simplicity now.
        # Consider asyncio.gather or background tasks for production.
        log.info(f"Processing {len(request.images)} images for weapons detection sequentially...")
        for img_input in tqdm(request.images, desc="Processing Images"):
            results = await process_single_weapons_image(img_input, request)
            print(f"\n\nResults: {results}\n\n")
            final_results.append(results)
            crop_url = results.cropped_face_path

    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid app type.")
    
    mqtt_client = get_mqtt_client()
    
    if image_url != "":
        try:
            mqtt_client.connect() # Ensure connection is established

            # Example payload
            test_payload = {
                "code": code,
                "app_type": app_type,
                "image_url": image_url,
                "crop_url": crop_url
            }

            # Use the process topic from settings
            topic_to_publish = settings.MQTT_LLM_TOPIC

            # Publish the message
            success = mqtt_client.publish(topic_to_publish, test_payload)

            if success:
                print("Example: Message published successfully.")
            else:
                print("Example: Message publication failed.")

        except Exception as e:
            log.error(f"Error publishing message: {e}")

        # send event to mqtt
        log.info(f"Finished processing {len(request.images)} images.")

    return final_results


# Add other endpoints like get by ID if needed 
@router.get("/processed-images", response_model=PaginatedProcessedImagesResponse) # Define response model
async def get_processed_images(
    limit: int = Query(20, ge=1, le=100, description="Number of records per page."),
    offset: int = Query(0, ge=0, description="Number of records to skip for pagination."),
    # optional query param
    app_type: str = Query(None, description="Type of application (face or weapons)")
)->PaginatedProcessedImagesResponse:
    """Retrieves a paginated list of processed image records, ordered by newest first."""
    db_records = await processed_image_crud.get_all_processed_images(limit=limit, offset=offset, app_type=app_type)
    total_count = await processed_image_crud.get_processed_images_count(app_type=app_type)
    
    response_items = []
    for record in db_records:
        try:
            # Parse the stored JSON string back into the structure
            # We assume the stored JSON matches ImageProcessingResult structure
            result_data = json.loads(record.result_json)
            # Create the response item, merging DB fields and JSON fields
            item = ProcessedImageRecord(
                db_id=record.id,
                saved_image_path=record.saved_image_path,
                processing_timestamp=record.processing_timestamp,
                has_blacklist_match=record.has_blacklist_match,
                cropped_face_path=record.cropped_path,
                # Fields from the parsed JSON
                image_path_or_identifier=result_data.get('image_path_or_identifier'),
                faces=result_data.get('faces', []),
                error=result_data.get('error')
            )
            response_items.append(item)
        except json.JSONDecodeError:
            log.error(f"Failed to decode result_json for processed_image ID {record.id}")
            # Optionally skip this record or add a placeholder with an error
        except Exception as e:
             log.error(f"Error processing record ID {record.id}: {e}")
             # Optionally skip

    return PaginatedProcessedImagesResponse(
         total_items=total_count,
         items=response_items,
         limit=limit,
         offset=offset
    )
