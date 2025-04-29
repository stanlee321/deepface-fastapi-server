from fastapi import APIRouter, Query
import json
import logging
from pydantic import BaseModel

from crud import processed_image_crud
from models import PaginatedProcessedImagesResponse, ProcessedImageRecord

log = logging.getLogger(__name__)
router = APIRouter()


@router.get("/", response_model=PaginatedProcessedImagesResponse) # Define response model
async def get_processed_images(
    limit: int = Query(20, ge=1, le=100, description="Number of records per page."),
    offset: int = Query(0, ge=0, description="Number of records to skip for pagination.")
):
    """Retrieves a paginated list of processed image records, ordered by newest first."""
    db_records = await processed_image_crud.get_all_processed_images(limit=limit, offset=offset)
    total_count = await processed_image_crud.get_processed_images_count()
    
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

# Add other endpoints like get by ID if needed 