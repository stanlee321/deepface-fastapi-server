import logging
import json
from typing import Optional, Union
from datetime import datetime

from database import database, processed_images_table
from models import FaceImageProcessingResult, WeaponImageProcessingResult # Import the Pydantic model for type hinting
from sqlalchemy import func
from sqlalchemy import select

log = logging.getLogger(__name__)

async def add_processed_image(
    input_identifier: str,
    saved_image_path: str,
    code: str,
    app_type: str,
    result: Union[FaceImageProcessingResult, WeaponImageProcessingResult], # Use the Pydantic model
    cropped_path: Optional[str] = None # Add new parameter
) -> Optional[int]:
    """Adds a record of a processed image and its results to the database."""
    
    # Determine if any blacklist match occurred
    has_match = False
    
    if app_type == "faces":
        if result.faces:
            for face in result.faces:
                if face.blacklist_matches:
                    has_match = True
                break
                
    # Serialize the full result model to JSON string
    try:
        # Use model_dump_json for Pydantic v2
        result_json_str = result.model_dump_json()
    except Exception as e:
        log.error(f"Error serializing processing result to JSON: {e}")
        result_json_str = json.dumps({"error": "Failed to serialize result", "details": str(e)})

    query = processed_images_table.insert().values(
        input_identifier=input_identifier, # Truncate
        saved_image_path=saved_image_path,
        processing_timestamp=datetime.now(), # Use current time
        code=code,
        app_type=app_type,
        has_blacklist_match=has_match,
        result_json=result_json_str,
        cropped_path=cropped_path, # Save the new path
    )
    
    try:
        last_record_id = await database.execute(query=query)
        log.info(f"Saved processing result to DB for image: {saved_image_path} (ID: {last_record_id})")
        return last_record_id
    except Exception as e:
        log.error(f"DB Error adding processed image record for {saved_image_path}: {e}")
        return None

# --- Optional: Add functions to retrieve records if needed --- 
# async def get_processed_image_by_id(id: int): ...

async def get_all_processed_images(limit: int = 20, offset: int = 0):
    """Retrieves processed image records with pagination, newest first."""
    query = (
        processed_images_table.select()
        .order_by(processed_images_table.c.processing_timestamp.desc()) # Order by newest first
        .limit(limit)
        .offset(offset)
    )
    try:
        results = await database.fetch_all(query=query)
        return results # Returns list of DB records (RowProxy)
    except Exception as e:
        log.error(f"DB Error retrieving processed images: {e}")
        return [] # Return empty list on error

async def get_processed_images_count() -> int:
    """Gets the total count of processed image records."""
    # Use select() with func.count() applied to a primary key column for counting
    query = select(func.count(processed_images_table.c.id)) # Count based on primary key
    try:
         # fetch_val will get the value from the first column of the first row
         count = await database.fetch_val(query=query)
         return count if count is not None else 0
    except Exception as e:
        log.error(f"DB Error counting processed images: {e}")
        return 0

# async def get_processed_images_with_matches(): ... 