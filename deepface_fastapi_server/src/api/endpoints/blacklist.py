from fastapi import APIRouter, HTTPException, Path, Depends, status, UploadFile, File, Form
from typing import List, Optional
from src.models import BlacklistCreate, BlacklistRecord
from src.crud import blacklist_crud, face_crud # Import face_crud for refresh
from src.config import BLACKLIST_DB_PATH # Get path from config
import logging
import os
import shutil # For deleting directories
import aiofiles # For async file operations

router = APIRouter()

async def save_upload_file(upload_file: UploadFile, destination: str):
    """Asynchronously saves an uploaded file."""
    try:
        async with aiofiles.open(destination, 'wb') as out_file:
            while content := await upload_file.read(1024 * 1024):  # Read chunk by chunk 1MB
                await out_file.write(content)
    except Exception as e:
        logging.error(f"Error saving file {destination}: {e}")
        raise # Re-raise exception to handle it in the endpoint
    finally:
         await upload_file.close()

@router.post("/", response_model=BlacklistRecord, status_code=status.HTTP_201_CREATED)
async def add_to_blacklist(
    # Use Form(...) for metadata alongside files
    name: str = Form(...),
    reason: Optional[str] = Form(None),
    # Accept multiple images
    images: List[UploadFile] = File(..., description="One or more image files for the blacklist entry.")
):
    """
    Adds a new entry to the blacklist database and saves associated images.
    Triggers an asynchronous refresh of the DeepFace index.
    """
    if not images:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image files provided."
        )

    # Use the payload model internally
    payload = BlacklistCreate(name=name, reason=reason)

    # Optional: Check if name already exists before inserting
    existing = await blacklist_crud.get_person_by_name(payload.name)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Blacklist entry with name '{payload.name}' already exists."
        )

    # 1. Add entry to database
    record_id = await blacklist_crud.add_person(payload)
    if record_id is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to add blacklist entry to database")

    # 2. Save uploaded images to a folder named after the record ID
    person_folder = os.path.join(BLACKLIST_DB_PATH, str(record_id))
    try:
        os.makedirs(person_folder, exist_ok=True)
        logging.info(f"Created directory: {person_folder}")

        saved_files_count = 0
        for image in images:
            # Basic validation (consider adding more robust checks)
            if not image.content_type.startswith("image/"):
                 logging.warning(f"Skipping non-image file: {image.filename} (type: {image.content_type})")
                 continue
            
            # Sanitize filename (optional but recommended)
            safe_filename = image.filename.replace("..", "").replace("/", "") # Basic sanitization
            file_destination = os.path.join(person_folder, safe_filename)
            await save_upload_file(image, file_destination)
            logging.info(f"Saved image: {file_destination}")
            saved_files_count += 1

        if saved_files_count == 0:
             # Clean up DB entry if no valid images were saved
             await blacklist_crud.delete_person(record_id)
             shutil.rmtree(person_folder) # Clean up folder
             raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No valid image files were provided or saved.")

    except Exception as e:
        logging.error(f"Failed to save images for blacklist entry {record_id}: {e}")
        # Clean up: remove database entry and any potentially created folder/files
        await blacklist_crud.delete_person(record_id)
        if os.path.exists(person_folder):
            shutil.rmtree(person_folder)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to save uploaded images: {e}")

    # 3. Update the DB record with the image directory path
    relative_image_dir = str(record_id) # Directory is named after the ID
    updated_id = await blacklist_crud.update_person(record_id, payload, image_dir_path=relative_image_dir)
    if updated_id is None:
         # Log warning, but proceed as the main entry and images exist
         logging.warning(f"Could not update blacklist entry {record_id} with image directory path.")

    # 4. Trigger DeepFace index refresh (asynchronously)
    # Note: This runs in the background (in the thread pool) but the API call waits.
    # For very long refreshes, consider a background task runner (Celery, ARQ).
    await face_crud.refresh_blacklist_index(BLACKLIST_DB_PATH)

    # Fetch the created record to return it
    created_record = await blacklist_crud.get_person(record_id)
    if not created_record:
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve created blacklist entry after adding.")
    return created_record

@router.get("/", response_model=List[BlacklistRecord])
async def get_blacklist():
    """Retrieves all entries from the blacklist."""
    return await blacklist_crud.get_all_persons()

@router.get("/{id}/", response_model=BlacklistRecord)
async def get_blacklist_person(id: int = Path(..., gt=0, description="The ID of the blacklist entry to retrieve.")):
    """Retrieves a specific blacklist entry by its ID."""
    person = await blacklist_crud.get_person(id)
    if not person:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Blacklist entry not found")
    return person

@router.put("/{id}/", response_model=BlacklistRecord)
async def update_blacklist_person_metadata(
    # Use BlacklistCreate model for update payload (name, reason)
    payload: BlacklistCreate, 
    id: int = Path(..., gt=0, description="The ID of the blacklist entry to update.")
):
    """Updates an existing blacklist entry's name and reason (metadata only)."""
    existing_person = await blacklist_crud.get_person(id)
    if not existing_person:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Blacklist entry not found")

    if payload.name != existing_person.name:
        potential_conflict = await blacklist_crud.get_person_by_name(payload.name)
        if potential_conflict and potential_conflict.id != id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Another blacklist entry with name '{payload.name}' already exists."
            )

    updated_id = await blacklist_crud.update_person(id, payload)
    if updated_id is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update blacklist entry in database")

    updated_record = await blacklist_crud.get_person(id)
    if not updated_record:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve updated blacklist entry after update.")
    
    # Renaming doesn't affect the folder named by ID, so no image action needed.
    # No index refresh needed for metadata-only changes.
    return updated_record

@router.delete("/{id}/", response_model=BlacklistRecord)
async def remove_from_blacklist(
    id: int = Path(..., gt=0, description="The ID of the blacklist entry to delete.")
):
    """
    Deletes a blacklist entry by its ID, removes associated image folder,
    and triggers an asynchronous refresh of the DeepFace index.
    """
    person = await blacklist_crud.get_person(id)
    if not person:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Blacklist entry not found")

    # 1. Delete associated image folder first (safer in case DB delete fails)
    person_folder = os.path.join(BLACKLIST_DB_PATH, str(id))
    if os.path.isdir(person_folder):
        try:
            shutil.rmtree(person_folder)
            logging.info(f"Deleted image folder: {person_folder}")
        except OSError as e:
            logging.error(f"Error deleting folder {person_folder}: {e}. Proceeding with DB delete.")
            # Decide if this should prevent DB deletion? For now, log and continue.

    # 2. Delete from database
    deleted_count = await blacklist_crud.delete_person(id)
    if deleted_count == 0:
        # This implies the record was already gone, maybe log a warning
        logging.warning(f"Attempted to delete blacklist entry ID {id}, but it was already gone from DB.")
        # Return the initially fetched record, but maybe indicate it was already gone?
        # Or raise 404 again?
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Blacklist entry not found (or already deleted)")
    if deleted_count is None:
         # DB error during delete
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete blacklist entry from database")

    # 3. Trigger DeepFace index refresh (asynchronously)
    await face_crud.refresh_blacklist_index(BLACKLIST_DB_PATH)

    logging.info(f"Deleted blacklist entry ID: {id}, Name: {person.name}")
    # Return the record that was fetched before deletion
    return person

@router.post("/{id}/images", status_code=status.HTTP_200_OK)
async def add_images_to_blacklist_entry(
    id: int = Path(..., gt=0, description="The ID of the blacklist entry to add images to."),
    images: List[UploadFile] = File(..., description="One or more image files to add to the entry.")
):
    """
    Adds one or more reference images to an existing blacklist entry.
    Triggers an asynchronous refresh of the DeepFace index.
    """
    if not images:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image files provided."
        )

    # 1. Check if the blacklist entry exists
    person = await blacklist_crud.get_person(id)
    if not person:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Blacklist entry not found")

    # 2. Determine the correct folder and ensure it exists
    person_folder = os.path.join(BLACKLIST_DB_PATH, str(id))
    if not os.path.isdir(person_folder):
        # This could happen if images were somehow deleted manually after creation
        logging.warning(f"Directory {person_folder} not found for existing entry ID {id}. Creating it.")
        try:
             os.makedirs(person_folder, exist_ok=True)
        except OSError as e:
            logging.error(f"Failed to create directory {person_folder}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to access image storage for entry ID {id}")

    # 3. Save uploaded images
    saved_files_count = 0
    errors = []
    for image in images:
        if not image.content_type or not image.content_type.startswith("image/"):
            logging.warning(f"Skipping non-image file: {image.filename} (type: {image.content_type})")
            errors.append(f"Skipped non-image file: {image.filename}")
            continue
            
        safe_filename = image.filename.replace("..", "").replace("/", "") # Basic sanitization
        file_destination = os.path.join(person_folder, safe_filename)
        
        # Check for duplicates before saving (optional)
        if os.path.exists(file_destination):
            logging.warning(f"Image {safe_filename} already exists for entry ID {id}. Skipping.")
            errors.append(f"Skipped duplicate file: {safe_filename}")
            continue
            
        try:
            await save_upload_file(image, file_destination) # Use existing helper
            logging.info(f"Added image to ID {id}: {file_destination}")
            saved_files_count += 1
        except Exception as e:
            logging.error(f"Failed to save image {safe_filename} for entry ID {id}: {e}")
            errors.append(f"Failed to save file: {safe_filename}")
            # Decide if one failure should stop the whole process? For now, continue.

    if saved_files_count == 0 and errors:
         # Only raise error if no files were saved AND there were errors
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No valid new images were provided or saved. Errors: " + "; ".join(errors))
    elif saved_files_count == 0:
        return {"message": "No new images were added (either empty list, duplicates, or non-images).", "errors": errors}

    # 4. Trigger DeepFace index refresh if new files were added
    await face_crud.refresh_blacklist_index(BLACKLIST_DB_PATH)

    return {"message": f"Successfully added {saved_files_count} image(s) to blacklist entry ID {id}. Index refreshed.", "errors": errors} 