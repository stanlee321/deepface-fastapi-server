from fastapi import APIRouter, HTTPException, Path, status, UploadFile, File, Form
from typing import List, Optional
import uuid # Import uuid
import asyncio # Import asyncio

from models import BlacklistCreate, BlacklistRecord
# Import specific CRUD functions as needed
from crud import blacklist_crud, face_crud
# Import AWS crud if needed
from crud import aws_rekognition_crud
# Import the settings instance
from config import settings
import logging
import os
import shutil # For deleting directories
import aiofiles # For async file operations
from pathlib import Path as SystemPath # Use SystemPath to avoid clash with fastapi.Path

router = APIRouter()
log = logging.getLogger(__name__)

async def save_upload_file(upload_file: UploadFile, destination: str):
    """Asynchronously saves an uploaded file."""
    try:
        # Ensure directory exists
        SystemPath(destination).parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(destination, 'wb') as out_file:
            while content := await upload_file.read(1024 * 1024):  # Read chunk by chunk 1MB
                await out_file.write(content)
        log.info(f"Successfully saved uploaded file to: {destination}")
    except Exception as e:
        log.error(f"Error saving file {destination}: {e}")
        raise # Re-raise exception to handle it in the endpoint
    finally:
         await upload_file.close()

# Helper function to get image bytes (reusable)
async def get_image_bytes(file_path: str) -> Optional[bytes]:
    try:
        async with aiofiles.open(file_path, 'rb') as f:
            return await f.read()
    except Exception as e:
        log.error(f"Error reading image bytes from {file_path}: {e}")
        return None

@router.post("/", response_model=BlacklistRecord, status_code=status.HTTP_201_CREATED)
async def add_to_blacklist(
    name: str = Form(...),
    reason: Optional[str] = Form(None),
    images: List[UploadFile] = File(..., description="One or more image files for the blacklist entry.")
):
    if not images:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image files provided."
        )
    payload = BlacklistCreate(name=name, reason=reason)
    existing = await blacklist_crud.get_person_by_name(payload.name)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Blacklist entry with name '{payload.name}' already exists."
        )

    record_id = await blacklist_crud.add_person(payload)
    log.info(f"DATABASE CHECK 1: add_person called for '{payload.name}'. Returned ID: {record_id}")
    if record_id is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to add blacklist entry to database")

    # --- Image Handling & Backend Indexing --- 
    person_folder_local = os.path.join(settings.BLACKLIST_DB_PATH, str(record_id))
    aws_s3_prefix = f"blacklist/{record_id}/" # S3 prefix for this entry's images
    saved_local_paths = []
    aws_indexing_errors = []
    aws_s3_keys = []

    try:
        os.makedirs(person_folder_local, exist_ok=True)
        log.info(f"Created local directory: {person_folder_local}")

        valid_images_processed = 0
        for image in images:
            if not image.content_type or not image.content_type.startswith("image/"):
                 log.warning(f"Skipping non-image file: {image.filename} (type: {image.content_type})")
                 continue
            
            safe_filename = str(uuid.uuid4()) + "_" + image.filename.replace("..", "").replace("/", "") # Add UUID
            local_file_destination = os.path.join(person_folder_local, safe_filename)
            
            # 1. Save locally (needed for DeepFace, might be temporary for AWS)
            await save_upload_file(image, local_file_destination)
            saved_local_paths.append(local_file_destination)
            valid_images_processed += 1
            
            # 2. If AWS backend, upload to S3 and index
            if settings.FACE_PROCESSING_BACKEND == 'aws_rekognition':
                img_bytes = await get_image_bytes(local_file_destination)
                if img_bytes:
                    s3_object_key = aws_s3_prefix + safe_filename
                    upload_success = await aws_rekognition_crud.upload_to_s3(img_bytes, s3_object_key, bucket=settings.AWS_S3_BUCKET_NAME)
                    if upload_success:
                        aws_s3_keys.append(s3_object_key)
                        # Index the face in Rekognition
                        face_record = await aws_rekognition_crud.index_face_aws(
                            collection_id=settings.AWS_REKOGNITION_COLLECTION_ID,
                            s3_bucket=settings.AWS_S3_BUCKET_NAME,
                            s3_key=s3_object_key,
                            external_id=str(record_id) # Link face to our DB ID
                        )
                        if face_record is None:
                            log.error(f"Failed to index face for {s3_object_key} in Rekognition.")
                            aws_indexing_errors.append(f"Failed indexing {safe_filename}")
                            # Optionally: delete the S3 object if indexing fails?
                    else:
                        log.error(f"Failed to upload {safe_filename} to S3.")
                        aws_indexing_errors.append(f"Failed S3 upload for {safe_filename}")
                else:
                     aws_indexing_errors.append(f"Failed read bytes for {safe_filename}")

        if valid_images_processed == 0:
             await blacklist_crud.delete_person(record_id)
             if os.path.exists(person_folder_local): shutil.rmtree(person_folder_local)
             # Also delete any S3 objects if uploads occurred before failure
             if aws_s3_keys: await aws_rekognition_crud.delete_from_s3(aws_s3_keys, bucket=settings.AWS_S3_BUCKET_NAME)
             raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No valid image files were provided or saved.")

    except Exception as e:
        log.error(f"Failed during image processing/indexing for blacklist entry {record_id}: {e}")
        await blacklist_crud.delete_person(record_id)
        if os.path.exists(person_folder_local): shutil.rmtree(person_folder_local)
        if aws_s3_keys: await aws_rekognition_crud.delete_from_s3(aws_s3_keys, bucket=settings.AWS_S3_BUCKET_NAME)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed processing images: {e}")

    # 3. Update DB with image directory path (relevant for DeepFace mostly)
    # For AWS, we might store S3 prefix or rely on convention
    relative_image_dir = str(record_id) # Keep local convention for now
    updated_id = await blacklist_crud.update_person(record_id, payload, image_dir_path=relative_image_dir)
    log.info(f"DATABASE CHECK 2: update_person called for ID {record_id}. Returned ID: {updated_id}")
    # ... (handle update failure if needed)

    # 4. Trigger backend-specific post-processing
    if settings.FACE_PROCESSING_BACKEND == 'deepface':
        log.info(f"Triggering DeepFace index refresh for {settings.BLACKLIST_DB_PATH}")
        await face_crud.refresh_blacklist_index(settings.BLACKLIST_DB_PATH)
    elif settings.FACE_PROCESSING_BACKEND == 'aws_rekognition':
        if aws_indexing_errors:
             log.warning(f"AWS Rekognition indexing finished with errors for entry {record_id}: {aws_indexing_errors}")
             # Decide if we should raise an error or just log
        else:
             log.info(f"AWS Rekognition indexing completed for entry {record_id}.")
        # Optionally: If local files aren't needed after S3 upload, delete person_folder_local?

    created_record = await blacklist_crud.get_person(record_id)
    log.info(f"DATABASE CHECK 3: get_person called for ID {record_id}. Found: {created_record is not None}")
    if not created_record:
         # This case is unlikely if DB add succeeded, but handle defensively
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
    
    return updated_record

@router.delete("/{id}/", response_model=BlacklistRecord)
async def remove_from_blacklist(
    id: int = Path(..., gt=0, description="The ID of the blacklist entry to delete.")
):
    person = await blacklist_crud.get_person(id)
    if not person:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Blacklist entry not found")

    # 1. Backend-specific cleanup
    cleanup_success = True
    if settings.FACE_PROCESSING_BACKEND == 'deepface':
        person_folder_local = os.path.join(settings.BLACKLIST_DB_PATH, str(id))
        if os.path.isdir(person_folder_local):
            try:
                shutil.rmtree(person_folder_local)
                log.info(f"Deleted local image folder: {person_folder_local}")
            except OSError as e:
                log.error(f"Error deleting local folder {person_folder_local}: {e}. Proceeding with DB delete.")
                # Set flag? cleanup_success = False
    elif settings.FACE_PROCESSING_BACKEND == 'aws_rekognition':
        log.info(f"Performing AWS cleanup for blacklist entry ID: {id}")
        # Delete faces from Rekognition collection
        rekognition_deleted = await aws_rekognition_crud.delete_all_faces_for_external_id(
            collection_id=settings.AWS_REKOGNITION_COLLECTION_ID,
            external_id=str(id)
        )
        if not rekognition_deleted:
            log.error(f"Failed to delete faces from Rekognition for external_id {id}. Check logs.")
            cleanup_success = False # Mark cleanup as potentially incomplete
        
        # Delete images from S3
        # List objects with prefix and delete them
        s3_prefix = f"blacklist/{id}/"
        try:
            s3_client = aws_rekognition_crud.get_s3_client()
            objects_to_delete = []
            paginator = s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(Bucket=settings.AWS_S3_BUCKET_NAME, Prefix=s3_prefix)
            for page in page_iterator:
                if "Contents" in page:
                    for obj in page['Contents']:
                        objects_to_delete.append(obj['Key']) 
            
            if objects_to_delete:
                log.info(f"Found {len(objects_to_delete)} S3 objects with prefix {s3_prefix} to delete.")
                s3_deleted = await aws_rekognition_crud.delete_from_s3(objects_to_delete, bucket=settings.AWS_S3_BUCKET_NAME)
                if not s3_deleted:
                    log.error(f"Failed to delete some/all S3 objects for prefix {s3_prefix}. Check logs.")
                    cleanup_success = False
            else:
                 log.info(f"No S3 objects found with prefix {s3_prefix} to delete.")
                 
        except Exception as e:
             log.exception(f"Error listing or deleting S3 objects for prefix {s3_prefix}: {e}")
             cleanup_success = False

    # 2. Delete from database
    deleted_count = await blacklist_crud.delete_person(id)
    if deleted_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Blacklist entry not found (or already deleted)")
    if deleted_count is None:
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete blacklist entry from database")

    # 3. Trigger DeepFace index refresh (only if using deepface)
    if settings.FACE_PROCESSING_BACKEND == 'deepface':
        log.info(f"Triggering DeepFace index refresh after deleting ID: {id}")
        await face_crud.refresh_blacklist_index(settings.BLACKLIST_DB_PATH)

    log.info(f"Deleted blacklist entry ID: {id}, Name: {person.name}. Backend cleanup success: {cleanup_success}")
    return person

@router.post("/{id}/images", status_code=status.HTTP_200_OK)
async def add_images_to_blacklist_entry(
    id: int = Path(..., gt=0, description="The ID of the blacklist entry to add images to."),
    images: List[UploadFile] = File(..., description="One or more image files to add to the entry.")
):
    if not images:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image files provided."
        )
    person = await blacklist_crud.get_person(id)
    if not person:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Blacklist entry not found")

    person_folder_local = os.path.join(settings.BLACKLIST_DB_PATH, str(id))
    aws_s3_prefix = f"blacklist/{id}/"
    saved_files_count = 0
    errors = []
    aws_indexing_errors = []

    # Ensure local directory exists (for DeepFace or temporary storage)
    try:
        os.makedirs(person_folder_local, exist_ok=True)
    except OSError as e:
        log.error(f"Failed to ensure directory {person_folder_local}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to access image storage for entry ID {id}")

    for image in images:
        if not image.content_type or not image.content_type.startswith("image/"):
            errors.append(f"Skipped non-image file: {image.filename}")
            continue
            
        safe_filename = str(uuid.uuid4()) + "_" + image.filename.replace("..", "").replace("/", "")
        local_file_destination = os.path.join(person_folder_local, safe_filename)
        
        if os.path.exists(local_file_destination) and settings.FACE_PROCESSING_BACKEND == 'deepface':
            # Avoid re-saving if file exists locally for deepface
            errors.append(f"Skipped duplicate file (local): {safe_filename}")
            continue
            
        try:
            # Save locally first
            await save_upload_file(image, local_file_destination)
            
            # If AWS, upload and index
            if settings.FACE_PROCESSING_BACKEND == 'aws_rekognition':
                img_bytes = await get_image_bytes(local_file_destination)
                if img_bytes:
                    s3_object_key = aws_s3_prefix + safe_filename
                    upload_success = await aws_rekognition_crud.upload_to_s3(img_bytes, s3_object_key, bucket=settings.AWS_S3_BUCKET_NAME)
                    if upload_success:
                        face_record = await aws_rekognition_crud.index_face_aws(
                            collection_id=settings.AWS_REKOGNITION_COLLECTION_ID,
                            s3_bucket=settings.AWS_S3_BUCKET_NAME,
                            s3_key=s3_object_key,
                            external_id=str(id)
                        )
                        if face_record is None:
                            aws_indexing_errors.append(f"Failed indexing {safe_filename}")
                    else:
                         aws_indexing_errors.append(f"Failed S3 upload for {safe_filename}")
                else:
                    aws_indexing_errors.append(f"Failed read bytes for {safe_filename}")
                # Optionally delete local file now if only needed for AWS?
                # try: os.remove(local_file_destination) except OSError: pass
                    
            saved_files_count += 1 # Count successful processing for this image
            
        except Exception as e:
            log.error(f"Failed processing image {safe_filename} for entry ID {id}: {e}")
            errors.append(f"Failed to process file: {safe_filename}")

    if saved_files_count == 0 and (errors or aws_indexing_errors):
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No valid new images processed. Errors: " + "; ".join(errors + aws_indexing_errors))
    elif saved_files_count == 0:
        return {"message": "No new images were added (either empty list, duplicates, or non-images).", "errors": errors + aws_indexing_errors}

    # Trigger backend-specific post-processing if new files were added
    if settings.FACE_PROCESSING_BACKEND == 'deepface':
        log.info(f"Triggering DeepFace index refresh after adding images to ID: {id}")
        await face_crud.refresh_blacklist_index(settings.BLACKLIST_DB_PATH)
    elif settings.FACE_PROCESSING_BACKEND == 'aws_rekognition':
        if aws_indexing_errors:
             log.warning(f"AWS Rekognition indexing finished with errors while adding images to entry {id}: {aws_indexing_errors}")
        else:
             log.info(f"AWS Rekognition indexing completed for images added to entry {id}.")

    return {"message": f"Successfully processed {saved_files_count} image(s) for blacklist entry ID {id}.", "errors": errors + aws_indexing_errors} 