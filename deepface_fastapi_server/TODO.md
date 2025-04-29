# TODO: Integrate AWS Rekognition as Alternative Backend

This document outlines the tasks required to integrate AWS Rekognition as an alternative face processing backend alongside the existing DeepFace implementation, allowing switching between modes via configuration.

**Goal:** Maintain the existing API contract (endpoints, request/response structures) while adding AWS support.

## Phase 1: Setup and AWS Module

- [ ] **Dependencies:** Add `boto3` to `src/requirements.txt`.
- [ ] **Configuration (`src/config.py`):**
    - [ ] Add `FACE_PROCESSING_BACKEND` variable (e.g., 'deepface' or 'aws_rekognition', default 'deepface').
    - [ ] Add AWS-specific configurations (e.g., `AWS_REGION`, `AWS_S3_BUCKET_NAME`, `AWS_REKOGNITION_COLLECTION_ID`). Consider how credentials will be handled (environment variables, IAM roles if running on EC2/ECS).
- [ ] **AWS Rekognition Module (`src/crud/aws_rekognition_crud.py`):**
    - [ ] Create a new file `aws_rekognition_crud.py`.
    - [ ] Implement functions based on `docs/face-aws.md` to interact with Rekognition using `boto3`.
        - [ ] `init_rekognition_client()`: Initialize boto3 client.
        - [ ] `ensure_collection_exists(collection_id)`: Create collection if it doesn't exist.
        - [ ] `upload_to_s3(file_path_or_bytes, object_key)`: Helper to upload images to S3.
        - [ ] `index_face_aws(collection_id, s3_bucket, s3_key, external_id)`: Index a face from S3. Map response to a common internal format if needed.
        - [ ] `delete_face_aws(collection_id, face_id)`: Delete face(s) from collection.
        - [ ] `delete_all_faces_for_external_id(collection_id, external_id)`: Helper to find and delete all `FaceId`s associated with a blacklist entry ID (needs `list_faces` and potentially iteration).
        - [ ] `search_face_aws(collection_id, image_bytes_or_s3, threshold)`: Search for matching faces. **Crucially, map the Rekognition response (`FaceMatches`, `Similarity`) to the structure currently returned by `face_crud.find_matches_in_blacklist` (List of Dicts with `identity`, `distance`, etc.).** This might involve converting similarity to a distance metric.
    - [ ] Add error handling (`ClientError`) for all AWS calls.

## Phase 2: Abstraction and Integration

- [ ] **Face Processing Service (`src/services/face_processing_service.py`):**
    - [ ] Create a new directory `src/services`.
    - [ ] Create `face_processing_service.py`.
    - [ ] Define functions that mirror the core operations needed by the API endpoints (e.g., `find_matches`, potentially others like `extract_faces` if needed, though Rekognition combines detection/search).
    - [ ] Inside these service functions, check `config.FACE_PROCESSING_BACKEND`.
        - If 'deepface', call the corresponding function in `src/crud/face_crud.py`.
        - If 'aws_rekognition', call the corresponding function in `src/crud/aws_rekognition_crud.py`.
    - [ ] Ensure the *return format* from both branches is identical and matches the existing API contract.
- [ ] **Adapt Blacklist CRUD (`src/crud/blacklist_crud.py`):**
    - [ ] Modify `create_blacklist_entry`:
        - If AWS mode: After saving metadata to DB and saving images locally (or directly uploading?), call `index_face_aws` for each image, associating them with the DB entry's ID (`external_id`). Store `FaceId`s if needed, or rely on `ExternalImageId`. Handle S3 upload. Update logic for `reference_image_dir`.
    - [ ] Modify `delete_blacklist_entry_by_id`:
        - If AWS mode: Before deleting from DB, call `delete_all_faces_for_external_id` (or similar logic using stored `FaceId`s) to remove faces from the Rekognition collection. Delete associated images from S3.
    - [ ] Modify `add_images_to_blacklist_entry`:
        - If AWS mode: Call `index_face_aws` for newly uploaded images. Handle S3 upload.
    - [ ] Modify `refresh_blacklist_index`:
        - This function becomes irrelevant in AWS mode (Rekognition manages its index). Add logic to skip if in AWS mode or adapt it to potentially re-sync S3/Collection if needed (though `index_faces`/`delete_faces` should handle this).
- [ ] **Adapt Processing Endpoint (`src/api/endpoints/processing.py`):**
    - [ ] Modify the main processing logic (likely within `/process-images`).
    - [ ] Instead of calling `face_crud` functions directly, call the corresponding functions in `src/services/face_processing_service.py`.
    - [ ] Ensure image input handling (path, URL, base64) works for both modes. For AWS, this might involve downloading URLs/decoding base64 *before* uploading to S3 or passing bytes directly to Rekognition if supported/preferred.
    - [ ] The logic for saving results to the `processed_images` table via `processed_image_crud` should remain largely the same, as the service layer ensures consistent data format.

## Phase 3: Testing and Documentation

- [ ] **Testing:**
    - [ ] Test `/process-images` endpoint in 'deepface' mode.
    - [ ] Configure AWS credentials/S3/Collection.
    - [ ] Test `/process-images` endpoint in 'aws_rekognition' mode. Verify response structure matches 'deepface' mode.
    - [ ] Test `/blacklist` CRUD endpoints in 'deepface' mode.
    - [ ] Test `/blacklist` CRUD endpoints in 'aws_rekognition' mode (add entry, add images, delete entry). Verify corresponding actions in S3 and Rekognition Collection.
    - [ ] Test `/processed-images/` endpoint works correctly regardless of the backend used during processing.
- [x] **Documentation (`README.md`):**
    - [x] Add documentation for the new `FACE_PROCESSING_BACKEND` setting.
    - [x] Add documentation for required AWS configuration variables (`AWS_REGION`, `AWS_S3_BUCKET_NAME`, `AWS_REKOGNITION_COLLECTION_ID`) and credential setup.
    - [x] Mention any behavioral differences between modes if unavoidable (e.g., exact confidence/distance scales might differ slightly even after mapping). 