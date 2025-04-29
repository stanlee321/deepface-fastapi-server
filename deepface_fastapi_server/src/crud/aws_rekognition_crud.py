import boto3
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
import logging
import io
from typing import Optional, List, Dict, Any

# Import the settings instance
from config import settings

log = logging.getLogger(__name__)

# --- Global Clients (Initialized on first use or explicitly) ---
# It's often better to initialize clients within functions or using a dependency injection system,
# but for simplicity here, we'll use global placeholders or initialize in relevant functions.
rekognition_client = None
s3_client = None

def ensure_collection(rekog, collection_id):
    try:
        rekog.create_collection(CollectionId=collection_id)
    except rekog.exceptions.ResourceAlreadyExistsException:
        print("Collection already exists")




def get_rekognition_client():
    """Initializes and returns the Rekognition client."""
    global rekognition_client
    if rekognition_client is None:
        try:
            log.info(f"Initializing Rekognition client for region: {settings.AWS_REGION}")
            rekognition_client = boto3.client('rekognition', region_name=settings.AWS_REGION)
            # …when initializing your client:
            ensure_collection(rekognition_client, settings.AWS_REKOGNITION_COLLECTION_ID)
            # Quick check to ensure credentials are valid
            rekognition_client.list_collections(MaxResults=1)
            log.info("Rekognition client initialized successfully.")
        except (NoCredentialsError, PartialCredentialsError) as e:
            log.exception("AWS credentials not found or incomplete. Configure credentials (e.g., via environment variables or IAM role).")
            raise e
        except ClientError as e:
            # Handle potential issues like invalid region, but allow specific errors later
            if e.response['Error']['Code'] == 'UnrecognizedClientException':
                 log.exception(f"Invalid AWS region configured: {settings.AWS_REGION}")
                 raise e
            # Log other client errors during init check
            log.error(f"Error during Rekognition client initialization check: {e}")
            # Don't raise here, allow functions to handle specific errors
        except Exception as e:
             log.exception(f"Unexpected error initializing Rekognition client: {e}")
             raise e
    return rekognition_client

def get_s3_client():
    """Initializes and returns the S3 client."""
    global s3_client
    if s3_client is None:
        try:
            log.info(f"Initializing S3 client for region: {settings.AWS_REGION}")
            # S3 client region might not strictly matter for bucket operations if bucket exists,
            # but specifying it is good practice.
            s3_client = boto3.client('s3', region_name=settings.AWS_REGION)
            # Quick check (optional) - e.g., list buckets if permissions allow
            # s3_client.list_buckets()
            log.info("S3 client initialized successfully.")
        except (NoCredentialsError, PartialCredentialsError) as e:
            log.exception("AWS credentials not found or incomplete for S3.")
            raise e
        except Exception as e:
             log.exception(f"Unexpected error initializing S3 client: {e}")
             raise e
    return s3_client


async def verify_s3_bucket_access(bucket: str = settings.AWS_S3_BUCKET_NAME):
    """Verifies access to the configured S3 bucket."""
    s3 = get_s3_client()
    try:
        # head_bucket is a low-cost way to check existence and permissions
        s3.head_bucket(Bucket=bucket)
        log.info(f"Successfully verified access to S3 bucket: {bucket}")
        return True
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code == '404': # Not Found
            log.exception(f"S3 bucket '{bucket}' not found. Please create it or check the name.")
        elif error_code == '403': # Forbidden
            log.exception(f"Access denied to S3 bucket '{bucket}'. Check bucket policy and IAM permissions.")
        else:
            log.exception(f"Error verifying access to S3 bucket '{bucket}': {e}")
        return False
    except Exception as e:
        log.exception(f"Unexpected error verifying S3 bucket '{bucket}': {e}")
        return False

async def ensure_collection_exists(collection_id: str = settings.AWS_REKOGNITION_COLLECTION_ID):
    """Creates the Rekognition collection if it doesn't exist."""
    rekog = get_rekognition_client()
    try:
        response = rekog.list_collections()
        if collection_id not in response.get('CollectionIds', []):
            log.info(f"Collection '{collection_id}' not found. Attempting to create.")
            create_response = rekog.create_collection(CollectionId=collection_id)
            status_code = create_response.get('StatusCode')
            if 200 <= status_code < 300:
                 log.info(f"Successfully created collection '{collection_id}', ARN: {create_response.get('CollectionArn')}")
            else:
                 log.error(f"Failed to create collection '{collection_id}', Status Code: {status_code}")
                 # Consider raising an exception here if creation is critical
        else:
            log.info(f"Rekognition collection '{collection_id}' already exists.")
    except ClientError as e:
        log.exception(f"Error checking or creating collection '{collection_id}': {e}")
        # Depending on the error, you might want to raise it

async def upload_to_s3(file_bytes: bytes, object_key: str, bucket: str = settings.AWS_S3_BUCKET_NAME) -> Optional[str]:
    """Uploads image bytes to S3. Returns the object key on success, None on failure."""
    # Optional: Verify bucket access before upload attempt
    # if not await verify_s3_bucket_access(bucket):
    #     return None # Or raise an error
    s3 = get_s3_client()
    try:
        s3.put_object(Bucket=bucket, Key=object_key, Body=file_bytes)
        log.info(f"Successfully uploaded image to s3://{bucket}/{object_key}")
        return object_key
    except ClientError as e:
        log.exception(f"Error uploading image to s3://{bucket}/{object_key}: {e}")
        return None
    except Exception as e:
        log.exception(f"Unexpected error uploading image to s3://{bucket}/{object_key}: {e}")
        return None

async def index_face_aws(collection_id: str, s3_bucket: str, s3_key: str, external_id: str) -> Optional[Dict[str, Any]]:
    """Index a face from an S3 object into the Rekognition collection.
       Returns the FaceRecord on success, None on failure.
    """
    rekog = get_rekognition_client()
    try:
        response = rekog.index_faces(
            CollectionId=collection_id,
            Image={'S3Object': {'Bucket': s3_bucket, 'Name': s3_key}},
            ExternalImageId=external_id, # Use blacklist entry ID (or similar unique ID)
            DetectionAttributes=[], # Or ['DEFAULT'] if needed later
            MaxFaces=1, # Assume one face per blacklist reference image
            QualityFilter='AUTO' # Or 'MEDIUM', 'HIGH' depending on needs
        )
        if response.get('FaceRecords'):
            face_record = response['FaceRecords'][0]
            log.info(f"Indexed face ID: {face_record['Face']['FaceId']} for External ID: {external_id} from s3://{s3_bucket}/{s3_key}")
            return face_record # Return the full record
        else:
            # Handle cases where no face was detected or indexed
            log.warning(f"No face record returned for External ID: {external_id} from s3://{s3_bucket}/{s3_key}. Response: {response}")
            return None
    except ClientError as e:
        log.exception(f"Error indexing face for External ID {external_id} from s3://{s3_bucket}/{s3_key}: {e}")
        return None

async def delete_face_aws(collection_id: str, face_ids: List[str]) -> bool:
    """Deletes specific FaceIds from the Rekognition collection."""
    if not face_ids:
        log.warning("No face IDs provided for deletion.")
        return True # Nothing to delete

    rekog = get_rekognition_client()
    try:
        response = rekog.delete_faces(CollectionId=collection_id, FaceIds=face_ids)
        deleted_count = len(response.get('DeletedFaces', []))
        log.info(f"Attempted to delete {len(face_ids)} face(s). Successfully deleted {deleted_count} face(s) from collection '{collection_id}'.")
        if deleted_count != len(face_ids):
             log.warning(f"Mismatch in deleted faces. Requested: {len(face_ids)}, Deleted: {deleted_count}. Response: {response}")
        return deleted_count > 0 # Return True if at least one was deleted
    except ClientError as e:
        log.exception(f"Error deleting faces {face_ids} from collection '{collection_id}': {e}")
        return False

async def delete_all_faces_for_external_id(collection_id: str, external_id: str) -> bool:
    """Finds all faces associated with an external_id and deletes them."""
    rekog = get_rekognition_client()
    face_ids_to_delete = []
    try:
        # Rekognition doesn't directly filter list_faces by external_id.
        # We need to list all faces and filter locally. This can be slow for large collections.
        # Consider storing FaceId -> external_id mapping in your DB if performance is critical.
        paginator = rekog.get_paginator('list_faces')
        page_iterator = paginator.paginate(CollectionId=collection_id)
        log.info(f"Listing faces in collection '{collection_id}' to find matches for External ID: {external_id}")
        for page in page_iterator:
            for face in page.get('Faces', []):
                if face.get('ExternalImageId') == external_id:
                    face_ids_to_delete.append(face['FaceId'])

        if not face_ids_to_delete:
            log.warning(f"No faces found associated with External ID: {external_id} in collection '{collection_id}'.")
            return True # Nothing to delete

        log.info(f"Found {len(face_ids_to_delete)} face(s) for External ID {external_id}. Attempting deletion.")
        return await delete_face_aws(collection_id, face_ids_to_delete)

    except ClientError as e:
        log.exception(f"Error listing faces while trying to delete by External ID {external_id}: {e}")
        return False

async def search_face_aws(collection_id: str, image_bytes: bytes, threshold: float = 90.0, max_matches: int = 5) -> List[Dict[str, Any]]:
    """Searches for faces in the collection matching the provided image bytes.
       Maps the result to the format expected by the existing API (similar to DeepFace.find).
    """
    rekog = get_rekognition_client()
    results_mapped = []
    try:
        response = rekog.search_faces_by_image(
            CollectionId=collection_id,
            Image={'Bytes': image_bytes},
            FaceMatchThreshold=threshold, # Rekognition uses Similarity Threshold (0-100)
            MaxFaces=max_matches
        )

        face_matches = response.get('FaceMatches', [])
        log.info(f"Rekognition search returned {len(face_matches)} matches above threshold {threshold}%")

        for match in face_matches:
            similarity = match.get('Similarity') # 0-100
            face = match.get('Face', {})
            external_id = face.get('ExternalImageId') # This is our link back to the blacklist entry
            face_id = face.get('FaceId')

            if external_id is None:
                 log.warning(f"Skipping match with FaceId {face_id} because it has no ExternalImageId.")
                 continue

            # --- Crucial Mapping ---
            # Convert Similarity (higher is better) to Distance (lower is better)
            # Simple inverse mapping: distance = 100 - similarity.
            # NOTE: This is NOT equivalent to cosine distance or euclidean distance.
            # The scale and meaning are different. This is a basic translation for consistency.
            # A more accurate mapping might require calibration or using a fixed Rekognition threshold
            # and mapping that to the equivalent DeepFace threshold conceptually.
            distance = (100.0 - similarity) / 100.0 # Normalize to roughly 0.0 - 1.0 range

            # Construct the dictionary matching DeepFace.find output structure
            # We may not have all the same fields (e.g., target face coords from index are complex)
            mapped_match = {
                # Essential fields:
                "identity": f"aws_rekognition_external_id:{external_id}", # Indicate source and use external ID
                "distance": distance,
                "threshold": (100.0 - threshold) / 100.0, # Map threshold similarly

                # Optional/Placeholder fields (might be derivable or omitted):
                "model": "aws_rekognition",
                "detector_backend": "aws_rekognition",
                "similarity_metric": "aws_similarity_mapped",
                # Rekognition search_faces_by_image gives bounding box of the *searched* face,
                # but not easily the bounding box of the *matched* face in the collection.
                "source_x": response.get('SearchedFaceBoundingBox', {}).get('Left'), # Example mapping needed
                "source_y": response.get('SearchedFaceBoundingBox', {}).get('Top'),
                "source_w": response.get('SearchedFaceBoundingBox', {}).get('Width'),
                "source_h": response.get('SearchedFaceBoundingBox', {}).get('Height'),
                # Target coords are not directly available from search_faces_by_image
                "target_x": None,
                "target_y": None,
                "target_w": None,
                "target_h": None,
                # Add Rekognition specific info if useful
                "aws_similarity": similarity,
                "aws_face_id": face_id,
            }
            results_mapped.append(mapped_match)

    except ClientError as e:
        # Handle specific errors like InvalidParameterException (bad image format?)
        log.exception(f"Error searching faces with AWS Rekognition: {e}")
        # Return empty list on error to maintain function signature
    except Exception as e:
        log.exception(f"Unexpected error during AWS face search: {e}")

    # Sort by distance (ascending) like DeepFace.find
    results_mapped.sort(key=lambda x: x['distance'])

    return results_mapped

# --- Helper to delete S3 objects (needed for blacklist cleanup) ---
async def delete_from_s3(object_keys: List[str], bucket: str = settings.AWS_S3_BUCKET_NAME) -> bool:
    """Deletes multiple objects from S3."""
    if not object_keys:
        return True
    s3 = get_s3_client()
    objects_to_delete = [{'Key': key} for key in object_keys]
    try:
        response = s3.delete_objects(
            Bucket=bucket,
            Delete={'Objects': objects_to_delete, 'Quiet': False}
        )
        deleted_count = len(response.get('Deleted', []))
        error_count = len(response.get('Errors', []))
        log.info(f"S3 Delete Request for {len(object_keys)} keys in bucket '{bucket}'. Deleted: {deleted_count}, Errors: {error_count}")
        if error_count > 0:
            log.error(f"Errors occurred during S3 deletion: {response.get('Errors')}")
        return error_count == 0 # Return True only if all deletions succeeded without error
    except ClientError as e:
        log.exception(f"Error deleting objects from s3://{bucket}/: {e}")
        return False
    except Exception as e:
        log.exception(f"Unexpected error deleting objects from s3://{bucket}/: {e}")
        return False 