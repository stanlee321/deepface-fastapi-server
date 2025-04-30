from deepface import DeepFace
from typing import List, Union, Dict, Any, Optional
import numpy as np
import logging
import base64
import validators # Need to install: pip install validators
import requests # Need to install: pip install requests
import os
import uuid # For unique filenames
import shutil # For copying files
import aiofiles # For async saving of uploads/downloads
import asyncio # For async operations

# Import the model used in type hint
from models import FacialArea

from config import settings


log = logging.getLogger(__name__)

def is_base64(s: str) -> bool:
    """Check if a string is likely base64 encoded."""
    try:
        # Check if it starts with data URI prefix
        if s.startswith('data:image'):
            s = s.split(',', 1)[1]
        # Attempt to decode
        base64.b64decode(s, validate=True)
        return True
    except (base64.binascii.Error, IndexError, ValueError):
        return False

def is_url(s: str) -> bool:
    """Check if a string is a valid URL."""
    return validators.url(s)

def download_image(url: str) -> Optional[np.ndarray]:
    """Downloads an image from a URL and returns it as a NumPy array (BGR)."""
    try:
        response = requests.get(url, stream=True, timeout=10) # Added timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        # Read image directly into numpy array
        image_array = np.asarray(bytearray(response.raw.read()), dtype=np.uint8)
        # Decode image using OpenCV
        img_np = cv2.imdecode(image_array, cv2.IMREAD_COLOR) # Use cv2
        if img_np is None:
             logging.error(f"Failed to decode image from URL: {url}")
             return None
        return img_np
    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading image from {url}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error processing image from {url}: {e}")
        return None

def resolve_image_input(img_input: str) -> Optional[Union[str, np.ndarray]]:
    """Determines if input is path, URL, or base64 and returns appropriate format for DeepFace."""
    if is_url(img_input):
        logging.info(f"Input identified as URL: {img_input}")
        # Download and return as numpy array (BGR)
        # Need opencv: pip install opencv-python
        global cv2
        try:
            import cv2
        except ImportError:
             logging.error("OpenCV is required for URL image processing. pip install opencv-python")
             return None
        return download_image(img_input)
    elif is_base64(img_input):
        # logging.info("Input identified as base64 string.")
        # DeepFace handles base64 strings directly
        return img_input
    elif os.path.exists(img_input):
        logging.info(f"Input identified as existing file path: {img_input}")
        # DeepFace handles file paths directly
        return img_input
    else:
        logging.warning(f"Input '{img_input}' is not a valid path, URL, or base64 string.")
        return None

def extract_faces_from_image(
    img_data: Union[str, np.ndarray],
    detector_backend: str = settings.DETECTOR_BACKEND,
    enforce_detection: bool = False, # Default to False to handle images w/o faces gracefully
    align: bool = True
) -> List[Dict[str, Any]]:
    """Extracts faces using DeepFace. Handles path, URL, base64."""
    
    processed_input = resolve_image_input(img_data) if isinstance(img_data, str) else img_data
    if processed_input is None:
         # If input was a string and couldn't be resolved
         if isinstance(img_data, str):
              logging.error(f"Could not resolve image input: {img_data}")
              # Raise an error or return empty list based on desired behavior
              # For now, returning empty list to avoid breaking batch processing
              return [] 
         # If input was already numpy array (e.g., from URL download)
         else:
             # This case implies the input was already processed, so proceed?
             # Or maybe the array itself was invalid? Let DeepFace handle it.
             processed_input = img_data

    try:
        # Note: extract_faces returns RGB float images by default in the 'face' key
        faces = DeepFace.extract_faces(
            img_path=processed_input, # Use the resolved input
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            anti_spoofing=False # Keep False for now, can be added as param
        )
        # Ensure result is always a list, even if empty or None
        return faces if faces else []
    except Exception as e:
        # Log the specific image identifier if possible
        img_identifier = img_data if isinstance(img_data, str) else "numpy_array"
        logging.error(f"Error in DeepFace extract_faces for {img_identifier}: {e}")
        return [] # Return empty list on error

def find_matches_in_blacklist(
    img_data: Union[str, np.ndarray],
    db_path: str = settings.BLACKLIST_DB_PATH,
    model_name: str = settings.MODEL_NAME,
    distance_metric: str = settings.DISTANCE_METRIC,
    detector_backend: str = settings.DETECTOR_BACKEND, # Use same detector for consistency
    threshold: Optional[float] = None, # Allow overriding default, but don't use config value here
    align: bool = True,
    refresh_database: bool = False # Keep False for performance unless explicitly needed
) -> List[List[Dict[str, Any]]]:
    """Compares faces in img_data against the blacklist db using DeepFace.find."""

    # Resolve input image first
    processed_input = resolve_image_input(img_data) if isinstance(img_data, str) else img_data
    if processed_input is None:
        img_identifier = img_data if isinstance(img_data, str) else "numpy_array"
        logging.error(f"Could not resolve image input for blacklist check: {img_identifier}")
        return []
        
    # Use configured threshold if not provided
    # REMOVED: Let DeepFace.find use its internal default by passing None
    # effective_threshold = threshold if threshold is not None else BLACKLIST_THRESHOLD 

    try:
        # Use batched=True for potentially better performance with multiple faces
        matches = DeepFace.find(
            img_path=processed_input, # Pass the resolved input
            db_path=db_path,
            model_name=model_name,
            distance_metric=distance_metric,
            detector_backend=detector_backend,
            enforce_detection=False, # Don't fail if no face found in input
            align=align,
            threshold=threshold, # Pass the explicit threshold from request, or None for default
            silent=True,
            refresh_database=refresh_database,
            # batched=True # REMOVED: Now returns List[pd.DataFrame]
        )
        # Ensure result is always a list, even if empty or None
        return matches if matches else []
    except ValueError as ve:
        # Handle specific case where db_path might be empty or invalid
        if "does not exist" in str(ve) or "is not a directory" in str(ve):
             logging.error(f"Blacklist path error for {db_path}: {ve}")
        elif "No item found in db_path" in str(ve):
             logging.warning(f"Blacklist path {db_path} is empty.")
        else:
            img_identifier = img_data if isinstance(img_data, str) else "numpy_array"
            logging.error(f"ValueError in DeepFace find for : {ve}")
        return [] # Return empty list on known ValueErrors
    except Exception as e:
        img_identifier = img_data if isinstance(img_data, str) else "numpy_array"
        logging.error(f"Error in DeepFace find for : {e}")
        return [] # Return empty list on other errors 

async def refresh_blacklist_index(db_path: str = settings.BLACKLIST_DB_PATH):
    """
    Forces DeepFace to refresh its index (.pkl file) for the given database path.
    This is done by calling DeepFace.find with refresh_database=True.
    We use a dummy image path as we only care about the refresh side effect.
    """
    # Find requires a valid image, even if we don't care about the result.
    # Create a dummy small black image array if no real image is easily accessible.
    dummy_img = np.zeros((10, 10, 3), dtype=np.uint8) 
    
    # Check if db_path is accessible and non-empty before attempting refresh
    if not os.path.isdir(db_path):
        logging.warning(f"Cannot refresh index: Blacklist path '{db_path}' is not a valid directory.")
        return
    if not os.listdir(db_path):
        logging.info(f"Skipping index refresh: Blacklist path '{db_path}' is empty.")
        # Delete any potentially orphaned .pkl file if the directory is now empty
        pkl_files = [f for f in os.listdir(os.path.dirname(db_path)) if f.startswith("ds_") and f.endswith(".pkl") and db_path in f]
        for pkl_file in pkl_files:
             try:
                 pkl_path = os.path.join(os.path.dirname(db_path), pkl_file)
                 os.remove(pkl_path)
                 logging.info(f"Removed orphaned index file: {pkl_path}")
             except OSError as e:
                 logging.error(f"Error removing orphaned index file {pkl_path}: {e}")
        return

    logging.info(f"Attempting to refresh DeepFace index for '{db_path}'...")
    try:
        # Call find just for the refresh side effect. Parameters should match find_matches.
        _ = DeepFace.find(
            img_path=dummy_img, # Use dummy image
            db_path=db_path,
            model_name=settings.MODEL_NAME, # Use configured model
            distance_metric=settings.DISTANCE_METRIC, # Use configured metric
            detector_backend=settings.DETECTOR_BACKEND, # Use configured detector
            enforce_detection=False,
            align=True,
            threshold=None, # Use internal default threshold for refresh check
            silent=True,
            refresh_database=True # THE IMPORTANT PART
        )
        logging.info(f"DeepFace index refresh for '{db_path}' completed.")
    except ValueError as ve:
        # Catch potential errors if db_path is empty after deletion etc.
         if "No item found in db_path" in str(ve):
             logging.info(f"Index refresh unnecessary: Blacklist path '{db_path}' is empty after check.")
         else:
            logging.error(f"ValueError during DeepFace index refresh for '{db_path}': {ve}")
    except Exception as e:
        logging.error(f"Error during DeepFace index refresh for '{db_path}': {e}") 

async def save_incoming_image(img_input: str, output_dir: str = settings.PROCESSED_IMAGES_OUTPUT_DIR) -> Optional[str]:
    """
    Saves a copy of the input image (path, URL, or base64) to the specified output directory.
    Uses a UUID for a unique filename.
    Returns the absolute path to the saved file within the server/container environment, or None on failure.
    """
    os.makedirs(output_dir, exist_ok=True)
    img_extension = ".jpg" # Default extension, try to get better one
    unique_filename = f"{uuid.uuid4()}{img_extension}"
    saved_file_path = os.path.abspath(os.path.join(output_dir, unique_filename))

    try:
        if is_url(img_input):
            logging.info(f"Saving image from URL: {img_input}")
            # Try to get extension from URL
            parsed_url = requests.utils.urlparse(img_input)
            _, ext = os.path.splitext(parsed_url.path)
            if ext and len(ext) < 6: # Basic check for valid extension
                 unique_filename = f"{uuid.uuid4()}{ext.lower()}"
                 saved_file_path = os.path.abspath(os.path.join(output_dir, unique_filename))
            
            response = await asyncio.to_thread(requests.get, img_input, stream=True, timeout=15) # Use asyncio.to_thread for blocking request
            response.raise_for_status()
            async with aiofiles.open(saved_file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    await f.write(chunk)
            logging.info(f"Saved URL image to: {saved_file_path}")
            return saved_file_path

        elif is_base64(img_input):
            # logging.info("Saving image from base64 string.")
            # Extract data and potential format
            try:
                 header, encoded = img_input.split(',', 1)
                 # e.g., data:image/png;base64
                 img_format = header.split('/')[1].split(';')[0]
                 if img_format and len(img_format) < 6:
                     img_extension = f".{img_format}"
                     unique_filename = f"{uuid.uuid4()}{img_extension}"
                     saved_file_path = os.path.abspath(os.path.join(output_dir, unique_filename))
            except ValueError:
                encoded = img_input # Assume raw base64 if no header
            
            img_data = base64.b64decode(encoded, validate=True)
            async with aiofiles.open(saved_file_path, 'wb') as f:
                await f.write(img_data)
            # logging.info(f"Saved base64 image to: {saved_file_path}")
            return saved_file_path

        elif os.path.exists(img_input):
            logging.info(f"Saving image from file path: {img_input}")
            _, ext = os.path.splitext(img_input)
            if ext:
                 unique_filename = f"{uuid.uuid4()}{ext.lower()}"
                 saved_file_path = os.path.abspath(os.path.join(output_dir, unique_filename))
            # Use shutil.copy for file paths (consider making async if very large files expected)
            await asyncio.to_thread(shutil.copy, img_input, saved_file_path) 
            logging.info(f"Saved file path image to: {saved_file_path}")
            return saved_file_path
            
        else:
            logging.warning(f"Cannot save input: Not a valid path, URL, or base64 string: {img_input}")
            return None
            
    except Exception as e:
        logging.exception(f"Error saving incoming image '{img_input[:50]}...' to {output_dir}: {e}")
        # Clean up potentially partially created file
        if os.path.exists(saved_file_path):
             try: os.remove(saved_file_path) 
             except OSError: pass
        return None 

# === Image Drawing Utility ===

def draw_bounding_box_on_image(image_path: str, box_coords: FacialArea, match_status: bool):
    """Draws a bounding box on an image file.

    Args:
        image_path: Path to the image file.
        box_coords: FacialArea object containing coordinates.
        match_status: Boolean indicating if it was a blacklist match (for color).
    """
    try:
        # Check if cv2 is available (might not be if only base64/url used previously)
        global cv2
        if 'cv2' not in globals():
            import cv2
            
        img = cv2.imread(image_path)
        if img is None:
            log.error(f"[Drawing] Failed to read image: {image_path}")
            return
            
        img_h, img_w = img.shape[:2]
        
        # Coordinates from FacialArea (can be int or float)
        x = box_coords.x
        y = box_coords.y
        w = box_coords.w
        h = box_coords.h
        
        if None in [x, y, w, h]:
             log.warning(f"[Drawing] Skipping draw for {image_path}, missing coordinates.")
             return

        # Convert fractional coordinates (0.0-1.0) to absolute pixel values if necessary
        if isinstance(x, float) and x <= 1.0: x = int(x * img_w)
        if isinstance(y, float) and y <= 1.0: y = int(y * img_h)
        if isinstance(w, float) and w <= 1.0: w = int(w * img_w)
        if isinstance(h, float) and h <= 1.0: h = int(h * img_h)
        
        # Ensure coordinates are integers for drawing
        try:
             x, y, w, h = int(x), int(y), int(w), int(h)
        except (TypeError, ValueError):
            log.error(f"[Drawing] Invalid coordinate types for {image_path}: x={x}, y={y}, w={w}, h={h}")
            return

        # Define color based on match status (BGR)
        color = (0, 0, 255) if match_status else (0, 255, 0) # Red for match, Green otherwise
        thickness = 2

        # Draw the rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
        
        # Overwrite the original image file
        success = cv2.imwrite(image_path, img)
        if success:
             log.info(f"[Drawing] Successfully drew bounding box on: {image_path}")
        else:
             log.error(f"[Drawing] Failed to save annotated image: {image_path}")
             
    except ImportError:
         log.warning("OpenCV not installed, cannot draw bounding boxes. Please install opencv-python.")
    except Exception as e:
        log.exception(f"[Drawing] Error drawing bounding box on {image_path}: {e}")

# === Image Cropping Utility ===

def crop_and_save_face(original_image_path: str, face_coords: FacialArea, output_dir: str) -> Optional[str]:
    """Crops a face region from an image and saves it to a specified directory.

    Args:
        original_image_path: Path to the source image file.
        face_coords: FacialArea object containing coordinates for the crop.
        output_dir: Base directory to save the cropped face image.

    Returns:
        The absolute path to the saved cropped image file, or None on failure.
    """
    try:
        # Ensure OpenCV is available
        global cv2
        if 'cv2' not in globals():
            import cv2

        img = cv2.imread(original_image_path)
        if img is None:
            log.error(f"[Crop] Failed to read image for cropping: {original_image_path}")
            return None
            
        img_h, img_w = img.shape[:2]
        
        # Get coordinates
        x = face_coords.x
        y = face_coords.y
        w = face_coords.w
        h = face_coords.h
        
        if None in [x, y, w, h]:
             log.warning(f"[Crop] Skipping crop for {original_image_path}, missing coordinates.")
             return None

        # Convert fractional coordinates (0.0-1.0) to absolute pixel values if necessary
        if isinstance(x, float) and x <= 1.0: x = int(x * img_w)
        if isinstance(y, float) and y <= 1.0: y = int(y * img_h)
        if isinstance(w, float) and w <= 1.0: w = int(w * img_w)
        if isinstance(h, float) and h <= 1.0: h = int(h * img_h)
        
        # Ensure coordinates are integers for slicing
        try:
             x, y, w, h = int(x), int(y), int(w), int(h)
        except (TypeError, ValueError):
            log.error(f"[Crop] Invalid coordinate types for {original_image_path}: x={x}, y={y}, w={w}, h={h}")
            return None

        # === Add Padding ===
        padding_ratio = settings.CROPPED_FACE_PADDING_RATIO #padding
        padding_w = int(w * padding_ratio)
        padding_h = int(h * padding_ratio)
        
        # Adjust coordinates (subtract half padding from top-left, add full to size)
        padded_x = x - (padding_w // 2)
        padded_y = y - (padding_h // 2)
        padded_w = w + padding_w
        padded_h = h + padding_h
        # === End Padding ===
        
        # Calculate crop boundaries using padded coordinates, ensuring they are within image dimensions
        x1 = max(0, padded_x)
        y1 = max(0, padded_y)
        x2 = min(img_w, padded_x + padded_w) 
        y2 = min(img_h, padded_y + padded_h)
        
        # Check if calculated crop area is valid
        if x1 >= x2 or y1 >= y2:
            log.warning(f"[Crop] Invalid crop dimensions for {original_image_path} after clamping. x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            return None

        # Perform cropping
        cropped_face = img[y1:y2, x1:x2]

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate unique filename for the cropped image
        original_filename = os.path.splitext(os.path.basename(original_image_path))[0]
        # Use the UUID from the original saved name if possible (remove extension first)
        try:
            original_uuid_part = original_filename.split('_')[0] if '_' in original_filename else original_filename
            if len(uuid.UUID(original_uuid_part).hex) == 32:
                 crop_filename = f"{original_uuid_part}_cropped.png" # Use original UUID + suffix
            else:
                 raise ValueError("Not a valid UUID part")
        except ValueError:
            crop_filename = f"{uuid.uuid4()}_cropped.png" # Fallback to new UUID
            
        output_path = os.path.abspath(os.path.join(output_dir, crop_filename))

        # Save the cropped image
        success = cv2.imwrite(output_path, cropped_face)
        if success:
            log.info(f"[Crop] Successfully saved cropped face to: {output_path}")
            return output_path
        else:
            log.error(f"[Crop] Failed to save cropped face image: {output_path}")
            return None

    except ImportError:
         log.warning("OpenCV not installed, cannot crop faces. Please install opencv-python.")
         return None
    except Exception as e:
        log.exception(f"[Crop] Error cropping face from {original_image_path}: {e}")
        return None 