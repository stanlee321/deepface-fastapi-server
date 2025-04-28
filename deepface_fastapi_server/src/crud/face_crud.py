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

# Import config to get default parameters
from src.config import (
    DETECTOR_BACKEND, MODEL_NAME, DISTANCE_METRIC, BLACKLIST_DB_PATH,
    PROCESSED_IMAGES_OUTPUT_DIR # Import the new output dir config
)

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
        logging.info("Input identified as base64 string.")
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
    detector_backend: str = DETECTOR_BACKEND,
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
    db_path: str = BLACKLIST_DB_PATH,
    model_name: str = MODEL_NAME,
    distance_metric: str = DISTANCE_METRIC,
    detector_backend: str = DETECTOR_BACKEND, # Use same detector for consistency
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
            logging.error(f"ValueError in DeepFace find for {img_identifier}: {ve}")
        return [] # Return empty list on known ValueErrors
    except Exception as e:
        img_identifier = img_data if isinstance(img_data, str) else "numpy_array"
        logging.error(f"Error in DeepFace find for {img_identifier}: {e}")
        return [] # Return empty list on other errors 

async def refresh_blacklist_index(db_path: str = BLACKLIST_DB_PATH):
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
            model_name=MODEL_NAME, # Use configured model
            distance_metric=DISTANCE_METRIC, # Use configured metric
            detector_backend=DETECTOR_BACKEND, # Use configured detector
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

async def save_incoming_image(img_input: str, output_dir: str = PROCESSED_IMAGES_OUTPUT_DIR) -> Optional[str]:
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
            logging.info("Saving image from base64 string.")
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
            logging.info(f"Saved base64 image to: {saved_file_path}")
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