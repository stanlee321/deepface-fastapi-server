import os
import base64
import validators
import base64
import validators
import requests
import cv2
import numpy as np
import logging
from typing import Optional, Union
import uuid
import aiofiles
import shutil
import asyncio

from models import FacialArea, WeaponArea

from config import settings

log = logging.getLogger(__name__)



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
            # log.info(f"Saving image from URL: {img_input}")
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
            log.info(f"Saved URL image to: {saved_file_path}")
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
            # log.info(f"Saving image from file path: {img_input}")
            _, ext = os.path.splitext(img_input)
            if ext:
                 unique_filename = f"{uuid.uuid4()}{ext.lower()}"
                 saved_file_path = os.path.abspath(os.path.join(output_dir, unique_filename))
            # Use shutil.copy for file paths (consider making async if very large files expected)
            await asyncio.to_thread(shutil.copy, img_input, saved_file_path) 
            log.info(f"Saved file path image to: {saved_file_path}")
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
        print(f"Input identified as existing file path: {img_input}")
        logging.info(f"Input identified as existing file path: {img_input}")
        # DeepFace handles file paths directly
        return img_input
    else:
        logging.warning(f"Input '{img_input}' is not a valid path, URL, or base64 string.")
        return None


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
    
    
    
# === Image Drawing Utility ===

def draw_bounding_box_on_image(image_path: str, box_coords: Union[FacialArea, WeaponArea], match_status: bool = False):
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

def crop_and_save_object(original_image_path: str, object_coords: Union[FacialArea, WeaponArea], output_dir: str) -> Optional[str]:
    """Crops a face region from an image and saves it to a specified directory.

    Args:
        original_image_path: Path to the source image file.
        object_coords: FacialArea or WeaponArea object containing coordinates for the crop.
        output_dir: Base directory to save the cropped object image.

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
        x = object_coords.x
        y = object_coords.y
        w = object_coords.w
        h = object_coords.h
        
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