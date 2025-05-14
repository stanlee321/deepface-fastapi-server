import numpy as np

from typing import List, Union, Dict, Any
from config import settings
from crud import common
from crud.weapon_detection import get_weapon_detector

import logging
log = logging.getLogger(__name__)


def extract_weapons_from_image(
    img_data: Union[str, np.ndarray],
) -> List[Dict[str, Any]]:
    """Extracts weapons using DeepFace. Handles path, URL, base64."""
    
    processed_input = common.resolve_image_input(img_data) if isinstance(img_data, str) else img_data
    if processed_input is None:
         # If input was a string and couldn't be resolved
         if isinstance(img_data, str):
            #   logging.error(f"Could not resolve image input: {img_data}")
              # Raise an error or return empty list based on desired behavior
              # For now, returning empty list to avoid breaking batch processing
              return [] 
         # If input was already numpy array (e.g., from URL download)
         else:
             # This case implies the input was already processed, so proceed?
             # Or maybe the array itself was invalid? Let DeepFace handle it.
             processed_input = img_data
    weapon_detector = get_weapon_detector()
    # Note: extract_weapons returns RGB float images by default in the 'weapon' key
    weapons = weapon_detector.process_frame(processed_input, 
                                            confidence_threshold=settings.WEAPON_DETECTION_CONFIDENCE_THRESHOLD)
    print("WEAPONS DETECTED...weapons", weapons)
    # Ensure result is always a list, even if empty or None
    return weapons if weapons else []




