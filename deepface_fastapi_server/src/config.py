import os
from deepface import DeepFace

# --- DeepFace Configuration ---
DETECTOR_BACKEND = os.environ.get("DETECTOR_BACKEND", "retinaface")
MODEL_NAME = os.environ.get("MODEL_NAME", "Facenet")
DISTANCE_METRIC = os.environ.get("DISTANCE_METRIC", "cosine")

# --- Blacklist Configuration ---
# IMPORTANT: Update this path to the actual location of your blacklist images folder
# It should be relative to the project root or an absolute path.
BLACKLIST_DB_PATH = os.environ.get("BLACKLIST_DB_PATH", "/Users/stanleysalvatierra/Desktop/2024/lucam/face/deepface_fastapi_server/blacklist_db")

# Calculate threshold based on model and metric
# REMOVED: Threshold will be determined internally by DeepFace.find
# try:
#     BLACKLIST_THRESHOLD = DeepFace.find_threshold(MODEL_NAME, DISTANCE_METRIC)
# except ValueError as e:
#     print(f"Warning: Could not automatically find threshold for {MODEL_NAME}/{DISTANCE_METRIC}. Using default 0.40. Error: {e}")
#     BLACKLIST_THRESHOLD = 0.40 # Default fallback

# --- Server Configuration ---
API_TITLE = "DeepFace Enhanced API"
API_VERSION = "0.1.0"

# --- Output Configuration ---
# Directory to save copies of processed images
# Path relative to the project root. Ensure it's mounted in Docker.
PROCESSED_IMAGES_OUTPUT_DIR = os.environ.get("PROCESSED_IMAGES_OUTPUT_DIR", "/Users/stanleysalvatierra/Desktop/2024/lucam/face/deepface_fastapi_server/processed_images_output") 