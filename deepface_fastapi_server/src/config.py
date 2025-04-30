import os
from typing import Literal
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env file. If a variable is set in both .env and environment, environment takes precedence.
load_dotenv()

class Settings(BaseSettings):
    # --- General Face Processing Backend Configuration ---
    # Determines which backend to use: 'deepface' (on-premise) or 'aws_rekognition' (cloud)
    FACE_PROCESSING_BACKEND: Literal['deepface', 'aws_rekognition'] = os.getenv("FACE_PROCESSING_BACKEND", "deepface")

    # --- DeepFace Configuration (Used when FACE_PROCESSING_BACKEND='deepface') ---
    DETECTOR_BACKEND: str =  os.getenv("DETECTOR_BACKEND", "retinaface")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "Facenet")
    DISTANCE_METRIC: str = os.getenv("DISTANCE_METRIC", "cosine")

    # --- Blacklist Configuration ---
    # Path for DeepFace backend's image folder structure
    # NOTE: Default path might need adjustment depending on deployment.
    BLACKLIST_DB_PATH: str = os.getenv("BLACKLIST_DB_PATH","")

    # --- AWS Rekognition Configuration (Used when FACE_PROCESSING_BACKEND='aws_rekognition') ---
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    AWS_S3_BUCKET_NAME: str = os.getenv("AWS_S3_BUCKET_NAME", "face-lucam") # Updated default bucket name
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "AKIAX555555555555555")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "5555555555555555555555555555555555555555")
    AWS_REKOGNITION_COLLECTION_ID: str = os.getenv("AWS_REKOGNITION_COLLECTION_ID", "face_blacklist_collection")
    # AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN) are read automatically
    # by boto3 from environment variables or IAM roles. No need to define them here.

    # --- Server Configuration ---
    API_TITLE: str = os.getenv("API_TITLE", "DeepFace Enhanced API")
    API_VERSION: str = os.getenv("API_VERSION", "0.1.0")

    # --- Output Configuration ---
    # Directory to save copies of processed images
    # NOTE: Default path might need adjustment depending on deployment.
    PROCESSED_IMAGES_OUTPUT_DIR: str = os.getenv("PROCESSED_IMAGES_OUTPUT_DIR", "")
    # Directory to save cropped face images (if feature enabled implicitly by logic)
    CROPPED_FACES_OUTPUT_DIR: str = os.getenv("CROPPED_FACES_OUTPUT_DIR", "")

    # Optional: Database URL (if moving away from hardcoded database.py)
    # DATABASE_URL: str = "sqlite+aiosqlite:///./blacklist.db"

    # --- Feature Flags ---
    DRAW_BOUNDING_BOXES: bool = os.getenv("DRAW_BOUNDING_BOXES", "true").lower() == "true"

    # --- Padding Configuration ---
    CROPPED_FACE_PADDING_RATIO: float = os.getenv("CROPPED_FACE_PADDING_RATIO", 0.40)

    # --- Pydantic Settings Configuration ---
    class Config:
        # Optional: Specify .env file explicitly if needed
        env_file = '.env'
        env_file_encoding = 'utf-8'
        # For case-insensitive environment variables matching
        case_sensitive = False

# Create a single, importable instance of the settings
settings = Settings()

# --- Remove old DeepFace threshold calculation (was already commented out) ---
# from deepface import DeepFace # Keep import removed

# Calculate threshold based on model and metric
# REMOVED: Threshold will be determined internally by DeepFace.find
# try:
#     BLACKLIST_THRESHOLD = DeepFace.find_threshold(MODEL_NAME, DISTANCE_METRIC)
# except ValueError as e:
#     print(f"Warning: Could not automatically find threshold for {MODEL_NAME}/{DISTANCE_METRIC}. Using default 0.40. Error: {e}")
#     BLACKLIST_THRESHOLD = 0.40 # Default fallback 