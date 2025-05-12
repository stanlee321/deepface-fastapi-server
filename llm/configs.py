import os
from typing import Literal
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env file. If a variable is set in both .env and environment, environment takes precedence.
load_dotenv()

class Settings(BaseSettings):
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_TEMPERATURE: float = os.getenv("OPENAI_TEMPERATURE", 0.4)
    OPENAI_MAX_TOKENS: int = os.getenv("OPENAI_MAX_TOKENS", 1000)
    OPENAI_STORE_RESULTS: bool = os.getenv("OPENAI_STORE_RESULTS", False)
    OPENAI_SYSTEM_PROMPT: str = os.getenv("OPENAI_SYSTEM_PROMPT", "")

    PIPELINE_INTERVAL: int = os.getenv("PIPELINE_INTERVAL", 2)


    MQTT_BROKER_HOST: str = os.getenv("MQTT_BROKER_HOST", "localhost")
    MQTT_BROKER_PORT: int = os.getenv("MQTT_BROKER_PORT", 1883)
    MQTT_USERNAME: str = os.getenv("MQTT_USERNAME", "admin")
    MQTT_PASSWORD: str = os.getenv("MQTT_PASSWORD", "admin")
    MQTT_CLIENT_ID: str = os.getenv("MQTT_CLIENT_ID", "llm")
    
    MQTT_PROCESS_TOPIC: str = os.getenv("MQTT_PROCESS_TOPIC", "llm/process")

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