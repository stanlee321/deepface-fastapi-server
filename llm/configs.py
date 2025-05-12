import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env file. If a variable is set in both .env and environment, environment takes precedence.
load_dotenv()

class Settings(BaseSettings):
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")
    OPENAI_TEMPERATURE: float = os.getenv("OPENAI_TEMPERATURE", 0.4)
    OPENAI_MAX_TOKENS: int = os.getenv("OPENAI_MAX_TOKENS", 1000)
    OPENAI_STORE_RESULTS: bool = os.getenv("OPENAI_STORE_RESULTS", False)
    OPENAI_SYSTEM_PROMPT: str = os.getenv("OPENAI_SYSTEM_PROMPT", "")

    PIPELINE_INTERVAL: int = os.getenv("PIPELINE_INTERVAL", 2)


    MQTT_BROKER_HOST: str = os.getenv("MQTT_BROKER_HOST", "localhost")
    MQTT_BROKER_PORT: int = os.getenv("MQTT_BROKER_PORT", 1883)
    MQTT_USERNAME: Optional[str] = os.getenv("MQTT_USERNAME", None)
    MQTT_PASSWORD: Optional[str] = os.getenv("MQTT_PASSWORD", None)
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
