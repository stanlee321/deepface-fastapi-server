from pathlib import Path

# Import services
from services.llm import describe_image, encode_image_to_data_uri

# Import settings and db module relative to libs
from libs import db

async def process_llm_request_event_data(payload: dict):
    """
    Processes the initial MQTT event: copies images, updates DB status,
    and publishes a new event to trigger processing.
    
    Args:
        payload (dict): The dictionary parsed from the incoming MQTT message.
        fast_mqtt (FastMQTT): The FastMQTT client instance used for publishing.
    """
    # print(f"MQTT Background task started: Processing data {payload}")
    
    infraction_code = payload.get("code")
    event_type = payload.get("event_type")
    app_type = payload.get("app_type") # Extract app_type
    image_url = payload.get("image_url")

    try:
        db_id = db.create_raw_description("", image_url, infraction_code, app_type, 'RECEIVED_MQTT')
    except Exception as e:
        # Print in red
        print(f"\033[91mError storing initial MQTT payload to DB for {infraction_code}: {e}\033[0m")
        return

    # --- Describe Image ---
    try:
        local_image_path = Path(image_url)
        if local_image_path and local_image_path.exists():
            data_uri_local = encode_image_to_data_uri(local_image_path)
            if data_uri_local:
                description = describe_image(data_uri_local)
    except Exception as e:
        print(f"Error describing image for {infraction_code}: {e}")
        db.update_raw_description_status_and_code(db_id, description, 'IMAGE_DESCRIEBER_FAILED', infraction_code)
        return
    # --- Update DB Status & Publish Processing Event ---
    try:
        db.update_raw_description_status_and_code(db_id, description, 'IMAGE_DESCRIEBER_SUCCESS', infraction_code)
    except Exception as e:
        print(f"Error updating status or publishing process event for {infraction_code}: {e}")
        db.update_raw_description_status_and_code(db_id, description, 'IMAGE_DESCRIEBER_FAILED', infraction_code)
        return
