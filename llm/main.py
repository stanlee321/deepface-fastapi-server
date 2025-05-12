import json # Added for parsing MQTT payload
from contextlib import asynccontextmanager # Added for lifespan
import asyncio # Import asyncio

from jobs.pipeline import pipeline

# APScheduler imports
from apscheduler.schedulers.asyncio import AsyncIOScheduler


from fastapi import (FastAPI,
                     HTTPException, 
                     Depends) # Added BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm
from fastapi_mqtt import FastMQTT, MQTTConfig # Added MQTT imports

# Import the settings object
from configs import settings


# Import our DB helper functions from our data_processor module.
from libs import db

# Import the specific processing function from the new location
from libs.event_processor import process_llm_request_event_data

# Import the specific processing function from the new locatio
from libs.auth import signJWT


# MQTT Client Setup - Use settings attributes
mqtt_config = MQTTConfig(
    host = settings.MQTT_BROKER_HOST,
    port= settings.MQTT_BROKER_PORT,
    username= settings.MQTT_USERNAME,
    password= settings.MQTT_PASSWORD,
    client_id= settings.MQTT_CLIENT_ID,
    keepalive=60,
    # Add ssl context if needed
    # ssl=...
)

fast_mqtt = FastMQTT(config=mqtt_config)

# Initialize the scheduler
scheduler = AsyncIOScheduler()

# Lifespan manager for MQTT connection and Scheduler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Connect to MQTT broker using the correct method
    print("Connecting to MQTT broker...")
    await fast_mqtt.mqtt_startup() # Correct method based on docs
    print("MQTT Connected.")

    # Initialize DB (keep existing logic)
    app.state.db = db.init_db()

    # Add the job to the scheduler
    # Run pipeline every 2 minutes
    # Note: APScheduler runs synchronous functions in a thread pool by default,
    # so it won't block the main asyncio event loop.
    scheduler.add_job(pipeline, 'interval', minutes=settings.PIPELINE_INTERVAL,  id='pipeline_job', replace_existing=True)
    print(f"Scheduled pipeline job every {settings.PIPELINE_INTERVAL} minutes.")

    # Start the scheduler
    scheduler.start()
    print("Scheduler started.")

    yield  # Application runs here

    # Shutdown: Disconnect from MQTT broker using the correct method
    print("Disconnecting from MQTT broker...")
    await fast_mqtt.mqtt_shutdown() # Correct method based on docs
    print("MQTT Disconnected.")

    # Shutdown the scheduler gracefully
    print("Shutting down scheduler...")
    scheduler.shutdown()
    print("Scheduler shut down.")

app = FastAPI(lifespan=lifespan) # Use the lifespan manager
# app.state.db = db.init_db() # Moved db init into lifespan

# MQTT Message Handler (Initial Event)
@fast_mqtt.on_connect()
def connect(client, flags, rc, properties):
    print("FastMQTT Connected callback: ", client, flags, rc, properties)
    # Subscribe upon successful connection
    fast_mqtt.client.subscribe(settings.MQTT_PROCESS_TOPIC) # Also subscribe to the process topic


# # NEW MQTT Handler for Processing Requests
@fast_mqtt.subscribe(settings.MQTT_PROCESS_TOPIC)
async def handle_process_request(client, topic, payload, qos, properties):
    """
    Handles messages received on the MQTT_PROCESS_TOPIC.
    Calls the Face API for detection.
    """
    print(f"Received message on process topic '{topic}': {payload.decode()}")
    try:
        data = json.loads(payload.decode())

        task = asyncio.create_task(process_llm_request_event_data(data))
        task.add_done_callback(lambda t: print(f"Task process_smart_request finished. Exception: {t.exception()}") if t.exception() else None)
        
        print(f"Scheduled background processing using asyncio for message from topic '{topic}'")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from process topic '{topic}': {payload.decode()}")
    except Exception as e:
        print(f"Error handling message from process topic '{topic}': {e}")

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Simple login endpoint for demonstration purposes.
    Replace this with your actual authentication logic.
    NOTE: Assumes username 'test', password 'test'. JWT_SECRET is loaded via settings.
    """
    # We need JWT_SECRET here, loaded via settings
    if form_data.username == "test" and form_data.password == "test":
        # Pass the secret from settings to signJWT if needed
        # Adjust signJWT in libs/auth.py to accept the secret as an argument
        # Or have libs/auth.py import settings itself (potential circular import risk)
        # For now, assuming signJWT can access it or doesn't need it passed explicitly
        # If signJWT needs the secret, modify libs/auth.py to import settings
        return signJWT(form_data.username)
    else:
        raise HTTPException(status_code=401, detail="Invalid username or password")

