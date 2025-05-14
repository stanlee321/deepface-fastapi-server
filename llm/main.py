import json # Added for parsing MQTT payload
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
from libs.event_processor import  process_llm_request_event_data

# Import the specific processing function from the new locatio
from libs.auth import signJWT

# Import Vanna related functions
from services.vanna_service import initialize_vanna, get_vanna, MyVanna # MyVanna for type hinting

# Import the models
from models import NLQueryRequest, NLQueryResponse

# --- Import Pydantic for payload modeling --- 
from pydantic import BaseModel
from typing import List, Any, Optional # Added Optional

# --- Define Pydantic model for smart_process payload --- 
class SmartProcessPayload(BaseModel):
    # Define expected fields here, e.g.:
    request_id: str
    data: dict
    # Add other fields as necessary based on what process_smart_request expects

# --- Define Pydantic models for paginated responses ---
class RawDescriptionItem(BaseModel):
    id: int
    raw_description: str
    image_url: str
    code: str
    app_type: str
    status: str
    created_at: str # Assuming datetime is returned as string
    updated_at: str # Assuming datetime is returned as string

class ProcessedDescriptionItem(BaseModel):
    id: int
    processed_description: str
    code: str
    app_type: str
    status: str
    created_at: str # Assuming datetime is returned as string
    updated_at: str # Assuming datetime is returned as string

class PaginatedRawDescriptionsResponse(BaseModel):
    total_count: int
    limit: int
    offset: int
    data: List[RawDescriptionItem]

class PaginatedProcessedDescriptionsResponse(BaseModel):
    total_count: int
    limit: int
    offset: int
    data: List[ProcessedDescriptionItem]

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

# Refactored Lifespan manager
async def lifespan(app: FastAPI):
    # Startup: Connect to MQTT broker using the correct method
    print("Connecting to MQTT broker...")
    await fast_mqtt.mqtt_startup() # Correct method based on docs
    print("MQTT Connected.")

    # Initialize DB (keep existing logic)
    app.state.db = db.init_db()
    
    # Initialize Vanna
    print("Initializing Vanna service...")
    # Store the Vanna instance on app.state for easy access if needed elsewhere,
    # but we'll primarily use the get_vanna() dependency.
    app.state.vanna_instance = initialize_vanna()
    print("Vanna service initialized.")

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
    TODO: SECURITY RISK - Replace hardcoded credentials with secure authentication!
    Checks against username 'test', password 'test'.
    """
    # We need JWT_SECRET here, loaded via settings
    # TODO: Ensure signJWT has access to settings.JWT_SECRET
    if form_data.username == "test" and form_data.password == "test":
        # Pass the secret from settings to signJWT if needed
        # Adjust signJWT in libs/auth.py to accept the secret as an argument
        # Or have libs/auth.py import settings itself (potential circular import risk)
        # For now, assuming signJWT can access it or doesn't need it passed explicitly
        # If signJWT needs the secret, modify libs/auth.py to import settings
        return signJWT(form_data.username)
    else:
        raise HTTPException(status_code=401, detail="Invalid username or password")

@app.post("/query_processed_descriptions_nl", response_model=NLQueryResponse)
async def query_processed_descriptions_nl(
    request_body: NLQueryRequest, 
    vn: MyVanna = Depends(get_vanna) # Use dependency injection
):
    """
    Ask a natural language question about the processed_descriptions table.
    """
    question = request_body.question
    print(f"Received natural language query: {question}")
    try:
        # Vanna's ask method generates SQL, runs it, and can also generate a plot (which we ignore here)
        # It returns a tuple: (sql, df, fig)
        # sql, df, fig = vn.ask(question=question) # vn.ask() can also auto-train if successful
        
        # Or, for more control:
        # Add allow_llm_to_see_data=True to let Vanna's LLM inspect data if needed for the query
        sql_query = vn.generate_sql(question=question, allow_llm_to_see_data=True)
        print(f"Generated SQL: {sql_query}")

        if not sql_query:
            return NLQueryResponse(question=question, sql_query="", error="Could not generate SQL query.")

        # Vanna will use its connected SQLite database to run this
        df_results = vn.run_sql(sql=sql_query)
        
        if df_results is not None:
            # Convert DataFrame to a list of dicts for JSON response
            results_list = df_results.to_dict(orient="records")
            # Auto-train Vanna on successful queries
            vn.train(question=question, sql=sql_query)
            return NLQueryResponse(question=question, sql_query=sql_query, results=results_list)
        else:
            return NLQueryResponse(question=question, sql_query=sql_query, results=None, error="Query executed but returned no data.")

    except Exception as e:
        print(f"Error processing natural language query: {e}")
        # Optionally, try to get Vanna to fix the SQL if there's an error
        # This is a more advanced Vanna feature not shown in the basic README example
        # For now, just return the error.
        # if 'sql_query' in locals() and sql_query:
        #     try:
        #         # vn.train(question=question, sql=sql_query) # Add to training data even if it failed, for review
        #         # fixed_sql = vn.fix_sql(question=question, sql=sql_query, error=str(e))
        #         # ... then try running fixed_sql
        #     except Exception as fix_e:
        #          print(f"Error trying to fix SQL: {fix_e}")
        error_message = str(e)
        if 'sql_query' not in locals():
            sql_query = "Error before SQL generation"
        return NLQueryResponse(question=question, sql_query=sql_query, results=None, error=error_message)

@app.get("/raw_descriptions/", response_model=PaginatedRawDescriptionsResponse)
async def get_raw_descriptions_paginated_endpoint(
    page: int = 1, 
    page_size: int = 10, 
    code: Optional[str] = None
):
    """
    Get paginated raw descriptions.
    Optionally filter by code.
    """
    offset = (page - 1) * page_size
    items, total_count = db.get_raw_descriptions_paginated(limit=page_size, offset=offset, code=code)
    
    # Convert tuple results to RawDescriptionItem
    data = [
        RawDescriptionItem(
            id=item[0], 
            raw_description=item[1], 
            image_url=item[2], 
            code=item[3], 
            app_type=item[4], 
            status=item[5],
            created_at=str(item[6]), # Ensure datetime is stringified
            updated_at=str(item[7])  # Ensure datetime is stringified
        ) for item in items
    ]
    return PaginatedRawDescriptionsResponse(
        total_count=total_count,
        limit=page_size,
        offset=offset,
        data=data
    )

@app.get("/processed_descriptions/", response_model=PaginatedProcessedDescriptionsResponse)
async def get_processed_descriptions_paginated_endpoint(
    page: int = 1, 
    page_size: int = 10, 
    code: Optional[str] = None
):
    """
    Get paginated processed descriptions.
    Optionally filter by code.
    """
    offset = (page - 1) * page_size
    items, total_count = db.get_processed_descriptions_paginated(limit=page_size, offset=offset, code=code)
    
    # Convert tuple results to ProcessedDescriptionItem
    data = [
        ProcessedDescriptionItem(
            id=item[0], 
            processed_description=item[1], 
            code=item[2], 
            app_type=item[3], 
            status=item[4],
            created_at=str(item[5]), # Ensure datetime is stringified
            updated_at=str(item[6])  # Ensure datetime is stringified
        ) for item in items
    ]
    return PaginatedProcessedDescriptionsResponse(
        total_count=total_count,
        limit=page_size,
        offset=offset,
        data=data
    )



