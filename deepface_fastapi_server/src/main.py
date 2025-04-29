from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import connect_db, disconnect_db, create_db_and_tables
from api.router import api_router
from config import settings

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler() # Log to console
        # Optionally add FileHandler here
        # logging.FileHandler("app.log"),
    ]
)
log = logging.getLogger(__name__)

# --- Database Setup ---
# Create tables synchronously before the app starts accepting connections
# Note: In production, use migrations (e.g., Alembic) for schema changes.
log.info("Attempting to create database tables...")
create_db_and_tables()
log.info("Database table check complete.")





# --- FastAPI App Initialization ---
app = FastAPI(title=settings.API_TITLE, version=settings.API_VERSION)

# --- CORS Middleware ---
# Allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
# --- END CORS Middleware ---

@app.on_event("startup")
async def startup_event():
    log.info("Starting up API...")
    await connect_db()
    log.info("Database connection established.")
    # You could preload models or blacklist data here if needed
    # e.g., preload_blacklist_embeddings()

@app.on_event("shutdown")
async def shutdown_event():
    log.info("Shutting down API...")
    await disconnect_db()
    log.info("Database connection closed.")

# --- Include API Routers ---
app.include_router(api_router, prefix="/api/v1") # Add a version prefix

# --- Root Endpoint --- (Optional)
@app.get("/", tags=["Health Check"])
async def root():
    return {"message": f"Welcome to the {settings.API_TITLE}"}

# Add any other middleware or configurations as needed
# Example CORS middleware:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
) 