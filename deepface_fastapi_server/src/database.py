import os
from databases import Database
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, DateTime, Boolean, TEXT
from sqlalchemy.sql import func
import logging

DATABASE_URL = "sqlite+aiosqlite:///./blacklist.db" # Use async sqlite driver
database = Database(DATABASE_URL)
metadata = MetaData()

# Example blacklist table definition
blacklist_table = Table(
    "blacklist",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(100), unique=True, index=True), # Name or identifier
    Column("reason", String(255), nullable=True),
    # Store the relative path to the directory holding reference images for this entry
    Column("reference_image_dir", String(512), nullable=True),
    # Storing embeddings directly in SQLite is generally inefficient for searching.
    # Better to store image paths and use DeepFace's .pkl or a vector DB.
    # For simplicity here, we might just store names/IDs.
    # Column("embedding_path", String(255), nullable=True), # Path to embedding file?
    Column("added_date", DateTime, default=func.now())
)

# New table for processed image results
processed_images_table = Table(
    "processed_images",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("input_identifier", String(1024)), # Original path, URL, or truncated base64
    Column("saved_image_path", String(1024), index=True), # Absolute path within container/server
    Column("processing_timestamp", DateTime, default=func.now(), index=True),
    Column("has_blacklist_match", Boolean, default=False, index=True),
    Column("result_json", TEXT), # Use TEXT for potentially large JSON strings
    Column("cropped_face_path", String(512), nullable=True) # Path to the cropped face image
)

# Create a synchronous engine for metadata creation if needed (databases doesn't do it)
# Remove the async driver part for the sync engine URL
sync_db_url = DATABASE_URL.replace("+aiosqlite", "")
# Ensure the URL starts correctly for SQLAlchemy's create_engine
if sync_db_url.startswith("sqlite:///"): 
    # Correct format for relative path
    pass 
elif sync_db_url.startswith("sqlite:"): 
    # Adjust if it only has one slash
    sync_db_url = "sqlite://" + sync_db_url[len("sqlite:"):]

sync_engine = create_engine(sync_db_url)

def create_db_and_tables():
    try:
        # Create the database directory if it doesn't exist
        db_dir = os.path.dirname(sync_db_url.replace("sqlite:///", ""))
        if db_dir and not os.path.exists(db_dir):
             os.makedirs(db_dir)
             logging.info(f"Created database directory: {db_dir}")
        
        metadata.create_all(sync_engine)
        logging.info("Database tables created (if they didn't exist).")
    except Exception as e:
        logging.error(f"Error creating database tables: {e}")

async def connect_db():
    try:
        await database.connect()
        logging.info("Database connected.")
    except Exception as e:
        logging.error(f"Error connecting to database: {e}")


async def disconnect_db():
    try:
        await database.disconnect()
        logging.info("Database disconnected.")
    except Exception as e:
        logging.error(f"Error disconnecting from database: {e}") 