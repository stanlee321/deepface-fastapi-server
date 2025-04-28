# Example using 'databases' library
from src.database import database, blacklist_table
from src.models import BlacklistCreate, BlacklistRecord # Ensure correct import
from typing import List, Optional
import logging

async def add_person(payload: BlacklistCreate) -> Optional[int]:
    """Adds a person to the blacklist table (without image dir initially)."""
    query = blacklist_table.insert().values(name=payload.name, reason=payload.reason)
    try:
        # The execute method returns the primary key of the inserted row
        last_record_id = await database.execute(query=query)
        return last_record_id
    except Exception as e:
        # Log error e.g., duplicate name constraint (IntegrityError)
        logging.error(f"DB Error adding person '{payload.name}': {e}")
        # Depending on DB/driver, IntegrityError might need specific handling
        # from sqlalchemy.exc import IntegrityError
        # except IntegrityError:
        #     logging.warning(f"Blacklist entry with name '{payload.name}' already exists.")
        #     return None
        return None

async def get_person(id: int) -> Optional[BlacklistRecord]:
    """Retrieves a single blacklist entry by its ID."""
    query = blacklist_table.select().where(blacklist_table.c.id == id)
    result = await database.fetch_one(query=query)
    # Use model_validate for Pydantic v2
    return BlacklistRecord.model_validate(result) if result else None

async def get_person_by_name(name: str) -> Optional[BlacklistRecord]:
    """Retrieves a single blacklist entry by its name."""
    query = blacklist_table.select().where(blacklist_table.c.name == name)
    result = await database.fetch_one(query=query)
    # Ensure model_validate handles potentially missing reference_image_dir
    return BlacklistRecord.model_validate(result) if result else None

async def get_all_persons() -> List[BlacklistRecord]:
    """Retrieves all entries from the blacklist table."""
    query = blacklist_table.select()
    results = await database.fetch_all(query=query)
    # Use list comprehension with model_validate
    # Ensure model_validate handles potentially missing reference_image_dir
    return [BlacklistRecord.model_validate(result) for result in results]

async def delete_person(id: int) -> Optional[int]:
    """Deletes a blacklist entry by its ID. Returns number of rows deleted (0 or 1)."""
    query = blacklist_table.delete().where(blacklist_table.c.id == id)
    try:
        # Execute returns the number of rows matched by the where clause
        num_deleted = await database.execute(query=query)
        return num_deleted
    except Exception as e:
        logging.error(f"DB Error deleting person with ID {id}: {e}")
        return None

# Add update function if needed
async def update_person(
    id: int, 
    payload: BlacklistCreate, 
    image_dir_path: Optional[str] = None # Add optional image dir path
) -> Optional[int]:
    """Updates a blacklist entry by ID. Can optionally set image dir path."""
    values_to_update = {"name": payload.name, "reason": payload.reason}
    if image_dir_path is not None:
        values_to_update["reference_image_dir"] = image_dir_path
        
    query = (
        blacklist_table.update()
        .where(blacklist_table.c.id == id)
        .values(**values_to_update)
        # Optionally, add returning(blacklist_table.c.id) if driver supports it and needed
    )
    try:
        num_updated = await database.execute(query=query)
        if num_updated > 0:
            return id
        else:
            return None # Indicate not found or no change
    except Exception as e:
        logging.error(f"DB Error updating person with ID {id}: {e}")
        return None 