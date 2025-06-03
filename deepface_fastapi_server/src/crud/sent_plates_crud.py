import logging
from typing import Optional
from datetime import datetime

from database import database, sent_plates_table

log = logging.getLogger(__name__)

async def add_sent_plate(
    plate_text: str,
    code: str,
    confidence: float,
    internal_code: str,
    parking_api_response: str = None
) -> Optional[int]:
    """
    Records that a plate was sent to the parking API.
    
    Args:
        plate_text: The plate text that was sent
        code: The code/session where it was detected
        confidence: The confidence score
        internal_code: The internal code sent to parking API
        parking_api_response: HTTP response code from parking API (e.g., "201")
        
    Returns:
        The ID of the inserted record, or None if failed
    """
    query = sent_plates_table.insert().values(
        plate_text=plate_text.upper(),  # Always store uppercase for consistency
        code=code,
        confidence=str(confidence),
        internal_code=internal_code,
        parking_api_response=parking_api_response,
        sent_timestamp=datetime.now()
    )
    
    try:
        record_id = await database.execute(query=query)
        log.info(f"Recorded sent plate: '{plate_text}' for code '{code}' (ID: {record_id})")
        return record_id
    except Exception as e:
        log.error(f"DB Error recording sent plate '{plate_text}': {e}")
        return None

async def check_plate_sent_globally(plate_text: str) -> bool:
    """
    Check if a specific plate has ever been sent to the parking API.
    
    Args:
        plate_text: The plate text to check (case-insensitive)
        
    Returns:
        bool: True if this plate was already sent, False otherwise
    """
    query = (
        sent_plates_table.select()
        .where(sent_plates_table.c.plate_text == plate_text.upper())
        .limit(1)
    )
    
    try:
        result = await database.fetch_one(query=query)
        if result:
            log.info(f"Global duplicate found: plate '{plate_text}' was already sent on {result.sent_timestamp}")
            return True
        else:
            log.info(f"Global check passed: plate '{plate_text}' has never been sent")
            return False
    except Exception as e:
        log.error(f"DB Error checking global plate duplicate: {e}")
        return False  # Fail-safe: allow sending if there's a DB error

async def get_all_sent_plates(limit: int = 100, offset: int = 0):
    """
    Retrieve all sent plates with pagination.
    
    Args:
        limit: Maximum number of records to return
        offset: Number of records to skip
        
    Returns:
        List of sent plate records
    """
    query = (
        sent_plates_table.select()
        .order_by(sent_plates_table.c.sent_timestamp.desc())
        .limit(limit)
        .offset(offset)
    )
    
    try:
        results = await database.fetch_all(query=query)
        return results
    except Exception as e:
        log.error(f"DB Error retrieving sent plates: {e}")
        return []

async def get_sent_plates_by_code(code: str):
    """
    Retrieve all plates sent for a specific code.
    
    Args:
        code: The code to filter by
        
    Returns:
        List of sent plate records for this code
    """
    query = (
        sent_plates_table.select()
        .where(sent_plates_table.c.code == code)
        .order_by(sent_plates_table.c.sent_timestamp.desc())
    )
    
    try:
        results = await database.fetch_all(query=query)
        return results
    except Exception as e:
        log.error(f"DB Error retrieving sent plates for code '{code}': {e}")
        return []

async def check_code_already_processed(code: str) -> bool:
    """
    Check if a specific code has already had a plate sent to the parking API.
    
    This allows us to skip expensive AlpAPI processing for codes that already
    have a successful plate detection sent to the parking system.
    
    Args:
        code: The code/session to check
        
    Returns:
        bool: True if this code already has a sent plate, False otherwise
    """
    query = (
        sent_plates_table.select()
        .where(sent_plates_table.c.code == code)
        .limit(1)
    )
    
    try:
        result = await database.fetch_one(query=query)
        if result:
            log.info(f"Code '{code}' already processed: plate '{result.plate_text}' was sent on {result.sent_timestamp}")
            return True
        else:
            log.info(f"Code '{code}' not yet processed - continuing with plate detection")
            return False
    except Exception as e:
        log.error(f"DB Error checking if code '{code}' was processed: {e}")
        return False  # Fail-safe: continue processing if there's a DB error 