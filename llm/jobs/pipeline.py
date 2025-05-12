
from time import time

from libs import db

from services.llm import describe_merged_descriptions


async def pipeline():
    """
    Pipeline for the data server.
    """
    start_time = time()
    
    # --- Process images for lowres and crop --- 
    tasks = db.get_raw_descriptions_by_status('IMAGE_DESCRIEBER_SUCCESS')
    
    descriptions = []
    for task in tasks:
        _id, raw_description, image_url, code, app_type, status, created_at, updated_at = task
        descriptions.append({
            'raw_description': raw_description,
            'code': code,
        })
        
    if len(descriptions) > 0:
        # --- Process descriptions --- 
        
        input_descriptions = [data['raw_description'] for data in descriptions]
        print(f"Input descriptions: {input_descriptions}")
        merged_description = describe_merged_descriptions(input_descriptions)
        
        # --- Update DB Status & Publish Processing Event ---
        db.create_processed_description(merged_description, code, app_type, "MERGED_DESCRIPTION_SUCCESS")
        
        
        for task in tasks:
            _id, raw_description, image_url, code, app_type, status, created_at, updated_at = task
            db.update_raw_description_status(_id, 'MERGED_DESCRIPTION_SUCCESS')

        
    end_time = time()
    # print in RED
    print(f"\033[91mPipeline completed in {end_time - start_time} seconds\033[0m")