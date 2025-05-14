
from time import time

from libs import db
from tqdm import tqdm

from services.llm import describe_merged_descriptions


def group_descriptions_by_code(descriptions):
    # Create a dictionary to store the grouped data
    grouped_dict = {}
    
    # Iterate through each description
    for desc in descriptions:
        code = desc['code']
        raw_description = desc['raw_description']
        _id = desc['raw_id']
        image_url = desc['image_url']
        # If code doesn't exist in dictionary, create new entry
        if code not in grouped_dict:
            grouped_dict[code] = {
                'code': code,
                'raw_descriptions': [],
                'raw_ids': [],
                'image_urls': []
            }
        
        # Append the raw description to the corresponding code
        grouped_dict[code]['raw_descriptions'].append(raw_description)
        grouped_dict[code]['raw_ids'].append(_id)
        grouped_dict[code]['image_urls'].append(image_url)
    # Convert dictionary to list of dictionaries
    result = list(grouped_dict.values())
    return result

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
            'raw_id': _id,
            'image_url': image_url
        })
    results = group_descriptions_by_code(descriptions)

    # print in green
    print(f"\033[92mMerging this size of descriptions: {len(descriptions)} \033[0m")
    for result in tqdm(results, desc="Processing descriptions"):
        # --- Process descriptions --
        input_descriptions = result['raw_descriptions']
        code = result['code']
        _ids = result['raw_ids']
        _last_image_url = result['image_urls'][-1]
        
        merged_description = describe_merged_descriptions(input_descriptions)
        
        # --- Update DB Status & Publish Processing Event ---
        db.create_processed_description(merged_description, code, app_type, "MERGED_DESCRIPTION_SUCCESS", _last_image_url)
        
        
        for _id in _ids:
            db.update_raw_description_status(_id, 'MERGED_DESCRIPTION_SUCCESS')

    end_time = time()
    # print in RED
    print(f"\033[91mPipeline completed in {end_time - start_time} seconds\033[0m")