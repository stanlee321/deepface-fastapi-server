import os
from pathlib import Path

from libs.core import ImageDescriber

from configs import settings


# Example of how to use the class (replaces the original code)
if __name__ == "__main__":
    # --- Example Usage ---

    # --- Option 1: Use a publicly accessible URL --- (already handled)
    image_url_direct = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

    # --- Option 2: Use a local file path --- 
    # Create a dummy file for testing or use a real path
    local_image_path_str = "../data/sofi.jpeg" # Relative to workspace root usually
    local_image_path = Path(local_image_path_str)
   
    # --- Option 3: Use a URL that needs encoding --- 
    image_url_to_encode = "https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png"


    # --- Initialize Describer --- 
    # Check if OPENAI_API_KEY is set (optional, OpenAI client does this too)
    if not os.getenv("OPENAI_API_KEY"):
       print("Error: OPENAI_API_KEY environment variable not set.")
       # exit(1) # Or handle appropriately

    try:
        describer = ImageDescriber(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL
            ) # Assumes API key is in env var

        # # --- Describe Image from Direct URL --- 
        # print(f"Describing image from direct URL: {image_url_direct[:60]}...")
        # description_url = describer.describe_image(image_url_direct)
        # if description_url:
        #     print("\n--- URL Image Description ---")
        #     print(description_url)
        #     print("---------------------------\n")
        # else:
        #     print("Failed to get description from URL.")

        # --- Describe Image from Local File --- 
        if local_image_path and local_image_path.exists():
            print(f"Encoding local image: {local_image_path}")
            data_uri_local = ImageDescriber.encode_image_to_data_uri(local_image_path)
            if data_uri_local:
                print(f"Describing image from local file (data URI): {data_uri_local[:60]}...")
                description_local = describer.describe_image(data_uri_local)
                if description_local:
                    print("\n--- Local Image Description ---")
                    print(description_local)
                    print("-----------------------------\n")
                else:
                    print("Failed to get description from local file.")
            else:
                print("Failed to encode local image.")
        else:
            print("Skipping local file test (file not found or created).")

        # --- Describe Image from Encoded URL --- 
        print(f"Encoding URL image: {image_url_to_encode}")
        data_uri_encoded_url = ImageDescriber.encode_image_to_data_uri(image_url_to_encode)
        if data_uri_encoded_url:
            print(f"Describing image from encoded URL (data URI): {data_uri_encoded_url[:60]}...")
            # description_encoded_url = describer.describe_image(data_uri_encoded_url)
            # if description_encoded_url:
            #      print("\n--- Encoded URL Image Description ---")
            #      print(description_encoded_url)
            #      print("-----------------------------------\n")
            # else:
            #     print("Failed to get description from encoded URL.")
        else:
            print("Failed to encode URL image.")

    except ValueError as ve: # Catch API key error from constructor
        print(f"Configuration Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")