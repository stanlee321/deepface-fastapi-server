import gradio as gr
import requests
import math
import os
import json # For potentially parsing detailed results
import cv2 # Import OpenCV
import numpy as np # Import Numpy for image manipulation

# --- Configuration ---
# Should match the running FastAPI server address
# If running Gradio locally and FastAPI in Docker, use http://localhost:8000
# If both are local, use http://localhost:8000
# If both are in Docker (on same network), use http://deepface_fastapi_service:8000 (service name)
FASTAPI_BASE_URL = os.environ.get("FASTAPI_URL", "http://localhost:8000/api/v1")
# Path where Gradio app expects to find the saved processed images.
# This needs to align with how the FastAPI server saves paths and how Gradio accesses them.
# If Gradio runs locally and FastAPI in Docker (with volume mount), this local path must match the host part of the volume mount.
PROCESSED_IMAGES_BASE_PATH = os.environ.get("PROCESSED_IMAGES_PATH", "/Users/stanleysalvatierra/Desktop/2024/lucam/face/deepface_fastapi_server/processed_images_output")
# Base path where Gradio app can find the blacklist image directories
# Adjust if your deployment structure is different
BLACKLIST_DB_BASE_PATH = os.environ.get("BLACKLIST_DB_IMG_PATH", "/Users/stanleysalvatierra/Desktop/2024/lucam/face/deepface_fastapi_server/blacklist_db")


# --- New Helper Function: Draw Annotations ---
def draw_annotations(image_path: str, faces_data: list, has_match: bool):
    """Loads an image, draws bounding boxes and landmarks, returns annotated image as NumPy array."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image for annotation: {image_path}")
            # Return a placeholder or raise an error? For now, return None
            return None 
        
        # Define colors (BGR format)
        box_color = (0, 0, 255) if has_match else (0, 255, 0) # Red if match, Green if no match
        landmark_color = (0, 255, 0)   # Green for landmarks
        thickness = 2

        if isinstance(faces_data, list):
            for face in faces_data:
                if isinstance(face, dict):
                    facial_area = face.get('facial_area')
                    if isinstance(facial_area, dict):
                        x = facial_area.get('x')
                        y = facial_area.get('y')
                        w = facial_area.get('w')
                        h = facial_area.get('h')
                        # Draw bounding box
                        if all(isinstance(i, int) for i in [x, y, w, h]):
                           cv2.rectangle(img, (x, y), (x + w, y + h), box_color, thickness)
                           
                        # Draw landmarks if available
                        left_eye = facial_area.get('left_eye')
                        right_eye = facial_area.get('right_eye')
                        if left_eye and isinstance(left_eye, list) and len(left_eye) == 2:
                            cv2.circle(img, tuple(left_eye), radius=2, color=landmark_color, thickness=-1)
                        if right_eye and isinstance(right_eye, list) and len(right_eye) == 2:
                             cv2.circle(img, tuple(right_eye), radius=2, color=landmark_color, thickness=-1)
        
        # Convert BGR (OpenCV default) to RGB for Gradio display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb # Return the annotated image (NumPy array RGB)

    except Exception as e:
        print(f"Error during annotation drawing for {image_path}: {e}")
        # Optionally return original image or None
        # Try loading original again just in case annotation failed mid-way
        try: 
            original_img = cv2.imread(image_path)
            if original_img is not None:
                 return cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB) # Convert original too
            else:
                 return None
        except: 
            return None

# --- API Client Functions ---

# Function to get current blacklist
def get_blacklist_entries():
    """Fetches blacklist entries from the API.
       Returns: Tuple (status_string, raw_entries_list, dropdown_choices_list)
    """
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/blacklist/")
        response.raise_for_status()
        entries = response.json()
        # Format for display (list of strings for dropdown)
        if not entries:
            # Return format: status_string, raw_data, dropdown_choices
            return "Blacklist is currently empty.", [], [] 
        
        dropdown_choices = []
        for entry in entries:
            # Simple format for dropdown
            dropdown_choices.append(f"ID: {entry['id']}, Name: {entry['name']}") 

        # Return format: status_string, raw_data, dropdown_choices
        return f"{len(entries)} entries loaded.", entries, dropdown_choices 
    except requests.exceptions.RequestException as e:
        error_msg = f"Error fetching blacklist: {e} - {e.response.text if e.response else 'No response'}"
        print(error_msg)
        return error_msg, [], []
    except Exception as e:
        error_msg = f"Error processing blacklist data: {e}"
        print(error_msg)
        return error_msg, [], []

# Function to add to blacklist
def add_blacklist_entry(name: str, reason: str, image_files: list):
    """Adds a new entry with images to the blacklist via API."""
    if not name or not image_files:
        list_status, raw_entries, choices = get_blacklist_entries()
        return "Error: Name and at least one image file are required.", raw_entries, gr.update(choices=choices)

    files_to_upload = []
    opened_files = [] # Keep track of files to close them
    try:
        for img_file_obj in image_files:
            # img_file_obj is a TemporaryFileWrapper object from Gradio
            file_path = img_file_obj.name
            filename = os.path.basename(file_path)
            # Determine MIME type (basic)
            mime_type = 'image/jpeg' if filename.lower().endswith(('.jpg', '.jpeg')) else 'image/png' if filename.lower().endswith('.png') else 'application/octet-stream'
            
            f = open(file_path, 'rb')
            opened_files.append(f)
            files_to_upload.append(('images', (filename, f, mime_type)))

    except Exception as e:
        # Close any files that were opened
        for f in opened_files:
            f.close()
        print(f"Error preparing files for upload: {e}")
        list_status, raw_entries, choices = get_blacklist_entries()
        return f"Error preparing files for upload: {e}", raw_entries, gr.update(choices=choices)

    data = {'name': name, 'reason': reason or ""} # Ensure reason is string

    try:
        print(f"Sending POST to {FASTAPI_BASE_URL}/blacklist/")
        response = requests.post(
            f"{FASTAPI_BASE_URL}/blacklist/",
            files=files_to_upload,
            data=data,
            timeout=60 # Add a timeout
        )
        print(f"Response status: {response.status_code}")
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        status_msg = f"Successfully added '{name}' to blacklist. Index refreshed."

    except requests.exceptions.RequestException as e:
        error_text = e.response.text if e.response else 'No response details'
        print(f"Error adding to blacklist: {e} - {error_text}")
        status_msg = f"Error adding to blacklist: {e} - {error_text}"
    except Exception as e:
         print(f"Unexpected error adding to blacklist: {e}")
         status_msg = f"Unexpected error: {e}"
    finally:
        # Ensure all opened files are closed
        for f in opened_files:
            f.close()

    # Get updated list regardless of success/failure of add operation
    list_status, raw_entries, choices = get_blacklist_entries()
    # Return format: status_msg, raw_data, dropdown_update
    return status_msg, raw_entries, gr.update(choices=choices, value=None) 

# Function to delete from blacklist
def delete_blacklist_entry(entry_to_delete: str):
    """Deletes a blacklist entry via API using the formatted string from dropdown."""
    if not entry_to_delete:
         list_status, raw_entries, choices = get_blacklist_entries()
         return "Error: Please select an entry ID to delete.", raw_entries, gr.update(choices=choices)
    try:
        # Extract ID from the formatted string "ID: 1, Name: ..."
        actual_id = entry_to_delete.split(",")[0].split(":")[1].strip()
        print(f"Sending DELETE to {FASTAPI_BASE_URL}/blacklist/{actual_id}/")
        response = requests.delete(f"{FASTAPI_BASE_URL}/blacklist/{actual_id}/", timeout=60)
        print(f"Response status: {response.status_code}")
        response.raise_for_status()
        deleted_info = response.json()
        status_msg = f"Successfully deleted entry ID: {actual_id} (Name: {deleted_info.get('name', 'N/A')}). Index refreshed."
    except (IndexError, ValueError):
         status_msg = f"Error: Invalid ID format selected: {entry_to_delete}"
    except requests.exceptions.RequestException as e:
        error_text = e.response.text if e.response else 'No response details'
        print(f"Error deleting from blacklist: {e} - {error_text}")
        status_msg = f"Error deleting entry ID {actual_id}: {e} - {error_text}"
    except Exception as e:
         print(f"Unexpected error deleting from blacklist: {e}")
         status_msg = f"Unexpected error deleting entry ID {actual_id}: {e}"

    # Get updated list
    list_status, raw_entries, choices = get_blacklist_entries()
    # Return format: status_msg, raw_data, dropdown_update
    return status_msg, raw_entries, gr.update(choices=choices, value=None) 


# --- New Function: Add Images to Existing Entry ---
def add_images_to_entry(selected_entry_str: str, new_image_files: list):
    """Adds newly uploaded images to an existing blacklist entry via API."""
    if not selected_entry_str:
        return "Error: No blacklist entry selected from the dropdown."
    if not new_image_files:
        return "Error: No new image files were uploaded."
        
    try:
        # Extract ID from the formatted string "ID: 1, Name: ..."
        actual_id = selected_entry_str.split(",")[0].split(":")[1].strip()
    except (IndexError, ValueError):
         return f"Error: Invalid ID format selected: {selected_entry_str}"

    files_to_upload = []
    opened_files = []
    try:
        for img_file_obj in new_image_files:
            file_path = img_file_obj.name
            filename = os.path.basename(file_path)
            mime_type = 'image/jpeg' if filename.lower().endswith(('.jpg', '.jpeg')) else 'image/png' if filename.lower().endswith('.png') else 'application/octet-stream'
            f = open(file_path, 'rb')
            opened_files.append(f)
            files_to_upload.append(('images', (filename, f, mime_type)))

    except Exception as e:
        for f in opened_files:
            f.close()
        print(f"Error preparing files for upload: {e}")
        return f"Error preparing files for upload: {e}"

    try:
        api_url = f"{FASTAPI_BASE_URL}/blacklist/{actual_id}/images"
        print(f"Sending POST to {api_url}")
        response = requests.post(
            api_url,
            files=files_to_upload,
            timeout=60 # Add a timeout
        )
        print(f"Response status: {response.status_code}")
        response.raise_for_status()
        response_data = response.json()
        status_msg = response_data.get("message", "Operation completed.")
        if response_data.get("errors"): # Append errors if any
             status_msg += f"\nErrors: {'; '.join(response_data['errors'])}"

    except requests.exceptions.RequestException as e:
        error_text = e.response.text if e.response else 'No response details'
        print(f"Error adding images: {e} - {error_text}")
        status_msg = f"Error adding images: {e} - {error_text}"
    except Exception as e:
         print(f"Unexpected error adding images: {e}")
         status_msg = f"Unexpected error adding images: {e}"
    finally:
        for f in opened_files:
            f.close()

    # We don't directly return the updated list here, maybe refresh separately?
    return status_msg 


# --- New Function to Show Blacklist Details ---
def show_blacklist_details(selected_entry_str: str, all_entries_raw: list):
    """Given the selected dropdown string and raw entry data, displays details and images."""
    if not selected_entry_str or not all_entries_raw:
        return "No entry selected.", "", "", [] # Clear details

    try:
        selected_id = int(selected_entry_str.split(",")[0].split(":")[1].strip())
    except (IndexError, ValueError):
        return "Invalid selection format.", "", "", []

    selected_entry_data = None
    for entry in all_entries_raw:
        if entry.get('id') == selected_id:
            selected_entry_data = entry
            break

    if not selected_entry_data:
        return f"Entry with ID {selected_id} not found in data.", "", "", []

    name = selected_entry_data.get('name', 'N/A')
    reason = selected_entry_data.get('reason', 'N/A')
    img_dir_relative = selected_entry_data.get('reference_image_dir')

    image_paths_for_gallery = []
    detail_text = f"ID: {selected_id}\nName: {name}\nReason: {reason}"

    if img_dir_relative:
        detail_text += f"\nImage Directory: {img_dir_relative}"
        try:
            full_img_dir_path = os.path.abspath(os.path.join(BLACKLIST_DB_BASE_PATH, img_dir_relative))
            if os.path.isdir(full_img_dir_path):
                files = [f for f in os.listdir(full_img_dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if files:
                    image_paths_for_gallery = [os.path.abspath(os.path.join(full_img_dir_path, fname)) for fname in files]
                    # Check existence again for absolute paths
                    image_paths_for_gallery = [p for p in image_paths_for_gallery if os.path.exists(p)]
                    if not image_paths_for_gallery:
                        print(f"Warning: No image files found or accessible in {full_img_dir_path}")
                        detail_text += "\nFiles: (No valid image files found)"
                else:
                    detail_text += "\nFiles: (Directory empty)"
            else:
                detail_text += "\nFiles: (Directory not found)"
                print(f"Warning: Blacklist image directory not found by Gradio: {full_img_dir_path}")
        except Exception as e:
            detail_text += f"\nFiles: (Error listing files: {e})"
            print(f"Error listing files in {img_dir_relative}: {e}")
    else:
         detail_text += "\nImage Directory: Not set"

    return detail_text, name, reason, image_paths_for_gallery


# Function to get processed images (paginated)
def get_processed_images(page_num: int, page_size: int):
    """Fetches processed images data from the API for pagination.
       Returns: Tuple (gallery_data, raw_items_list, page_num, total_pages, page_info_text, prev_btn_update, next_btn_update)
    """
    offset = (page_num - 1) * page_size
    limit = page_size
    print(f"Fetching processed images: limit={limit}, offset={offset}")
    try:
        response = requests.get(f"{FASTAPI_BASE_URL}/processed-images/", params={'limit': limit, 'offset': offset}, timeout=30)
        response.raise_for_status()
        data = response.json()
        items = data.get('items', []) # This is the raw list of dicts from API
        total_items = data.get('total_items', 0)
        total_pages = math.ceil(total_items / page_size) if page_size > 0 else 1
        total_pages = max(1, total_pages) # Ensure at least 1 page

        print(f"Received {len(items)} items. Total items: {total_items}. Total pages: {total_pages}")

        # Prepare data for gallery - list of (annotated_image_array, caption) tuples
        gallery_data = []
        for item in items:
            img_path_relative = item.get('saved_image_path') 
            if not img_path_relative:
                 print("Warning: Item missing saved_image_path")
                 continue

            filename = os.path.basename(img_path_relative)
            img_path_for_gradio = os.path.abspath(os.path.join(PROCESSED_IMAGES_BASE_PATH, filename))

            # Construct caption - Simplified
            caption_lines = []
            caption_lines.append(f"DB ID: {item.get('db_id')}")
            caption_lines.append(f"Timestamp: {item.get('processing_timestamp')}")
            
            is_match = item.get('has_blacklist_match', False)
            if is_match:
                 caption_lines.append("\n**🚨 BLACKLIST MATCH! 🚨**")
                 faces_data_for_caption = item.get('faces', []) # Get face data for caption
                 if isinstance(faces_data_for_caption, list):
                     for face in faces_data_for_caption:
                         if isinstance(face, dict) and face.get('blacklist_matches'):
                             matches = face.get('blacklist_matches', [])
                             if isinstance(matches, list): 
                                 for match in matches:
                                     if isinstance(match, dict):
                                         identity = match.get('identity', 'Unknown')
                                         distance = match.get('distance', -1)
                                         caption_lines.append(f"  - Match: {os.path.basename(identity)} (Dist: {distance:.4f})")

            caption = "\n".join(caption_lines)

            # Draw annotations
            annotated_image = None
            if os.path.exists(img_path_for_gradio):
                 annotated_image = draw_annotations(img_path_for_gradio, item.get('faces', []), is_match)
            else:
                 print(f"Warning: Image path not accessible/found by Gradio: {img_path_for_gradio} (Original API path: {img_path_relative})")
            
            if annotated_image is not None:
                 # Gallery expects list of (image, caption)
                 gallery_data.append((annotated_image, caption))
            # else: # Optionally append a placeholder if annotation failed
                 # gallery_data.append((placeholder_img_array, caption + "\n(Error loading/annotating image)"))

        # Update pagination controls visibility/values
        prev_btn_interactive = page_num > 1
        next_btn_interactive = page_num < total_pages
        page_info_text = f"Page {page_num} of {total_pages} ({total_items} total items)"

        # Return the gallery data AND raw items list
        return gallery_data, items, page_num, total_pages, page_info_text, gr.update(interactive=prev_btn_interactive), gr.update(interactive=next_btn_interactive)

    except requests.exceptions.RequestException as e:
        error_msg = f"Error fetching processed images: {e} - {e.response.text if e.response else 'No response'}"
        print(error_msg)
        # Return empty items list on error
        return [], [], 1, 1, error_msg, gr.update(interactive=False), gr.update(interactive=False)
    except Exception as e:
         error_msg = f"Error processing processed images data: {e}"
         print(error_msg)
         # Return empty items list on error
         return [], [], 1, 1, error_msg, gr.update(interactive=False), gr.update(interactive=False)


# --- Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft(), title="DeepFace Admin") as demo:
    gr.Markdown("## DeepFace Admin Dashboard")
    
    # Hidden state to store raw blacklist data
    raw_blacklist_state = gr.State(value=[])

    with gr.Tabs():
        # --- Blacklist Management Tab ---
        with gr.TabItem("Blacklist Management"):
            with gr.Row():
                # Left column: Add controls
                with gr.Column(scale=1, min_width=300):
                    gr.Markdown("### Add New Blacklist Entry")
                    bl_name_input = gr.Textbox(label="Name / Identifier", placeholder="Enter unique name")
                    bl_reason_input = gr.Textbox(label="Reason (Optional)", placeholder="Enter reason for blacklisting")
                    bl_image_input = gr.File(
                        label="Reference Images (Upload one or more)",
                        file_count="multiple",
                        file_types=["image"]
                    )
                    bl_add_button = gr.Button("Add to Blacklist", variant="primary", scale=1)
                    bl_status_output = gr.Textbox(label="Add/Delete Status", interactive=False, lines=2)
                    
                # Right column: View/Delete Existing
                with gr.Column(scale=2, min_width=400):
                    gr.Markdown("### View / Delete Existing Entries")
                    with gr.Row():
                        bl_view_delete_dropdown = gr.Dropdown(label="Select Entry", choices=[], interactive=True, scale=4)
                        bl_refresh_button = gr.Button("🔄 Refresh List", scale=1)
                    
                    gr.Markdown("#### Selected Entry Details")
                    bl_detail_text = gr.Textbox(label="Info", lines=4, interactive=False)
                    bl_detail_gallery = gr.Gallery(
                         label="Reference Images",
                         show_label=True,
                         columns=4,
                         height=200,
                         object_fit="contain",
                         preview=True
                    )
                    
                    # --- Section to Add More Images ---
                    gr.Markdown("##### Add More Images to Selected Entry")
                    bl_add_more_images_input = gr.File(
                        label="Upload New Images",
                        file_count="multiple",
                        file_types=["image"]
                    )
                    bl_add_more_images_button = gr.Button("Add Uploaded Images to this Entry", scale=1)
                    # --- End Add More Images Section ---
                    
                    bl_delete_button = gr.Button("Delete Selected Entry", variant="stop")

            # Define interactions for Blacklist Tab
            bl_add_button.click(
                fn=add_blacklist_entry,
                inputs=[bl_name_input, bl_reason_input, bl_image_input],
                # Outputs: status, raw_data_state, dropdown_update
                outputs=[bl_status_output, raw_blacklist_state, bl_view_delete_dropdown] 
            )

            bl_delete_button.click(
                 fn=delete_blacklist_entry,
                 inputs=[bl_view_delete_dropdown], # Input is the selected dropdown value
                 # Outputs: status, raw_data_state, dropdown_update
                 outputs=[bl_status_output, raw_blacklist_state, bl_view_delete_dropdown] 
            )
            
            # When dropdown selection changes, show details
            bl_view_delete_dropdown.change(
                fn=show_blacklist_details,
                inputs=[bl_view_delete_dropdown, raw_blacklist_state],
                outputs=[bl_detail_text, bl_name_input, bl_reason_input, bl_detail_gallery] # Update detail text & gallery
                # We also update name/reason inputs for potential (manual) editing scenario, though update isn't implemented
            )

            # --- Interaction for Adding More Images ---
            def add_images_and_refresh_details(selected_entry_str, new_image_files, all_entries_raw):
                # 1. Call the API to add images
                status = add_images_to_entry(selected_entry_str, new_image_files)
                # 2. Refresh the details view for the *same* selected entry
                detail_text, _, _, gallery_images = show_blacklist_details(selected_entry_str, all_entries_raw)
                # 3. Return updates: status message, cleared file input, updated gallery
                return status, None, gallery_images
                
            bl_add_more_images_button.click(
                 fn=add_images_and_refresh_details, # Wrapper function
                 inputs=[bl_view_delete_dropdown, bl_add_more_images_input, raw_blacklist_state],
                 outputs=[bl_status_output, bl_add_more_images_input, bl_detail_gallery] # Update status, clear file input, refresh gallery
            )

            # Refresh button action
            def refresh_blacklist_ui():
                 """Gets list and updates dropdown and raw state"""
                 list_status, raw_entries, choices = get_blacklist_entries()
                 # Return updates for: raw_state, dropdown, status_textbox
                 return raw_entries, gr.update(choices=choices, value=None), list_status 

            bl_refresh_button.click(
                fn=refresh_blacklist_ui,
                inputs=[],
                # Outputs: raw_state, dropdown, status_textbox
                outputs=[raw_blacklist_state, bl_view_delete_dropdown, bl_status_output]
            )

        # --- Processed Images Tab ---
        with gr.TabItem("Processed Images"):
            gr.Markdown("### Processed Images Viewer")
            # State variables
            page_num_state = gr.State(value=1)
            page_size_state = gr.State(value=12) 
            total_pages_state = gr.State(value=1)
            # State to hold the full data items for the current page
            current_page_data_state = gr.State(value=[]) 

            with gr.Row():
                # Left Column: Detail View
                with gr.Column(scale=1, min_width=300):
                    gr.Markdown("#### Selected Image Details")
                    selected_processed_image = gr.Image(label="Selected Processed Image", show_label=False, height=300)
                    selected_image_info = gr.Textbox(label="Image Info", lines=4, interactive=False)
                    gr.Markdown("##### Blacklist Match Images (if any)")
                    blacklist_match_gallery = gr.Gallery(
                        label="Blacklist Match Reference Images",
                        show_label=False,
                        columns=3,
                        height=200,
                        object_fit="contain",
                        preview=True
                    )
                    
                # Right Column: Main Gallery & Pagination
                with gr.Column(scale=2, min_width=400):
                    gr.Markdown("#### Recently Processed (Browse & Select)")
                    processed_gallery = gr.Gallery(
                        label="Recently Processed Images",
                        show_label=False,
                        columns=4, 
                        height=550,
                        object_fit="contain",
                        preview=True # Keep preview for quick look
                    )
                    page_info_display = gr.Textbox(label="Pagination Info", interactive=False)
                    # Pagination controls
                    with gr.Row():
                         prev_button = gr.Button("⬅️ Previous", interactive=False, scale=1)
                         next_button = gr.Button("Next ➡️", interactive=False, scale=1)

            # Define interactions for Processed Images Tab
            # Function to load a page of data
            def load_page_wrapper(current_page, page_size):
                """Wrapper to call the API fetch function for gallery and store raw items"""
                print(f"UI: Loading page {current_page}")
                # Returns: gallery_data, raw_items, page_num, total_pages, page_info, prev_update, next_update
                return get_processed_images(current_page, page_size)

            # Previous button click
            def go_prev_page(current_page, page_size):
                 print(f"UI: Go Previous from page {current_page}")
                 new_page = max(1, current_page - 1)
                 # Returns: gallery_data, raw_items, page_num, total_pages, page_info, prev_update, next_update
                 return load_page_wrapper(new_page, page_size)

            prev_button.click(
                 fn=go_prev_page,
                 inputs=[page_num_state, page_size_state],
                 # Outputs need to include the raw data state
                 outputs=[processed_gallery, current_page_data_state, page_num_state, total_pages_state, page_info_display, prev_button, next_button] 
            )

            # Next button click
            def go_next_page(current_page, total_pages, page_size):
                 print(f"UI: Go Next from page {current_page}, total pages {total_pages}")
                 new_page = min(total_pages, current_page + 1)
                 # Returns: gallery_data, raw_items, page_num, total_pages, page_info, prev_update, next_update
                 return load_page_wrapper(new_page, page_size)

            next_button.click(
                 fn=go_next_page,
                 inputs=[page_num_state, total_pages_state, page_size_state],
                 # Outputs need to include the raw data state
                 outputs=[processed_gallery, current_page_data_state, page_num_state, total_pages_state, page_info_display, prev_button, next_button] 
            )
            
            # --- New Interaction: Handle Gallery Selection --- 
            def display_selected_details(selection_data: gr.SelectData, page_items: list):
                """Updates the left panel when an image is selected in the main gallery."""
                if not page_items or selection_data.index >= len(page_items):
                    print(f"Selection index {selection_data.index} out of bounds for page data size {len(page_items)}")
                    return None, "Error: Selection index out of bounds.", []
                selected_item = page_items[selection_data.index] 
                
                # Construct Gradio path for the selected processed image
                saved_image_relative = selected_item.get('saved_image_path')
                selected_img_path_for_gradio = None
                annotated_selected_image = None # Variable for the annotated image array
                
                if saved_image_relative:
                    filename = os.path.basename(saved_image_relative)
                    selected_img_path_for_gradio = os.path.abspath(os.path.join(PROCESSED_IMAGES_BASE_PATH, filename))
                    if os.path.exists(selected_img_path_for_gradio):
                        # Draw annotations on the selected image for the detail view
                        annotated_selected_image = draw_annotations(
                            selected_img_path_for_gradio, 
                            selected_item.get('faces', []), 
                            selected_item.get('has_blacklist_match', False)
                        )
                    else:
                         print(f"Warning: Selected processed image not found by Gradio: {selected_img_path_for_gradio}")
                
                # Initialize outputs
                info_lines = []
                info_lines.append(f"DB ID: {selected_item.get('db_id')}")
                info_lines.append(f"Timestamp: {selected_item.get('processing_timestamp')}")
                match_gallery_output = []
                
                if selected_item.get('has_blacklist_match'):
                    info_lines.append("\n**🚨 BLACKLIST MATCH! 🚨**")
                    faces_data = selected_item.get('faces', [])
                    if isinstance(faces_data, list):
                        for face in faces_data:
                            if isinstance(face, dict) and face.get('blacklist_matches'):
                                matches = face.get('blacklist_matches', [])
                                if isinstance(matches, list):
                                    for match in matches:
                                        if isinstance(match, dict):
                                            identity = match.get('identity', 'Unknown') 
                                            distance = match.get('distance', -1)
                                            identity_basename = os.path.basename(identity)
                                            info_lines.append(f"  - Match: {identity_basename} (Dist: {distance:.4f})")
                                            
                                            try:
                                                parts = identity.strip('./').split('/')
                                                if len(parts) >= 3 and parts[0] == 'blacklist_db':
                                                    match_id_dir = parts[1]
                                                    match_img_folder_path = os.path.abspath(os.path.join(BLACKLIST_DB_BASE_PATH, match_id_dir))
                                                    match_img_path = os.path.abspath(os.path.join(match_img_folder_path, identity_basename))
                                                    if os.path.exists(match_img_path):
                                                        match_gallery_output.append(match_img_path)
                                                    else:
                                                        print(f"Warning: Blacklist match ref image not found by Gradio: {match_img_path}")
                                                else:
                                                    print(f"Warning: Could not parse blacklist ID dir from identity: {identity}")
                                            except Exception as e:
                                                print(f"Error constructing path for matched image {identity}: {e}")
                                                
                # Return the annotated image array instead of the path
                return annotated_selected_image, "\n".join(info_lines), match_gallery_output
                
            processed_gallery.select(
                fn=display_selected_details,
                inputs=[current_page_data_state], # Pass the full items list for the current page
                outputs=[selected_processed_image, selected_image_info, blacklist_match_gallery]
            )

            # Load initial data
            demo.load(
                 fn=load_page_wrapper,
                 inputs=[page_num_state, page_size_state],
                 # Outputs need to include the new state variable
                 outputs=[processed_gallery, current_page_data_state, page_num_state, total_pages_state, page_info_display, prev_button, next_button] 
            )


    # Initial load of blacklist on app start
    demo.load(
         fn=refresh_blacklist_ui,
         inputs=[],
         # Outputs: raw_state, dropdown, status_textbox
         outputs=[raw_blacklist_state, bl_view_delete_dropdown, bl_status_output]
    )

# Launch the Gradio app
if __name__ == "__main__":
    # Ensure the base paths for images are absolute for allowed_paths
    allowed_processed_img_path = os.path.abspath(PROCESSED_IMAGES_BASE_PATH)
    allowed_blacklist_img_path = os.path.abspath(BLACKLIST_DB_BASE_PATH)
    print(f"Gradio allowing processed image paths from: {allowed_processed_img_path}")
    print(f"Gradio allowing blacklist image paths from: {allowed_blacklist_img_path}")
    demo.launch(
        server_name="0.0.0.0", # Listen on all interfaces
        # Add both image directories to allowed paths
        allowed_paths=[allowed_processed_img_path, allowed_blacklist_img_path] 
    ) 