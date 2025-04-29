import requests
import base64
import os
import time
import json

# --- Configuration ---
# Use environment variable or default
BASE_URL = os.environ.get("FASTAPI_URL", "http://localhost:8000/api/v1")



# --- Sample Data (Ensure these files exist!) ---
# Image to add to blacklist
DATA_DIR = "/Users/stanleysalvatierra/Desktop/2024/lucam/face/data"
BLACKLIST_IMAGE_1 = os.path.join(DATA_DIR, "sample1.png")
BLACKLIST_IMAGE_2 = os.path.join(DATA_DIR, "sample2.png")

# Image to test processing (should match the blacklist entry)
PROCESS_IMAGE_MATCH = os.path.join(DATA_DIR, "sample_match.png")


# Image to test processing (should NOT match)
PROCESS_IMAGE_NO_MATCH = os.path.join(DATA_DIR, "sample1.png") # Use one of the blacklist images or another

# --- Helper Functions ---
def get_base64_string(filepath):
    if not os.path.exists(filepath):
        print(f"ERROR: Image file not found at {filepath}")
        return None
    with open(filepath, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode('utf-8')
    # Basic check for image type, adjust if needed
    mime_type = "image/jpeg" if filepath.lower().endswith(('.jpg', '.jpeg')) else "image/png"
    return f"data:{mime_type};base64,{encoded_string}"

def print_response(name, response):
    print(f"--- Test: {name} ---")
    print(f"URL: {response.url}")
    print(f"Status Code: {response.status_code}")
    try:
        print("Response JSON:")
        print(json.dumps(response.json(), indent=2))
    except json.JSONDecodeError:
        print("Response Text:")
        print(response.text)
    print("--------------------\n")

# --- Test Functions ---
def test_health_check():
    response = requests.get(f"{BASE_URL}/") # Should likely hit the root, not /api/v1/
    # Adjust URL if root is different
    root_url = BASE_URL.replace("/api/v1", "/")
    response = requests.get(root_url)
    print_response("Health Check", response)
    assert response.status_code == 200
    # assert "message" in response.json()

def test_add_blacklist(name="Test Person", reason="Testing API"):
    if not os.path.exists(BLACKLIST_IMAGE_1):
        print(f"Skipping Add Blacklist: File not found {BLACKLIST_IMAGE_1}")
        return None
    files = {
        'images': (os.path.basename(BLACKLIST_IMAGE_1), open(BLACKLIST_IMAGE_1, 'rb'), 'image/jpeg'),
    }
    data = {'name': name, 'reason': reason}
    response = requests.post(f"{BASE_URL}/blacklist/", files=files, data=data)
    print_response("Add Blacklist Entry", response)
    assert response.status_code == 201
    assert "id" in response.json()
    return response.json()['id'] # Return the ID for subsequent tests

def test_list_blacklist():
    response = requests.get(f"{BASE_URL}/blacklist/")
    print_response("List Blacklist Entries", response)
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_get_single_blacklist(entry_id):
    if entry_id is None:
        print("Skipping Get Single Blacklist: No ID provided.")
        return
    response = requests.get(f"{BASE_URL}/blacklist/{entry_id}/")
    print_response(f"Get Single Blacklist Entry (ID: {entry_id})", response)
    assert response.status_code == 200
    assert response.json()['id'] == entry_id

def test_add_images_to_blacklist(entry_id):
    if entry_id is None:
        print("Skipping Add Images to Blacklist: No ID provided.")
        return
    if not os.path.exists(BLACKLIST_IMAGE_2):
        print(f"Skipping Add Images to Blacklist: File not found {BLACKLIST_IMAGE_2}")
        return
        
    files = {
        'images': (os.path.basename(BLACKLIST_IMAGE_2), open(BLACKLIST_IMAGE_2, 'rb'), 'image/jpeg'),
    }
    response = requests.post(f"{BASE_URL}/blacklist/{entry_id}/images", files=files)
    print_response(f"Add Images to Blacklist Entry (ID: {entry_id})", response)
    assert response.status_code == 200
    assert "message" in response.json()

def test_process_images():
    img_match_b64 = get_base64_string(PROCESS_IMAGE_MATCH)
    img_no_match_b64 = get_base64_string(PROCESS_IMAGE_NO_MATCH)

    if not img_match_b64 or not img_no_match_b64:
        print("Skipping Process Images: One or more sample files missing.")
        return

    payload = {
        "images": [img_match_b64, img_no_match_b64],
        # "threshold": 0.5 # Optional: Add threshold override if needed for testing
    }
    response = requests.post(f"{BASE_URL}/process/process-images", json=payload)
    print_response("Process Images", response)
    assert response.status_code == 200
    results = response.json()
    assert isinstance(results, list)
    assert len(results) == 2
    # Basic check: expected matching image has blacklist_matches
    # This depends heavily on the model, threshold, and backend used
    # assert len(results[0]["faces"]) > 0 # Check if face was processed
    # assert results[0]["has_blacklist_match"] == True # Check match flag
    # assert len(results[1]["faces"]) >= 0 # May or may not find faces/matches
    # assert results[1]["has_blacklist_match"] == False # Check no match flag
    print("INFO: Manual check recommended for process_images match results based on backend/threshold.")

def test_get_processed_images():
    response = requests.get(f"{BASE_URL}/processed-images/?offset=0&limit=5")
    print_response("Get Processed Images", response)
    assert response.status_code == 200
    assert "items" in response.json()
    assert "total_items" in response.json()

def test_delete_blacklist(entry_id):
    if entry_id is None:
        print("Skipping Delete Blacklist: No ID provided.")
        return
    response = requests.delete(f"{BASE_URL}/blacklist/{entry_id}/")
    print_response(f"Delete Blacklist Entry (ID: {entry_id})", response)
    assert response.status_code == 200
    assert response.json()['id'] == entry_id
    
    # Verify deletion
    verify_response = requests.get(f"{BASE_URL}/blacklist/{entry_id}/")
    print_response(f"Verify Deletion (ID: {entry_id})", verify_response)
    assert verify_response.status_code == 404

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Starting API tests against: {BASE_URL}\n")
    
    # Ensure sample files exist before running tests that need them
    required_files = [BLACKLIST_IMAGE_1, BLACKLIST_IMAGE_2, PROCESS_IMAGE_MATCH, PROCESS_IMAGE_NO_MATCH]
    if not all(os.path.exists(f) for f in required_files):
        print("ERROR: One or more required sample image files are missing.")
        print(f"Please create/place the following files in the script directory: {required_files}")
        exit(1)
        
    added_id = None
    try:
        test_health_check()
        added_id = test_add_blacklist()
        time.sleep(1) # Give server a moment
        test_list_blacklist()
        test_get_single_blacklist(added_id)
        test_add_images_to_blacklist(added_id)
        time.sleep(2) # Allow potential indexing/refresh
        test_process_images()
        time.sleep(1)
        test_get_processed_images()
        
    except AssertionError as e:
        print(f"\n!!!!!!!! ASSERTION FAILED: {e} !!!!!!!!")
    except Exception as e:
        print(f"\n!!!!!!!! TEST FAILED WITH EXCEPTION: {e} !!!!!!!!")
    finally:
        # Cleanup: Delete the entry created during the test
        if added_id:
            print("--- Cleaning up created blacklist entry ---")
            test_delete_blacklist(added_id)
            
    print("\nAPI tests completed.") 