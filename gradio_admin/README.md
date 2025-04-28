# DeepFace FastAPI - Gradio Admin Frontend

This Gradio application provides a web-based administrative interface for the DeepFace FastAPI server.

## Features

*   **Blacklist Management:**
    *   View the current list of blacklisted individuals stored in the backend database.
    *   Add new individuals to the blacklist by providing a name, an optional reason, and uploading one or more reference images.
    *   Delete existing individuals from the blacklist.
    *   Automatically refreshes the DeepFace index on the backend after adding or deleting entries.
*   **Processed Image Viewer:**
    *   View images that have been processed by the backend's `/process-images` endpoint.
    *   Displays images in a paginated gallery, ordered by the most recent first.
    *   Shows metadata associated with each processed image (database ID, timestamp, original source identifier).
    *   Highlights images that resulted in a blacklist match and displays details about the match.

## Setup and Running

**Prerequisites:**

1.  **Running DeepFace FastAPI Server:** The backend server (`deepface_fastapi_server`) must be running and accessible from where you run this Gradio app. Note the URL and port it's running on (e.g., `http://localhost:8000`).
2.  **Python 3.8+:** A compatible Python version.
3.  **Shared File Access (for Processed Image Viewer):** The directory where the FastAPI server saves processed image copies (`processed_images_output` by default) must be accessible from where the Gradio app is run. See Configuration below.

**Steps:**

1.  **Navigate to the `gradio_admin` directory:**
    ```bash
    cd gradio_admin
    ```
2.  **(Optional) Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure API and Image Paths (if necessary):**
    *   The app attempts to connect to the FastAPI server at `http://localhost:8000/api/v1` by default.
    *   It expects the processed images to be accessible at `../deepface_fastapi_server/processed_images_output` relative to its own directory.
    *   If your FastAPI server is at a different address OR the processed images directory is located elsewhere *relative to where you run `python app.py`*, set the following environment variables before running:
        ```bash
        # Example for different API URL:
        export FASTAPI_URL="http://192.168.1.100:8000/api/v1"
        # Example for different processed images path:
        export PROCESSED_IMAGES_PATH="/path/to/shared/processed_images_output"

        python app.py
        ```
5.  **Run the Gradio App:**
    ```bash
    python app.py
    ```
6.  **Access:** Open your web browser to the URL provided by Gradio (typically `http://127.0.0.1:7860`).
7.  **Stopping the App:** Press `Ctrl+C` in the terminal where the app is running.

## Usage

*   **Blacklist Management Tab:**
    *   The current list is displayed on the right. Use the "Refresh List" button to update it manually if needed.
    *   To add an entry, fill in the name (required) and reason (optional), upload one or more image files using the file component, and click "Add to Blacklist".
    *   To delete an entry, select it from the dropdown list (which is populated based on the current list display) and click "Delete Selected Entry".
    *   Status messages will appear below the add/delete buttons.
*   **Processed Images Tab:**
    *   The gallery displays recently processed images, newest first.
    *   Use the "Previous" and "Next" buttons to navigate through pages.
    *   Images with detected blacklist matches will have a caption indicating "**BLACKLIST MATCH FOUND!**" along with details of the match (matched identity file path and distance).

## Important Considerations

*   **API URL (`FASTAPI_BASE_URL`):** Ensure this constant (or environment variable) correctly points to your running FastAPI backend, including the `/api/v1` prefix.
*   **Processed Image Path (`PROCESSED_IMAGES_BASE_PATH`):** This is crucial for the gallery view. The Gradio app needs read access to the physical image files saved by the FastAPI server. The path defined here must be the correct path *from the perspective of the machine running `gradio_admin/app.py`*. Misconfiguration will result in broken image icons in the gallery. 