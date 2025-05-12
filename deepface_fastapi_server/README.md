# DeepFace FastAPI Server

This project provides a FastAPI server that wraps the `deepface` library to offer face detection, coordinate extraction, and blacklist comparison functionalities via a REST API.

It uses SQLite for managing blacklist metadata (ID, name, reason) and relies on `deepface`'s file-based representation (`.pkl` file) for efficient blacklist searching within the `DeepFace.find` function.

## Features

*   **Face Detection & Coordinate Extraction:** Detects faces in images (provided as paths, URLs, or base64 strings) and returns their bounding box coordinates.
*   **Blacklist Comparison:** Compares detected faces against a pre-defined blacklist image database using `DeepFace.find`.
*   **Blacklist Management:** Provides CRUD endpoints to manage blacklist entries (name, reason) in an SQLite database.
*   **Asynchronous API:** Built with FastAPI for potentially handling concurrent requests efficiently (especially I/O bound parts like downloading URLs).
*   **Dockerized:** Includes `Dockerfile` and `docker-compose.yml` for easy setup and deployment.
*   **Results Persistence:** Saves a copy of processed images and logs processing results (including blacklist matches) to an SQLite database.

## Project Structure

```
deepface_fastapi_server/
├── blacklist_db/         # Folder for blacklist reference images (one subfolder per person)
│   └── .keep
├── src/
│   ├── api/
│   │   ├── endpoints/    # API route definitions (processing, blacklist)
│   │   └── router.py     # Main API router aggregation
│   ├── crud/
│   │   ├── blacklist_crud.py # DB interaction logic for blacklist table
│   │   └── face_crud.py    # Wrappers around DeepFace functions
│   │   └── processed_image_crud.py # DB interaction logic for processed images table
│   ├── config.py         # Configuration settings (paths, model defaults)
│   ├── database.py       # Database connection & table setup (SQLite)
│   ├── main.py           # FastAPI application entrypoint & events
│   ├── models.py         # Pydantic models for API validation
│   ├── __init__.py
│   ├── Dockerfile        # Instructions to build the Docker image
│   └── requirements.txt  # Python dependencies
├── blacklist.db          # SQLite database file (created automatically)
├── processed_images_output/ # Folder where copies of processed images are saved
│   └── .keep             # Placeholder
├── docker-compose.yml    # Docker Compose configuration
└── README.md             # This file
```

## Setup and Running

There are two primary ways to run the server: using Docker (recommended for consistency) or running locally.

**Important:** Before running, ensure you have configured the necessary environment variables, especially for the chosen `FACE_PROCESSING_BACKEND` (see Configuration section).

### Option 1: Running with Docker (Recommended)

**Prerequisites:**

*   Docker: [Install Docker](https://docs.docker.com/get-docker/)
*   Docker Compose: Usually included with Docker Desktop.

**Steps:**

1.  **Clone the repository (if applicable).**
2.  **Populate Blacklist Database:**
    *   Place reference images for blacklisted individuals inside the `blacklist_db` folder.
    *   Create one sub-folder for each person (e.g., `blacklist_db/person_A/img1.jpg`, `blacklist_db/person_A/img2.jpg`, `blacklist_db/person_B/photo.png`).
    *   *Note:* If implementing image uploads via the API (see Limitations section), this manual step might become unnecessary for *new* entries.
    *   `DeepFace.find` will use these images to build its internal representation (`.pkl` file) the first time it runs against this folder (or when refreshed).
3.  **Build and Run with Docker Compose:**
    ```bash
    cd deepface_fastapi_server
    docker-compose up --build -d
    ```
    *   `--build`: Builds the Docker image based on `src/Dockerfile`.
    *   `-d`: Runs the container in detached mode (in the background).
4.  **Server Access:** The API will be available at `http://localhost:8000`.
5.  **Interactive Docs (Swagger UI):** Access `http://localhost:8000/docs` in your browser.
6.  **Stopping the Server:**
    ```bash
    docker-compose down
    ```

### Option 2: Running Locally (for Development/Testing)

**Prerequisites:**

*   Python 3.8+ (matching the version in the Dockerfile is recommended, e.g., 3.9)
*   `pip` and `venv`
*   **AWS Credentials** (if using `aws_rekognition` backend): Configure via environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` [optional]) or an IAM role.
*   **AWS Resources** (if using `aws_rekognition` backend): An S3 bucket specified by `AWS_S3_BUCKET_NAME` must exist. The Rekognition collection (`AWS_REKOGNITION_COLLECTION_ID`) will be created automatically if it doesn't exist.

**Steps:**

1.  **Clone the repository (if applicable).**
2.  **Navigate to the project root directory:**
    ```bash
    cd deepface_fastapi_server
    ```
3.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
4.  **Install dependencies:**
    ```bash
    pip install -r src/requirements.txt
    ```
    *   *Note on Apple Silicon (MPS):* If you want to attempt GPU acceleration, you might need to manually install `tensorflow-macos` and `tensorflow-metal` compatible with your system *after* installing the other requirements. Check the TensorFlow documentation for Apple Silicon.
5.  **Populate Blacklist Database:**
    *   Ensure the `blacklist_db` folder exists in the project root.
    *   Place reference images inside, structured as described in the Docker setup.
6.  **Run the Uvicorn server:**
    ```bash
    uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
    ```
    *   `--reload`: Enables auto-reload on code changes (useful for development).
7.  **Server Access:** The API will be available at `http://localhost:8000`.
8.  **Interactive Docs (Swagger UI):** Access `http://localhost:8000/docs` in your browser.
9.  **Stopping the Server:** Press `Ctrl+C` in the terminal where Uvicorn is running.
10. **Deactivate virtual environment (when done):**
    ```bash
    deactivate
    ```

## API Endpoints

All endpoints are prefixed with `/api/v1`.

**Health Check**

*   `GET /`: Simple health check endpoint (redirects to `/docs` by default with FastAPI).
    *   **Curl Example:**
        ```bash
        curl http://localhost:8000/api/v1/
        ```

**Image Processing (`/process`)**

*   `POST /process-images`
    *   **Purpose:** Processes a list of input images to detect faces and check them against the blacklist.
    *   **Request Body:** (`application/json`)
        ```json
        {
          "images": [
            "/path/on/server/image1.jpg", // Or URL, or base64 string
            "data:image/jpeg;base64,/9j/...",
            "https://example.com/image2.png"
          ],
          "detector_backend": "mtcnn", // Optional: Override default detector
          "model_name": "VGG-Face", // Optional: Override default model
          "distance_metric": "euclidean_l2", // Optional: Override default metric
          "threshold": 0.6 // Optional: Override default threshold
        }
        ```
    *   **Response:** (`200 OK`) A list of `ImageProcessingResult` objects, one per input image. The results are also saved to the `processed_images` table in `blacklist.db`, and a copy of the image is saved in `processed_images_output`.
        ```json
        [
          {
            "image_path_or_identifier": "/path/on/server/image1.jpg",
            "faces": [
              {
                "face_index": 0,
                "facial_area": {"x": 50, "y": 60, "w": 100, "h": 120, ...},
                "confidence": 0.99,
                "blacklist_matches": [
                  {
                    "identity": "blacklist_db/person_A/img1.jpg",
                    "target_x": 10, ..., "source_x": 50, ..., 
                    "threshold": 0.40,
                    "distance": 0.25
                  }
                  // ... other potential matches below threshold
                ]
              }
              // ... other faces found in image1
            ],
            "error": null
          },
          // ... results for other input images
        ]
        ```
    *   **Curl Example (using base64):**
        ```bash
        # Replace '/path/to/your/image.jpg' with an actual image path
        IMG_BASE64=$(base64 /path/to/your/image.jpg)

        curl -X POST http://localhost:8000/api/v1/process/process-images \
        -H "Content-Type: application/json" \
        -d '{
              "images": [
                "data:image/jpeg;base64,'"$IMG_BASE64"'",
                "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/OrteliusWorldMap1570.jpg/1280px-OrteliusWorldMap1570.jpg"
              ]
            }'
        ```
        Example:

        ```bash
        IMG_BASE64_1=$(base64 ../data/1.png)
        IMG_BASE64_2=$(base64 ../data/2.png)
        IMG_BASE64_3=$(base64 ../data/3.png)

        # Option A: tell base64 where the input is
        IMG_BASE64_1=$(base64 -i ../data/1.png)
        IMG_BASE64_2=$(base64 -i ../data/2.png)
        IMG_BASE64_3=$(base64 -i ../data/3.png)

        curl -X POST http://localhost:8000/api/v1/process/process-images \
        -H "Content-Type: application/json" \
        -d '{
              "images": [
                "data:image/png;base64,'"$IMG_BASE64_1"'",
                "data:image/png;base64,'"$IMG_BASE64_2"'",
                "data:image/png;base64,'"$IMG_BASE64_3"'"
              ]
            }'
        ```

    *   **Notes:**
        *   Input images can be file paths *accessible from the server environment* (local path when running locally, path inside container when running in Docker), public URLs, or base64 encoded strings.
        *   Uses `asyncio.gather` internally, which provides concurrency for I/O tasks like downloading URLs but not true CPU parallelism for DeepFace model inference.

*   `GET /processed-images/`
    *   **Purpose:** Retrieve a paginated list of previously processed image results stored in the database.
    *   **Request Query Parameters:**
        *   `offset` (int, optional, default=0): Number of records to skip for pagination.
        *   `limit` (int, optional, default=10): Maximum number of records to return (adjust default based on actual implementation if known, using 10 as a placeholder).
    *   **Response:** (`200 OK`) A JSON object containing:
        *   `total_items` (int): The total number of records available.
        *   `items` (list): A list of `ImageProcessingResult`-like objects for the requested page.
    *   **Curl Example:**
        ```bash
        curl http://localhost:8000/api/v1/processed-images/?offset=0&limit=10
        ```

**Face Detection (`/detect`)**

*   `POST /detect-face`
    *   **Purpose:** Detects faces in a single image and returns their coordinates.
    *   **Request Body:** (`application/json`)
        ```json
        {
          "image": "/path/on/server/image1.jpg", // Or URL, or base64 string
          "detector_backend": "mtcnn" // Optional: Override default detector from config
        }
        ```
    *   **Response:** (`200 OK`) A list of detected faces.
        ```json
        [
          {
            "facial_area": {
              "x": 120,
              "y": 55,
              "w": 90,
              "h": 115,
              "left_eye": null,
              "right_eye": null,
              "nose": null,
              "mouth_left": null,
              "mouth_right": null
            },
            "confidence": 0.987
          }
          // ... other detected faces
        ]
        ```
    *   **Curl Example (using base64):**
        ```bash
        IMG_BASE64=$(base64 /path/to/your/image.jpg)

        curl -X POST http://localhost:8000/api/v1/detect/detect-face \
        -H "Content-Type: application/json" \
        -d '{
              "image": "data:image/jpeg;base64,'"$IMG_BASE64"'"
            }'
        ```

**Blacklist Management (`/blacklist`)**

*   `POST /`
    *   **Purpose:** Add a new person to the blacklist database. **Requires image file upload(s).**
    *   **Request Body:** (`multipart/form-data`) - Use form fields for metadata and file parts for images.
    *   **Curl Example:**
        ```bash
        # Replace '/path/to/person_c_img1.jpg' and '/path/to/person_c_img2.png' with actual image paths
        curl -X POST http://localhost:8000/api/v1/blacklist/ \\
        -F "name=Person C" \\
        -F "reason=Suspicious activity" \\
        -F "images=@/path/to/person_c_img1.jpg;type=image/jpeg" \\
        -F "images=@/path/to/person_c_img2.png;type=image/png"
        ```

        Example:
        ```bash
        curl -X POST http://localhost:8000/api/v1/blacklist/ \
          -F "name=Stan" \
          -F "reason=Suspicious activity" \
          -F "images=@./data/1.png;type=image/png"
        ```

    *   **Response:** (`201 Created`) The created `BlacklistRecord`.
*   `POST /{id}/images`
    *   **Purpose:** Add one or more reference images to an *existing* blacklist entry.
    *   **Request Body:** (`multipart/form-data`) - Use file parts for images.
    *   **Curl Example (add images to entry ID 1):**
        ```bash
        # Replace paths with actual image files
        curl -X POST http://localhost:8000/api/v1/blacklist/1/images \\
        -F "images=@/path/to/new_image1.jpg;type=image/jpeg" \\
        -F "images=@/path/to/new_image2.png;type=image/png"
        ```
    *   **Response:** (`200 OK`) Success message indicating how many images were added.
*   `GET /`
    *   **Purpose:** Retrieve all entries from the blacklist database.
    *   **Response:** (`200 OK`) A list of `BlacklistRecord` objects.
    *   **Curl Example:**
        ```bash
        curl http://localhost:8000/api/v1/blacklist/
        ```
*   `GET /{id}/`
    *   **Purpose:** Retrieve a specific blacklist entry by its ID.
    *   **Response:** (`200 OK`) The `BlacklistRecord` or `404 Not Found`.
    *   **Curl Example (for ID 1):**
        ```bash
        curl http://localhost:8000/api/v1/blacklist/1/
        ```
*   `PUT /{id}/`
    *   **Purpose:** Update an existing blacklist entry (metadata only).
    *   **Request Body:** (`application/json`) - *Note: This endpoint was changed to use JSON for metadata update.*
        ```json
        {\"name\": \"Updated Name\", \"reason\": \"Updated reason\"}
        ```
    *   **Curl Example (for ID 1):**
        ```bash
        curl -X PUT http://localhost:8000/api/v1/blacklist/1/ \\
        -H "Content-Type: application/json" \\
        -d '{"name": "Updated Person C Name", "reason": "Reason changed"}'
        ```
    *   **Response:** (`200 OK`) The updated `BlacklistRecord` or `404 Not Found`.
*   `DELETE /{id}/`
    *   **Purpose:** Delete a blacklist entry by its ID (also removes associated images).
    *   **Response:** (`200 OK`) The deleted `BlacklistRecord` or `404 Not Found`.
    *   **Curl Example (for ID 1):**
        ```bash
        curl -X DELETE http://localhost:8000/api/v1/blacklist/1/
        ```

## Configuration

Configuration settings are primarily managed via environment variables, checked in `src/config.py`.

**Core Settings:**

*   `FACE_PROCESSING_BACKEND`: Determines the backend used for face analysis. 
    *   `deepface` (Default): Uses the local DeepFace library. Requires images in `BLACKLIST_DB_PATH`.
    *   `aws_rekognition`: Uses AWS Rekognition service. Requires AWS credentials and S3 configuration.

**DeepFace Specific Settings (used when `FACE_PROCESSING_BACKEND='deepface'`):**

*   `DETECTOR_BACKEND`: Default face detector for DeepFace (e.g., `retinaface`, `mtcnn`).
*   `MODEL_NAME`: Default face recognition model for DeepFace (e.g., `Facenet`, `VGG-Face`).
*   `DISTANCE_METRIC`: Default distance metric for DeepFace (e.g., `cosine`, `euclidean_l2`).
*   `BLACKLIST_DB_PATH`: Path to the folder containing DeepFace blacklist images (structured with subfolders per ID).
*   `PROCESSED_IMAGES_OUTPUT_DIR`: Directory where copies of processed images are saved.

**AWS Rekognition Specific Settings (used when `FACE_PROCESSING_BACKEND='aws_rekognition'`):**

*   `AWS_REGION`: The AWS region for Rekognition and S3 services (e.g., `us-east-1`).
*   `AWS_S3_BUCKET_NAME`: The name of the S3 bucket where blacklist images will be stored.
*   `AWS_REKOGNITION_COLLECTION_ID`: The ID for the AWS Rekognition collection used for the blacklist.
*   **AWS Credentials:** Must be configured externally (e.g., environment variables `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, or an IAM Role if running in AWS).

**Overriding Defaults:**

*   **Environment Variables:** Set any of the above variables in your environment (e.g., shell, `.env` file loaded by your process, `docker-compose.yml`) to override defaults.
*   **Request Parameters:** The `/process-images` endpoint accepts optional `detector_backend`, `model_name`, `distance_metric`, and `threshold` fields in the request body. *Note: These primarily affect the DeepFace backend or how thresholds are interpreted for the AWS backend.* 

## Limitations & Potential Improvements

*   **Backend Differences:** 
    *   **Blacklist Storage:** DeepFace uses local folders (`BLACKLIST_DB_PATH`), while AWS uses an S3 bucket (`AWS_S3_BUCKET_NAME`) and a Rekognition Collection.
    *   **Indexing:** DeepFace requires index refreshes (automatic via API), while AWS manages indexing internally.
    *   **Result Mapping:** The mapping between AWS Rekognition's similarity score and DeepFace's distance metrics is approximate. Threshold values might need different tuning depending on the active backend.
    *   **Facial Area in Results:** The `/process-images` response structure was slightly adapted. It now returns a single `DetectedFaceResult` per image (if matches are found), containing all matches. The `facial_area` in this result typically corresponds to the bounding box of the face *found in the input image* that generated the matches (specifically, the box from the first match in the list). This differs from the previous implementation which could list multiple detected faces separately.
*   **AWS Cost:** Using the `aws_rekognition` backend incurs AWS service costs for Rekognition API calls and S3 storage.
*   **Local File Storage (AWS Mode):** Currently, images uploaded via the blacklist API are still saved locally *before* being uploaded to S3 in AWS mode. This local copy could potentially be removed after successful S3 upload and indexing to save disk space, if not needed for other purposes.
*   **DeepFace Representation Refresh:** The index (`.pkl` file) used by `DeepFace.find` is automatically refreshed when adding or deleting entries via the `/api/v1/blacklist` endpoints **when using the `deepface` backend**. Manual changes to the `blacklist_db` folder still require a refresh.
*   **Alternative Blacklist Comparison:** For very large blacklists, calling `DeepFace.find` on each request might be inefficient. Consider implementing the alternative approach: pre-calculate embeddings for the blacklist (using `DeepFace.represent`) and store them (e.g., in memory, files, or a vector database). Then, in the processing endpoint, get the embedding for the input face and compare it directly against the loaded blacklist embeddings using distance metrics (`DeepFace.verify` logic or `scipy.spatial.distance`).
*   **Database:** Uses SQLite for simplicity. For production or higher concurrency, consider switching to PostgreSQL with an async driver (`psycopg[binary]`) and potentially `pgvector` for storing embeddings directly in the database.
*   **Results Querying:** The API doesn't currently expose endpoints to query the saved processing results from the `processed_images` table. This could be added if needed.
*   **CPU Parallelism:** `asyncio.gather` helps with I/O, but DeepFace inference is CPU-bound. For true parallelism across multiple images in a single request, explore using Python's `multiprocessing` module within the endpoint, potentially managed with `concurrent.futures.ProcessPoolExecutor` run via `asyncio.to_thread` or similar async bridging.
*   **Apple Silicon (MPS):** Compatibility depends on installing the correct `tensorflow-macos` and `tensorflow-metal` versions (commented out in `requirements.txt`) and potentially using an ARM64-compatible base image in the `Dockerfile`. Testing is required. 

## Refs:
- https://stackoverflow.com/questions/70981334/how-to-install-deepface-python-face-recognition-package-on-m1-mac

