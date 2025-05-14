# LLM Service

## Overview

This service provides a FastAPI application that integrates with MQTT for receiving events, schedules background processing jobs using APScheduler, and exposes HTTP endpoints for tasks like authentication and triggering specific processes.

## Setup and Running

1.  **Environment Variables:**
    Ensure the following environment variables are set (e.g., in a `.env` file loaded by your environment or Uvicorn):
    *   `MQTT_BROKER_HOST`: Hostname or IP of your MQTT broker.
    *   `MQTT_BROKER_PORT`: Port of your MQTT broker (e.g., 1883, 8883 for TLS).
    *   `MQTT_USERNAME`: Username for MQTT authentication.
    *   `MQTT_PASSWORD`: Password for MQTT authentication.
    *   `MQTT_CLIENT_ID`: Unique client ID for this service.
    *   `MQTT_TOPIC`: The main MQTT topic to subscribe to for initial events.
    *   `MQTT_PROCESS_TOPIC`: The MQTT topic for processing requests.
    *   `DATABASE_URL`: Connection string for your database (e.g., `sqlite+aiosqlite:///./llm_descriptions.db`).
    *   `PERMANENT_IMAGE_FOLDER`: Path to the folder for storing images.
    *   `JWT_SECRET`: A strong, secret key for signing JWT tokens.
    *   `PIPELINE_INTERVAL`: Interval in minutes for running the processing pipeline (e.g., `2`).
    *   `OPENAI_API_KEY`: If any part of the service (like `image_describer`) uses OpenAI, set this key.

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Server:**
    Use Uvicorn to run the FastAPI application:
    ```bash
    uvicorn main:app --reload --host 0.0.0.0 --port 8000 
    # Adjust host and port as needed
    ```
    The `--reload` flag is useful for development.

## API Endpoints

### 1. Login

*   **Method:** `POST`
*   **Path:** `/login`
*   **Description:** Authenticates a user based on username and password. **Warning:** Currently uses hardcoded credentials (`test`/`test`) - **DO NOT USE IN PRODUCTION WITHOUT REPLACING AUTH LOGIC.** Returns a JWT token upon successful authentication.
*   **Request Body:** `application/x-www-form-urlencoded`
    *   `username`: The user's username.
    *   `password`: The user's password.
*   **Response Body:** `application/json`
    *   `access_token`: The JWT access token.
    *   `token_type`: Type of token (e.g., "bearer").

*   **Curl Example:**
    ```bash
    curl -X POST http://localhost:8000/login \
         -H "Content-Type: application/x-www-form-urlencoded" \
         -d "username=test&password=test"
    ```

### 2. Query Processed Descriptions (Natural Language)

*   **Method:** `POST`
*   **Path:** `/query_processed_descriptions_nl`
*   **Description:** Accepts a natural language question about processed descriptions, uses Vanna.io to generate a corresponding SQL query (if possible), executes the query against the database, and returns the results.
*   **Authentication:** Requires a valid JWT token in the `Authorization: Bearer <token>` header.
*   **Request Body:** `application/json`
    *   Requires a JSON object matching the `NLQueryRequest` model:
        ```json
        {
          "question": "What were the main topics described yesterday?"
        }
        ```
*   **Response Body:** `application/json`
    *   Returns a JSON object matching the `NLQueryResponse` model:
        ```json
        {
          "question": "What were the main topics described yesterday?",
          "sql_query": "SELECT ... FROM processed_descriptions WHERE ...", 
          "results": [ ... ], // Query results or null
          "error": null // Error message if query generation or execution failed
        }
        ```

*   **Curl Example (Replace `YOUR_JWT_TOKEN`):**
    ```bash
    # First, get a token from /login
    # TOKEN=$(curl -s -X POST http://localhost:8000/login -H "Content-Type: application/x-www-form-urlencoded" -d "username=test&password=test" | jq -r .access_token)
    # echo "Using token: $TOKEN"

    curl -X POST http://localhost:8004/query_processed_descriptions_nl \
         -H "Content-Type: application/json" \
         -H "Authorization: Bearer YOUR_JWT_TOKEN" \
         -d '{
           "question": "Alguna arma?"
         }'
    ```

### 3. Get Raw Descriptions (Paginated)

*   **Method:** `GET`
*   **Path:** `/raw_descriptions/`
*   **Description:** Retrieves a paginated list of raw descriptions. Can be optionally filtered by a `code`.
*   **Authentication:** Requires a valid JWT token in the `Authorization: Bearer <token>` header.
*   **Query Parameters:**
    *   `page` (optional, integer, default: `1`): The page number to retrieve.
    *   `page_size` (optional, integer, default: `10`): The number of items per page.
    *   `code` (optional, string): Filters descriptions by the exact code.
*   **Response Body:** `application/json`
    *   Returns a JSON object matching the `PaginatedRawDescriptionsResponse` model:
        ```json
        {
          "total_count": 100,
          "limit": 10,
          "offset": 0,
          "data": [
            {
              "id": 1,
              "raw_description": "Initial raw description of an item.",
              "image_url": "http://example.com/image.jpg",
              "code": "ITEM001",
              "app_type": "inventory",
              "status": "pending",
              "created_at": "2023-01-01T12:00:00Z",
              "updated_at": "2023-01-01T12:00:00Z"
            }
            // ... more items
          ]
        }
        ```
*   **Curl Examples (Replace `YOUR_JWT_TOKEN`):**
    *   Get first page (default size 10):
        ```bash
        # First, get a token from /login if you don't have one
        # TOKEN=$(curl -s -X POST http://localhost:8000/login -H "Content-Type: application/x-www-form-urlencoded" -d "username=test&password=test" | jq -r .access_token)
        # echo "Using token: $TOKEN"

        curl -X GET "http://localhost:8000/raw_descriptions/" \
             -H "Authorization: Bearer YOUR_JWT_TOKEN"
        ```
    *   Get page 2 with page size 5:
        ```bash
        curl -X GET "http://localhost:8000/raw_descriptions/?page=2&page_size=5" \
             -H "Authorization: Bearer YOUR_JWT_TOKEN"
        ```
    *   Get first page, filtered by code "XYZ123":
        ```bash
        curl -X GET "http://localhost:8000/raw_descriptions/?code=XYZ123" \
             -H "Authorization: Bearer YOUR_JWT_TOKEN"
        ```

### 4. Get Processed Descriptions (Paginated)

*   **Method:** `GET`
*   **Path:** `/processed_descriptions/`
*   **Description:** Retrieves a paginated list of processed descriptions. Can be optionally filtered by a `code`.
*   **Authentication:** Requires a valid JWT token in the `Authorization: Bearer <token>` header.
*   **Query Parameters:**
    *   `page` (optional, integer, default: `1`): The page number to retrieve.
    *   `page_size` (optional, integer, default: `10`): The number of items per page.
    *   `code` (optional, string): Filters descriptions by the exact code.
*   **Response Body:** `application/json`
    *   Returns a JSON object matching the `PaginatedProcessedDescriptionsResponse` model:
        ```json
        {
          "total_count": 50,
          "limit": 10,
          "offset": 0,
          "data": [
            {
              "id": 1,
              "processed_description": "This is a processed description.",
              "code": "PROC001",
              "app_type": "reporting",
              "status": "completed",
              "created_at": "2023-01-02T14:00:00Z",
              "updated_at": "2023-01-02T14:00:00Z"
            }
            // ... more items
          ]
        }
        ```
*   **Curl Examples (Replace `YOUR_JWT_TOKEN`):**
    *   Get first page (default size 10):
        ```bash
        # First, get a token from /login if you don't have one
        # TOKEN=$(curl -s -X POST http://localhost:8000/login -H "Content-Type: application/x-www-form-urlencoded" -d "username=test&password=test" | jq -r .access_token)
        # echo "Using token: $TOKEN"
        
        curl -X GET "http://localhost:8000/processed_descriptions/" \
             -H "Authorization: Bearer YOUR_JWT_TOKEN"
        ```
    *   Get page 3 with page size 20, filtered by code "ABC987":
        ```bash
        curl -X GET "http://localhost:8000/processed_descriptions/?page=3&page_size=20&code=ABC987" \
             -H "Authorization: Bearer YOUR_JWT_TOKEN"
        ```

## MQTT Integration

*   The service connects to the configured MQTT broker on startup.
*   It subscribes to the following topics:
    *   `settings.MQTT_TOPIC`: Messages received here trigger `process_lucam_event_data` in the background.
    *   `settings.MQTT_PROCESS_TOPIC`: Messages received here trigger `process_smart_request` in the background.

## Scheduled Jobs (APScheduler)

*   The service runs a background scheduler.
*   The `jobs.pipeline.pipeline` function is scheduled to run at a regular interval defined by `settings.PIPELINE_INTERVAL` (in minutes).

