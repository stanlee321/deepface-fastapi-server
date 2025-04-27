Okay, let's create a comprehensive documentation guide for the `serengil/deepface` library based on the provided information.

---

# **DeepFace Library: Comprehensive Guide**

**Version:** 0.0.94 (Based on provided `package_info.json`)

## 1. Introduction

DeepFace is a lightweight yet powerful Python framework for face recognition and facial attribute analysis. It wraps state-of-the-art deep learning models for tasks including:

*   **Face Verification:** Determining if two images belong to the same person.
*   **Face Recognition (Identification):** Finding known faces from a database that match a given face.
*   **Facial Attribute Analysis:** Predicting age, gender, emotion, and race from facial images.
*   **Face Detection & Extraction:** Locating and extracting facial regions from images.
*   **Real-time Analysis:** Applying recognition and analysis to video streams.
*   **Vector Embedding Generation:** Representing faces as numerical vectors (embeddings).
*   **Anti-Spoofing:** Detecting presentation attacks (e.g., photos, masks).

DeepFace aims to make cutting-edge facial analysis accessible with a simple API, handling complex pipeline stages (detection, alignment, normalization, representation, verification/identification) internally.

This guide provides a comprehensive overview of its functionalities, parameters, and use cases, intended for developers and AI practitioners.

## 2. Installation

### Using pip (Recommended)

The easiest way is to install from PyPI, which includes necessary dependencies:

```bash
pip install deepface
```

### From Source

To get the latest features or contribute, install from the cloned repository:

```bash
git clone https://github.com/serengil/deepface.git
cd deepface
pip install -e .
```

### Importing

Once installed, import the main class:

```python
from deepface import DeepFace
```

## 3. Core Functionalities

DeepFace provides several high-level functions for common tasks.

### 3.1. Face Verification (`DeepFace.verify`)

Determines if two images contain the same person.

```python
result = DeepFace.verify(
    img1_path: Union[str, np.ndarray, IO[bytes], List[float]],
    img2_path: Union[str, np.ndarray, IO[bytes], List[float]],
    model_name: str = "VGG-Face",
    detector_backend: str = "opencv",
    distance_metric: str = "cosine",
    enforce_detection: bool = True,
    align: bool = True,
    expand_percentage: int = 0,
    normalization: str = "base",
    silent: bool = False,
    threshold: Optional[float] = None,
    anti_spoofing: bool = False,
)
```

**Parameters:**

*   `img1_path`, `img2_path`: Paths (str), NumPy arrays (BGR format), file objects (binary read mode), base64 encoded strings, or pre-calculated embedding vectors (List[float]) for the two images to compare.
*   `model_name`: Face recognition model (See Section 4.1). Default: "VGG-Face".
*   `detector_backend`: Face detector (See Section 4.2). Default: "opencv". Use "skip" if providing already cropped faces or embeddings.
*   `distance_metric`: Metric for comparing embeddings (See Section 4.3). Default: "cosine".
*   `enforce_detection`: If `True`, raise an error if no face is detected. If `False`, proceeds even if no face is found (useful for low-res images or pre-cropped faces). Default: `True`.
*   `align`: Perform face alignment based on eye landmarks. Default: `True`.
*   `expand_percentage`: Percentage to expand the detected facial bounding box. Default: 0.
*   `normalization`: Preprocessing normalization technique (See Section 4.4). Default: "base".
*   `silent`: Suppress console logs. Default: `False`.
*   `threshold`: Custom distance threshold for verification. If `None`, uses pre-tuned defaults for the model/metric pair. Default: `None`.
*   `anti_spoofing`: Enable presentation attack detection. Default: `False`.

**Returns:**

A dictionary containing:

*   `verified` (bool): `True` if the distance is below the threshold, `False` otherwise.
*   `distance` (float): Calculated distance between embeddings. Lower means more similar.
*   `threshold` (float): The threshold used for the decision.
*   `model` (str): Name of the recognition model used.
*   `detector_backend` (str): Name of the detector used.
*   `similarity_metric` (str): Name of the distance metric used.
*   `facial_areas` (dict): Bounding boxes for detected faces in `img1` and `img2`. Contains keys 'img1' and 'img2', each with a dict like `{'x': int, 'y': int, 'w': int, 'h': int, 'left_eye': tuple, 'right_eye': tuple}`. Coordinates are `None` if input was an embedding.
*   `time` (float): Execution time in seconds.

**Example:**

```python
from deepface import DeepFace

result = DeepFace.verify(img1_path="dataset/img1.jpg", img2_path="dataset/img2.jpg")
print(f"Are the images the same person? {result['verified']}")
print(f"Distance: {result['distance']:.4f}, Threshold: {result['threshold']:.2f}")

# Example with embeddings
embedding1 = DeepFace.represent("dataset/img1.jpg")[0]['embedding']
embedding2 = DeepFace.represent("dataset/img2.jpg")[0]['embedding']
result_emb = DeepFace.verify(img1_path=embedding1, img2_path=embedding2)
print(f"Verification using embeddings: {result_emb['verified']}")
```

### 3.2. Face Recognition / Identification (`DeepFace.find`)

Identifies faces in an image by comparing them against a database of known faces.

```python
results = DeepFace.find(
    img_path: Union[str, np.ndarray, IO[bytes]],
    db_path: str,
    model_name: str = "VGG-Face",
    distance_metric: str = "cosine",
    enforce_detection: bool = True,
    detector_backend: str = "opencv",
    align: bool = True,
    expand_percentage: int = 0,
    threshold: Optional[float] = None,
    normalization: str = "base",
    silent: bool = False,
    refresh_database: bool = True,
    anti_spoofing: bool = False,
    batched: bool = False,
)
```

**Parameters:**

*   `img_path`: Input image (path, NumPy array, file object, base64). Can contain multiple faces.
*   `db_path`: Path to the folder containing the database images. Subfolders are supported (e.g., `database/person1/img1.jpg`).
*   `model_name`, `distance_metric`, `enforce_detection`, `detector_backend`, `align`, `expand_percentage`, `threshold`, `normalization`, `silent`, `anti_spoofing`: Same as `DeepFace.verify`.
*   `refresh_database`: If `True`, checks for new/deleted/modified images in `db_path` and updates the internal representation database (`.pkl` file). If `False`, uses the existing `.pkl` file, potentially ignoring changes. Default: `True`.
*   `batched`: If `True`, uses optimized batch processing suitable for large databases or inputs with many faces, returning a list of lists of dictionaries. If `False`, returns a list of pandas DataFrames. Default: `False`.

**Returns:**

*   If `batched=False` (Default): A list of pandas DataFrames. Each DataFrame corresponds to a face detected in `img_path` and contains matching identities from `db_path` sorted by distance.
*   If `batched=True`: A list of lists of dictionaries. Each inner list corresponds to a face detected in `img_path`, and each dictionary represents a matching identity.

**DataFrame/Dictionary Columns/Keys:**

*   `identity`: Path to the matched image in the database.
*   `hash`: Hash of the database image file (used internally for `refresh_database`).
*   `target_x`, `target_y`, `target_w`, `target_h`: Bounding box of the face in the matched database image.
*   `source_x`, `source_y`, `source_w`, `source_h`: Bounding box of the face detected in the input `img_path`.
*   `threshold`: Distance threshold used.
*   `distance`: Calculated distance between the source face and the target face.

**Internal Representation (`.pkl` file):**

`DeepFace.find` creates a pickle file (e.g., `ds_model_vggface_...pkl`) in the `db_path` to store pre-computed embeddings for faster subsequent searches. The `refresh_database` parameter controls its update behavior.

**Example:**

```python
from deepface import DeepFace
import pandas as pd

# Ensure db_path exists and contains images (e.g., in subfolders per person)
# Example structure:
# my_database/
#   person_a/
#     img1.jpg
#     img2.jpg
#   person_b/
#     img3.jpg

try:
    dfs = DeepFace.find(img_path="dataset/img1.jpg", db_path="my_database")
    if dfs and not dfs[0].empty:
        print("Found matches:")
        print(dfs[0].head()) # Display top matches for the first detected face
    else:
        print("No matches found or no faces detected in input.")
except ValueError as e:
    print(f"Error: {e}") # Handle cases like empty db_path

# Example using batched mode (returns list of lists of dicts)
results_batched = DeepFace.find(img_path="dataset/couple.jpg", db_path="my_database", batched=True)
print(f"\nBatched results for {len(results_batched)} detected faces:")
for i, face_matches in enumerate(results_batched):
    print(f"  Face {i+1}: Found {len(face_matches)} matches.")
    if face_matches:
        print(f"    Top match: {face_matches[0]['identity']} (Distance: {face_matches[0]['distance']:.4f})")

```

### 3.3. Facial Attribute Analysis (`DeepFace.analyze`)

Predicts age, gender, emotion, and race for faces in an image.

```python
results = DeepFace.analyze(
    img_path: Union[str, np.ndarray, IO[bytes], List[str], List[np.ndarray], List[IO[bytes]]],
    actions: Union[tuple, list] = ("emotion", "age", "gender", "race"),
    enforce_detection: bool = True,
    detector_backend: str = "opencv",
    align: bool = True,
    expand_percentage: int = 0,
    silent: bool = False,
    anti_spoofing: bool = False,
)
```

**Parameters:**

*   `img_path`: Input image(s) (path, NumPy array, file object, base64, or a list/batch of these).
*   `actions`: A tuple or list specifying which attributes to analyze. Options: 'age', 'gender', 'emotion', 'race'. Default: `("emotion", "age", "gender", "race")`.
*   `enforce_detection`, `detector_backend`, `align`, `expand_percentage`, `silent`, `anti_spoofing`: Same as `DeepFace.verify`.

**Returns:**

*   If input is a single image: A list of dictionaries, one for each detected face.
*   If input is a batch (list of images/arrays): A list of lists of dictionaries.

**Dictionary Keys:**

*   `region` (dict): Bounding box `{'x', 'y', 'w', 'h'}`.
*   `face_confidence` (float): Confidence score from the face detector.
*   `age` (float): Estimated apparent age.
*   `dominant_gender` (str): "Woman" or "Man".
*   `gender` (dict): Probabilities for "Woman" and "Man".
*   `dominant_emotion` (str): Most likely emotion (e.g., "happy", "sad", "neutral").
*   `emotion` (dict): Probabilities for all detected emotions (angry, disgust, fear, happy, sad, surprise, neutral).
*   `dominant_race` (str): Most likely race (e.g., "white", "asian", "black").
*   `race` (dict): Probabilities for all detected races (asian, indian, black, white, middle eastern, latino hispanic).

**Example:**

```python
from deepface import DeepFace

analysis_results = DeepFace.analyze(img_path="dataset/img4.jpg", actions=['age', 'gender'])

for i, result in enumerate(analysis_results):
    print(f"--- Face {i+1} ---")
    print(f"Region: {result['region']}")
    print(f"Age: {result['age']}")
    print(f"Dominant Gender: {result['dominant_gender']}")
    print(f"Gender Confidence: Man={result['gender']['Man']:.2f}%, Woman={result['gender']['Woman']:.2f}%")

# Example with batch input
batch_results = DeepFace.analyze(img_path=["dataset/img1.jpg", "dataset/img2.jpg"])
print(f"\nAnalyzed {len(batch_results)} images in batch.")
```

### 3.4. Face Representation (`DeepFace.represent`)

Generates numerical vector embeddings for faces in an image.

```python
results = DeepFace.represent(
    img_path: Union[str, np.ndarray, IO[bytes], Sequence[Union[str, np.ndarray, IO[bytes]]]],
    model_name: str = "VGG-Face",
    enforce_detection: bool = True,
    detector_backend: str = "opencv",
    align: bool = True,
    expand_percentage: int = 0,
    normalization: str = "base",
    anti_spoofing: bool = False,
    max_faces: Optional[int] = None,
)
```

**Parameters:**

*   `img_path`: Input image(s) (path, NumPy array, file object, base64, or a sequence/batch of these).
*   `model_name`, `enforce_detection`, `detector_backend`, `align`, `expand_percentage`, `normalization`, `anti_spoofing`: Same as `DeepFace.verify`.
*   `max_faces`: Optional limit on the number of faces to process per image. If more faces are detected, only the largest `max_faces` are processed. Default: `None` (process all).

**Returns:**

*   If input is a single image: A list of dictionaries, one for each detected face.
*   If input is a batch (sequence of images/arrays): A list of lists of dictionaries.

**Dictionary Keys:**

*   `embedding` (List[float]): The vector embedding. Its dimension depends on `model_name`.
*   `facial_area` (dict): Bounding box `{'x', 'y', 'w', 'h', 'left_eye', 'right_eye'}`. Coordinates are nonsensical if `detector_backend='skip'`.
*   `face_confidence` (float): Confidence from the detector. 0 if `detector_backend='skip'`.

**Example:**

```python
from deepface import DeepFace

embedding_objs = DeepFace.represent(img_path="dataset/img1.jpg", model_name="Facenet")

if embedding_objs:
    print(f"Generated {len(embedding_objs[0]['embedding'])}-dimensional embedding for the first face.")
    # print(embedding_objs[0]['embedding']) # Uncomment to see the vector

# Example with batch input
batch_embeddings = DeepFace.represent(img_path=["dataset/img1.jpg", "dataset/img2.jpg"], model_name="Facenet")
print(f"\nGenerated embeddings for {len(batch_embeddings)} images in batch.")
```

### 3.5. Face Extraction (`DeepFace.extract_faces`)

Detects and extracts facial regions from an image, optionally performing alignment and normalization.

```python
face_objs = DeepFace.extract_faces(
    img_path: Union[str, np.ndarray, IO[bytes]],
    detector_backend: str = "opencv",
    enforce_detection: bool = True,
    align: bool = True,
    expand_percentage: int = 0,
    grayscale: bool = False, # Deprecated
    color_face: str = "rgb",
    normalize_face: bool = True,
    anti_spoofing: bool = False,
)
```

**Parameters:**

*   `img_path`: Input image (path, NumPy array, file object, base64).
*   `detector_backend`, `enforce_detection`, `align`, `expand_percentage`, `anti_spoofing`: Same as `DeepFace.verify`.
*   `grayscale`: **Deprecated**. Use `color_face='gray'` instead.
*   `color_face`: Output color format for the extracted face image. Options: 'rgb', 'bgr', 'gray'. Default: 'rgb'.
*   `normalize_face`: If `True`, normalizes pixel values of the extracted face to [0, 1]. Default: `True`.

**Returns:**

A list of dictionaries, one for each detected face.

**Dictionary Keys:**

*   `face` (np.ndarray): The extracted face image in the specified `color_face` format.
*   `facial_area` (dict): Bounding box `{'x', 'y', 'w', 'h', 'left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right'}`. Landmarks (`left_eye`, etc.) are only present if the detector provides them.
*   `confidence` (float): Confidence score from the detector.
*   `is_real` (bool, optional): Result of anti-spoofing analysis (only if `anti_spoofing=True`).
*   `antispoof_score` (float, optional): Anti-spoofing score (only if `anti_spoofing=True`).

**Example:**

```python
from deepface import DeepFace
import matplotlib.pyplot as plt

face_objs = DeepFace.extract_faces(img_path="dataset/img1.jpg", detector_backend="mtcnn")

if face_objs:
    print(f"Detected {len(face_objs)} face(s).")
    first_face = face_objs[0]['face']
    print(f"First face shape: {first_face.shape}")
    print(f"Facial area: {face_objs[0]['facial_area']}")
    print(f"Confidence: {face_objs[0]['confidence']:.2f}")

    # Display the first extracted face (assuming RGB output)
    # plt.imshow(first_face)
    # plt.axis("off")
    # plt.show()
else:
    print("No faces detected.")
```

### 3.6. Real-time Stream Analysis (`DeepFace.stream`)

Performs real-time face recognition and optional facial attribute analysis on a video stream (e.g., webcam).

```python
DeepFace.stream(
    db_path: str = "",
    model_name: str = "VGG-Face",
    detector_backend: str = "opencv",
    distance_metric: str = "cosine",
    enable_face_analysis: bool = True,
    source: Any = 0,
    time_threshold: int = 5,
    frame_threshold: int = 5,
    anti_spoofing: bool = False,
    output_path: Optional[str] = None,
    debug: bool = False,
)
```

**Parameters:**

*   `db_path`: Path to the face database folder for recognition. If empty, only analysis is performed (if enabled).
*   `model_name`, `detector_backend`, `distance_metric`, `anti_spoofing`: Same as `DeepFace.find`.
*   `enable_face_analysis`: If `True`, performs age, gender, and emotion analysis. Default: `True`.
*   `source`: Video source. Can be an integer (webcam ID, usually 0) or a video file path (str). Default: 0.
*   `time_threshold`: How many seconds to display analysis results before re-analyzing. Default: 5.
*   `frame_threshold`: How many consecutive frames a face must be detected before analysis is triggered. Default: 5.
*   `output_path`: Optional path to save the processed video stream as an MP4 file. Default: `None`.
*   `debug`: If `True`, saves intermediate frame processing steps as images (for debugging). Default: `False`.

**Behavior:**

Opens a window displaying the video stream. Detects faces, performs recognition against `db_path` (if provided), and optionally analyzes attributes. Results are overlaid on the video feed. Press 'q' to quit.

**Example:**

```python
from deepface import DeepFace

# Example: Use webcam, recognize against 'my_database', analyze attributes
# DeepFace.stream(db_path="my_database")

# Example: Use video file, only do recognition (no attribute analysis)
# DeepFace.stream(db_path="my_database", source="my_video.mp4", enable_face_analysis=False)

# Example: Use webcam, only do attribute analysis (no recognition)
# DeepFace.stream(db_path="", enable_face_analysis=True)

# Example: Use webcam, recognize, analyze, enable anti-spoofing, save output
# DeepFace.stream(db_path="my_database", anti_spoofing=True, output_path="output.mp4")
```

## 4. Key Concepts and Parameters

Understanding these parameters allows for fine-tuning DeepFace's behavior.

### 4.1. Face Recognition Models (`model_name`)

DeepFace supports various models, each with different architectures, embedding sizes, and performance characteristics.

*   **VGG-Face:** (Default) Based on VGG16. Robust, 4096-dim embedding.
*   **Facenet:** Google's model, 128-dim embedding. Good balance.
*   **Facenet512:** Variant of Facenet with 512-dim embedding. Often higher accuracy.
*   **ArcFace:** Popular model using additive angular margin loss. 512-dim embedding. High accuracy.
*   **SFace:** Lightweight model from OpenCV Zoo. 128-dim embedding. Fast.
*   **GhostFaceNet:** Efficient model design. 512-dim embedding. Good accuracy/speed trade-off.
*   **OpenFace:** Early deep learning model. 128-dim embedding.
*   **DeepFace:** Facebook's model. 4096-dim embedding.
*   **DeepID:** Early deep learning model. 160-dim embedding.
*   **Dlib:** ResNet-based model from Dlib library. 128-dim embedding. Requires `dlib` installation.
*   **Buffalo_L:** InsightFace model. 512-dim embedding. Requires `insightface` installation.

**Selection:** Choose based on accuracy requirements, computational resources, and embedding size needs. Facenet512, ArcFace, VGG-Face, and Dlib generally offer high accuracy (see Benchmarks).

### 4.2. Face Detection Backends (`detector_backend`)

Different algorithms for locating faces in images.

*   **opencv:** (Default) Haar Cascade classifier. Fast but less accurate, especially with non-frontal faces or challenging lighting.
*   **ssd:** Single Shot Detector. Good balance of speed and accuracy.
*   **mtcnn:** Multi-task Cascaded Convolutional Networks. Generally accurate, good at finding smaller faces, but slower.
*   **fastmtcnn:** Faster implementation of MTCNN using `facenet-pytorch`. Requires `facenet-pytorch` installation.
*   **dlib:** HOG-based detector. Decent speed and accuracy. Requires `dlib` installation.
*   **retinaface:** High accuracy, robust to scale variations, provides more landmarks (eyes, nose, mouth corners). Slower than OpenCV/SSD.
*   **mediapipe:** Google's solution. Fast and reasonably accurate. Requires `mediapipe` installation.
*   **yolov8:** YOLO (You Only Look Once) v8 face detector. Requires `ultralytics` installation. Fast and accurate.
*   **yolov11n, yolov11s, yolov11m:** YOLO-Face v11 variants (nano, small, medium). Requires `ultralytics` installation. Offer different speed/accuracy trade-offs.
*   **yunet:** Lightweight and fast detector from OpenCV Zoo. Requires `opencv-contrib-python`.
*   **centerface:** Anchor-free detector. Good accuracy.
*   **skip:** Bypasses detection. Use only when providing pre-cropped face images or embeddings to functions like `represent` or `verify`.

**Selection:** Trade-off between speed and accuracy. `retinaface` and `mtcnn` are often most accurate but slower. `opencv`, `ssd`, `yunet` are faster alternatives.

### 4.3. Distance Metrics (`distance_metric`)

Used to quantify the similarity between two face embeddings. Lower distance means higher similarity.

*   **cosine:** (Default) Measures the cosine of the angle between two vectors. Ranges from 0 (identical) to 2 (opposite). Effective for high-dimensional spaces.
*   **euclidean:** Standard Euclidean distance (straight-line distance). Sensitive to vector magnitude.
*   **euclidean_l2:** Euclidean distance calculated *after* L2 normalization of the vectors. This makes it similar in behavior to cosine distance but potentially faster to compute in some scenarios.
*   **angular:** Calculates the angular separation between vectors, normalized to [0, 1]. `arccos(cosine_similarity) / pi`.

**Selection:** `cosine` and `euclidean_l2` are commonly used and often perform well. Default thresholds are pre-tuned for each combination.

### 4.4. Normalization (`normalization`)

Preprocessing step applied to the input image *before* feeding it to the recognition model.

*   **base:** (Default) Simple scaling to [0, 1].
*   **raw:** No normalization applied (pixels likely in [0, 255]).
*   **Facenet:** Mean/standard deviation normalization.
*   **Facenet2018:** Scales to [-1, 1].
*   **VGGFace:** Specific mean subtraction based on VGG-Face training data.
*   **VGGFace2:** Specific mean subtraction based on VGG-Face2 training data.
*   **ArcFace:** Scales to [-1, 1] (`(pixel - 127.5) / 128`).

**Selection:** Generally, keep the default ("base") unless you have specific reasons or are replicating results from studies that used different normalization for a particular model. Using the model-specific normalization (e.g., "ArcFace" for the ArcFace model) might slightly improve accuracy if the model was trained with that scheme.

### 4.5. Alignment (`align`)

Aligns the face based on detected eye landmarks before feeding it to the recognition model. This typically improves accuracy by making faces more consistent in orientation and scale. Enabled by default (`True`).

### 4.6. Enforce Detection (`enforce_detection`)

If `True` (default), DeepFace raises a `ValueError` if no face is detected in an input image. If `False`, it attempts to proceed (e.g., by using the whole image in `represent` or returning an empty result in `find`/`analyze`). Set to `False` if you expect some inputs might not contain faces or if you are providing pre-cropped images with `detector_backend='skip'`.

### 4.7. Anti-Spoofing (`anti_spoofing`)

Uses the FasNet model to determine if the presented face is real or a presentation attack (e.g., photo, video replay). When enabled (`True`), functions like `verify`, `find`, `analyze`, and `extract_faces` will include spoofing detection. `extract_faces` will add `is_real` (bool) and `antispoof_score` (float) keys to its output. Other functions may raise an error if a spoof is detected. Requires `torch` installation.

## 5. Advanced Use Cases & Examples

### 5.1. Identifying Players in a Group Photo/Video (Example from `script_version.py`)

This demonstrates combining DeepFace functions with custom logic. The goal is to identify known players in a group photo by comparing detected faces to pre-computed average embeddings for each player.

**Steps:**

1.  **Prepare Reference Data:**
    *   Organize reference images in folders, one folder per player (e.g., `database/player_name/img1.jpg`, `database/player_name/img2.jpg`).
    *   For each player, calculate the average embedding from their reference images.

2.  **Process Target Image/Video Frame:**
    *   Detect all faces in the target image/frame.
    *   For each detected face:
        *   Calculate its embedding.
        *   Compare this embedding to the average embedding of *each* known player using cosine similarity.
        *   Find the player with the highest similarity score.
        *   If the highest similarity exceeds a chosen threshold (e.g., 0.3-0.5) and that player hasn't already been assigned in the *current* frame/image, label the face with the player's name. Mark the player as assigned for this frame/image to prevent multiple assignments.
        *   If no match meets the threshold or the best match is already assigned, label as "Desconocido" (Unknown).

3.  **Annotate:** Draw bounding boxes and labels on the image/frame.

**Python Implementation Sketch:**

```python
import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from deepface import DeepFace
# from matplotlib import pyplot as plt # For displaying results

def calculate_average_embedding(img_paths, model_name='Facenet'):
    """Calculates the average embedding for a list of images."""
    embeddings = []
    valid_paths = []
    for img_path in img_paths:
        try:
            # Use enforce_detection=False if some reference images might not have clear faces
            representation = DeepFace.represent(img_path=img_path, model_name=model_name, enforce_detection=True)
            # represent returns a list, take the first face's embedding
            if representation and 'embedding' in representation[0]:
                 embeddings.append(representation[0]['embedding'])
                 valid_paths.append(img_path)
            else:
                 print(f"Warning: Could not get embedding for {img_path}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}") # Handle potential errors during represent

    if not embeddings:
         print(f"Warning: No valid embeddings found for paths: {img_paths}")
         return None # Or handle as appropriate, e.g., raise error or return zero vector

    print(f"Calculated average embedding from {len(embeddings)} images.")
    return np.mean(embeddings, axis=0)

# --- Configuration ---
TARGET_IMAGE_PATH = "/path/to/your/group/photo.jpg" # Or video path
PLAYERS_DB_FOLDER = "/path/to/your/player/database/" # Contains subfolders per player
MODEL_NAME = 'Facenet' # Or another model like 'VGG-Face'
DETECTOR_BACKEND = 'retinaface' # Choose a suitable detector
SIMILARITY_THRESHOLD = 0.3 # Adjust based on testing

# --- 1. Calculate Reference Embeddings ---
player_embeddings = {}
print("Calculating reference embeddings...")
for player_name in os.listdir(PLAYERS_DB_FOLDER):
    player_folder = os.path.join(PLAYERS_DB_FOLDER, player_name)
    if os.path.isdir(player_folder):
        img_paths = [os.path.join(player_folder, f) for f in os.listdir(player_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if img_paths:
            print(f"Processing player: {player_name} with {len(img_paths)} images.")
            avg_embedding = calculate_average_embedding(img_paths, model_name=MODEL_NAME)
            if avg_embedding is not None:
                 player_embeddings[player_name] = avg_embedding
        else:
            print(f"No images found for player: {player_name}")
print("Reference embeddings calculated.")

# --- 2. Process Target Image ---
print(f"Processing target image: {TARGET_IMAGE_PATH}")
img = cv2.imread(TARGET_IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Could not read image: {TARGET_IMAGE_PATH}")

try:
    # Use extract_faces to get face images and coordinates
    faces_detected = DeepFace.extract_faces(
        img_path=TARGET_IMAGE_PATH,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=False # Don't fail if no faces are found initially
    )
except Exception as e:
     print(f"Error during face extraction: {e}")
     faces_detected = []


print(f"Detected {len(faces_detected)} faces.")

assigned_players_in_image = set() # Track assignments for this specific image

for face_obj in faces_detected:
    x, y, w, h = (face_obj['facial_area']['x'],
                  face_obj['facial_area']['y'],
                  face_obj['facial_area']['w'],
                  face_obj['facial_area']['h'])

    # Use the already extracted face image (in RGB format from extract_faces)
    # Convert face_obj['face'] (0-1 float RGB) back to BGR uint8 if needed for represent,
    # or ensure represent handles float RGB correctly.
    # DeepFace.represent expects BGR numpy array by default.
    # face_img_rgb_float = face_obj['face']
    # face_img_bgr_uint8 = (face_img_rgb_float[:, :, ::-1] * 255).astype(np.uint8)

    try:
        # Pass the extracted face directly, skip detection
        face_representation = DeepFace.represent(
            img_path=face_obj['face'], # Pass the numpy array face
            model_name=MODEL_NAME,
            enforce_detection=False, # Already detected
            detector_backend='skip'
        )

        if not face_representation:
             print("Could not represent face, skipping.")
             continue

        face_embedding = face_representation[0]['embedding']

        best_similarity = -1 # Use -1 as cosine similarity can be negative
        best_player = None

        # Compare with reference embeddings
        for player_name, player_avg_embedding in player_embeddings.items():
            if player_name not in assigned_players_in_image:
                similarity = cosine_similarity([face_embedding], [player_avg_embedding])[0][0]
                # print(f"  Comparing with {player_name}, Similarity: {similarity:.4f}") # Debugging
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_player = player_name

        # Assign label based on threshold and assignment status
        if best_player and best_similarity > SIMILARITY_THRESHOLD:
            assigned_players_in_image.add(best_player)
            label = best_player
            print(f"  Assigned {label} (Similarity: {best_similarity:.4f})")
        else:
            label = "Unknown"
            print(f"  No match found above threshold {SIMILARITY_THRESHOLD} (Best: {best_similarity:.4f})")

        # --- 3. Annotate ---
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    except Exception as e:
        print(f"Error processing a detected face: {e}")


# --- Display Result ---
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.axis("off")
# plt.show()

# Or save the result
output_path = "identified_players.jpg"
cv2.imwrite(output_path, img)
print(f"Result saved to {output_path}")

```

**Note on Video Processing:** The provided `script_version.py` includes logic for processing video frames. It accumulates embeddings over several frames (e.g., 10 frames) for each detected face track, averages them, and then performs the comparison. This temporal averaging can improve robustness against variations in single frames. Implementing robust face tracking across frames is necessary for this approach.

### 5.2. Using the Built-in API

DeepFace includes a Flask-based REST API.

**Running the API:**

1.  Navigate to the `scripts` directory in your cloned repository.
2.  Run using Gunicorn (recommended for stability):
    ```bash
    cd deepface/api/src
    gunicorn --workers=1 --timeout=3600 --bind=0.0.0.0:5005 "app:create_app()"
    ```
    (Adjust port `5005` if needed). The service will typically run on port 5005, while the internal Flask app uses 5000.
3.  Alternatively, run via Docker (see `Dockerfile` and `scripts/dockerize.sh`):
    ```bash
    cd scripts
    ./dockerize.sh # Builds and runs the container
    ```
    This usually maps container port 5000 to host port 5005.

**Endpoints:**

*   **`POST /verify`**: Performs face verification.
*   **`POST /represent`**: Generates face embeddings.
*   **`POST /analyze`**: Performs facial attribute analysis.

**Input Data:**

The API accepts input images via JSON payload or `multipart/form-data`.

*   **JSON:**
    *   Pass image data as a key (e.g., `"img"`, `"img1"`, `"img2"`).
    *   The value can be:
        *   An absolute file path on the server running the API (e.g., `"/path/on/server/image.jpg"`).
        *   A publicly accessible URL (e.g., `"https://example.com/image.jpg"`).
        *   A base64 encoded image string (e.g., `"data:image/jpeg;base64,/9j/..."`).
    *   Other parameters (`model_name`, `detector_backend`, `actions`, etc.) are passed as regular JSON key-value pairs.

*   **Multipart/form-data:**
    *   Use this method to upload image files directly.
    *   The image file should be associated with the corresponding key (e.g., `img`, `img1`, `img2`).
    *   Other parameters are sent as regular form fields. Note that list/tuple parameters like `actions` might need special formatting when sent as text fields (e.g., `"[\"age\", \"gender\"]"`).

**Example (using `requests` in Python):**

```python
import requests
import base64

API_URL = "http://localhost:5005" # Adjust if your API runs elsewhere

# --- Example: Verify using JSON (image paths on server) ---
data_verify_json = {
    "img1_path": "/path/on/server/img1.jpg",
    "img2_path": "/path/on/server/img2.jpg",
    "model_name": "Facenet"
}
# response = requests.post(f"{API_URL}/verify", json=data_verify_json)
# print("Verify (JSON paths):", response.json())

# --- Example: Represent using JSON (base64) ---
with open("dataset/img1.jpg", "rb") as f:
    img1_b64 = base64.b64encode(f.read()).decode('utf-8')
data_represent_b64 = {
    "img": f"data:image/jpeg;base64,{img1_b64}",
    "model_name": "Facenet"
}
response = requests.post(f"{API_URL}/represent", json=data_represent_b64)
print("Represent (JSON base64):", response.json())

# --- Example: Analyze using multipart/form-data (file upload) ---
files = {'img': ('img1.jpg', open('dataset/img1.jpg', 'rb'), 'image/jpeg')}
data_analyze_form = {
    'actions': '["age", "gender"]' # Send list as a string
}
response = requests.post(f"{API_URL}/analyze", files=files, data=data_analyze_form)
print("Analyze (Form-data file):", response.json())
```

Refer to the `deepface-api.postman_collection.json` file for more detailed request examples.

### 5.3. Large-Scale Face Recognition

For databases with millions or billions of faces, iterating through all embeddings (`DeepFace.find`'s default behavior) becomes too slow. Approximate Nearest Neighbor (ANN) search using vector databases or indexes is necessary.

**Concept:**

1.  **Indexing:** Use `DeepFace.represent` to generate embeddings for all faces in your database. Store these embeddings in a specialized vector index/database.
2.  **Searching:** When a new face needs identification, generate its embedding using `DeepFace.represent`. Query the vector index/database to find the *closest* (most similar) embeddings within milliseconds, even among billions.

**Tools:**

*   **Vector Indexes (Libraries):** Faiss (Facebook), Annoy (Spotify), NMSLIB, ScaNN (Google), Voyager. Integrate directly into your Python application.
*   **Vector Databases:** Elasticsearch (with vector search), Milvus, Pinecone, Weaviate, Qdrant, Postgres (with `pgvector` extension), Redis (with RediSearch). Offer managed storage, indexing, and querying.

**DeepFace Integration:** Use `DeepFace.represent` for embedding generation. The indexing and querying steps involve using the chosen vector database/library's API.

### 5.4. Encrypted Embeddings (Homomorphic Encryption)

For high-security applications where embeddings must remain encrypted even during comparison (e.g., on an untrusted cloud server), Homomorphic Encryption (HE) can be used. DeepFace's README demonstrates integration with Partially Homomorphic Encryption (PHE) using the `LightPHE` library.

**Concept (PHE Example):**

1.  **On-Premise (Client):**
    *   Generate embeddings for source (`alpha`) and target (`beta`) images using `DeepFace.represent`.
    *   Initialize a PHE cryptosystem (e.g., Paillier).
    *   Encrypt the source embedding (`encrypted_alpha`).
2.  **Cloud (Server):**
    *   Receive `encrypted_alpha` (encrypted) and `beta` (plain).
    *   Compute the dot product homomorphically: `encrypted_cosine_similarity = encrypted_alpha @ beta`. The server *cannot* decrypt this result.
3.  **On-Premise (Client):**
    *   Receive `encrypted_cosine_similarity`.
    *   Decrypt it using the private key to get the actual similarity score.
    *   Compare the decrypted score to the threshold.

**Code Snippet (from README):**

```python
# Requires: pip install lightphe
from lightphe import LightPHE
from deepface import DeepFace

# Assume threshold is defined (e.g., threshold = 0.40 for Facenet cosine)
threshold = DeepFace.find_threshold("Facenet", "cosine") # Example

# 1. On-Premise: Setup and Embeddings
cs = LightPHE(algorithm_name="Paillier", precision=19) # Precision might need tuning
alpha_emb = DeepFace.represent("dataset/img1.jpg", model_name="Facenet")[0]["embedding"]
beta_emb = DeepFace.represent("dataset/img2.jpg", model_name="Facenet")[0]["embedding"]

# 2. On-Premise: Encrypt source
encrypted_alpha = cs.encrypt(alpha_emb)

# --- Data sent to Cloud: encrypted_alpha, beta_emb ---

# 3. Cloud: Homomorphic Calculation (No private key needed)
# Note: LightPHE's @ operator might require specific vector handling.
# Conceptual representation:
encrypted_dot_product = encrypted_alpha @ beta_emb # This performs encrypted-plain dot product

# --- Result sent back to On-Premise: encrypted_dot_product ---

# 4. On-Premise: Decrypt Result (Private key needed)
# LightPHE decryption might return a list, access the first element for the scalar dot product
decrypted_dot_product = cs.decrypt(encrypted_dot_product)[0]

# 5. On-Premise: Calculate Cosine Similarity (if needed) and Verify
# Note: PHE usually works directly with dot products. If cosine is strictly needed,
# norms would also need handling (potentially complex with PHE).
# Assuming the threshold is adapted for dot product or embeddings are pre-normalized:

# Example verification based on dot product (assuming normalized embeddings)
# Cosine similarity = dot product for normalized vectors
calculated_similarity = decrypted_dot_product
verified = calculated_similarity >= (1 - threshold) # Cosine distance = 1 - similarity

print(f"Calculated Similarity (Encrypted): {calculated_similarity}")
print("Same person" if verified else "Different persons")
```

**Note:** Fully Homomorphic Encryption (FHE) allows computations on *two* encrypted inputs (e.g., encrypted embedding vs. encrypted embedding) but is significantly more computationally expensive. See the `CipherFace` library mentioned in the README for FHE examples.

## 6. Performance and Benchmarks

*   **Model Choice:** Significantly impacts accuracy and speed. Models like Facenet512, ArcFace, VGG-Face generally offer higher accuracy. Lightweight models like SFace or GhostFaceNet are faster.
*   **Detector Choice:** Accuracy vs. Speed trade-off. RetinaFace/MTCNN are often more accurate but slower than OpenCV/SSD/YOLO/YuNet.
*   **Alignment:** Slightly increases processing time but usually improves accuracy.
*   **Hardware:** GPUs significantly accelerate deep learning model inference.

Refer to the `benchmarks/` directory in the repository for detailed performance comparisons between different model/detector/metric combinations on datasets like LFW. The README also summarizes key accuracy findings.

## 7. Contribution

Contributions are welcome via Pull Requests.

*   Run tests and linting locally before submitting: `make test && make lint`.
*   Follow the structure in `.github/pull_request_template.md`.
*   Adhere to the `.github/CODE_OF_CONDUCT.md`.

## 8. Support

*   Star the GitHub repository.
*   Financial support: [Patreon](https://www.patreon.com/serengil?repo=deepface), [GitHub Sponsors](https://github.com/sponsors/serengil), [Buy Me a Coffee](https://buymeacoffee.com/serengil).

## 9. Citation

If DeepFace helps your research, please cite the relevant papers listed in `CITATION.md`.

**Primary Benchmark Paper:**

```bibtex
@article{serengil2024lightface,
  title     = {A Benchmark of Facial Recognition Pipelines and Co-Usability Performances of Modules},
  author    = {Serengil, Sefik and Ozpinar, Alper},
  journal   = {Journal of Information Technologies},
  volume    = {17},
  number    = {2},
  pages     = {95-107},
  year      = {2024},
  doi       = {10.17671/gazibtd.1399077},
  url       = {https://dergipark.org.tr/en/pub/gazibtd/issue/84331/1399077},
  publisher = {Gazi University}
}
```
*(See `CITATION.md` for others related to specific functionalities like attribute analysis or older framework versions).*

Also, add `deepface` to your `requirements.txt` if used in GitHub projects.

## 10. License

DeepFace itself is licensed under the MIT License.

However, it wraps several external models and detectors, each with its own license (VGG-Face, Facenet, OpenFace, DeepFace (FB), DeepID, ArcFace, Dlib, SFace, GhostFaceNet, Buffalo_L, OpenCV, SSD, MTCNN, Fast MTCNN, RetinaFace, MediaPipe, YuNet, YOLO, CenterFace, FasNet). **Please verify the licenses of the specific models/detectors you intend to use, especially for commercial applications.**

---