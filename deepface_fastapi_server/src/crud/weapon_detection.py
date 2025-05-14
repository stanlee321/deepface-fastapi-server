import cv2
import torch
from ultralytics import YOLO
from rfdetr import RFDETRBase

import supervision as sv
from typing import List, Dict, Any,Optional

from config import settings



def parse_weapon_detections(detections, image_identifier, dt_type: str):
    
    if dt_type == "yolo":
        results = {"image_path_or_identifier": image_identifier, "objects": [], "error": None}
        for i, (xyxy, conf, class_id, class_name) in enumerate(zip(detections.xyxy, detections.confidence, detections.class_id, detections.data.get("class_name", []))):
            x1, y1, x2, y2 = xyxy
        w = x2 - x1
        h = y2 - y1
        results["objects"].append({
            "object_index": i,
                "bounding_box": {"x": int(x1), "y": int(y1), "w": int(w), "h": int(h)},
                "confidence": float(conf),
                "class_id": int(class_id) if class_id is not None else None,
                "class_name": class_name if class_name else None
            })
    elif dt_type == "rfdetr":
        results = []

        if not (hasattr(detections, 'xyxy') and \
                hasattr(detections, 'confidence') and \
                hasattr(detections, 'class_id')):
            error_message = (f"RFDETR detections are not in the expected 'Detections' object format "
                             f"(missing one of: xyxy, confidence, class_id). Got type: {type(detections)}. No objects processed.")
            results["error"] = error_message
            print(f"Error: {error_message} - Detections content: {str(detections)[:200]}")
            return results

        if not hasattr(detections.xyxy, '__len__') or len(detections.xyxy) == 0:
            return results

        num_detections = len(detections.xyxy)
        
        class_names_list = [None] * num_detections
        if hasattr(detections, 'data') and isinstance(detections.data, dict) and "class_name" in detections.data:
            retrieved_class_names = detections.data["class_name"]
            valid_class_names_found = False
            if isinstance(retrieved_class_names, list) and len(retrieved_class_names) == num_detections:
                class_names_list = retrieved_class_names
                valid_class_names_found = True
            elif hasattr(retrieved_class_names, 'shape') and hasattr(retrieved_class_names, 'tolist') and \
                 len(retrieved_class_names.shape) == 1 and retrieved_class_names.shape[0] == num_detections:
                class_names_list = retrieved_class_names.tolist() # Convert NumPy array to list
                valid_class_names_found = True
            
            if not valid_class_names_found:
                actual_type_str = str(type(retrieved_class_names))
                actual_len_shape_str = 'N/A'
                if hasattr(retrieved_class_names, 'shape'): # Primarily for NumPy arrays
                    actual_len_shape_str = f"shape {retrieved_class_names.shape}"
                elif hasattr(retrieved_class_names, '__len__'): # For lists and other sequences
                    try: actual_len_shape_str = f"length {len(retrieved_class_names)}"
                    except TypeError: pass

                print(f"Warning: For RFDETR, 'detections.data[\"class_name\"]' was present but not a compatible list or NumPy array of length {num_detections}. "
                      f"Actual type: {actual_type_str}, Actual_len/shape: {actual_len_shape_str}. Using None for class names.")
        
        for i, (xyxy_val, conf_val, class_id_val, c_name) in enumerate(zip(detections.xyxy, detections.confidence, detections.class_id, class_names_list)):
            try:
                valid_xyxy = False
                if isinstance(xyxy_val, (list, tuple)) and len(xyxy_val) == 4:
                    valid_xyxy = True
                elif hasattr(xyxy_val, 'shape') and hasattr(xyxy_val, 'tolist'): # Check for NumPy-like array
                    if len(xyxy_val.shape) == 1 and xyxy_val.shape[0] == 4:
                        valid_xyxy = True
                
                if not valid_xyxy:
                    type_str = str(type(xyxy_val))
                    shape_str = str(xyxy_val.shape) if hasattr(xyxy_val, 'shape') else "N/A"
                    val_str = str(xyxy_val)[:100]
                    print(f"Warning: RFDETR detection item {i} has invalid 'xyxy' format. Expected list/tuple of 4, or NumPy-like array of shape (4,). "
                          f"Got Type: {type_str}, Shape: {shape_str}, Value: {val_str}. Skipping.")
                    continue
                
                x1_f, y1_f, x2_f, y2_f = map(float, xyxy_val)
                w_f = x2_f - x1_f
                h_f = y2_f - y1_f
                
                data = {
                    "image_path_or_identifier": image_identifier, 
                    "objects": [], 
                    "error": None
                }
                
                data["objects"].append({
                    "object_index": i,
                    "weapon_area": {"x": int(x1_f), "y": int(y1_f), "w": int(w_f), "h": int(h_f)},
                    "confidence": float(conf_val),
                    "class_id": int(class_id_val) if class_id_val is not None else None,
                    "class_name": c_name if c_name else None
                })

                results.append(data)
                
            except (TypeError, ValueError) as e:
                print(f"Warning: Error processing RFDETR detection item {i} (xyxy: {str(xyxy_val)[:50]}, conf: {conf_val}, id: {class_id_val}). Error: {e}. Skipping.")
                continue
    return results


class CoreDetector:
    def __init__(self, model_path="best_11n.pt", rfdetr_model_path: Optional[str] = None):
        
        self.model = YOLO(model_path)
        self.rfdetr_model = None
        if rfdetr_model_path:
            self.rfdetr_model = RFDETRBase(pretrain_weights=rfdetr_model_path)
        self.device = self.get_gpu_device()
        
    @staticmethod
    def get_gpu_device():
        """
        Get the GPU device.
        
        if NVIDIA GPU is available,  return "gpu"
        if MPS (Apple Silicon GPU) is available, return "mps"
        otherwise, return "cpu"
        """
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            print(torch.cuda.get_device_name(0))
            return "cuda"
        else:
            return "cpu"
    def calculate_centroid(self, xyxy):
        """
        From the detection coordinates (xyxy), calculate the centroid.
        """
        x1, y1, x2, y2 = xyxy # Unpack coordinates directly
        
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        
        return x, y
    
    def draw_centroid(self, frame, detections: sv.Detections):
        # Iterate through the bounding box coordinates of each detection
        for xyxy_det in detections.xyxy: 
            x, y = self.calculate_centroid(xyxy_det) # Pass the coordinates
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
        return frame
    
    @staticmethod
    def parse_weapon_detections(detections, image_identifier)->List[Dict[str, Any]]:
        """
        Parse the detections into a list of dictionaries.
        Each dictionary contains the image identifier, the objects detected, and the error.
        Args:
            detections: The detections to parse.
            image_identifier: The identifier of the image.
        Returns:
            A list of dictionaries, each containing the image identifier, the objects detected, and the error.
        """
        results = []
    
        for i, (xyxy, conf, class_id, class_name) in enumerate(zip(detections.xyxy, detections.confidence, detections.class_id, detections.data.get("class_name", []))):
            x1, y1, x2, y2 = xyxy
            w = x2 - x1
            h = y2 - y1
            
            data = {
                "image_path_or_identifier": image_identifier, 
                "objects": [], 
                "error": None
            }
            data["objects"].append({
                "object_index": i,
                "weapon_area": {"x": int(x1), "y": int(y1), "w": int(w), "h": int(h)},
                "confidence": float(conf),
                "class_id": int(class_id) if class_id is not None else None,
                "class_name": class_name if class_name else None
            })
            results.append(data)
    
        return results
    
    def process_frame(self, frame, confidence_threshold=settings.WEAPON_DETECTION_CONFIDENCE_THRESHOLD)->List[Dict[str, Any]]:
        """
        Process a frame and return the detections.
        Args:
            frame: The frame to process ().
            confidence_threshold: The confidence threshold to use.
        Returns:
            A list of dictionaries, each containing the image identifier, the objects detected, and the error.
        """
        
        print(f"Processing frame with confidence threshold: {confidence_threshold}")
        
        results = self.model.predict(frame,
                                     save=False, 
                                     device=self.device, 
                                     conf=confidence_threshold, show=False)
        
        detections = sv.Detections.from_ultralytics(results[0])
        
        print(f"Detections: {detections}")
        # Handle case where no detections are found (or left after filtering)
        if len(detections) == 0:
            return []
        return parse_weapon_detections(detections, frame, dt_type="rfdetr")
    
    
weapon_detector = None

# TODO: FastAPI Scalability Improvement
# The current singleton implementation of CoreDetector, while memory-efficient,
# can become a bottleneck in a concurrent FastAPI environment due to the
# synchronous nature of `process_frame` (especially `model.predict`).
# Under load, requests might queue up, increasing latency.
#
# Future improvements to consider:
# 1. Offload synchronous work: Wrap the call to `weapon_detector.process_frame`
#    within FastAPI endpoints using `fastapi.concurrency.run_in_threadpool`.
#    Example in an endpoint:
#    ```python
#    # from fastapi.concurrency import run_in_threadpool
#    # processed_frame, centroids, detections = await run_in_threadpool(
#    #     weapon_detector.process_frame, frame, confidence_threshold=0.5
#    # )
#    ```
# 2. For truly CPU-bound tasks that hold the GIL, a multiprocessing Pool
#    (e.g., from `concurrent.futures.ProcessPoolExecutor`) might be more
#    effective than a thread pool for the `process_frame` execution.
#    This would require careful management of the model instance across processes
#    or loading it in each worker process.
# 3. Dedicated Model Serving: For very high throughput requirements, consider
#    deploying the model using a dedicated model serving solution like
#    NVIDIA Triton Inference Server, TensorFlow Serving, TorchServe, or BentoML.
#    The FastAPI server would then make requests to this dedicated service.

def get_weapon_detector()->CoreDetector:
    global weapon_detector
    if weapon_detector is None:
        weapon_detector = CoreDetector(
            model_path="/home/stanley/Desktop/2024/lucam/deepface-fastapi-server/weapons/weights_weapons_v1.pt",
            rfdetr_model_path=settings.WEAPON_DETECTION_MODEL_PATH
            # model_path=settings.WEAPON_DETECTION_MODEL_PATH
            
        )
    return weapon_detector
    
