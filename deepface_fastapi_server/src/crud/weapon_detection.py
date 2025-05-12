import cv2
import torch
from ultralytics import YOLO
import supervision as sv
from typing import List, Dict, Any

from config import settings

class CoreDetector:
    def __init__(self, model_path="best_11n.pt"):
        
        self.model = YOLO(model_path)
        
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
        return self.parse_weapon_detections(detections, frame)
    
    
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
            model_path=settings.WEAPON_DETECTION_MODEL_PATH
        )
    return weapon_detector
    
