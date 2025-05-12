from ultralytics import YOLO
import supervision as sv
import cv2

class CoreDetector:
    def __init__(self, model_path="best_11n.pt"):
        
        self.model = YOLO(model_path)

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
    
    def process_frame(self, frame, confidence_threshold=0.4, device="cpu"):
        results = self.model.predict(frame, save=False, device=device, conf=confidence_threshold, show=False)
        detections = sv.Detections.from_ultralytics(results[0])
        
        # Handle case where no detections are found (or left after filtering)
        if len(detections) == 0:
            return frame.copy(), [] # Return original frame, empty centroids list


        # Start with the original frame or Ultralytics annotated frame
        # annotated_frame = frame.copy() # Option 1: Start with clean frame
        annotated_frame = results[0].plot(kpt_radius=2) # Option 2: Use Ultralytics plot
        
        # Draw centroids on the frame
        annotated_frame = self.draw_centroid(annotated_frame, detections) 
        
        # Calculate centroids for all detections
        centroids = [self.calculate_centroid(xyxy_det) for xyxy_det in detections.xyxy]

        
        return annotated_frame, centroids, detections