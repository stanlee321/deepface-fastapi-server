import cv2
from libs.core import CoreDetector
from libs.utils import get_gpu_device


GLOBAL_RESOLUTION = (640, 640)
THRESHOLD = 0.3
DEVICE = get_gpu_device()
# VIDEO_SOURCE = "./video_prueba_1.mp4"
VIDEO_SOURCE = "prueba1.mp4"

# Modelo de detección
# best_11n.pt es el modelo de detección  mas pequeño (n de nano)
# best_11s.pt es el modelo de detección mas grande (s de small)

detector = CoreDetector(
    model_path="./weights_weapons_1.pt",
    # rfdetr_model_path="./weights_weapons_12_may.pt"
)



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
        results = {"image_path_or_identifier": image_identifier, "objects": [], "error": None}

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
                
                results["objects"].append({
                    "object_index": i,
                    "bounding_box": {"x": int(x1_f), "y": int(y1_f), "w": int(w_f), "h": int(h_f)},
                    "confidence": float(conf_val),
                    "class_id": int(class_id_val) if class_id_val is not None else None,
                    "class_name": c_name
                })
            except (TypeError, ValueError) as e:
                print(f"Warning: Error processing RFDETR detection item {i} (xyxy: {str(xyxy_val)[:50]}, conf: {conf_val}, id: {class_id_val}). Error: {e}. Skipping.")
                continue
    return results


# Función principal
def main():

    # Continuar con el procesamiento del video
    video = cv2.VideoCapture(VIDEO_SOURCE)
    
    # Verificar si la cámara se abrió correctamente
    if not video.isOpened():
        print("Error: No se pudo abrir la cámara o el video.")
        exit()
    
    # Configurar la resolución del video
    video.set(cv2.CAP_PROP_FRAME_WIDTH, GLOBAL_RESOLUTION[0])
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, GLOBAL_RESOLUTION[1])
    
    print("Video inicializado correctamente")
    
    while True:
        ret, frame = video.read()
        if not ret:
            print("Error: No se pudo leer el frame.")
            # Reiniciar el video cuando llegue al final
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        print(f"DEVICE: {get_gpu_device()}")

        # Redimensionar el frame a GLOBAL_RESOLUTION
        frame = cv2.resize(frame, GLOBAL_RESOLUTION)
        
        annotated_frame, detections = detector.process_frame(frame, confidence_threshold=THRESHOLD, device=DEVICE)
        results = parse_weapon_detections(detections, VIDEO_SOURCE, "rfdetr")
        print("RESULTS", results)

        # Mostrar el resultado final
        # cv2.imshow('Zonas y ROI', frame_with_overlays) # No longer needed
        cv2.imshow('Deteccion YOLO con Zonas', annotated_frame)

        # Presionar 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Liberar recursos
    video.release()
    cv2.destroyAllWindows()
    
# Ejecutar el programa
if __name__ == "__main__":
    main()
