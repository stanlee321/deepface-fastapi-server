import cv2
from libs.core import CoreDetector
from libs.utils import get_gpu_device


GLOBAL_RESOLUTION = (640, 640)
THRESHOLD = 0.3
DEVICE = get_gpu_device()
# VIDEO_SOURCE = "./video_prueba_1.mp4"
VIDEO_SOURCE = 0

# Modelo de detección
# best_11n.pt es el modelo de detección  mas pequeño (n de nano)
# best_11s.pt es el modelo de detección mas grande (s de small)

detector = CoreDetector(
    model_path="/home/stanley/Desktop/2024/lucam/deepface-fastapi-server/weapons/weights_weapons_v1.pt"
)



def parse_weapon_detections(detections, image_identifier):
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
        
        annotated_frame, centroids, detections = detector.process_frame(frame, confidence_threshold=THRESHOLD, device=DEVICE)
        results = parse_weapon_detections(detections, VIDEO_SOURCE)
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
