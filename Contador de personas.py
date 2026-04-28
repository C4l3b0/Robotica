import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Realizamos las configuraciones del detector de personas
options = vision.ObjectDetectorOptions(
    base_options = python.BaseOptions(model_asset_path = "efficientdet_lite0.tflite"),
    running_mode = vision.RunningMode.IMAGE, 
    max_results = 10, 
    score_threshold = 0.2
)

# Inicializamos el detector
detector = vision.ObjectDetector.create_from_options(options)

# Inicialización de la captura de video
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

print("Iniciando conteo de weones")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape   
    # Convertimos el video BGR a RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Realizamos la detección
    detection_result = detector.detect(mp_frame)

    personas_actuales = [d for d in detection_result.detections if d.categories[0].category_name == "person"]
    contador_personas = len(personas_actuales)

    # Procesamos las detecciones para dibujar los cuadros
    for detection in personas_actuales:
        category = detection.categories[0]
        bbox = detection.bounding_box
        coords = np.array([bbox.origin_x, bbox.origin_y, bbox.width, bbox.height], dtype=int)
        x, y, w_box, h_box = coords

        # Dibujamos el cuadrado para cada persona
        cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
        
        # Etiqueta de confianza
        display_text = f"P: {round(category.score, 2)}"
        cv2.putText(frame, display_text, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Colacamos el contador de personas
    cv2.putText(frame, f"Personas: {contador_personas}", (w - 200, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Mostramos el video en vivo
    cv2.imshow('Deteccion de Personas con Contador', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

detector.close()
cap.release()
cv2.destroyAllWindows()
