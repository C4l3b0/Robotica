import cv2
import numpy as np
from ultralytics import YOLO

# Cargamos el modelo a utilizar
model = YOLO("yolo26n.pt") 

# Iniciamos la captura de video
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

print("Iniciando YOLO26... Espera a que cargue el modelo.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    results = model.predict(frame, classes=[0], conf=0.3, verbose=False, stream=True)
    contador_personas = 0

    for r in results:
        boxes = r.boxes
        for box in boxes:
            contador_personas += 1
            
            coords = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords
            
            conf = float(box.conf[0])

            # Dibujamos el cuadrado
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Etiqueta
            label = f"Persona: {round(conf, 2)}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Agregamos el contador en el video
    cv2.putText(frame, f"Personas: {contador_personas}", (w - 210, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Mostrar video
    cv2.imshow('Deteccion de Personas - YOLO26', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
