import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Definir las opciones usando la ruta completa y segura
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)
# --- Funciones de Lógica de Señas ---
# Nota: En MediaPipe, Y crece hacia ABAJO y X crece hacia la DERECHA.
def check_direction(landmarks):
    # Usamos el dedo índice: TIP (8) y MCP (5)
    tip = landmarks[8]
    mcp = landmarks[5]
    
    # Calculamos distancia euclidiana como umbral de sensibilidad
    distance = math.sqrt((tip.x - mcp.x)**2 + (tip.y - mcp.y)**2)
    threshold = distance * 0.8

    if tip.y < mcp.y - threshold:
        return "Up"
    elif tip.y > mcp.y + threshold:
        return "Down"
    elif tip.x < mcp.x - threshold:
        return "Left"
    elif tip.x > mcp.x + threshold:
        return "Right"
    return None

# --- Configuración del Landmarker ---
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO, # Modo específico para video/camara
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Bucle Principal ---
cam = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    while cam.isOpened():
        ret, frame = cam.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        # Convertir BGR a RGB
        rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Necesitamos un timestamp en milisegundos para el modo VIDEO
        frame_timestamp_ms = int(time.time() * 1000)
        
        # Procesar frame
        result = landmarker.detect_for_video(rgb_frame, frame_timestamp_ms)

        # Dibujar y procesar lógica
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                # Dibujar conexiones (opcional, usando la utilidad antigua que sigue siendo compatible)
                # O puedes iterar y dibujar círculos manualmente con cv2.circle
                
                # Lógica de dirección
                direction = check_direction(hand_landmarks)
                if direction:
                    cv2.putText(frame, direction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    print(f"Comando Drone: {direction}")

        cv2.imshow("Hand Control - Tasks API", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break

cam.release()
cv2.destroyAllWindows()