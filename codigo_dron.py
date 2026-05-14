# Importamos las librerias
import numpy as np
from ultralytics import YOLO
from pupil_apriltags import Detector
from djitellopy import Tello
import cv2, time, os, sys, signal, platform

# Funcion para ver si la persona esta en el area (70%)
def esta_en_zona(poligono, x1, y1, x2, y2, h, w):
    mask_poly = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_poly, [poligono], 255)
    persona_roi = mask_poly[y1:y2, x1:x2]
    if persona_roi.size == 0: return 0
    pixeles_dentro = np.sum(persona_roi == 255)
    area_total = (x2 - x1) * (y2 - y1)
    return pixeles_dentro / area_total

# Cargamos el modelo YOLO26
model = YOLO("yolo26n.pt")

# Detector de AprilTags (Familia estandar)
at_detector = Detector(families='tag36h11')

# =======================
# CONFIGURACION TELLO
# =======================
tello = Tello()
tello.connect()
tello.streamoff()
tello.streamon()
frame_read = tello.get_frame_read()
time.sleep(2)

# Referencia metrica: 10 cm por lado del tag
TAG_SIZE_METERS = 0.1

# =======================
# TECLADO (OS DEPENDANT)
# =======================
OS = platform.system()
USE_PYNPUT = (OS == "Darwin")
if USE_PYNPUT:
    from pynput import keyboard
    keys = set()
    def on_press(key):
        try: keys.add(key.char)
        except: 
            if key == keyboard.Key.esc: keys.add('esc')
    def on_release(key):
        try: keys.discard(key.char)
        except:
            if key == keyboard.Key.esc: keys.discard('esc')
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

# =======================
# FUNCIONES DE SEGURIDAD
# =======================
def safe_land():
    print("ATERRIZAJE DE EMERGENCIA/SEGURIDAD...")
    for _ in range(5):
        tello.send_rc_control(0,0,0,0)
        time.sleep(0.05)
    time.sleep(0.3)
    try:
        tello.land()
    except:
        tello.emergency()

def handler(sig, frame):
    safe_land()
    tello.streamoff()
    tello.end()
    sys.exit(0)

signal.signal(signal.SIGINT, handler)

# Despegue inicial
tello.takeoff()
time.sleep(2)

# =======================
# LOOP PRINCIPAL
# =======================
speed = 40
last_rc_time = 0
rc_interval = 0.05

tinicial = time.time()

while True:
    # 1. Captura y Redimensionamiento
    # Redimensionamos AL PRINCIPIO para que toda la vision y el UI esten en la misma escala
    frame_raw = frame_read.frame
    if frame_raw is None: continue
    
    # Redimensionamos a 640x480 para asegurar que se vea todo en pantalla y vaya fluido
    frame = cv2.resize(frame_raw, (640, 480))
    frame2 = frame.copy()
    h, w, _ = frame.shape
    
    # 2. Seguridad de Bateria
    battery = tello.get_battery()
    if battery < 20:
        print(f"BATERIA CRITICA ({battery}%). ATERRIZANDO...")
        safe_land()
        break

    # 3. Vision: AprilTags
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detecciones = at_detector.detect(gray)
    
    puntos_tags = []
    pixeles_por_metro = 0
    
    for d in detecciones:
        cx, cy = int(d.center[0]), int(d.center[1])
        puntos_tags.append((cx, cy))
        
        # Calculo metrico basado en el tamaño del tag
        c = d.corners
        lado_px = (np.linalg.norm(c[0]-c[1]) + np.linalg.norm(c[1]-c[2])) / 2
        if lado_px > 0:
            pixeles_por_metro = lado_px / TAG_SIZE_METERS

        # Dibujo de tag
        pts_tag = [tuple(p.astype(int)) for p in d.corners]
        for i in range(4):
            cv2.line(frame, pts_tag[i], pts_tag[(i+1)%4], (255, 0, 0), 2)
        cv2.putText(frame, f"ID:{d.tag_id}", (pts_tag[0][0], pts_tag[0][1]-10), 0, 0.5, (0, 0, 255), 2)

    # 4. Construccion de Area y Calculos Metricos
    poligono = None
    area_m2 = 0
    if len(puntos_tags) >= 3:
        pts_hull = np.array(puntos_tags)
        hull = cv2.convexHull(pts_hull)
        poligono = hull.reshape(-1, 2)
        
        # Area en metros cuadrados
        area_px = cv2.contourArea(poligono)
        if pixeles_por_metro > 0:
            area_m2 = area_px / (pixeles_por_metro ** 2)
        
        # Visualizacion del area
        overlay = frame.copy()
        cv2.fillPoly(overlay, [poligono], (0, 255, 255))
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.polylines(frame, [poligono], True, (0, 255, 255), 2)

    # 5. Deteccion de Personas YOLO
    results = model.predict(frame2, classes=[0], conf=0.4, verbose=False)
    conteo_personas = 0
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            if poligono is not None:
                if esta_en_zona(poligono, x1, y1, x2, y2, h, w) >= 0.70:
                    conteo_personas += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Calculo de densidad
    densidad = conteo_personas / area_m2 if area_m2 > 0 else 0

    # 6. INTERFAZ FINAL (Dual Panel)
    # PANEL MONITOR (Top-Left)
    cv2.rectangle(frame, (10, 10), (250, 105), (40, 40, 40), -1)
    cv2.rectangle(frame, (10, 10), (250, 105), (200, 200, 200), 1)
    cv2.putText(frame, "DRONE MONITOR", (20, 30), 0, 0.5, (0, 255, 255), 1)
    cv2.putText(frame, f"Tags Detectados: {len(puntos_tags)}", (20, 50), 0, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, f"Area Area: {area_m2:.2f} m2", (20, 65), 0, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, f"Personas: {conteo_personas}", (20, 80), 0, 0.4, (0, 255, 0), 1)
    cv2.putText(frame, f"Densidad: {densidad:.2f} p/m2", (20, 95), 0, 0.4, (0, 255, 255), 1)

    # PANEL TELEMETRIA (Bottom-Right)
    tw, th = 200, 75
    cv2.rectangle(frame, (w-tw-10, h-th-10), (w-10, h-10), (40, 40, 40), -1)
    cv2.rectangle(frame, (w-tw-10, h-th-10), (w-10, h-10), (200, 200, 200), 1)
    cv2.putText(frame, "TELEMETRIA", (w-tw, h-60), 0, 0.5, (255, 128, 0), 1)
    cv2.putText(frame, f"Bateria: {battery}%", (w-tw, h-45), 0, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, f"Altura: {tello.get_height()/100:.2f} m", (w-tw, h-30), 0, 0.4, (255, 255, 255), 1)
    tvuelo = int(time.time()- tinicial)
    cv2.putText(frame, f"Tiempo: {tvuelo} s", (w-tw, h-15), 0, 0.4, (255, 255, 255), 1)
    # 7. DISPLAY (NO convertir a RGB, imshow usa BGR)
    cv2.imshow("TELLO MISSION CONTROL", cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    
    key = cv2.waitKey(1) & 0xFF
    if USE_PYNPUT:
        cv2.pollKey()
        pressed = keys.copy()
    else:
        pressed = set()
        if key != 255: pressed.add(chr(key))

    lr, fb, ud, yaw = 0, 0, 0, 0
    if 'w' in pressed: fb = speed
    if 's' in pressed: fb = -speed
    if 'a' in pressed: lr = -speed
    if 'd' in pressed: lr = speed
    if 'r' in pressed: ud = speed
    if 'f' in pressed: ud = -speed
    if 'q' in pressed: yaw = -speed
    if 'e' in pressed: yaw = speed

    now = time.time()
    if now - last_rc_time > rc_interval:
        tello.send_rc_control(lr, fb, ud, yaw)
        last_rc_time = now

    if 'l' in pressed or 'esc' in pressed or key == 27:
        safe_land()
        break

# Cleanup
tello.streamoff()
tello.end()
cv2.destroyAllWindows()
