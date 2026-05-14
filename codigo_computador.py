import cv2
import numpy as np
from ultralytics import YOLO
from pupil_apriltags import Detector
import time

# Funcion para ver si la persona esta en el area
def esta_en_zona(poligono, x1, y1, x2, y2, h, w):
    mask_poly = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_poly, [poligono], 255)
    persona_roi = mask_poly[y1:y2, x1:x2]
    if persona_roi.size == 0: return 0
    pixeles_dentro = np.sum(persona_roi == 255)
    area_total = (x2 - x1) * (y2 - y1)
    return pixeles_dentro / area_total

def iniciar():
    at_detector = Detector(families='tag36h11')
    model = YOLO("yolo26n.pt")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): cap = cv2.VideoCapture(1)

    # --- SIMULACION ---
    bateria_sim = 100
    tiempo_inicio = time.time()
    TAG_SIZE_METERS = 0.1  # 10 cm por lado

    print("--- Simulador Metric-Vision (ESC: Salir) ---")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # ELIMINAMOS EL FLIP: El detector de AprilTags no funciona bien con imagenes espejadas
        h, w, _ = frame.shape
        
        # Telemetria simulada
        segundos_vuelo = int(time.time() - tiempo_inicio)
        bateria_sim = 100 - (segundos_vuelo // 10)
        altura_sim = 1.2 # metros
        
        # A. Deteccion de Tags (Sobre la imagen original, sin flip)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detecciones = at_detector.detect(gray)
        
        puntos_tags = []
        pixeles_por_metro = 0
        
        for d in detecciones:
            cx, cy = int(d.center[0]), int(d.center[1])
            puntos_tags.append((cx, cy))
            
            # Calcular tamaño del tag en pixeles (promedio de lados)
            c = d.corners
            lado1 = np.linalg.norm(c[0] - c[1])
            lado2 = np.linalg.norm(c[1] - c[2])
            lado_prom_px = (lado1 + lado2) / 2
            
            # Factor de conversion: Pixeles por Metro
            if lado_prom_px > 0:
                pixeles_por_metro = lado_prom_px / TAG_SIZE_METERS

            # Dibujo basico
            pts = [tuple(p.astype(int)) for p in d.corners]
            for i in range(4):
                cv2.line(frame, pts[i], pts[(i+1)%4], (255, 0, 0), 2)

        # B. Area y Distancia
        poligono = None
        area_m2 = 0
        distancia_tags_m = 0
        
        if len(puntos_tags) >= 2 and pixeles_por_metro > 0:
            # Distancia entre los dos primeros tags detectados
            d_px = np.linalg.norm(np.array(puntos_tags[0]) - np.array(puntos_tags[1]))
            distancia_tags_m = d_px / pixeles_por_metro

        if len(puntos_tags) >= 3:
            pts_hull = np.array(puntos_tags)
            hull = cv2.convexHull(pts_hull)
            poligono = hull.reshape(-1, 2)
            
            # Calcular area en m^2
            area_px = cv2.contourArea(poligono)
            if pixeles_por_metro > 0:
                area_m2 = area_px / (pixeles_por_metro ** 2)
            
            # Visualización
            overlay = frame.copy()
            cv2.fillPoly(overlay, [poligono], (0, 255, 255))
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            cv2.polylines(frame, [poligono], True, (0, 255, 255), 2)

        # C. Personas y Densidad
        results = model.predict(frame, classes=[0], conf=0.4, verbose=False)
        conteo = 0
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                if poligono is not None:
                    if esta_en_zona(poligono, x1, y1, x2, y2, h, w) >= 0.30:
                        conteo += 1
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        densidad = conteo / area_m2 if area_m2 > 0 else 0

        # --- D. INTERFAZ ---
        # 1. Panel Monitor (Top-Left)
        cv2.rectangle(frame, (10, 10), (250, 100), (40, 40, 40), -1)
        cv2.rectangle(frame, (10, 10), (250, 100), (200, 200, 200), 1)
        cv2.putText(frame, "DRONE MONITOR", (20, 30), 0, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"Tags: {len(puntos_tags)}", (20, 50), 0, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"Area: {area_m2:.2f} m2", (20, 65), 0, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"Personas: {conteo}", (20, 80), 0, 0.4, (0, 255, 0), 1)
        cv2.putText(frame, f"Densidad: {densidad:.2f} p/m2", (20, 95), 0, 0.4, (0, 255, 255), 1)

        # 2. Panel Telemetria (Bottom-Right)
        tw, th = 200, 75
        cv2.rectangle(frame, (w-tw-10, h-th-10), (w-10, h-10), (40, 40, 40), -1)
        cv2.rectangle(frame, (w-tw-10, h-th-10), (w-10, h-10), (200, 200, 200), 1)
        cv2.putText(frame, "TELEMETRIA", (w-tw, h-60), 0, 0.5, (255, 100, 0), 1)
        cv2.putText(frame, f"Bateria: {bateria_sim}%", (w-tw, h-45), 0, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"Altura: {altura_sim:.1f} m", (w-tw, h-30), 0, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"Tiempo: {segundos_vuelo}s", (w-tw, h-15), 0, 0.4, (255, 255, 255), 1)

        cv2.imshow('Deteccion Drone - Metrica', frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    iniciar()
