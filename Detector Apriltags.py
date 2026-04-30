import cv2
from pupil_apriltags import Detector

def iniciar_deteccion():
    # 1. Configurar el detector
    # Usamos la familia tag36h11 que es el estándar
    at_detector = Detector(families='tag36h11')

    # 2. Captura de video
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: No se detecta la cámara.")
        return

    print("--- Buscando AprilTags (Presiona 'q' para salir) ---")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # El detector necesita la imagen en blanco y negro (escala de grises)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 3. Detectar los tags en el frame
        detecciones = at_detector.detect(gray)

        # 4. Dibujar la información en pantalla
        for d in detecciones:
            # Extraer esquinas para dibujar el cuadro
            (ptA, ptB, ptC, ptD) = d.corners
            ptA = (int(ptA[0]), int(ptA[1]))
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))

            # Dibujar líneas verdes alrededor del tag
            cv2.line(frame, ptA, ptB, (255, 0, 0), 2)
            cv2.line(frame, ptB, ptC, (255, 0, 0), 2)
            cv2.line(frame, ptC, ptD, (255, 0, 0), 2)
            cv2.line(frame, ptD, ptA, (255, 0, 0), 2)

            # Escribir el ID del tag detectado
            texto = f"ID: {d.tag_id}"
            cv2.putText(frame, texto, (ptA[0], ptA[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Mostrar el resultado
        cv2.imshow("Vision por Computadora - AprilTags", frame)

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    iniciar_deteccion()