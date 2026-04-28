# Importamos las librerias

import cv2              # Libreria de vizualización de camara
import mediapipe as mp  # Libreria de detección de señas
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Definimos las funciones de las señas

def dron_up(landmarks,finger_tip,finger_mcp):
    
    distance = math.sqrt((landmarks[finger_tip].x - landmarks[finger_mcp].x)**2 + (landmarks[finger_tip].y - landmarks[finger_mcp].y)**2)
    return landmarks[finger_tip].y + distance*0.8 < landmarks[finger_mcp].y

def dron_down(landmarks,finger_tip,finger_mcp):
    distance = math.sqrt((landmarks[finger_tip].x - landmarks[finger_mcp].x)**2 + (landmarks[finger_tip].y - landmarks[finger_mcp].y)**2)
    return landmarks[finger_tip].y > landmarks[finger_mcp].y + distance*0.8 

def dron_left(landmarks,finger_tip,finger_mcp):
    distance = math.sqrt((landmarks[finger_tip].x - landmarks[finger_mcp].x)**2 + (landmarks[finger_tip].y - landmarks[finger_mcp].y)**2)
    return landmarks[finger_tip].x + distance*0.8 < landmarks[finger_mcp].x

def dron_right(landmarks,finger_tip,finger_mcp):
    distance = math.sqrt((landmarks[finger_tip].x - landmarks[finger_mcp].x)**2 + (landmarks[finger_tip].y - landmarks[finger_mcp].y)**2)
    return landmarks[finger_tip].x  >landmarks[finger_mcp].x+  distance*0.8

# Por alguna razon las funcionen operan invertidas, por eso estan como estan

# Activamos la camara

cam = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence = 0.5, 
                    min_tracking_confidence = 0.5, max_num_hands = 2) as hands:
    finger_state = [False]*6

    while cam.isOpened():
        ret, frame = cam.read() # Retorna el frame actual y el estado de retorno
        if not ret:
            print("Ignorando frame")
            continue

        frame = cv2.flip(frame,1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Funcion para pasar de BGR a RGB y evitar conflictos con mediapipe
        results = hands.process(rgb_frame)

        # Definimos la funcion para graficar los landmarks en la camara

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                finger_tip = 8
                finger_mcp = 5
                
                if dron_up(hand_landmarks.landmark,finger_tip,finger_mcp) == True:
                    print("Up") # Aca iran las acciones del up

                elif dron_down(hand_landmarks.landmark,finger_tip,finger_mcp) == True:
                    print("Down") # Aca iran las acciones del down

                elif dron_left(hand_landmarks.landmark,finger_tip,finger_mcp) == True:
                    print("Left") # Aca iran las acciones del left

                elif dron_right(hand_landmarks.landmark,finger_tip,finger_mcp) == True:
                    print("Right") # Aca iran las acciones del right

        cv2.imshow("Hand detection", frame)
        if cv2.waitKey(1) & 0xFF == 27: # Terminamos la ejecución con ESC
            break

cam.release()
cv2.destroyAllWindows()