import cv2
import mediapipe as mp
from mediapipe.python.solutions import hands, drawing_utils

video = cv2.VideoCapture(0)

mp_hands = hands.Hands(max_num_hands=1)
mp_drawing = drawing_utils

while True:
    _, frame = video.read()
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(img_rgb)
    hands_points = results.multi_hand_landmarks
    h, w, _ = frame.shape
    pontos = []
    
    if hands_points:
        for points in hands_points:
            
            mp_drawing.draw_landmarks(frame, points, hands.HAND_CONNECTIONS)
            for id, cord in enumerate(points.landmark):
                cx, cy = int(cord.x * w), int(cord.y * h)
                cv2.putText(frame, str(id), (cx, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                pontos.append((cx, cy))
                
            top_point_fingers = [8, 12, 16, 20]
            qt_fingers = 0
            if pontos:
                if pontos[4][0] < pontos[0][0]:                    
                    if pontos[4][0] < pontos[3][0]:
                        qt_fingers += 1
                else:
                    if pontos[4][0] > pontos[3][0]:
                        qt_fingers += 1
                        
                for x in top_point_fingers:
                    if pontos[x][1] < pontos[x -2][1]:
                        qt_fingers += 1
                        
            cv2.rectangle(frame, (80, 10), (200, 110), (255, 0, 0), -1)
            cv2.putText(frame, str(qt_fingers), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 5)
            
    cv2.imshow('imagem', frame)
    key = cv2.waitKey(1) & 0xFF
    
    if cv2.getWindowProperty('imagem', cv2.WND_PROP_VISIBLE) < 1:
        break
    
    if key == 27:  # Tecla 'ESC'
        break

video.release()
cv2.destroyAllWindows()