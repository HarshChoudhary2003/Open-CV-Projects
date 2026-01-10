import cv2
import mediapipe as mp
import numpy as np
import pygame

# ---------- AUDIO ----------
pygame.mixer.init()
pygame.mixer.music.load("sample.mp3")  # put any mp3 in same folder
pygame.mixer.music.play(-1)

# ---------- MEDIAPIPE ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ---------- VIDEO ----------
cap = cv2.VideoCapture(0)
width, height = 640, 480

intensity = 50

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (width, height))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            y = int(hand.landmark[8].y * height)
            intensity = int(np.interp(y, [0, height], [200, 20]))

    # ---------- VISUALIZER ----------
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(0, width, 20):
        bar_height = np.random.randint(20, intensity)
        cv2.rectangle(
            canvas,
            (i, height),
            (i + 10, height - bar_height),
            (0, 255, 255),
            -1
        )

    output = cv2.addWeighted(frame, 0.3, canvas, 0.7, 0)

    cv2.putText(
        output,
        "Gesture Music Visualizer | Q to Quit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("Music Visualizer", output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
pygame.mixer.music.stop()
cv2.destroyAllWindows()
