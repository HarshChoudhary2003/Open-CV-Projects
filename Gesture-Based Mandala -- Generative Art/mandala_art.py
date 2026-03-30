import cv2
import mediapipe as mp
import numpy as np
import math
import random

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Canvas
width, height = 640, 480
canvas = np.zeros((height, width, 3), dtype=np.uint8)

# Webcam
cap = cv2.VideoCapture(0)

prev_x, prev_y = None, None
symmetry_lines = 8  # mandala symmetry

def draw_mandala(x1, y1, x2, y2, color):
    cx, cy = width // 2, height // 2

    for i in range(symmetry_lines):
        angle = 2 * math.pi * i / symmetry_lines

        rx1 = int(math.cos(angle) * (x1 - cx) - math.sin(angle) * (y1 - cy) + cx)
        ry1 = int(math.sin(angle) * (x1 - cx) + math.cos(angle) * (y1 - cy) + cy)

        rx2 = int(math.cos(angle) * (x2 - cx) - math.sin(angle) * (y2 - cy) + cx)
        ry2 = int(math.sin(angle) * (x2 - cx) + math.cos(angle) * (y2 - cy) + cy)

        cv2.line(canvas, (rx1, ry1), (rx2, ry2), color, 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (width, height))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Index finger tip
            x = int(hand_landmarks.landmark[8].x * width)
            y = int(hand_landmarks.landmark[8].y * height)

            if prev_x is not None:
                color = (
                    random.randint(100, 255),
                    random.randint(100, 255),
                    random.randint(100, 255)
                )
                draw_mandala(prev_x, prev_y, x, y, color)

            prev_x, prev_y = x, y
    else:
        prev_x, prev_y = None, None

    output = cv2.add(frame, canvas)

    cv2.putText(
        output,
        "Mandala Art | Move Finger | C: Clear | Q: Quit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("Mandala Generative Art", output)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        canvas[:] = 0

cap.release()
cv2.destroyAllWindows()
