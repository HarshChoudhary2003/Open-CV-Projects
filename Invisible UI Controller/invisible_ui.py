import cv2
import mediapipe as mp
import numpy as np
import math
import time

# -----------------------------
# MediaPipe setup
# -----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

# -----------------------------
# Button class
# -----------------------------
class Button:
    def __init__(self, x, y, w, h, text):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.text = text
        self.clicked_time = 0

    def draw(self, img, is_hover=False):
        color = (200, 200, 200)
        if is_hover:
            color = (0, 255, 0)

        cv2.rectangle(
            img,
            (self.x, self.y),
            (self.x + self.w, self.y + self.h),
            color,
            -1
        )

        cv2.putText(
            img,
            self.text,
            (self.x + 20, self.y + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2
        )

    def is_inside(self, x, y):
        return self.x < x < self.x + self.w and self.y < y < self.y + self.h


# -----------------------------
# Create buttons
# -----------------------------
buttons = [
    Button(50, 50, 180, 60, "PLAY"),
    Button(270, 50, 180, 60, "PAUSE"),
    Button(50, 150, 180, 60, "VOL +"),
    Button(270, 150, 180, 60, "VOL -"),
    Button(50, 250, 180, 60, "NEXT"),
    Button(270, 250, 180, 60, "PREV")
]

last_click_time = 0
click_delay = 0.8  # seconds

# -----------------------------
# Main loop
# -----------------------------
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    finger_x, finger_y = None, None
    pinch = False

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Index finger tip (8)
            ix = int(hand_landmarks.landmark[8].x * w)
            iy = int(hand_landmarks.landmark[8].y * h)

            # Thumb tip (4)
            tx = int(hand_landmarks.landmark[4].x * w)
            ty = int(hand_landmarks.landmark[4].y * h)

            finger_x, finger_y = ix, iy

            # Draw pointer
            cv2.circle(frame, (ix, iy), 10, (255, 0, 255), -1)

            # Distance between thumb and index
            distance = math.hypot(ix - tx, iy - ty)

            if distance < 40:
                pinch = True

    # Draw buttons and detect interaction
    for button in buttons:
        hover = False
        if finger_x and button.is_inside(finger_x, finger_y):
            hover = True

            if pinch and time.time() - last_click_time > click_delay:
                button.clicked_time = time.time()
                last_click_time = time.time()
                print(f"{button.text} clicked")

        button.draw(frame, hover)

    # Display title
    cv2.putText(
        frame,
        "Invisible UI Controller",
        (50, 420),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.imshow("Invisible UI Controller", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
