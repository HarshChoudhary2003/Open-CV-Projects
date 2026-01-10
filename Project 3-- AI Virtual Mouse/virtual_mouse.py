import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Safety off to prevent mouse corner crash
pyautogui.FAILSAFE = False

# Screen size
screen_width, screen_height = pyautogui.size()

# MediaPipe Hands setup (CORRECT)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Mirror image
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert to RGB
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
            index_tip = hand_landmarks.landmark[8]
            # Thumb tip
            thumb_tip = hand_landmarks.landmark[4]

            ix, iy = int(index_tip.x * w), int(index_tip.y * h)
            tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)

            # Map to screen size
            screen_x = np.interp(ix, [0, w], [0, screen_width])
            screen_y = np.interp(iy, [0, h], [0, screen_height])

            # Move mouse
            pyautogui.moveTo(screen_x, screen_y, duration=0.01)

            # Distance between thumb and index
            distance = np.hypot(ix - tx, iy - ty)

            # Click gesture
            if distance < 40:
                pyautogui.click()
                cv2.putText(
                    frame,
                    "Click",
                    (ix, iy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

    cv2.imshow("AI Virtual Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
