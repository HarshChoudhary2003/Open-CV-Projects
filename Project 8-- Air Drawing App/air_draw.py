import cv2
import mediapipe as mp
import numpy as np

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
# Canvas & webcam
# -----------------------------
width, height = 640, 480
canvas = np.zeros((height, width, 3), dtype=np.uint8)
cap = cv2.VideoCapture(0)

prev_x, prev_y = None, None
draw_color = (255, 255, 255)
brush_thickness = 4
mode = "draw"  # draw / erase

# -----------------------------
# Toolbar buttons
# -----------------------------
# (x1, y1, x2, y2, label, color)
buttons = [
    (0, 0, 120, 60, "RED", (0, 0, 255)),
    (120, 0, 240, 60, "GREEN", (0, 255, 0)),
    (240, 0, 360, 60, "BLUE", (255, 0, 0)),
    (360, 0, 480, 60, "ERASE", (50, 50, 50)),
    (480, 0, 640, 60, "CLEAR", (100, 100, 100)),
]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (width, height))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # Draw toolbar
    for (x1, y1, x2, y2, label, color) in buttons:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.putText(
            frame, label,
            (x1 + 10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

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

            cv2.circle(frame, (x, y), 8, (0, 0, 0), -1)

            # Toolbar interaction
            if y < 60:
                for (x1, y1, x2, y2, label, color) in buttons:
                    if x1 < x < x2:
                        if label == "ERASE":
                            mode = "erase"
                        elif label == "CLEAR":
                            canvas[:] = 0
                        else:
                            mode = "draw"
                            draw_color = color
                        prev_x, prev_y = None, None
            else:
                if prev_x is None:
                    prev_x, prev_y = x, y

                if mode == "draw":
                    cv2.line(
                        canvas,
                        (prev_x, prev_y),
                        (x, y),
                        draw_color,
                        brush_thickness
                    )
                else:  # erase mode
                    cv2.line(
                        canvas,
                        (prev_x, prev_y),
                        (x, y),
                        (0, 0, 0),
                        30
                    )

                prev_x, prev_y = x, y
    else:
        prev_x, prev_y = None, None

    # Merge canvas with frame
    final = cv2.add(frame, canvas)

    cv2.putText(
        final,
        "AI Whiteboard",
        (10, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow("AI Whiteboard", final)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
