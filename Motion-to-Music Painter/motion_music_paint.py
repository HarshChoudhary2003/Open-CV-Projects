import cv2
import numpy as np
import random

cap = cv2.VideoCapture(0)

prev_gray = None
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
energy = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is None:
        prev_gray = gray
        continue

    # Motion detection
    diff = cv2.absdiff(prev_gray, gray)
    _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    motion_amount = np.sum(motion_mask) / 255
    energy = energy * 0.8 + motion_amount * 0.01

    # Map energy to brush size
    brush = int(min(40, max(5, energy)))

    # Random paint strokes based on motion
    if energy > 10:
        for _ in range(10):
            x = random.randint(0, 639)
            y = random.randint(0, 479)
            color = (
                random.randint(100, 255),
                random.randint(100, 255),
                random.randint(100, 255)
            )
            cv2.circle(canvas, (x, y), brush, color, -1)

    # Fade canvas slowly
    canvas = cv2.addWeighted(canvas, 0.96, np.zeros_like(canvas), 0.04, 0)

    output = cv2.add(frame, canvas)

    cv2.putText(
        output,
        f"Motion-to-Music Painter | Energy: {int(energy)} | Q to Quit",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.imshow("Motion Music Painter", output)

    prev_gray = gray

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
