import cv2
import numpy as np
import random

cap = cv2.VideoCapture(0)

prev_gray = None
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

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
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:
        if cv2.contourArea(cnt) > 400:
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2

            color = (
                random.randint(100, 255),
                random.randint(100, 255),
                random.randint(100, 255)
            )

            radius = random.randint(5, 20)
            cv2.circle(canvas, (cx, cy), radius, color, -1)

    # Fade effect
    canvas = cv2.addWeighted(canvas, 0.95, np.zeros_like(canvas), 0.05, 0)

    output = cv2.add(frame, canvas)

    cv2.putText(
        output,
        "Motion-Driven Generative Art | C Clear | Q Quit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("Generative Art Engine", output)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        canvas[:] = 0

    prev_gray = gray

cap.release()
cv2.destroyAllWindows()
