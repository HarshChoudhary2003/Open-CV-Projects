import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)
angle = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 640))

    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2

    # Circular mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), min(cx, cy), 255, -1)

    # Rotate slice
    angle += 0.5
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1)
    rotated = cv2.warpAffine(frame, M, (w, h))
    rotated = cv2.bitwise_and(rotated, rotated, mask=mask)

    # Mirror symmetry
    left = rotated[:, :cx]
    right = cv2.flip(left, 1)
    top = np.hstack([left, right])
    bottom = cv2.flip(top, 0)

    kaleido = np.vstack([top, bottom])

    # Overlay with original
    output = cv2.addWeighted(kaleido, 0.85, frame, 0.15, 0)

    cv2.putText(
        output,
        "Interactive Kaleidoscope | Move to Change Patterns | Q to Quit",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.imshow("Kaleidoscope World", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
