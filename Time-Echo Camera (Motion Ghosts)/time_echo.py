import cv2
import numpy as np
from collections import deque

cap = cv2.VideoCapture(0)

# Store past frames
buffer_size = 25
frame_buffer = deque(maxlen=buffer_size)

alpha_values = np.linspace(0.05, 0.9, buffer_size)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))

    frame_buffer.append(frame.copy())

    # Start with black canvas
    echo_frame = np.zeros_like(frame)

    # Blend old frames (ghost effect)
    for i, past in enumerate(frame_buffer):
        alpha = alpha_values[i]
        echo_frame = cv2.addWeighted(echo_frame, 1.0, past, alpha, 0)

    output = cv2.addWeighted(frame, 0.6, echo_frame, 0.4, 0)

    cv2.putText(
        output,
        "Time-Echo Camera | Motion Ghosts | Q to Quit",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.imshow("Time Echo Camera", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
