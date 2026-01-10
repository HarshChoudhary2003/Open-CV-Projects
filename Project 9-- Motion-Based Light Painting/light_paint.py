import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Canvas to store trails
trail = None

# Parameters
alpha = 0.2   # trail fading speed
threshold_value = 200  # brightness threshold

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    if trail is None:
        trail = np.zeros_like(frame)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect bright areas (light / finger with torch)
    _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    # Convert mask to color
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Add glow effect
    glow = cv2.GaussianBlur(mask_colored, (21, 21), 0)

    # Accumulate trails
    trail = cv2.addWeighted(trail, 1 - alpha, glow, alpha, 0)

    # Combine with original frame
    output = cv2.add(frame, trail)

    cv2.putText(
        output,
        "Light Painting - Press C to Clear | Q to Quit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("Light Painting", output)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        trail[:] = 0  # clear canvas

cap.release()
cv2.destroyAllWindows()
