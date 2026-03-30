import cv2
import numpy as np

cap = cv2.VideoCapture(0)
prev_gray = None
dream_layer = np.zeros((480, 640, 3), dtype=np.uint8)

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
    _, motion = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    motion = cv2.GaussianBlur(motion, (31, 31), 0)

    motion_energy = np.sum(motion) / 255
    intensity = min(1.0, motion_energy / 8000)

    # Soft blur (dream haze)
    blur = cv2.GaussianBlur(frame, (21, 21), 0)

    # Glow effect
    glow = cv2.addWeighted(frame, 0.6, blur, 0.4, 0)

    # Color shift (dream tone)
    dream = glow.copy()
    dream[:,:,0] = np.clip(dream[:,:,0] + 30 * intensity, 0, 255)
    dream[:,:,2] = np.clip(dream[:,:,2] + 50 * intensity, 0, 255)

    # Accumulate dream layer (memory effect)
    dream_layer = cv2.addWeighted(dream_layer, 0.92, dream, 0.08, 0)

    output = cv2.addWeighted(frame, 0.4, dream_layer, 0.6, 0)

    cv2.putText(
        output,
        "AI Dream Camera | Surreal Mode | Q to Quit",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.imshow("AI Dream Camera", output)

    prev_gray = gray

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
