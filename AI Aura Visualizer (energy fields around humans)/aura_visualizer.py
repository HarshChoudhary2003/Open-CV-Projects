import cv2
import numpy as np

cap = cv2.VideoCapture(0)
prev_gray = None
aura = np.zeros((480, 640, 3), dtype=np.uint8)

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
    intensity = int(min(255, motion_energy * 0.01))

    # Create aura color (blue → purple → pink)
    aura_color = (intensity, 50, 255 - intensity)

    # Expand aura
    aura_mask = cv2.dilate(motion, np.ones((25, 25), np.uint8), iterations=1)

    colored_aura = np.zeros_like(frame)
    colored_aura[aura_mask > 0] = aura_color

    aura = cv2.addWeighted(aura, 0.85, colored_aura, 0.15, 0)

    output = cv2.add(frame, aura)

    cv2.putText(
        output,
        "AI Aura Visualizer | Energy Field | Q to Quit",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.imshow("AI Aura Visualizer", output)

    prev_gray = gray

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
