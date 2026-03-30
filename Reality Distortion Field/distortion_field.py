import cv2
import numpy as np

cap = cv2.VideoCapture(0)
prev_gray = None

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
    _, motion = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    motion = cv2.GaussianBlur(motion, (21, 21), 0)

    # Create distortion maps
    h, w = gray.shape
    flow_x = np.zeros((h, w), np.float32)
    flow_y = np.zeros((h, w), np.float32)

    strength = np.clip(np.sum(motion) / 255 / 4000, 0, 5)

    # Radial distortion around motion
    for y in range(0, h, 10):
        for x in range(0, w, 10):
            if motion[y, x] > 0:
                dx = (x - w // 2) * 0.02 * strength
                dy = (y - h // 2) * 0.02 * strength
                flow_x[y:y+10, x:x+10] = dx
                flow_y[y:y+10, x:x+10] = dy

    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (map_x + flow_x).astype(np.float32)
    map_y = (map_y + flow_y).astype(np.float32)

    distorted = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)

    output = cv2.addWeighted(frame, 0.6, distorted, 0.4, 0)

    cv2.putText(
        output,
        "Reality Distortion Field | Move to Bend Space | Q to Quit",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.imshow("Reality Distortion Field", output)

    prev_gray = gray

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
