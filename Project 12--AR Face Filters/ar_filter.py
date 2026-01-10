import cv2
import numpy as np

# Load face cascade
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# Load PNG filter (must have transparency)
filter_img = cv2.imread("glasses.png", cv2.IMREAD_UNCHANGED)

cap = cv2.VideoCapture(0)

def overlay_png(background, overlay, x, y):
    h, w = overlay.shape[:2]

    if x < 0 or y < 0 or x + w > background.shape[1] or y + h > background.shape[0]:
        return background

    overlay_rgb = overlay[:, :, :3]
    overlay_alpha = overlay[:, :, 3] / 255.0

    for c in range(3):
        background[y:y+h, x:x+w, c] = (
            overlay_alpha * overlay_rgb[:, :, c] +
            (1 - overlay_alpha) * background[y:y+h, x:x+w, c]
        )

    return background

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Resize filter to face width
        filter_width = w
        filter_height = int(h * 0.35)

        resized_filter = cv2.resize(filter_img, (filter_width, filter_height))

        # Position filter (eye area)
        fx = x
        fy = y + int(h * 0.25)

        frame = overlay_png(frame, resized_filter, fx, fy)

    cv2.imshow("AR Face Filter", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
