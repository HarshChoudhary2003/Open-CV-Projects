import cv2
import numpy as np

# Load face detector
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

# Canvas
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Mouse drawing
drawing = False
px, py = None, None
color = (255, 255, 255)

def draw(event, x, y, flags, param):
    global drawing, px, py
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        px, py = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.line(canvas, (px, py), (x, y), color, 4)
        px, py = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        px, py = None, None

cv2.namedWindow("Emotion Paint")
cv2.setMouseCallback("Emotion Paint", draw)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Simple emotion proxy (demo-friendly)
    for (x, y, w, h) in faces:
        face_area = w * h

        if face_area > 25000:
            color = (0, 255, 255)   # Happy → Yellow
            emotion = "HAPPY"
        elif face_area < 18000:
            color = (255, 0, 0)     # Neutral → Blue
            emotion = "NEUTRAL"
        else:
            color = (0, 0, 255)     # Intense → Red
            emotion = "INTENSE"

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            frame, emotion, (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
        )

    output = cv2.add(frame, canvas)

    cv2.putText(
        output,
        "Emotion-Based Color Painting | C: Clear | Q: Quit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("Emotion Paint", output)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        canvas[:] = 0

cap.release()
cv2.destroyAllWindows()
