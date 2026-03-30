import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

canvas = np.zeros((480, 640, 3), dtype=np.uint8)
prev_face_center = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_small = cv2.resize(face, (w//10, h//10))
        face_paint = cv2.resize(face_small, (w, h), interpolation=cv2.INTER_LINEAR)

        # Edge strokes
        edges = cv2.Canny(face_paint, 80, 160)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        painterly = cv2.addWeighted(face_paint, 0.8, edges, 0.2, 0)

        canvas[y:y+h, x:x+w] = painterly

        cx, cy = x + w//2, y + h//2
        if prev_face_center:
            dx = cx - prev_face_center[0]
            dy = cy - prev_face_center[1]
            shift = int(min(20, abs(dx) + abs(dy)))
            canvas = np.roll(canvas, shift, axis=1)

        prev_face_center = (cx, cy)

    canvas = cv2.addWeighted(canvas, 0.94, np.zeros_like(canvas), 0.06, 0)
    output = cv2.addWeighted(frame, 0.4, canvas, 0.6, 0)

    cv2.putText(
        output,
        "Living Portrait Generator | Q to Quit",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.imshow("Living Portrait Generator", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
