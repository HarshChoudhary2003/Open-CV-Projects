import cv2
import mediapipe as mp
import numpy as np
import math
import random

cap = cv2.VideoCapture(0)

mp_face = mp.solutions.face_mesh
face = mp_face.FaceMesh(refine_landmarks=True)

canvas = np.zeros((480, 640, 3), dtype=np.uint8)
prev_center = None
energy = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = face.process(rgb)

    emotion = "NEUTRAL"

    if result.multi_face_landmarks:
        lm = result.multi_face_landmarks[0].landmark

        mouth_open = abs(lm[13].y - lm[14].y)
        brow = abs(lm[65].y - lm[158].y)

        if mouth_open > 0.03:
            emotion = "HAPPY"
        elif brow > 0.03:
            emotion = "INTENSE"

        cx = int(lm[1].x * w)
        cy = int(lm[1].y * h)

        if prev_center:
            dx = cx - prev_center[0]
            dy = cy - prev_center[1]
            energy = min(100, abs(dx) + abs(dy))
        prev_center = (cx, cy)

        if emotion == "HAPPY":
            for _ in range(5):
                cv2.circle(
                    canvas,
                    (cx + random.randint(-50,50), cy + random.randint(-50,50)),
                    random.randint(5,15),
                    (0,255,255),
                    -1
                )

        elif emotion == "INTENSE":
            for i in range(8):
                angle = i * math.pi / 4
                x = int(cx + math.cos(angle) * energy)
                y = int(cy + math.sin(angle) * energy)
                cv2.line(canvas, (cx, cy), (x, y), (0,0,255), 2)

        else:
            cv2.circle(canvas, (cx, cy), 10, (255,100,100), 1)

    canvas = cv2.addWeighted(canvas, 0.95, np.zeros_like(canvas), 0.05, 0)
    output = cv2.add(frame, canvas)

    cv2.putText(
        output,
        f"Emotion Mirror: {emotion}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        2
    )

    cv2.imshow("AI Emotion Mirror", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
