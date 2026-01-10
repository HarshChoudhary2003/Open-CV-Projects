import cv2
import mediapipe as mp
import numpy as np
import time

# MediaPipe Face Mesh
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

attentive_time = 0
distracted_time = 0
last_state_time = time.time()

# Eye landmarks (left & right iris centers)
LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]

def eye_center(landmarks, ids, w, h):
    x = int((landmarks[ids[0]].x + landmarks[ids[1]].x) / 2 * w)
    y = int((landmarks[ids[0]].y + landmarks[ids[1]].y) / 2 * h)
    return x, y

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    state = "NO FACE"

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]
        now = time.time()
        dt = now - last_state_time
        last_state_time = now

        lx, ly = eye_center(face.landmark, LEFT_EYE, w, h)
        rx, ry = eye_center(face.landmark, RIGHT_EYE, w, h)

        cx = (lx + rx) // 2

        # Simple attention logic
        if w * 0.4 < cx < w * 0.6:
            state = "ATTENTIVE"
            attentive_time += dt
            color = (0, 255, 0)
        else:
            state = "DISTRACTED"
            distracted_time += dt
            color = (0, 0, 255)

        cv2.circle(frame, (lx, ly), 5, color, -1)
        cv2.circle(frame, (rx, ry), 5, color, -1)

    total = attentive_time + distracted_time
    score = int((attentive_time / total) * 100) if total > 0 else 0

    cv2.putText(
        frame, f"State: {state}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
        (255, 255, 255), 2
    )
    cv2.putText(
        frame, f"Attention Score: {score}%",
        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
        (255, 255, 0), 2
    )

    cv2.imshow("AI Attention Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
