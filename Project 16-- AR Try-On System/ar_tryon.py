import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Face Mesh
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Load glasses PNG (transparent)
glasses = cv2.imread("glasses.png", cv2.IMREAD_UNCHANGED)

cap = cv2.VideoCapture(0)

def overlay_png(bg, overlay, x, y, w, h):
    overlay = cv2.resize(overlay, (w, h))
    if x < 0 or y < 0 or x+w > bg.shape[1] or y+h > bg.shape[0]:
        return bg

    overlay_rgb = overlay[:, :, :3]
    alpha = overlay[:, :, 3] / 255.0

    for c in range(3):
        bg[y:y+h, x:x+w, c] = (
            alpha * overlay_rgb[:, :, c] +
            (1 - alpha) * bg[y:y+h, x:x+w, c]
        )
    return bg

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]

        # Eye landmarks (left & right)
        left_eye = face.landmark[33]
        right_eye = face.landmark[263]

        lx, ly = int(left_eye.x * w), int(left_eye.y * h)
        rx, ry = int(right_eye.x * w), int(right_eye.y * h)

        glasses_width = int(1.5 * abs(rx - lx))
        glasses_height = int(glasses_width * 0.4)

        gx = int((lx + rx) / 2 - glasses_width / 2)
        gy = int((ly + ry) / 2 - glasses_height / 2)

        frame = overlay_png(
            frame, glasses, gx, gy, glasses_width, glasses_height
        )

    cv2.putText(
        frame,
        "AR Try-On (Glasses) | Q to Quit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("AR Try-On Pro", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
