import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Selfie Segmentation
mp_selfie = mp.solutions.selfie_segmentation
segment = mp_selfie.SelfieSegmentation(model_selection=1)

cap = cv2.VideoCapture(0)

# Load background image
bg_img = cv2.imread("bg.jpg")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    bg = cv2.resize(bg_img, (w, h))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = segment.process(rgb)

    mask = result.segmentation_mask
    condition = mask > 0.5

    output = np.where(condition[..., None], frame, bg)

    cv2.putText(
        output,
        "Background Replacement | Q to Quit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("Virtual Background", output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
