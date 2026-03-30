import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load models
face_net = cv2.dnn.readNet(
    "face_detector/deploy.prototxt",
    "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
)

mask_model = load_model("mask_detector.model")

# Labels & colors
labels = ["Mask", "No Mask"]
colors = [(0,255,0), (0,0,255)]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )

    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue

            face = cv2.resize(face, (224, 224))
            face = face / 255.0
            face = np.reshape(face, (1, 224, 224, 3))

            (mask, no_mask) = mask_model.predict(face)[0]
            idx = 0 if mask > no_mask else 1

            label = f"{labels[idx]}: {max(mask, no_mask)*100:.2f}%"
            color = colors[idx]

            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.putText(
                frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
            )

    cv2.imshow("Face Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
