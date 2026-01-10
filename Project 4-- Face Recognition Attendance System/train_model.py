import cv2
import os
import numpy as np

data_path = "dataset"
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

faces = []
labels = []
label_map = {}
current_label = 0

for person_name in os.listdir(data_path):
    person_path = os.path.join(data_path, person_name)

    if not os.path.isdir(person_path):
        continue

    label_map[current_label] = person_name

    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        detected_faces = face_cascade.detectMultiScale(img, 1.3, 5)

        for (x, y, w, h) in detected_faces:
            face_roi = img[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))  # IMPORTANT
            faces.append(face_roi)
            labels.append(current_label)

    current_label += 1

# Train model
model = cv2.face.LBPHFaceRecognizer_create()
model.train(faces, np.array(labels))
model.save("trainer.yml")

np.save("labels.npy", label_map)

print("✅ Model trained successfully")
