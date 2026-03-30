import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os

face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

model = cv2.face.LBPHFaceRecognizer_create()
model.read("trainer.yml")

label_map = np.load("labels.npy", allow_pickle=True).item()

if not os.path.exists("attendance.csv"):
    pd.DataFrame(columns=["Name", "Date", "Time"]).to_csv(
        "attendance.csv", index=False
    )

cap = cv2.VideoCapture(0)
marked = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        label, confidence = model.predict(face)

        name = label_map[label]

        if confidence < 80:
            if name not in marked:
                now = datetime.now()
                df = pd.read_csv("attendance.csv")
                df.loc[len(df)] = [
                    name,
                    now.strftime("%Y-%m-%d"),
                    now.strftime("%H:%M:%S")
                ]
                df.to_csv("attendance.csv", index=False)
                marked.add(name)

            text = f"{name}"
            color = (0, 255, 0)
        else:
            text = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            frame, text, (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
        )

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
