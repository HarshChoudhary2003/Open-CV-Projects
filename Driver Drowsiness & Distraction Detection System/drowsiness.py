import cv2
import time

# Load cascades
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    "haarcascade_eye.xml"
)

cap = cv2.VideoCapture(0)

# Drowsiness parameters
eye_closed_frames = 0
DROWSY_THRESHOLD = 30  # frames
alert_on = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        face_gray = gray[y:y+h, x:x+w]
        face_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(
            face_gray, scaleFactor=1.1, minNeighbors=5
        )

        if len(eyes) == 0:
            eye_closed_frames += 1
        else:
            eye_closed_frames = 0
            alert_on = False

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(
                    face_color,
                    (ex, ey),
                    (ex+ew, ey+eh),
                    (0, 255, 0),
                    2
                )

        if eye_closed_frames > DROWSY_THRESHOLD:
            alert_on = True
            cv2.putText(
                frame,
                "DROWSINESS ALERT!",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3
            )

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
