import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("digit_model.h5")

# Canvas
canvas = np.zeros((400, 400), dtype=np.uint8)
drawing = False

def draw(event, x, y, flags, param):
    global drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.circle(canvas, (x, y), 10, 255, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

cv2.namedWindow("Digit Draw")
cv2.setMouseCallback("Digit Draw", draw)

while True:
    img = canvas.copy()

    # Preprocess for prediction
    small = cv2.resize(canvas, (28, 28))
    small = cv2.bitwise_not(small)
    small = small / 255.0
    small = small.reshape(1, 28, 28, 1)

    prediction = model.predict(small, verbose=0)
    digit = np.argmax(prediction)

    cv2.putText(
        img,
        f"Prediction: {digit}",
        (10, 380),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255),
        2
    )

    cv2.imshow("Digit Draw", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("c"):
        canvas[:] = 0
    elif key == ord("q"):
        break

cv2.destroyAllWindows()
