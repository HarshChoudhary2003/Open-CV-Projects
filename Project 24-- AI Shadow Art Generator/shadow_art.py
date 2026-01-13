import cv2
import numpy as np

cap = cv2.VideoCapture(0)

bg_sub = cv2.createBackgroundSubtractorMOG2(
    history=300,
    varThreshold=50,
    detectShadows=True
)

canvas = np.zeros((480, 640, 3), dtype=np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))

    mask = bg_sub.apply(frame)

    # Clean mask
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours (silhouette)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:
        if cv2.contourArea(cnt) > 2000:
            hull = cv2.convexHull(cnt)
            cv2.drawContours(canvas, [hull], -1, (0, 0, 0), -1)

    # Fade canvas slowly (ink effect)
    canvas = cv2.addWeighted(canvas, 0.97, np.zeros_like(canvas), 0.03, 0)

    output = cv2.add(frame, canvas)

    cv2.putText(
        output,
        "AI Shadow Art | Move Slowly for Best Effect | Q to Quit",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.imshow("Shadow Art Generator", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
