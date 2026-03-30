import cv2
import mediapipe as mp
import numpy as np

# ------------------ SETUP ------------------
cap = cv2.VideoCapture(0)
width, height = 640, 480
canvas = np.zeros((height, width, 3), dtype=np.uint8)
mode = 1

# Face detection
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# AR filter
filter_img = cv2.imread("glasses.png", cv2.IMREAD_UNCHANGED)

prev_x, prev_y = None, None
draw_color = (255, 255, 255)

def overlay_png(bg, overlay, x, y):
    h, w = overlay.shape[:2]
    if x < 0 or y < 0 or x+w > bg.shape[1] or y+h > bg.shape[0]:
        return bg
    overlay_rgb = overlay[:, :, :3]
    alpha = overlay[:, :, 3] / 255.0
    for c in range(3):
        bg[y:y+h, x:x+w, c] = (
            alpha * overlay_rgb[:, :, c] +
            (1-alpha) * bg[y:y+h, x:x+w, c]
        )
    return bg

# ------------------ MAIN LOOP ------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (width, height))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # ---------- MODE 1: AIR DRAWING ----------
    if mode == 1 and result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            x = int(hand.landmark[8].x * width)
            y = int(hand.landmark[8].y * height)
            if prev_x:
                cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, 4)
            prev_x, prev_y = x, y

    # ---------- MODE 2: AR FACE FILTER ----------
    if mode == 2:
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            fw = w
            fh = int(h * 0.35)
            resized = cv2.resize(filter_img, (fw, fh))
            frame = overlay_png(frame, resized, x, y + int(h * 0.25))

    # ---------- MODE 3: LIGHT PAINT ----------
    if mode == 3:
        bright = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(bright, 200, 255, cv2.THRESH_BINARY)
        glow = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        canvas = cv2.addWeighted(canvas, 0.9, glow, 0.1, 0)

    # ---------- MODE 4: EMOTION COLOR PAINT ----------
    if mode == 4:
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            area = w * h
            if area > 25000:
                draw_color = (0, 255, 255)
                emotion = "HAPPY"
            elif area < 18000:
                draw_color = (255, 0, 0)
                emotion = "CALM"
            else:
                draw_color = (0, 0, 255)
                emotion = "INTENSE"
            cv2.putText(frame, emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, draw_color, 2)

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                x = int(hand.landmark[8].x * width)
                y = int(hand.landmark[8].y * height)
                if prev_x:
                    cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, 4)
                prev_x, prev_y = x, y

    # ---------- UI ----------
    final = cv2.add(frame, canvas)
    cv2.putText(final, f"Mode {mode} | 1 Draw 2 Filter 3 Light 4 Emotion",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)

    cv2.imshow("Creative CV Studio", final)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key in [ord("1"), ord("2"), ord("3"), ord("4")]:
        mode = int(chr(key))
        canvas[:] = 0
        prev_x = prev_y = None
    elif key == ord("c"):
        canvas[:] = 0

cap.release()
cv2.destroyAllWindows()
