import cv2
import mediapipe as mp
import numpy as np
import math
import colorsys

# ---------------- SYSTEM ----------------
W, H = 900, 650
cap = cv2.VideoCapture(0)
cap.set(3, W)
cap.set(4, H)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# ---------------- STATE ----------------
strokes = []
current = []

cam_yaw = 0.0
cam_pitch = 0.0
hue = 0.0

# ---------------- 3D ENGINE ----------------
def rotate_3d(x, y, z, yaw, pitch):
    # Yaw (Y axis)
    xz = x * math.cos(yaw) - z * math.sin(yaw)
    zz = x * math.sin(yaw) + z * math.cos(yaw)

    # Pitch (X axis)
    yz = y * math.cos(pitch) - zz * math.sin(pitch)
    zz2 = y * math.sin(pitch) + zz * math.cos(pitch)

    return xz, yz, zz2

def project(x, y, z):
    z += 6
    f = 400 / z
    px = int(x * f + W // 2)
    py = int(y * f + H // 2)
    return px, py, z

def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

# ---------------- DRAW GRID ----------------
def draw_grid(img):
    for i in range(-5, 6):
        p1 = project(i, 0, -5)
        p2 = project(i, 0, 5)
        cv2.line(img, p1[:2], p2[:2], (50,50,50), 1)

        p3 = project(-5, 0, i)
        p4 = project(5, 0, i)
        cv2.line(img, p3[:2], p4[:2], (50,50,50), 1)

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    draw_mode = False
    rotate_mode = False
    color_mode = False

    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0].landmark
        idx, thumb, mid, wrist = lm[8], lm[4], lm[12], lm[0]

        pinch = dist(idx, thumb)
        fist = dist(idx, wrist)
        two = dist(idx, mid)

        x = (idx.x - 0.5) * 5
        y = (idx.y - 0.5) * -5
        z = max(0.3, 2.2 - fist * 4)

        if pinch < 0.05:
            draw_mode = True
        elif fist < 0.25:
            rotate_mode = True
        elif two < 0.06:
            color_mode = True

        if draw_mode:
            current.append((x, y, z))
        else:
            if current:
                strokes.append(current)
                current = []

        if rotate_mode:
            cam_yaw += (idx.x - 0.5) * 0.05
            cam_pitch += (idx.y - 0.5) * 0.05

        if color_mode:
            hue = (hue + 0.01) % 1.0

    # Combine strokes
    all_strokes = strokes + ([current] if current else [])

    # Flatten points for depth sorting
    segments = []
    for stroke in all_strokes:
        for i in range(1, len(stroke)):
            segments.append((stroke[i-1], stroke[i]))

    # Sort by depth
    segments.sort(key=lambda s: s[0][2] + s[1][2], reverse=True)

    canvas = frame.copy()
    draw_grid(canvas)

    for (x1,y1,z1),(x2,y2,z2) in segments:
        rx1,ry1,rz1 = rotate_3d(x1,y1,z1,cam_yaw,cam_pitch)
        rx2,ry2,rz2 = rotate_3d(x2,y2,z2,cam_yaw,cam_pitch)

        p1 = project(rx1,ry1,rz1)
        p2 = project(rx2,ry2,rz2)

        depth = max(0.1, (p1[2]+p2[2])/2)
        thickness = int(np.clip(8 / depth, 1, 6))
        fade = int(np.clip(255 / depth, 80, 255))

        r,g,b = [int(c*255) for c in colorsys.hsv_to_rgb(hue,1,1)]
        cv2.line(canvas, p1[:2], p2[:2], (b*fade//255, g*fade//255, r*fade//255), thickness)

    # UI
    cv2.rectangle(canvas,(0,0),(W,70),(0,0,0),-1)
    cv2.putText(canvas,
        "LEVEL-C 3D AIR DRAWING | Pinch Draw | Fist Rotate | Two Fingers Color | C Clear | Q Quit",
        (10,45),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

    cv2.imshow("3D Air Drawing – Level C", canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        strokes.clear()
        current.clear()

cap.release()
cv2.destroyAllWindows()
