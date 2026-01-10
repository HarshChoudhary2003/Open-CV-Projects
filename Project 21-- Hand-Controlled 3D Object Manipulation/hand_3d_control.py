import cv2
import mediapipe as mp
import numpy as np
import math
import colorsys
import random

# ------------------ Setup ------------------
W, H = 640, 480
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# ------------------ Shapes ------------------
cube = np.array([
    [-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
    [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]
])

pyramid = np.array([
    [-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[0,0,1.5]
])

def sphere_points(n=40):
    pts=[]
    for i in range(n):
        th=random.uniform(0,2*math.pi)
        ph=random.uniform(0,math.pi)
        pts.append([
            math.sin(ph)*math.cos(th),
            math.sin(ph)*math.sin(th),
            math.cos(ph)
        ])
    return np.array(pts)

shapes = [cube, pyramid, sphere_points()]
shape_idx = 0
current_shape = shapes[0]

edges_cube = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7)
]

# ------------------ State ------------------
rx, ry = 0.0, 0.0
vx, vy = 0.0, 0.0
scale = 170
hue = 0.0
trail = np.zeros((H,W,3),dtype=np.uint8)
particles = []
prev_speed = 0

# ------------------ Helpers ------------------
def rotate(p, ax, ay):
    Rx = np.array([[1,0,0],[0,math.cos(ax),-math.sin(ax)],[0,math.sin(ax),math.cos(ax)]])
    Ry = np.array([[math.cos(ay),0,math.sin(ay)],[0,1,0],[-math.sin(ay),0,math.cos(ay)]])
    return p @ Rx.T @ Ry.T

def project(p):
    out=[]
    for x,y,z in p:
        z+=5
        f=scale/z
        out.append((int(x*f+W//2),int(y*f+H//2)))
    return out

def explode(x,y,color):
    for _ in range(40):
        particles.append([
            x,y,
            random.uniform(-4,4),
            random.uniform(-4,4),
            color,
            random.randint(20,40)
        ])

# ------------------ Loop ------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)
    frame = cv2.resize(frame,(W,H))
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    res = hands.process(rgb)

    speed = 0
    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0].landmark
        cx = lm[8].x - 0.5
        cy = lm[8].y - 0.5
        vx += cy * 0.06
        vy += cx * 0.06
        speed = abs(vx)+abs(vy)

    # Inertia
    rx += vx
    ry += vy
    vx *= 0.90
    vy *= 0.90

    # Color
    hue = (hue + 0.004 + speed*0.02) % 1
    r,g,b = [int(c*255) for c in colorsys.hsv_to_rgb(hue,1,1)]

    # Shape morph trigger
    if speed > 0.35 and prev_speed <= 0.35:
        shape_idx = (shape_idx + 1) % len(shapes)
        current_shape = shapes[shape_idx]
        explode(W//2,H//2,(r,g,b))

    prev_speed = speed

    rot = rotate(current_shape,rx,ry)
    proj = project(rot)

    trail = cv2.addWeighted(trail,0.85,np.zeros_like(trail),0.15,0)

    if current_shape.shape[0]==8:
        for e in edges_cube:
            cv2.line(trail,proj[e[0]],proj[e[1]],(b,g,r),3)
    else:
        for p in proj:
            cv2.circle(trail,p,3,(b,g,r),-1)

    # Particles
    for p in particles[:]:
        p[0]+=p[2]
        p[1]+=p[3]
        p[5]-=1
        cv2.circle(trail,(int(p[0]),int(p[1])),2,p[4],-1)
        if p[5]<=0:
            particles.remove(p)

    final = cv2.add(frame,trail)

    cv2.putText(final,"Neon XR Generative 3D ART",
                (10,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)
    cv2.putText(final,"Fast motion: Explode | Auto Shape Morph | Q Quit",
                (10,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,200),2)

    cv2.imshow("Neon XR Advanced",final)

    if cv2.waitKey(1)&0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
