import cv2, mediapipe as mp, numpy as np, math, time, threading
import speech_recognition as sr
import pyttsx3

# ===================== SYSTEM INIT =====================
W, H = 1200, 800
cap = cv2.VideoCapture(0)
cap.set(3, W)
cap.set(4, H)

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
hands = mp_hands.Hands(max_num_hands=1)
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

engine = pyttsx3.init()
recognizer = sr.Recognizer()
mic = sr.Microphone()

# ===================== STATE =====================
mode = "IDLE"      # IDLE | ANALYZE | DIAGNOSTICS | PANELS
yaw = 0.0
zoom = 1.0
pulse = 0.0
voice_command = ""

# ===================== VOICE THREAD =====================
def listen():
    global voice_command, mode
    while True:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
        try:
            cmd = recognizer.recognize_google(audio).lower()
            voice_command = cmd

            if "analyze" in cmd:
                mode = "ANALYZE"
                engine.say("Analyzing subject")
            elif "diagnostic" in cmd:
                mode = "DIAGNOSTICS"
                engine.say("Opening diagnostics")
            elif "panel" in cmd:
                mode = "PANELS"
                engine.say("Opening holographic panels")
            elif "reset" in cmd:
                mode = "IDLE"
                engine.say("System reset")

            engine.runAndWait()
        except:
            pass

threading.Thread(target=listen, daemon=True).start()

# ===================== HELPERS =====================
def polar(c, r, a):
    return int(c[0] + r * math.cos(a)), int(c[1] + r * math.sin(a))

def draw_ring(img, c, r, a, color, thick=2):
    for t in np.linspace(0, 2*math.pi, 180):
        x,y = polar(c, r, t + a)
        cv2.circle(img,(x,y),thick,color,-1)

def hud_text(img, text, x, y, s=0.8):
    cv2.putText(img,text,(x,y),
                cv2.FONT_HERSHEY_SIMPLEX,s,(0,255,255),2)

# ===================== MAIN LOOP =====================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)
    hud = frame.copy()
    pulse += 0.04
    yaw += 0.01

    center = (W//2, H//2)

    # ===================== FACE SCAN =====================
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    face_res = face_mesh.process(rgb)

    if mode == "ANALYZE" and face_res.multi_face_landmarks:
        for lm in face_res.multi_face_landmarks:
            for p in lm.landmark[::5]:
                x = int(p.x*W)
                y = int(p.y*H)
                cv2.circle(hud,(x,y),1,(0,255,255),-1)
        hud_text(hud,"FACE LOCKED",50,100)

    # ===================== DIAGNOSTICS =====================
    if mode == "DIAGNOSTICS":
        for i in range(6):
            h = int((math.sin(pulse+i)+1)*100)
            cv2.rectangle(hud,(100+i*60,600),
                          (140+i*60,600-h),(255,150,0),-1)
        hud_text(hud,"SYSTEM LOAD",100,560)

    # ===================== MULTI PANELS =====================
    if mode == "PANELS":
        panels = [
            ((200,200),"ENERGY"),
            ((W-200,200),"TARGETS"),
            ((200,H-200),"NETWORK"),
            ((W-200,H-200),"MAP")
        ]
        for (x,y),label in panels:
            draw_ring(hud,(x,y),60,yaw,(0,200,255))
            hud_text(hud,label,x-50,y+90,0.6)

    # ===================== CORE HUD =====================
    draw_ring(hud,center,int(160*zoom),yaw,(0,255,255))
    draw_ring(hud,center,int(220*zoom),-yaw*0.6,(255,180,0),1)
    draw_ring(hud,center,int(280*zoom),yaw*0.3,(200,200,200),1)

    # ===================== TOP BAR =====================
    cv2.rectangle(hud,(0,0),(W,60),(0,0,0),-1)
    hud_text(hud,"STARK-OS | JARVIS ONLINE",30,40)
    hud_text(hud,f"MODE: {mode}",W-300,40)

    if voice_command:
        hud_text(hud,f"VOICE: {voice_command}",30,80,0.6)

    cv2.imshow("STARK-OS",hud)

    if cv2.waitKey(1)&0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
