import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def fingers_up(hand):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    fingers.append(hand.landmark[tips[0]].x < hand.landmark[tips[0]-1].x)

    # Other fingers
    for i in range(1, 5):
        fingers.append(hand.landmark[tips[i]].y < hand.landmark[tips[i]-2].y)

    return fingers

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    sign_text = "None"

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        f = fingers_up(hand)

        # Gesture logic
        if f == [False, False, False, False, False]:
            sign_text = "A"
        elif f == [False, True, True, True, True]:
            sign_text = "B"
        elif f == [False, True, False, False, False]:
            sign_text = "POINT"
        elif f == [True, False, False, False, False]:
            sign_text = "THUMBS UP"
        elif f == [True, True, False, False, False]:
            sign_text = "OK"

    cv2.rectangle(frame, (0, 0), (300, 60), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Sign: {sign_text}",
        (10, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Sign Language Translator", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
