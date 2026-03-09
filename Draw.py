import cv2
import mediapipe as mp
import numpy as np

# Initialize mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Start camera
cap = cv2.VideoCapture(0)

canvas = None
prev_x, prev_y = 0, 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    if canvas is None:
        canvas = np.zeros_like(img)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            h, w, c = img.shape

            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)

            cv2.circle(img, (x,y), 10, (255,0,255), -1)

            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x, y

            cv2.line(canvas, (prev_x, prev_y), (x, y), (0,255,0), 5)

            prev_x, prev_y = x, y

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    img = cv2.add(img, canvas)

    cv2.imshow("Virtual Drawing Board", img)

    key = cv2.waitKey(1) & 0xFF

    # Press C to clear drawing
    if key == ord('c'):
        canvas = np.zeros_like(img)
        prev_x, prev_y = 0, 0

    # Press Q to quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()