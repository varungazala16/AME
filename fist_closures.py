import cv2
import mediapipe as mp
import numpy as np
import math

FINGERS = {
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20]
}

def angle_between(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    v1 /= np.linalg.norm(v1) if np.linalg.norm(v1) != 0 else 1
    v2 /= np.linalg.norm(v2) if np.linalg.norm(v2) != 0 else 1
    return math.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))

def calculate_finger_curl(landmarks, finger):
    mcp, pip, dip, tip = [np.array([landmarks[i].x, landmarks[i].y, landmarks[i].z]) for i in finger]
    angle1 = angle_between(pip - mcp, dip - pip)
    angle2 = angle_between(dip - pip, tip - dip)
    wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
    return (angle1 > 20 and angle2 > 20) or (np.linalg.norm(tip - wrist) < np.linalg.norm(mcp - wrist))

def is_fist_no_thumb(landmarks):
    curls = [calculate_finger_curl(landmarks, FINGERS[f]) for f in FINGERS]
    return sum(curls) >= 2

def count_fist_closures(video_path):
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    fist_closed_prev = False
    fist_closure_count = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)
            fist_closed = False

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    if is_fist_no_thumb(hand_landmarks.landmark):
                        fist_closed = True
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    break

            # Print the detected state (for debugging)
            # print("Fist closed:" if fist_closed else "Hand open")

            # Rising edge detection (open to closed)
            if not fist_closed_prev and fist_closed:
                fist_closure_count += 1

            fist_closed_prev = fist_closed

            cv2.imshow("Hand Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    return [str(fist_closure_count)]


