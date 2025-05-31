import cv2
import mediapipe as mp
import numpy as np
import math
import os
from collections import deque

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

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

def calculate_thumb_curl(landmarks):
    cmc = np.array([landmarks[1].x, landmarks[1].y, landmarks[1].z])
    mcp = np.array([landmarks[2].x, landmarks[2].y, landmarks[2].z])
    ip  = np.array([landmarks[3].x, landmarks[3].y, landmarks[3].z])
    tip = np.array([landmarks[4].x, landmarks[4].y, landmarks[4].z])
    angle1 = angle_between(mcp - cmc, ip - mcp)
    angle2 = angle_between(ip - mcp, tip - ip)
    hand_size = np.linalg.norm(np.array([landmarks[0].x - landmarks[9].x, landmarks[0].y - landmarks[9].y]))
    thumb_dist = np.linalg.norm(tip - np.array([landmarks[5].x, landmarks[5].y, landmarks[5].z]))
    return (angle1 > 15 and angle2 > 15) or (thumb_dist / hand_size < 0.5)

def is_fist(landmarks):
    curls = [calculate_finger_curl(landmarks, FINGERS[f]) for f in FINGERS]
    thumb = calculate_thumb_curl(landmarks)
    return sum(curls) >= 2 and thumb

class FistClosureDetector:
    def __init__(self, video_path):
        self.video_path = video_path
        self.fist_count = 0
        self.debug = True   # Set to True for print statements, False for silent

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error opening video.")
            return 0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 1e-2:
            fps = 30.0

        stable_buffer = deque(maxlen=2)
        stable_state = None
        fist_count = 0
        frame_idx = 0

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        ) as hands:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)
                current_state = "open"

                if result.multi_hand_landmarks:
                    for hand in result.multi_hand_landmarks:
                        if is_fist(hand.landmark):
                            current_state = "closed"
                        mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                stable_buffer.append(current_state)
                if stable_buffer.count("closed") >= 1:
                    if stable_state != "closed":
                        fist_count += 1
                        if self.debug:
                            print(f"Frame {frame_idx}: Fist Closed")
                    stable_state = "closed"
                elif stable_buffer.count("open") >= 1:
                    stable_state = "open"

        cap.release()
        if self.debug:
            print(f"Final fist closure count: {fist_count}")
        return fist_count

def count_fist_openClose(video_path):
    detector = FistClosureDetector(video_path)
    fist_count = detector.process_video()
    return [str(fist_count)]


