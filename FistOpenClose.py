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

    def process_video(self, start_time_sec=0.0, end_time_sec=None):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error opening video.")
            return 0

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 1e-2:
            fps = 30.0

        # Seek to start time (in milliseconds)
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time_sec * 1000)

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

                if timestamp < start_time_sec:
                    continue

                if end_time_sec is not None and timestamp > end_time_sec:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                current_state = "open"
                if result.multi_hand_landmarks:
                    for hand in result.multi_hand_landmarks:
                        if is_fist(hand.landmark):
                            current_state = "closed"
                            break  # Only one closed hand is enough to count

                stable_buffer.append(current_state)

                if stable_buffer.count("closed") >= 1:
                    if stable_state != "closed":
                        fist_count += 1
                        if self.debug:
                            print(f"Frame {frame_idx} ({timestamp:.2f}s): Fist Closed")
                    stable_state = "closed"
                elif stable_buffer.count("open") >= 1:
                    stable_state = "open"

        cap.release()

        if self.debug:
            print(f"Final fist closure count: {fist_count}")
        return fist_count

def count_fist_openClose(video_path):
    detectorL = FistClosureDetector(video_path)
    fist_countL = detectorL.process_video(4,14)
    detectorR = FistClosureDetector(video_path)
    fist_countR = detectorR.process_video(25,35)
    return [str(fist_countL), str(fist_countR)]


