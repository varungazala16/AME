import cv2
import mediapipe as mp
import numpy as np
from collections import deque

def count_flip_flops(video_path, hand, start_time_sec=0.0, end_time_sec=None):

    HAND_TO_TRACK = hand.lower()
    if HAND_TO_TRACK not in ['left', 'right']:
        raise ValueError("hand must be 'left' or 'right'")

    FLIP_THRESHOLD = -30
    FLOP_THRESHOLD = 30
    DEBOUNCE_FRAMES = 3

    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        model_complexity=1
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-2:
        fps = 30.0

    # Seek to start time
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time_sec * 1000)

    flip_flop_count = 0
    current_state = "neutral"
    state_history = deque(maxlen=DEBOUNCE_FRAMES)
    frame_idx = 0

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

        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness_classification = results.multi_handedness[i].classification[0]
                handedness_label = handedness_classification.label.lower()
                if handedness_label != HAND_TO_TRACK:
                    continue

                # Palm orientation angle calculation
                mid_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

                mid_pt = (int(mid_mcp.x * w), int(mid_mcp.y * h))
                pinky_pt = (int(pinky_mcp.x * w), int(pinky_mcp.y * h))

                palm_vector = np.array([mid_pt[0] - pinky_pt[0], mid_pt[1] - pinky_pt[1]])
                angle = np.degrees(np.arctan2(palm_vector[1], palm_vector[0]))

                if HAND_TO_TRACK == 'left':
                    angle = (angle - 90) % 360 - 180
                else:
                    angle = (angle + 90) % 360 - 180

                # Determine current state
                if angle < FLIP_THRESHOLD:
                    new_state = "flip"
                elif angle > FLOP_THRESHOLD:
                    new_state = "flop"
                else:
                    new_state = "neutral"

                state_history.append(new_state)

                if len(state_history) == DEBOUNCE_FRAMES:
                    if 'flip' in state_history and current_state != "flip":
                        if current_state == "flop":
                            flip_flop_count += 1
                        current_state = "flip"
                    elif 'flop' in state_history and current_state != "flop":
                        current_state = "flop"
                    elif 'neutral' in state_history and current_state != "neutral":
                        current_state = "neutral"

    cap.release()
    hands.close()
    return flip_flop_count


def flip_flops(video_path):
    left_pronation = count_flip_flops(video_path,"left", 5, 15) 
    right_pronation = count_flip_flops(video_path,"right", 24, 34)
    return [str(left_pronation), str(right_pronation)]
