import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

def count_flip_flops(video_path, hand):
    # Validate hand input
    HAND_TO_TRACK = hand.lower()
    if HAND_TO_TRACK not in ['left', 'right']:
        raise ValueError("hand must be 'left' or 'right'")

    FLIP_THRESHOLD = -30
    FLOP_THRESHOLD = 30
    DEBOUNCE_FRAMES = 3
    FRAME_DELAY = 1
    ZOOM_SIZE = 200
    RUN_TIME = 30  # seconds

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

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

    flip_flop_count = 0
    current_state = "neutral"
    state_history = deque(maxlen=DEBOUNCE_FRAMES)
    last_valid_pos = None

    start_time = time.time()
    end_processing_time = start_time + RUN_TIME
    frames_processed_count = 0

    while time.time() < end_processing_time:
        ret, frame = cap.read()
        if not ret:
            # End of video
            break

        frames_processed_count += 1
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        detected_hand_for_tracking_in_frame = False
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness_classification = results.multi_handedness[i].classification[0]
                handedness_label = handedness_classification.label.lower()
                if handedness_label != HAND_TO_TRACK:
                    continue

                detected_hand_for_tracking_in_frame = True

                # --- Palm orientation calculation ---
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                mid_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

                wrist_pt = (int(wrist.x * w), int(wrist.y * h))
                mid_pt = (int(mid_mcp.x * w), int(mid_mcp.y * h))
                pinky_pt = (int(pinky_mcp.x * w), int(pinky_mcp.y * h))
                palm_vector = np.array([mid_pt[0] - pinky_pt[0], mid_pt[1] - pinky_pt[1]])
                angle = np.degrees(np.arctan2(palm_vector[1], palm_vector[0]))
                if HAND_TO_TRACK == 'left':
                    angle = (angle - 90) % 360 - 180
                else:
                    angle = (angle + 90) % 360 - 180

                last_valid_pos = wrist_pt

                # Draw landmarks (optional, remove for headless/server use)
                # mp_drawing.draw_landmarks(
                #     frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                #     mp_drawing_styles.get_default_hand_landmarks_style(),
                #     mp_drawing_styles.get_default_hand_connections_style()
                # )

                # --- State machine ---
                if angle < FLIP_THRESHOLD:
                    new_state = "flip"
                elif angle > FLOP_THRESHOLD:
                    new_state = "flop"
                else:
                    new_state = "neutral"

                state_history.append(new_state)

                if len(state_history) == DEBOUNCE_FRAMES:
                    if 'flip' in state_history and current_state != "flip":
                        print("step-2 (flip region detected!)")
                        if current_state == "flop":
                            print("Increment flip_flop_count (transition flop → flip)")
                            flip_flop_count += 1
                        current_state = "flip"
                    elif 'flop' in state_history and current_state != "flop":
                        print("All FLOP detected, set current_state to flop")
                        current_state = "flop"
                    elif 'neutral' in state_history and current_state != "neutral":
                        print("All NEUTRAL detected, set current_state to neutral")
                        current_state = "neutral"

        # For speed (remove frame display for server/batch use)
        # if cv2.waitKey(FRAME_DELAY) & 0xFF == ord('q'):
        #     break

    cap.release()
    cv2.destroyAllWindows()
    return [str(flip_flop_count)]