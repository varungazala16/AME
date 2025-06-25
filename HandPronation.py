import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# --- Helper Function (Unchanged Logic) ---
def calculate_palm_orientation(hand_landmarks, image_width, image_height, hand_to_track):
    """
    Calculates the palm's orientation angle based on hand landmarks.
    This logic is identical to your original script.
    """
    wrist = hand_landmarks.landmark[0]
    mid_mcp = hand_landmarks.landmark[9]
    pinky_mcp = hand_landmarks.landmark[17]
    
    # This calculation remains unchanged.
    wrist_pt = (int(wrist.x * image_width), int(wrist.y * image_height))
    mid_pt = (int(mid_mcp.x * image_width), int(mid_mcp.y * image_height))
    pinky_pt = (int(pinky_mcp.x * image_width), int(pinky_mcp.y * image_height))
    
    palm_vector = np.array([mid_pt[0] - pinky_pt[0], mid_pt[1] - pinky_pt[1]])
    angle = np.degrees(np.arctan2(palm_vector[1], palm_vector[0]))
    
    # Adjust angle based on which hand is being tracked
    if hand_to_track == 'left':
        angle = (angle - 90) % 360 - 180
    else:
        angle = (angle + 90) % 360 - 180
    
    # wrist_pt is returned but will not be used in the headless version.
    return angle, wrist_pt


def count_flips_from_video(video_path: str, hand_to_track: str):
    """
    Processes a video file to count hand flip-flops without any GUI display.
    The core counting logic is identical to the original script.
    """
    HAND_TO_TRACK = hand_to_track.lower()
    if HAND_TO_TRACK not in ['left', 'right']:
        print("Error: hand_to_track must be 'left' or 'right'")
        return 0

    # --- Configuration (from original script, display-related vars removed) ---
    FLIP_THRESHOLD = -30
    FLOP_THRESHOLD = 30
    DEBOUNCE_FRAMES = 3

    # --- MediaPipe Initialization ---
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        model_complexity=1
    )

    # --- Video Capture Setup ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return 0

    # --- Tracking Variables (from original script) ---
    flip_flop_count = 0
    current_state = "neutral"
    state_history = deque(maxlen=DEBOUNCE_FRAMES)
    
    # --- Main Processing Loop ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # End of video. No need for a print statement here as the final one will suffice.
            break
        
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[i].classification[0].label.lower()
                if handedness != HAND_TO_TRACK:
                    continue
                    
                # Calculate orientation. The 'wrist_pt' is calculated but not used.
                palm_angle, wrist_pt = calculate_palm_orientation(hand_landmarks, w, h, HAND_TO_TRACK)
                
                # --- State Machine Logic (Unchanged) ---
                if palm_angle < FLIP_THRESHOLD:
                    new_state = "flip"
                elif palm_angle > FLOP_THRESHOLD:
                    new_state = "flop"
                else:
                    new_state = "neutral"
                
                state_history.append(new_state)
                
                if len(state_history) == DEBOUNCE_FRAMES:
                    if all(s == "flip" for s in state_history) and current_state != "flip":
                        if current_state == "flop":
                            flip_flop_count += 1
                        current_state = "flip"
                    elif all(s == "flop" for s in state_history) and current_state != "flop":
                        current_state = "flop"
                
                break  # Process only the first detected hand of the correct type
        
        # All cv2.imshow, cv2.waitKey, drawing, and text-drawing calls have been removed.

    # --- Cleanup ---
    cap.release()
    hands.close()
    
    return [str(flip_flop_count)]