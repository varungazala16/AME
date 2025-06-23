import cv2
import mediapipe as mp
import numpy as np
import os
import time

def analyze_tug_test(video_path, show_video=False):
    # --- MediaPipe Pose Setup (inside function for encapsulation) ---
    mp_pose = mp.solutions.pose
    pose_estimator = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # --- State Definitions & Parameters (Constants) ---
    STATE_UNKNOWN = -1
    STATE_SITTING = 0
    STATE_STANDING = 1
    STATE_TRANSITIONING = 2
    CONFIRMATION_WINDOW_FRAMES = 15
    HIP_Y_STABILITY_THRESHOLD = 0.02
    HIP_X_MOVEMENT_THRESHOLD = 0.015
    BUFFER_LEN = 10

    # --- Helper functions (nested for self-containment) ---
    def classify_pose(landmarks):
        try:
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
            right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

            hip_y = (left_hip.y + right_hip.y) / 2
            knee_y = (left_knee.y + right_knee.y) / 2
            ankle_y = (left_ankle.y + right_ankle.y) / 2
            shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            shoulder_hip_dist = abs(shoulder_y - hip_y)
            if shoulder_hip_dist < 1e-6: shoulder_hip_dist = 1.0
            hip_ankle_dist = abs(hip_y - ankle_y)
            hip_knee_diff = hip_y - knee_y

            if hip_ankle_dist < shoulder_hip_dist * 1.3: return STATE_SITTING
            elif hip_knee_diff < -0.05:
                return STATE_STANDING if hip_ankle_dist > shoulder_hip_dist * 0.8 else STATE_SITTING
            else:
                return STATE_SITTING if hip_knee_diff < 0.02 else STATE_TRANSITIONING
        except Exception:
            return STATE_UNKNOWN

    def get_avg_hip_coords(landmarks):
        if not landmarks: return None, None
        try:
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            if left_hip.visibility > 0.5 and right_hip.visibility > 0.5:
                return (left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2
            elif left_hip.visibility > 0.5: return left_hip.x, left_hip.y
            elif right_hip.visibility > 0.5: return right_hip.x, right_hip.y
            else: return None, None
        except: return None, None

    # --- Video Input Validation ---
    if not os.path.exists(video_path):
        # This print is useful for server-side debugging
        print(f"Error: The specified video file was not found: {video_path}")
        return None
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path} for analysis.")
        return None
    
    # --- Analysis State Variables ---
    current_state = STATE_UNKNOWN
    last_stable_state = STATE_UNKNOWN
    state_buffer = []

    first_confirmed_stand_up_time = None
    candidate_stand_up_time = None
    final_tug_time = None

    is_confirming_stand = False
    tug_timer_running = False
    
    hip_y_history_for_confirmation = []
    hip_x_history_for_confirmation = []

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break # End of video

            current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_estimator.process(image_rgb)

            frame_state = STATE_UNKNOWN
            current_hip_x, current_hip_y = None, None

            if results.pose_landmarks:
                frame_state = classify_pose(results.pose_landmarks.landmark)
                current_hip_x, current_hip_y = get_avg_hip_coords(results.pose_landmarks.landmark)

            # --- Original State Machine Logic ---
            state_buffer.append(frame_state)
            if len(state_buffer) > BUFFER_LEN:
                state_buffer.pop(0)

            stable_state = current_state
            if len(state_buffer) == BUFFER_LEN:
                valid_states = [s for s in state_buffer if s in (STATE_SITTING, STATE_STANDING)]
                if valid_states:
                    stable_state = max(set(valid_states), key=valid_states.count)

            if stable_state != current_state:
                if current_state == STATE_SITTING and stable_state == STATE_STANDING and not is_confirming_stand and first_confirmed_stand_up_time is None:
                    is_confirming_stand = True
                    candidate_stand_up_time = current_time_sec
                    hip_y_history_for_confirmation.clear()
                    hip_x_history_for_confirmation.clear()
                
                elif current_state == STATE_STANDING and stable_state == STATE_SITTING:
                    if tug_timer_running:
                        final_tug_time = current_time_sec - first_confirmed_stand_up_time
                        tug_timer_running = False
                        # Test is complete, we can exit the main loop now
                        break
                
                last_stable_state = current_state
                current_state = stable_state

            if is_confirming_stand and current_state == STATE_STANDING:
                if current_hip_x is not None and current_hip_y is not None:
                    hip_y_history_for_confirmation.append(current_hip_y)
                    hip_x_history_for_confirmation.append(current_hip_x)

                    if len(hip_y_history_for_confirmation) >= CONFIRMATION_WINDOW_FRAMES:
                        y_range = np.max(hip_y_history_for_confirmation) - np.min(hip_y_history_for_confirmation)
                        x_range = np.max(hip_x_history_for_confirmation) - np.min(hip_x_history_for_confirmation)

                        if y_range < HIP_Y_STABILITY_THRESHOLD and x_range > HIP_X_MOVEMENT_THRESHOLD:
                            if first_confirmed_stand_up_time is None:
                                first_confirmed_stand_up_time = candidate_stand_up_time
                                tug_timer_running = True
                        
                        # Whether confirmation succeeded or failed, the process is over
                        is_confirming_stand = False
                        candidate_stand_up_time = None
            
            # --- Visualization (only if enabled for local debugging) ---
            if show_video:
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                state_text = {STATE_SITTING: "SITTING", STATE_STANDING: "STANDING", STATE_TRANSITIONING: "TRANSITIONING"}.get(current_state, "UNKNOWN")
                if is_confirming_stand: state_text += " (Confirming Stand)"
                
                cv2.putText(frame, f"State: {state_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('TUG Test Analysis', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    finally:
        # --- Cleanup and Final Result Preparation ---
        cap.release()
        pose_estimator.close()
        if show_video:
            cv2.destroyAllWindows()

        # Check if the test was successfully completed
        if final_tug_time is not None and first_confirmed_stand_up_time is not None:
            
            # Convert final metrics to string format
            final_time_str = f"{final_tug_time:.2f}"
            stand_up_str = f"{first_confirmed_stand_up_time:.2f}"
            
    return [stand_up_str,final_time_str]