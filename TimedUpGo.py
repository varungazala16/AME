import cv2
import mediapipe as mp
import numpy as np
import os
import time

# --- State Definitions (Constants) ---
STATE_UNKNOWN = -1
STATE_SITTING = 0
STATE_STANDING = 1
STATE_TRANSITIONING = 2

# --- Parameters (Constants) ---
CONFIRMATION_WINDOW_FRAMES = 15
HIP_Y_STABILITY_THRESHOLD = 0.02
HIP_X_MOVEMENT_THRESHOLD = 0.015
BUFFER_LEN = 10

def analyze_tug_test(video_path, show_video=False):
    """
    Analyzes an existing video file for the Timed Up and Go (TUG) test.

    This function preserves the original analysis logic, processing a video to find a
    confirmed stand-up event, timing the duration until a final sit-down event,
    and returning the results as a single string.

    Args:
        video_path (str): The full path to the video file to be analyzed.
        show_video (bool, optional): If True, displays the video with live annotations.
                                     Requires a GUI. Defaults to False.

    Returns:
        str: A single string containing the final results, separated by ' | '.
             Format: "final_tug_time | stand_up_timestamp | sit_down_timestamp"
             Example: "12.34 | 2.56 | 14.90"
             Returns None if the video fails to open or if the test is not completed.
    """
    # --- MediaPipe Pose Setup (inside function for encapsulation) ---
    mp_pose = mp.solutions.pose
    pose_estimator = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

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
        except Exception as e:
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
        print(f"Error: The specified video file was not found: {video_path}")
        return None
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path} for analysis.")
        return None
    
    # --- This is the entire analysis loop from your original script ---
    current_state, last_stable_state = STATE_UNKNOWN, STATE_UNKNOWN
    state_buffer, sit_down_times = [], []
    first_confirmed_stand_up_time, candidate_stand_up_time, final_tug_time = None, None, None
    is_confirming_stand, tug_timer_running = False, False
    hip_y_history_for_confirmation, hip_x_history_for_confirmation = [], []
    
    final_result_string = None

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_estimator.process(image_rgb)

            if results.pose_landmarks:
                frame_state = classify_pose(results.pose_landmarks.landmark)
                current_hip_x, current_hip_y = get_avg_hip_coords(results.pose_landmarks.landmark)
            else:
                frame_state = STATE_UNKNOWN

            # --- Original State Machine and Logic ---
            state_buffer.append(frame_state)
            if len(state_buffer) > BUFFER_LEN: state_buffer.pop(0)

            stable_state = current_state
            if len(state_buffer) == BUFFER_LEN:
                valid_states = [s for s in state_buffer if s in (STATE_SITTING, STATE_STANDING)]
                if valid_states: stable_state = max(set(valid_states), key=valid_states.count)

            if stable_state != current_state:
                if current_state == STATE_SITTING and stable_state == STATE_STANDING and not is_confirming_stand and first_confirmed_stand_up_time is None:
                    print(f"Time: {current_time_sec:.2f}s - Candidate stand-up detected.")
                    is_confirming_stand, candidate_stand_up_time = True, current_time_sec
                    hip_y_history_for_confirmation.clear(), hip_x_history_for_confirmation.clear()
                elif current_state == STATE_STANDING and stable_state == STATE_SITTING:
                    print(f"Time: {current_time_sec:.2f}s - Sit-down detected.")
                    sit_down_times.append(current_time_sec)
                    if tug_timer_running:
                        final_tug_time = current_time_sec - first_confirmed_stand_up_time
                        tug_timer_running = False
                        print(f"--- TUG TIMER STOPPED. Final Time: {final_tug_time:.2f}s ---")
                        # Event is complete, we can exit the loop
                        break
                last_stable_state, current_state = current_state, stable_state

            if is_confirming_stand and current_state == STATE_STANDING:
                if current_hip_x is not None and current_hip_y is not None:
                    hip_y_history_for_confirmation.append(current_hip_y)
                    hip_x_history_for_confirmation.append(current_hip_x)
                    if len(hip_y_history_for_confirmation) == CONFIRMATION_WINDOW_FRAMES:
                        y_range = np.max(hip_y_history_for_confirmation) - np.min(hip_y_history_for_confirmation)
                        x_range = np.max(hip_x_history_for_confirmation) - np.min(hip_x_history_for_confirmation)
                        if y_range < HIP_Y_STABILITY_THRESHOLD and x_range > HIP_X_MOVEMENT_THRESHOLD and first_confirmed_stand_up_time is None:
                            first_confirmed_stand_up_time = candidate_stand_up_time
                            print(f"Time: {first_confirmed_stand_up_time:.2f}s - SUCCESSFUL STAND-UP CONFIRMED.")
                            print("--- TUG TIMER STARTED ---")
                            tug_timer_running = True
                        is_confirming_stand, candidate_stand_up_time = False, None
            
            # --- Visualization (if enabled) ---
            if show_video:
                image_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if results.pose_landmarks: mp_drawing.draw_landmarks(image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                state_text = {STATE_SITTING: "SITTING", STATE_STANDING: "STANDING", STATE_TRANSITIONING: "TRANSITIONING"}.get(current_state, "UNKNOWN")
                if is_confirming_stand: state_text += " (Confirming Stand)"
                cv2.putText(image_bgr, f"State: {state_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('TUG Test Analysis', image_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    finally:
        # --- Cleanup and Final Reporting ---
        cap.release()
        pose_estimator.close()
        if show_video: cv2.destroyAllWindows()

        # --- Prepare the final string for return ---
        sit_down_event_time = None
        if final_tug_time is not None and first_confirmed_stand_up_time is not None:
            sit_down_event_time = first_confirmed_stand_up_time + final_tug_time
            
            # Convert final metrics to string, using "N/A" for None values
            final_time_str = f"{final_tug_time:.2f}"
            stand_up_str = f"{first_confirmed_stand_up_time:.2f}"
            sit_down_str = f"{sit_down_event_time:.2f}"
            
            # Concatenate into a single string
            final_result_string = f"{final_time_str} | {stand_up_str} | {sit_down_str}"
            
            print("\n--- TUG Test Results ---")
            print(f">>> TOTAL TUG DURATION: {final_time_str} seconds <<<")
            
    return final_result_string