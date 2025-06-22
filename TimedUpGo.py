import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime, timedelta # Maintained for context, though not strictly needed without recording
import os
import time

# --- Mediapipe Pose Setup (Global as it's a resource) ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

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

def classify_pose(landmarks):
    # This function is from your original script, unchanged.
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

        if hip_ankle_dist < shoulder_hip_dist * 1.3:
            return STATE_SITTING
        elif hip_knee_diff < -0.05:
            if hip_ankle_dist > shoulder_hip_dist * 0.8:
                return STATE_STANDING
            else:
                return STATE_SITTING
        else:
            if hip_knee_diff < 0.02:
                return STATE_SITTING
            return STATE_TRANSITIONING
    except Exception as e:
        return STATE_UNKNOWN

def get_avg_hip_coords(landmarks):
    # This function is from your original script, unchanged.
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

def analyze_sit_to_stand(video_path):
    """
    Analyzes an existing video file for the TUG test using the original script's logic.
    The video recording functionality has been removed.

    Args:
        video_path (str): The full path to the video file to be analyzed.
    
    Returns:
        dict: A dictionary containing the final results.
    """
    # --- Initialize Pose Estimator ---
    # Done inside function for better encapsulation when imported
    pose_estimator = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # --- Video Input Validation ---
    if not os.path.exists(video_path):
        print(f"Error: The specified video file was not found: {video_path}")
        return None
        
    print(f"\n--- Starting Analysis on: {os.path.basename(video_path)} ---")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path} for analysis.")
        return None
    
    # --- This is the entire analysis loop from your original script ---
    current_state = STATE_UNKNOWN
    last_stable_state = STATE_UNKNOWN
    state_buffer = []

    sit_down_times = []
    first_confirmed_stand_up_time = None

    is_confirming_stand = False
    candidate_stand_up_time = None
    hip_y_history_for_confirmation = []
    hip_x_history_for_confirmation = []

    tug_timer_running = False
    final_tug_time = None

    frame_count = 0

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("End of video or error during analysis.")
                break

            frame_count += 1
            current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = pose_estimator.process(image_rgb)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            frame_state = STATE_UNKNOWN
            current_hip_x, current_hip_y = None, None

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )
                frame_state = classify_pose(results.pose_landmarks.landmark)
                current_hip_x, current_hip_y = get_avg_hip_coords(results.pose_landmarks.landmark)
            else:
                frame_state = STATE_UNKNOWN

            state_buffer.append(frame_state)
            if len(state_buffer) > BUFFER_LEN:
                state_buffer.pop(0)

            stable_state = current_state
            if len(state_buffer) == BUFFER_LEN:
                valid_states_in_buffer = [s for s in state_buffer if s != STATE_UNKNOWN and s != STATE_TRANSITIONING]
                if not valid_states_in_buffer:
                    if current_state == STATE_SITTING or current_state == STATE_STANDING:
                        stable_state = current_state
                    else:
                        stable_state = STATE_UNKNOWN
                else:
                    stable_state = max(set(valid_states_in_buffer), key=valid_states_in_buffer.count)

            if stable_state != current_state:
                if current_state == STATE_SITTING and stable_state == STATE_STANDING and not is_confirming_stand and first_confirmed_stand_up_time is None:
                    print(f"Time: {current_time_sec:.2f}s - Candidate stand-up detected.")
                    is_confirming_stand = True
                    candidate_stand_up_time = current_time_sec
                    hip_y_history_for_confirmation.clear()
                    hip_x_history_for_confirmation.clear()
                elif current_state == STATE_STANDING and stable_state == STATE_SITTING:
                    print(f"Time: {current_time_sec:.2f}s - Sit-down detected.")
                    sit_down_times.append(current_time_sec)
                    
                    if tug_timer_running:
                        final_tug_time = current_time_sec - first_confirmed_stand_up_time
                        tug_timer_running = False
                        print(f"--- TUG TIMER STOPPED. Final Time: {final_tug_time:.2f}s ---")

                    if is_confirming_stand:
                        print(f"Time: {current_time_sec:.2f}s - Stand-up attempt failed, sat back down.")
                        is_confirming_stand = False
                        candidate_stand_up_time = None
                last_stable_state = current_state
                current_state = stable_state

            if is_confirming_stand and current_state == STATE_STANDING:
                if current_hip_x is not None and current_hip_y is not None:
                    hip_y_history_for_confirmation.append(current_hip_y)
                    hip_x_history_for_confirmation.append(current_hip_x)

                    if len(hip_y_history_for_confirmation) > CONFIRMATION_WINDOW_FRAMES:
                        hip_y_history_for_confirmation.pop(0)
                        hip_x_history_for_confirmation.pop(0)

                    if len(hip_y_history_for_confirmation) == CONFIRMATION_WINDOW_FRAMES:
                        y_coords = np.array(hip_y_history_for_confirmation)
                        x_coords = np.array(hip_x_history_for_confirmation)
                        hip_y_range = np.max(y_coords) - np.min(y_coords)
                        hip_x_range = np.max(x_coords) - np.min(x_coords)

                        if hip_y_range < HIP_Y_STABILITY_THRESHOLD and hip_x_range > HIP_X_MOVEMENT_THRESHOLD:
                            if first_confirmed_stand_up_time is None:
                                first_confirmed_stand_up_time = candidate_stand_up_time
                                print(f"Time: {first_confirmed_stand_up_time:.2f}s - SUCCESSFUL STAND-UP CONFIRMED.")
                                print("--- TUG TIMER STARTED ---")
                                tug_timer_running = True
                            is_confirming_stand = False
                            candidate_stand_up_time = None

            elif is_confirming_stand and current_state != STATE_STANDING:
                print(f"Time: {current_time_sec:.2f}s - Stand-up confirmation aborted, state changed to {current_state}.")
                is_confirming_stand = False
                candidate_stand_up_time = None
                hip_y_history_for_confirmation.clear()
                hip_x_history_for_confirmation.clear()
            
            # --- Drawing on frame ---
            cv2.putText(image_bgr, f"Time: {current_time_sec:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)
            
            state_text = "UNKNOWN"
            if current_state == STATE_SITTING: state_text = "SITTING"
            elif current_state == STATE_STANDING: state_text = "STANDING"
            elif current_state == STATE_TRANSITIONING: state_text = "TRANSITIONING"
            if is_confirming_stand: state_text += " (Confirming Stand)"
            cv2.putText(image_bgr, f"State: {state_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            if first_confirmed_stand_up_time:
                cv2.putText(image_bgr, f"Stand Confirmed: {first_confirmed_stand_up_time:.2f}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)

            timer_text = ""
            timer_color = (255, 128, 0)
            if final_tug_time is not None:
                timer_text = f"TUG Time: {final_tug_time:.2f}s (Finished)"
                timer_color = (0, 255, 0) 
            elif tug_timer_running:
                elapsed_time = current_time_sec - first_confirmed_stand_up_time
                timer_text = f"TUG Timer: {elapsed_time:.2f}s"

            if timer_text:
                 cv2.putText(image_bgr, timer_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, timer_color, 2, cv2.LINE_AA)

            cv2.imshow('TUG Test Analysis', image_bgr)
            if cv2.waitKey(5) & 0xFF == 27: break
    
    finally:
        # --- Cleanup and Final Reporting ---
        cap.release()
        cv2.destroyAllWindows()
        pose_estimator.close()

        print("\n--- TUG Test Results ---")
        if first_confirmed_stand_up_time:
            print(f"Confirmed Stand-up Timestamp: {first_confirmed_stand_up_time:.2f}s")
        else:
            print("No confirmed stand-up was detected. Test did not start.")

        sit_down_event_time = None
        if final_tug_time is not None:
            sit_down_event_time = first_confirmed_stand_up_time + final_tug_time
            print(f"Final Sit-down Timestamp: {sit_down_event_time:.2f}s")
            print(f"\n>>> TOTAL TUG DURATION: {final_tug_time:.2f} seconds <<<")
        else:
            if first_confirmed_stand_up_time:
                print("Test was not completed (person did not sit back down before video ended).")
            else:
                print("Not enough data for TUG calculation.")
        
        # Prepare results to be returned
        return {
            "final_tug_time_seconds": final_tug_time,
            "stand_up_timestamp": first_confirmed_stand_up_time,
            "sit_down_timestamp": sit_down_event_time
        }

