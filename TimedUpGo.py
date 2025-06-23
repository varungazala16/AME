import cv2
import mediapipe as mp
import numpy as np
import os

def analyze_tug_from_video(video_path: str, show_video: bool = True):

    # --- Mediapipe Pose Setup ---
    mp_pose = mp.solutions.pose
    pose_estimator = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # --- State Definitions ---
    STATE_UNKNOWN = -1
    STATE_SITTING = 0
    STATE_STANDING = 1
    STATE_TRANSITIONING = 2

    # --- Parameters for Stand-up Confirmation ---
    CONFIRMATION_WINDOW_FRAMES = 15
    HIP_Y_STABILITY_THRESHOLD = 0.02
    HIP_X_MOVEMENT_THRESHOLD = 0.015
    BUFFER_LEN = 10

    # --- Helper Functions (nested for encapsulation) ---
    def classify_pose(landmarks):
        try:
            # Extract landmark coordinates
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
            right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

            # Calculate average y-coordinates
            hip_y = (left_hip.y + right_hip.y) / 2
            knee_y = (left_knee.y + right_knee.y) / 2
            ankle_y = (left_ankle.y + right_ankle.y) / 2
            shoulder_y = (left_shoulder.y + right_shoulder.y) / 2

            # Calculate vertical distances for classification logic
            shoulder_hip_dist = abs(shoulder_y - hip_y)
            if shoulder_hip_dist < 1e-6: shoulder_hip_dist = 1.0 # Avoid division by zero
            hip_ankle_dist = abs(hip_y - ankle_y)
            hip_knee_diff = hip_y - knee_y

            # Logic to classify pose based on relative landmark positions
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
        except Exception:
            return STATE_UNKNOWN

    def get_avg_hip_coords(landmarks):
        if not landmarks: return None, None
        try:
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            # Use average if both are visible, otherwise fall back to one
            if left_hip.visibility > 0.5 and right_hip.visibility > 0.5:
                return (left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2
            elif left_hip.visibility > 0.5:
                return left_hip.x, left_hip.y
            elif right_hip.visibility > 0.5:
                return right_hip.x, right_hip.y
            else:
                return None, None
        except Exception:
            return None, None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return {'status': 'error', 'message': f'Could not open video file {video_path}'}
    
    print(f"\n--- Starting Analysis on: {os.path.basename(video_path)} ---")

    # --- State variables for the analysis loop ---
    current_state = STATE_UNKNOWN
    state_buffer = []
    first_confirmed_stand_up_time = None
    is_confirming_stand = False
    candidate_stand_up_time = None
    hip_y_history_for_confirmation = []
    hip_x_history_for_confirmation = []
    tug_timer_running = False
    final_tug_time = None

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break # End of video

        # Get the timestamp of the current frame from the video file
        current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # Process the frame with MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose_estimator.process(image_rgb)
        
        frame_state = STATE_UNKNOWN
        current_hip_x, current_hip_y = None, None

        if results.pose_landmarks:
            frame_state = classify_pose(results.pose_landmarks.landmark)
            current_hip_x, current_hip_y = get_avg_hip_coords(results.pose_landmarks.landmark)
        
        # --- State Machine and TUG Timer Logic ---
        state_buffer.append(frame_state)
        if len(state_buffer) > BUFFER_LEN:
            state_buffer.pop(0)

        # Determine a stable state from the buffer to reduce noise
        stable_state = current_state
        if len(state_buffer) == BUFFER_LEN:
            valid_states = [s for s in state_buffer if s in [STATE_SITTING, STATE_STANDING]]
            if valid_states:
                stable_state = max(set(valid_states), key=valid_states.count)

        # Detect state transitions
        if stable_state != current_state:
            # Transition from SITTING to STANDING: A potential start of the TUG test
            if current_state == STATE_SITTING and stable_state == STATE_STANDING and not is_confirming_stand and first_confirmed_stand_up_time is None:
                is_confirming_stand = True
                candidate_stand_up_time = current_time_sec
                hip_y_history_for_confirmation.clear()
                hip_x_history_for_confirmation.clear()
            
            # Transition from STANDING to SITTING: The end of the TUG test
            elif current_state == STATE_STANDING and stable_state == STATE_SITTING:
                if tug_timer_running:
                    final_tug_time = current_time_sec - first_confirmed_stand_up_time
                    tug_timer_running = False
                if is_confirming_stand:
                    is_confirming_stand = False # Abort confirmation if they sit back down

            current_state = stable_state

        # Confirmation logic for a valid stand-up event
        if is_confirming_stand and current_state == STATE_STANDING:
            if current_hip_x is not None and current_hip_y is not None:
                hip_y_history_for_confirmation.append(current_hip_y)
                hip_x_history_for_confirmation.append(current_hip_x)

                if len(hip_y_history_for_confirmation) == CONFIRMATION_WINDOW_FRAMES:
                    y_range = np.max(hip_y_history_for_confirmation) - np.min(hip_y_history_for_confirmation)
                    x_range = np.max(hip_x_history_for_confirmation) - np.min(hip_x_history_for_confirmation)

                    # Check for vertical stability and horizontal movement
                    if y_range < HIP_Y_STABILITY_THRESHOLD and x_range > HIP_X_MOVEMENT_THRESHOLD:
                        if first_confirmed_stand_up_time is None:
                            first_confirmed_stand_up_time = candidate_stand_up_time
                            tug_timer_running = True
                            print(f"Time: {first_confirmed_stand_up_time:.2f}s - SUCCESSFUL STAND-UP CONFIRMED. TUG TIMER STARTED.")
                        is_confirming_stand = False
                    # Shift the window
                    hip_y_history_for_confirmation.pop(0)
                    hip_x_history_for_confirmation.pop(0)
        
        # Abort confirmation if state changes during the confirmation window
        elif is_confirming_stand and current_state != STATE_STANDING:
            is_confirming_stand = False

        # --- Visualization (if enabled) ---
        if show_video:
            # Draw pose landmarks on the frame
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Display status text
            state_text = {STATE_SITTING: "SITTING", STATE_STANDING: "STANDING", STATE_TRANSITIONING: "TRANSITIONING"}.get(current_state, "UNKNOWN")
            if is_confirming_stand: state_text += " (Confirming...)"
            cv2.putText(frame, f"State: {state_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Time: {current_time_sec:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
            
            if final_tug_time is not None:
                timer_text = f"TUG Time: {final_tug_time:.2f}s (Finished)"
                timer_color = (0, 255, 0)
            elif tug_timer_running:
                elapsed = current_time_sec - first_confirmed_stand_up_time
                timer_text = f"TUG Timer: {elapsed:.2f}s"
                timer_color = (255, 128, 0)
            else:
                timer_text = "TUG Timer: Not Started"
                timer_color = (0, 0, 255)
            cv2.putText(frame, timer_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, timer_color, 2)

            cv2.imshow('TUG Test Analysis', frame)
            if cv2.waitKey(5) & 0xFF == 27: # Press ESC to exit
                break

    # --- Cleanup and Result Generation ---
    cap.release()
    pose_estimator.close()
    if show_video:
        cv2.destroyAllWindows()

    print("\n--- Analysis Complete ---")
    if final_tug_time:
        end_time = first_confirmed_stand_up_time + final_tug_time
        print(f">>> TOTAL TUG DURATION: {final_tug_time:.2f} seconds <<<")
        return [
            str(round(first_confirmed_stand_up_time, 2)),
            str(round(final_tug_time, 2)),
            "success",
            str(round(end_time, 2))
        ]
    elif first_confirmed_stand_up_time:
        message = "Test was not completed (person did not sit back down before video ended)."
        print(message)
        return [
            str(round(first_confirmed_stand_up_time, 2)),
            "NA"
            "incomplete",
            "NA"
        ]
    else:
        message = "No confirmed stand-up was detected. TUG test did not start."
        print(message)
        return ["NA","NA","failure","NA"]