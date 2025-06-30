import cv2
import mediapipe as mp
import numpy as np
import os

def analyze_tug_from_video(video_path: str, show_video: bool = False):
    """
    Analyzes a video of a Timed Up and Go (TUG) test to measure the duration.
    This version measures to the *last* detected sit-down event.

    Args:
        video_path (str): The path to the video file to be analyzed.
        show_video (bool): If True, displays the video with real-time analysis overlays.
                           Defaults to True.

    Returns:
        list: A list containing [start_time, duration, status, end_time].
              - 'status' can be 'success', 'incomplete', or 'failure'.
              - 'NA' is used for values that could not be determined.
    """
    # --- Mediapipe Pose Setup ---
    mp_pose = mp.solutions.pose
    pose_estimator = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    
    landmark_drawing_spec = mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2)
    connection_drawing_spec = mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)

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
        except Exception:
            return STATE_UNKNOWN

    def get_avg_hip_coords(landmarks):
        if not landmarks: return None, None
        try:
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
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
    final_tug_duration = None # Renamed for clarity

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose_estimator.process(image_rgb)
        
        frame_state = STATE_UNKNOWN
        current_hip_x, current_hip_y = None, None

        if results.pose_landmarks:
            frame_state = classify_pose(results.pose_landmarks.landmark)
            current_hip_x, current_hip_y = get_avg_hip_coords(results.pose_landmarks.landmark)
        
        state_buffer.append(frame_state)
        if len(state_buffer) > BUFFER_LEN:
            state_buffer.pop(0)

        stable_state = current_state
        if len(state_buffer) == BUFFER_LEN:
            valid_states = [s for s in state_buffer if s in [STATE_SITTING, STATE_STANDING]]
            if valid_states:
                stable_state = max(set(valid_states), key=valid_states.count)

        if stable_state != current_state:
            if current_state == STATE_SITTING and stable_state == STATE_STANDING and not is_confirming_stand and first_confirmed_stand_up_time is None:
                is_confirming_stand = True
                candidate_stand_up_time = current_time_sec
                hip_y_history_for_confirmation.clear()
                hip_x_history_for_confirmation.clear()
            
            # ### MODIFICATION START ###
            # This logic now allows the end time to be continuously updated.
            elif current_state == STATE_STANDING and stable_state == STATE_SITTING:
                if tug_timer_running:
                    # Overwrite the duration with the time of the latest sit-down event.
                    final_tug_duration = current_time_sec - first_confirmed_stand_up_time
                    # The line `tug_timer_running = False` is REMOVED to allow updates.
                if is_confirming_stand:
                    is_confirming_stand = False
            # ### MODIFICATION END ###

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
                            print(f"Time: {first_confirmed_stand_up_time:.2f}s - SUCCESSFUL STAND-UP CONFIRMED. TUG TIMER STARTED.")
                        is_confirming_stand = False
                    hip_y_history_for_confirmation.pop(0)
                    hip_x_history_for_confirmation.pop(0)
        
        elif is_confirming_stand and current_state != STATE_STANDING:
            is_confirming_stand = False

        if show_video:
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          landmark_drawing_spec, connection_drawing_spec)
            overlay = frame.copy()
            cv2.rectangle(overlay, (5, 5), (450, 100), (20, 20, 20), -1)
            alpha = 0.7
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            
            state_map = {STATE_UNKNOWN: "UNKNOWN", STATE_SITTING: "SITTING", STATE_STANDING: "STANDING", STATE_TRANSITIONING: "TRANSITIONING"}
            state_text = f"Pose: {state_map.get(current_state, '...loading')}"
            cv2.putText(frame, state_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            timer_text = "TUG Time: --"
            timer_color = (255, 255, 0)
            if final_tug_duration:
                timer_text = f"TUG Time: {final_tug_duration:.2f}s"
                timer_color = (0, 255, 0)
            elif tug_timer_running:
                elapsed_time = current_time_sec - first_confirmed_stand_up_time
                timer_text = f"TUG Time: {elapsed_time:.2f}s"
                timer_color = (100, 255, 100)
            cv2.putText(frame, timer_text, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, timer_color, 2)

            status_text = "Status: Waiting for stand-up"
            if is_confirming_stand:
                status_text = "Status: Confirming stable stand..."
            elif first_confirmed_stand_up_time and not final_tug_duration:
                status_text = "Status: TUG test in progress..."
            elif final_tug_duration:
                status_text = "Status: Test complete."
            cv2.putText(frame, status_text, (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1)
            
            cv2.imshow('TUG Analysis', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
           
    cap.release()
    pose_estimator.close()
    if show_video:
        cv2.destroyAllWindows()

    print("\n--- Analysis Complete ---")
    if final_tug_duration:
        end_time = first_confirmed_stand_up_time + final_tug_duration
        print(f">>> TOTAL TUG DURATION: {final_tug_duration:.2f} seconds <<<")
        return [
            str(round(first_confirmed_stand_up_time, 2)),
            str(round(final_tug_duration, 2)),
            "success",
            str(round(end_time, 2))
        ]
    elif first_confirmed_stand_up_time:
        message = "Test was not completed (person did not sit back down before video ended)."
        print(message)
        return [
            str(round(first_confirmed_stand_up_time, 2)),
            "NA",
            "incomplete",
            "NA"
        ]
    else:
        message = "No confirmed stand-up was detected. TUG test did not start."
        print(message)
        return ["NA", "NA", "failure", "NA"]

def dual_attention(video_path: str, show_video: bool = False):
    """
    Analyzes a video of a Timed Up and Go (TUG) test to measure the duration.
    This version measures to the *last* detected sit-down event.

    Args:
        video_path (str): The path to the video file to be analyzed.
        show_video (bool): If True, displays the video with real-time analysis overlays.
                           Defaults to True.

    Returns:
        list: A list containing [start_time, duration, status, end_time].
              - 'status' can be 'success', 'incomplete', or 'failure'.
              - 'NA' is used for values that could not be determined.
    """
    # --- Mediapipe Pose Setup ---
    mp_pose = mp.solutions.pose
    pose_estimator = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    
    landmark_drawing_spec = mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2)
    connection_drawing_spec = mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)

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
        except Exception:
            return STATE_UNKNOWN

    def get_avg_hip_coords(landmarks):
        if not landmarks: return None, None
        try:
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
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
    final_tug_duration = None # Renamed for clarity

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose_estimator.process(image_rgb)
        
        frame_state = STATE_UNKNOWN
        current_hip_x, current_hip_y = None, None

        if results.pose_landmarks:
            frame_state = classify_pose(results.pose_landmarks.landmark)
            current_hip_x, current_hip_y = get_avg_hip_coords(results.pose_landmarks.landmark)
        
        state_buffer.append(frame_state)
        if len(state_buffer) > BUFFER_LEN:
            state_buffer.pop(0)

        stable_state = current_state
        if len(state_buffer) == BUFFER_LEN:
            valid_states = [s for s in state_buffer if s in [STATE_SITTING, STATE_STANDING]]
            if valid_states:
                stable_state = max(set(valid_states), key=valid_states.count)

        if stable_state != current_state:
            if current_state == STATE_SITTING and stable_state == STATE_STANDING and not is_confirming_stand and first_confirmed_stand_up_time is None:
                is_confirming_stand = True
                candidate_stand_up_time = current_time_sec
                hip_y_history_for_confirmation.clear()
                hip_x_history_for_confirmation.clear()
            
            # ### MODIFICATION START ###
            # This logic now allows the end time to be continuously updated.
            elif current_state == STATE_STANDING and stable_state == STATE_SITTING:
                if tug_timer_running:
                    # Overwrite the duration with the time of the latest sit-down event.
                    final_tug_duration = current_time_sec - first_confirmed_stand_up_time
                    # The line `tug_timer_running = False` is REMOVED to allow updates.
                if is_confirming_stand:
                    is_confirming_stand = False
            # ### MODIFICATION END ###

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
                            print(f"Time: {first_confirmed_stand_up_time:.2f}s - SUCCESSFUL STAND-UP CONFIRMED. TUG TIMER STARTED.")
                        is_confirming_stand = False
                    hip_y_history_for_confirmation.pop(0)
                    hip_x_history_for_confirmation.pop(0)
        
        elif is_confirming_stand and current_state != STATE_STANDING:
            is_confirming_stand = False

        if show_video:
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          landmark_drawing_spec, connection_drawing_spec)
            overlay = frame.copy()
            cv2.rectangle(overlay, (5, 5), (450, 100), (20, 20, 20), -1)
            alpha = 0.7
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            
            state_map = {STATE_UNKNOWN: "UNKNOWN", STATE_SITTING: "SITTING", STATE_STANDING: "STANDING", STATE_TRANSITIONING: "TRANSITIONING"}
            state_text = f"Pose: {state_map.get(current_state, '...loading')}"
            cv2.putText(frame, state_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            timer_text = "TUG Time: --"
            timer_color = (255, 255, 0)
            if final_tug_duration:
                timer_text = f"TUG Time: {final_tug_duration:.2f}s"
                timer_color = (0, 255, 0)
            elif tug_timer_running:
                elapsed_time = current_time_sec - first_confirmed_stand_up_time
                timer_text = f"TUG Time: {elapsed_time:.2f}s"
                timer_color = (100, 255, 100)
            cv2.putText(frame, timer_text, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, timer_color, 2)

            status_text = "Status: Waiting for stand-up"
            if is_confirming_stand:
                status_text = "Status: Confirming stable stand..."
            elif first_confirmed_stand_up_time and not final_tug_duration:
                status_text = "Status: TUG test in progress..."
            elif final_tug_duration:
                status_text = "Status: Test complete."
            cv2.putText(frame, status_text, (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1)
            
            cv2.imshow('TUG Analysis', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
           
    cap.release()
    pose_estimator.close()
    if show_video:
        cv2.destroyAllWindows()

    print("\n--- Analysis Complete ---")
    if final_tug_duration:
        end_time = first_confirmed_stand_up_time + final_tug_duration
        print(f">>> TOTAL TUG DURATION: {final_tug_duration:.2f} seconds <<<")
        return [
            str(round(final_tug_duration, 2)),
            "success",
            str(round(end_time, 2))
        ]
    elif first_confirmed_stand_up_time:
        message = "Test was not completed (person did not sit back down before video ended)."
        print(message)
        return [
            "NA",
            "incomplete",
            "NA"
        ]
    else:
        message = "No confirmed stand-up was detected. TUG test did not start."
        print(message)
        return ["NA", "failure", "NA"]