import cv2
import mediapipe as mp
import numpy as np
import time
import os

# --- Helper Functions (Defined outside to be clean) ---
def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    ba = a - b; bc = c - b
    norm_ba = np.linalg.norm(ba); norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0: return 0.0
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def get_landmark_coords(landmarks, landmark_enum, image_shape):
    if landmarks and landmarks.landmark[landmark_enum].visibility > 0.4:
        lm = landmarks.landmark[landmark_enum]
        return np.array([lm.x * image_shape[1], lm.y * image_shape[0]])
    return None

def get_vertical_midpoint(lm1_coords, lm2_coords):
    if lm1_coords is not None and lm2_coords is not None:
        return (lm1_coords[1] + lm2_coords[1]) / 2, (lm1_coords[0] + lm2_coords[0]) / 2
    return None, None

def analyze_sit_to_stand_RFC(video_path, show_video=False):
    """
    Analyzes a video file to measure the time and sway of a sit-to-stand event.

    This is a self-contained function that processes a video to find a stable sitting
    pose, times the duration until a stable standing pose is achieved, and calculates
    the horizontal sway during the event.

    Args:
        video_path (str): The full path to the video file to be analyzed.
        show_video (bool, optional): If True, displays the video with live annotations.
                                     Requires a GUI. Defaults to False.

    Returns:
        str: A single string containing the final results, separated by ' | '.
             Format: "time_seconds | sway_pixels" (e.g., "3.45 | 25.8").
             Returns None if the event is not completed or if the video fails to open.
    """
    # --- Configuration Constants ---
    MIN_DETECTION_CONFIDENCE = 0.5
    MIN_TRACKING_CONFIDENCE = 0.5
    SITTING_KNEE_ANGLE_THRESH_MAX = 130
    SITTING_HIP_ANGLE_THRESH_MAX = 130
    HIPS_BELOW_KNEES_FACTOR_SITTING = 0.9
    VELOCITY_THRESHOLD_STABLE_SIT = 0.03
    STABILITY_DURATION_SITTING_SEC = 0.5
    STANDING_KNEE_ANGLE_THRESH_MIN = 135
    STANDING_TORSO_ANGLE_THRESH_MAX_DEVIATION = 40
    HIPS_ABOVE_KNEES_FACTOR_STANDING = 0.95
    VELOCITY_THRESHOLD_FIRST_RISE_INTENT = -0.025
    VELOCITY_THRESHOLD_STABLE_STAND = 0.07
    STABILITY_CONFIRMATION_STANDING_SEC = 1.5

    # --- MediaPipe Initialization ---
    mp_pose = mp.solutions.pose
    pose_estimator = mp_pose.Pose(min_detection_confidence=MIN_DETECTION_CONFIDENCE,
                                  min_tracking_confidence=MIN_TRACKING_CONFIDENCE)
    mp_drawing = mp.solutions.drawing_utils

    # --- Video Handling ---
    if not os.path.exists(video_path):
        print(f"Error: Could not find video file: {video_path}")
        return None
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {video_path}")
        return None

    # --- State Machine and Variables ---
    STATE_WAITING_FOR_STABLE_SIT = "WAITING_FOR_STABLE_SIT"
    STATE_MONITORING_SIT_AND_TIMING = "MONITORING_SIT_AND_TIMING"
    STATE_ATTEMPTING_STAND_SEQUENCE = "ATTEMPTING_STAND_SEQUENCE"
    current_state = STATE_WAITING_FOR_STABLE_SIT

    overall_event_timer_start = None
    sway_data_points_for_event = []
    prev_hip_y_norm_for_velocity = None
    prev_time = time.time()
    stability_counter_start_time = None
    
    final_result_string = None

    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            image_height, image_width, _ = image.shape
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose_estimator.process(image_rgb)

            current_time = time.time()
            dt = current_time - prev_time
            if dt <= 0: dt = 1e-6
            prev_time = current_time

            person_detected = results.pose_landmarks is not None
            is_currently_sitting_pose = False
            is_currently_standing_pose = False
            temp_hip_y_norm = None
            temp_knee_y_norm = None
            current_hip_x_norm = None
            temp_hip_vy_norm = 0.0
            
            if person_detected:
                landmarks = results.pose_landmarks
                lm = landmarks.landmark
                
                if lm[mp_pose.PoseLandmark.LEFT_HIP].visibility > 0.4 and lm[mp_pose.PoseLandmark.RIGHT_HIP].visibility > 0.4:
                    temp_hip_y_norm = (lm[mp_pose.PoseLandmark.LEFT_HIP].y + lm[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2
                    current_hip_x_norm = (lm[mp_pose.PoseLandmark.LEFT_HIP].x + lm[mp_pose.PoseLandmark.RIGHT_HIP].x) / 2
                if lm[mp_pose.PoseLandmark.LEFT_KNEE].visibility > 0.4 and lm[mp_pose.PoseLandmark.RIGHT_KNEE].visibility > 0.4:
                    temp_knee_y_norm = (lm[mp_pose.PoseLandmark.LEFT_KNEE].y + lm[mp_pose.PoseLandmark.RIGHT_KNEE].y) / 2

                if temp_hip_y_norm is not None:
                    if prev_hip_y_norm_for_velocity is not None:
                        temp_hip_vy_norm = (temp_hip_y_norm - prev_hip_y_norm_for_velocity) / dt
                    prev_hip_y_norm_for_velocity = temp_hip_y_norm
                
                left_shoulder_px = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER, image.shape)
                right_shoulder_px = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER, image.shape)
                left_hip_px = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_HIP, image.shape)
                right_hip_px = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_HIP, image.shape)
                left_knee_px = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_KNEE, image.shape)
                right_knee_px = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE, image.shape)
                left_ankle_px = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE, image.shape)
                right_ankle_px = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE, image.shape)

                if all(p is not None for p in [left_hip_px, left_knee_px, left_ankle_px, right_hip_px, right_knee_px, right_ankle_px, left_shoulder_px, right_shoulder_px]):
                    temp_avg_knee_angle = (calculate_angle(left_hip_px, left_knee_px, left_ankle_px) + calculate_angle(right_hip_px, right_knee_px, right_ankle_px)) / 2
                    temp_avg_hip_angle = (calculate_angle(left_shoulder_px, left_hip_px, left_knee_px) + calculate_angle(right_shoulder_px, right_hip_px, right_knee_px)) / 2
                    mid_shoulder_y_px, mid_shoulder_x_px = get_vertical_midpoint(left_shoulder_px, right_shoulder_px)
                    mid_hip_y_px, mid_hip_x_px = get_vertical_midpoint(left_hip_px, right_hip_px)
                    if mid_shoulder_y_px is not None:
                        torso_vertical_ref_pt = np.array([mid_hip_x_px, mid_hip_y_px - 100])
                        mid_shoulder_pt = np.array([mid_shoulder_x_px, mid_shoulder_y_px])
                        mid_hip_pt = np.array([mid_hip_x_px, mid_hip_y_px])
                        temp_torso_angle = calculate_angle(torso_vertical_ref_pt, mid_hip_pt, mid_shoulder_pt)
                        
                        if temp_hip_y_norm is not None and temp_knee_y_norm is not None:
                            if temp_hip_y_norm > temp_knee_y_norm * HIPS_BELOW_KNEES_FACTOR_SITTING and temp_avg_knee_angle < SITTING_KNEE_ANGLE_THRESH_MAX and temp_avg_hip_angle < SITTING_HIP_ANGLE_THRESH_MAX:
                                is_currently_sitting_pose = True
                            if temp_hip_y_norm < temp_knee_y_norm * HIPS_ABOVE_KNEES_FACTOR_STANDING and temp_avg_knee_angle > STANDING_KNEE_ANGLE_THRESH_MIN and temp_torso_angle < STANDING_TORSO_ANGLE_THRESH_MAX_DEVIATION:
                                is_currently_standing_pose = True
                
                velocity_condition_met_for_stable_stand = abs(temp_hip_vy_norm) < VELOCITY_THRESHOLD_STABLE_STAND
                combined_condition_for_stable_stand = is_currently_standing_pose and velocity_condition_met_for_stable_stand

                if overall_event_timer_start is not None and current_hip_x_norm is not None:
                    sway_data_points_for_event.append(current_hip_x_norm)

                if current_state == STATE_WAITING_FOR_STABLE_SIT:
                    if is_currently_sitting_pose and abs(temp_hip_vy_norm) < VELOCITY_THRESHOLD_STABLE_SIT:
                        if stability_counter_start_time is None: stability_counter_start_time = current_time
                        elif current_time - stability_counter_start_time >= STABILITY_DURATION_SITTING_SEC:
                            overall_event_timer_start = current_time 
                            sway_data_points_for_event = [current_hip_x_norm] if current_hip_x_norm is not None else []
                            current_state = STATE_MONITORING_SIT_AND_TIMING
                            stability_counter_start_time = None 
                    else:
                        stability_counter_start_time = None 

                elif current_state == STATE_MONITORING_SIT_AND_TIMING:
                    if not is_currently_sitting_pose or temp_hip_vy_norm < VELOCITY_THRESHOLD_FIRST_RISE_INTENT: 
                        current_state = STATE_ATTEMPTING_STAND_SEQUENCE
                        stability_counter_start_time = None 
                
                elif current_state == STATE_ATTEMPTING_STAND_SEQUENCE:
                    if combined_condition_for_stable_stand: 
                        if stability_counter_start_time is None: stability_counter_start_time = current_time
                        if (current_time - stability_counter_start_time) >= STABILITY_CONFIRMATION_STANDING_SEC:
                            final_event_duration = current_time - overall_event_timer_start 
                            final_event_sway = np.std(sway_data_points_for_event) * image_width if len(sway_data_points_for_event) > 1 else 0.0
                            final_result_string = f"{final_event_duration:.2f} | {final_event_sway:.2f}"
                            break # Success, exit the loop
                    else: 
                        stability_counter_start_time = None

            else: # Person not detected
                if overall_event_timer_start is not None: # Reset if event was active
                    current_state = STATE_WAITING_FOR_STABLE_SIT
                    overall_event_timer_start = None
                    stability_counter_start_time = None
                    prev_hip_y_norm_for_velocity = None
                    sway_data_points_for_event = []

            if show_video:
                # This is the original display logic
                image_display = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_display = cv2.cvtColor(image_display, cv2.COLOR_RGB2BGR)
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(image_display, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                cv2.putText(image_display, f"State: {current_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow('Sit-to-Stand Analysis', image_display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
    finally:
        cap.release()
        pose_estimator.close()
        if show_video:
            cv2.destroyAllWindows()

    return final_result_string