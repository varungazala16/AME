import cv2
import mediapipe as mp
import numpy as np
import time
import sys

# --- Configuration Constants (Using Lenient Set for Robustness) ---
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
# Sitting Pose
SITTING_KNEE_ANGLE_THRESH_MAX = 130
SITTING_HIP_ANGLE_THRESH_MAX = 130
HIPS_BELOW_KNEES_FACTOR_SITTING = 0.9
VELOCITY_THRESHOLD_STABLE_SIT = 0.03
STABILITY_DURATION_SITTING_SEC = 0.5
# Standing Pose (MORE LENIENT)
STANDING_KNEE_ANGLE_THRESH_MIN = 135
STANDING_TORSO_ANGLE_THRESH_MAX_DEVIATION = 40
HIPS_ABOVE_KNEES_FACTOR_STANDING = 0.95
# Motion for Standing
VELOCITY_THRESHOLD_FIRST_RISE_INTENT = -0.025 # Still used to confirm they are *actually* rising from the monitored sit
VELOCITY_THRESHOLD_STABLE_STAND = 0.07
STABILITY_CONFIRMATION_STANDING_SEC = 1.5
# --- End of Configuration Constants ---

# --- Helper Functions (No Changes) ---
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
# --- End of Helper Functions ---

# --- Main Application ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

STATE_WAITING_FOR_STABLE_SIT = "WAITING_FOR_STABLE_SIT"
STATE_MONITORING_SIT_AND_TIMING = "MONITORING_SIT_AND_TIMING" # Timer starts here, waiting for rise
STATE_ATTEMPTING_STAND_SEQUENCE = "ATTEMPTING_STAND_SEQUENCE"

current_state = STATE_WAITING_FOR_STABLE_SIT
overall_event_timer_start = None
sway_data_points_for_event = []
prev_hip_y_norm_for_velocity = None
prev_time = time.time()
stability_counter_start_time = None

font = cv2.FONT_HERSHEY_SIMPLEX; font_scale = 0.6
font_color_default = (255,255,255); font_color_pass = (0,255,0); font_color_fail = (0,0,255)
line_type = 2

with mp_pose.Pose(min_detection_confidence=MIN_DETECTION_CONFIDENCE,
                  min_tracking_confidence=MIN_TRACKING_CONFIDENCE) as pose_estimator:
    while cap.isOpened():
        success, image = cap.read()
        if not success: print("End of video or camera error."); break

        image_height, image_width, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose_estimator.process(image_rgb)
        image_display = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        current_time = time.time()
        dt = current_time - prev_time
        if dt <= 0: dt = 1e-6
        fps_approx = 1.0 / dt
        prev_time = current_time

        person_detected = results.pose_landmarks is not None
        display_status_main = "No Person"
        overall_event_time_display = 0.0

        val_hip_y_n, val_knee_y_n = "N/A", "N/A"; val_avg_knee_angle, val_torso_angle, val_avg_hip_angle = "N/A", "N/A", "N/A"
        val_hip_vy_n = "N/A"; val_stable_stand_timer_display = "N/A"
        is_currently_sitting_pose = False; is_currently_standing_pose = False
        current_hip_x_norm = None; temp_hip_y_norm, temp_knee_y_norm = None, None
        temp_hip_vy_norm = 0.0; temp_avg_knee_angle, temp_torso_angle, temp_avg_hip_angle = 0.0, 0.0, 0.0
        velocity_condition_met_for_stable_stand = False; combined_condition_for_stable_stand = False

        if person_detected:
            landmarks = results.pose_landmarks; lm = landmarks.landmark
            left_shoulder_px = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER, image_display.shape)
            right_shoulder_px = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER, image_display.shape)
            left_hip_px = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_HIP, image_display.shape)
            right_hip_px = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_HIP, image_display.shape)
            left_knee_px = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_KNEE, image_display.shape)
            right_knee_px = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE, image_display.shape)
            left_ankle_px = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE, image_display.shape)
            right_ankle_px = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE, image_display.shape)

            if lm[mp_pose.PoseLandmark.LEFT_HIP].visibility > 0.4 and lm[mp_pose.PoseLandmark.RIGHT_HIP].visibility > 0.4:
                temp_hip_y_norm = (lm[mp_pose.PoseLandmark.LEFT_HIP].y + lm[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2
                current_hip_x_norm = (lm[mp_pose.PoseLandmark.LEFT_HIP].x + lm[mp_pose.PoseLandmark.RIGHT_HIP].x) / 2
                val_hip_y_n = f"{temp_hip_y_norm:.3f}"
            if lm[mp_pose.PoseLandmark.LEFT_KNEE].visibility > 0.4 and lm[mp_pose.PoseLandmark.RIGHT_KNEE].visibility > 0.4:
                temp_knee_y_norm = (lm[mp_pose.PoseLandmark.LEFT_KNEE].y + lm[mp_pose.PoseLandmark.RIGHT_KNEE].y) / 2
                val_knee_y_n = f"{temp_knee_y_norm:.3f}"

            new_hip_vy_norm = 0.0
            if temp_hip_y_norm is not None:
                if prev_hip_y_norm_for_velocity is not None:
                    new_hip_vy_norm = (temp_hip_y_norm - prev_hip_y_norm_for_velocity) / dt
                prev_hip_y_norm_for_velocity = temp_hip_y_norm
            temp_hip_vy_norm = new_hip_vy_norm
            val_hip_vy_n = f"{temp_hip_vy_norm:.4f}"

            if all(p is not None for p in [left_hip_px, left_knee_px, left_ankle_px, right_hip_px, right_knee_px, right_ankle_px, left_shoulder_px, right_shoulder_px]):
                temp_avg_knee_angle = (calculate_angle(left_hip_px, left_knee_px, left_ankle_px) + calculate_angle(right_hip_px, right_knee_px, right_ankle_px)) / 2
                val_avg_knee_angle = f"{temp_avg_knee_angle:.1f}"
                temp_avg_hip_angle = (calculate_angle(left_shoulder_px, left_hip_px, left_knee_px) + calculate_angle(right_shoulder_px, right_hip_px, right_knee_px)) / 2
                val_avg_hip_angle = f"{temp_avg_hip_angle:.1f}"
                mid_shoulder_y_px, mid_shoulder_x_px = get_vertical_midpoint(left_shoulder_px, right_shoulder_px)
                mid_hip_y_px, mid_hip_x_px = get_vertical_midpoint(left_hip_px, right_hip_px)
                if mid_shoulder_y_px is not None and mid_hip_y_px is not None and mid_shoulder_x_px is not None and mid_hip_x_px is not None :
                    torso_vertical_ref_pt = np.array([mid_hip_x_px, mid_hip_y_px - 100])
                    mid_shoulder_pt = np.array([mid_shoulder_x_px, mid_shoulder_y_px])
                    mid_hip_pt = np.array([mid_hip_x_px, mid_hip_y_px])
                    temp_torso_angle = calculate_angle(torso_vertical_ref_pt, mid_hip_pt, mid_shoulder_pt)
                    val_torso_angle = f"{temp_torso_angle:.1f}"

                if temp_hip_y_norm is not None and temp_knee_y_norm is not None:
                    if temp_hip_y_norm > temp_knee_y_norm * HIPS_BELOW_KNEES_FACTOR_SITTING and \
                       temp_avg_knee_angle < SITTING_KNEE_ANGLE_THRESH_MAX and \
                       temp_avg_hip_angle < SITTING_HIP_ANGLE_THRESH_MAX:
                        is_currently_sitting_pose = True
                    if temp_hip_y_norm < temp_knee_y_norm * HIPS_ABOVE_KNEES_FACTOR_STANDING and \
                       temp_avg_knee_angle > STANDING_KNEE_ANGLE_THRESH_MIN and \
                       temp_torso_angle < STANDING_TORSO_ANGLE_THRESH_MAX_DEVIATION:
                        is_currently_standing_pose = True
            
            current_hip_velocity_for_check = abs(temp_hip_vy_norm)
            velocity_condition_met_for_stable_stand = current_hip_velocity_for_check < VELOCITY_THRESHOLD_STABLE_STAND
            combined_condition_for_stable_stand = is_currently_standing_pose and velocity_condition_met_for_stable_stand

            # Collect sway data if timer has started
            if overall_event_timer_start is not None and current_hip_x_norm is not None:
                sway_data_points_for_event.append(current_hip_x_norm)


            # --- State Machine for Resilient Timer ---
            if current_state == STATE_WAITING_FOR_STABLE_SIT:
                display_status_main = "Waiting for Stable Sit"
                if is_currently_sitting_pose and abs(temp_hip_vy_norm) < VELOCITY_THRESHOLD_STABLE_SIT:
                    if stability_counter_start_time is None: stability_counter_start_time = current_time
                    elif current_time - stability_counter_start_time >= STABILITY_DURATION_SITTING_SEC:
                        # <<<< TIMER AND SWAY COLLECTION START HERE >>>>
                        overall_event_timer_start = current_time 
                        sway_data_points_for_event = [current_hip_x_norm] if current_hip_x_norm is not None else [] # Start sway collection
                        
                        current_state = STATE_MONITORING_SIT_AND_TIMING # New state name
                        display_status_main = "Timing Sit, Waiting for Rise"
                        print(f"[{time.strftime('%H:%M:%S')}] Stable Sit Confirmed. OVERALL TIMER STARTED. Waiting for Rise. FPS ~{fps_approx:.1f}")
                        stability_counter_start_time = None 
                else:
                    stability_counter_start_time = None 

            elif current_state == STATE_MONITORING_SIT_AND_TIMING:
                if overall_event_timer_start: overall_event_time_display = current_time - overall_event_timer_start
                display_status_main = f"Timing Sit, Waiting Rise...{overall_event_time_display:.2f}s"
                
                if not is_currently_sitting_pose and person_detected: 
                    # If they leave sitting pose without a clear "rise", it's a bit ambiguous.
                    # For robustness, let's assume they are starting to stand or readjusting.
                    # Transition to ATTEMPTING_STAND_SEQUENCE to catch the stand.
                    # This avoids getting stuck if the rise_intent velocity isn't perfectly caught.
                    print(f"[{time.strftime('%H:%M:%S')}] Left sitting pose during monitoring. Assuming stand attempt. FPS ~{fps_approx:.1f}")
                    current_state = STATE_ATTEMPTING_STAND_SEQUENCE
                    stability_counter_start_time = None # Reset for stand stability check
                elif temp_hip_vy_norm < VELOCITY_THRESHOLD_FIRST_RISE_INTENT: 
                    print(f"[{time.strftime('%H:%M:%S')}] Rise Intent Velocity Met. Transitioning to Attempting Stand. FPS ~{fps_approx:.1f}")
                    current_state = STATE_ATTEMPTING_STAND_SEQUENCE
                    stability_counter_start_time = None 
            
            elif current_state == STATE_ATTEMPTING_STAND_SEQUENCE:
                if overall_event_timer_start: overall_event_time_display = current_time - overall_event_timer_start
                display_status_main = f"Attempting Stand... {overall_event_time_display:.2f}s"
                # Sway data is already being collected if overall_event_timer_start is not None

                if combined_condition_for_stable_stand: 
                    if stability_counter_start_time is None: 
                        stability_counter_start_time = current_time
                    
                    val_stable_stand_timer_display = f"{(current_time - stability_counter_start_time):.1f}/{STABILITY_CONFIRMATION_STANDING_SEC:.1f}s"
                    time_elapsed_stable = current_time - stability_counter_start_time
                    
                    if time_elapsed_stable >= STABILITY_CONFIRMATION_STANDING_SEC:
                        final_event_duration = current_time - overall_event_timer_start 
                        final_event_sway = np.std(sway_data_points_for_event) * image_width if len(sway_data_points_for_event) > 1 else 0.0
                        
                        print(f"[{time.strftime('%H:%M:%S')}] STAND EVENT COMPLETED! SCRIPT TERMINATING.")
                        print(f"    Total Time (from stable sit): {final_event_duration:.2f} seconds")
                        print(f"    Sway during entire event: {final_event_sway:.2f} pixels (approx)")
                        
                        display_status_main = f"Success: {final_event_duration:.2f}s, Sway: {final_event_sway:.1f}px"
                        temp_image_final = image_display.copy()
                        cv2.putText(temp_image_final, f"Status: {display_status_main}", (10, 20), font, 0.7, (0,255,0), line_type, cv2.LINE_AA)
                        cv2.putText(temp_image_final, f"Time: {final_event_duration:.2f}s", (10, 20 + int(25 * 1.2)), font, 0.7, (255,255,255), line_type, cv2.LINE_AA)
                        cv2.putText(temp_image_final, f"Sway: {final_event_sway:.1f}px", (10, 20 + int(25 * 2.4)), font, 0.7, (255,255,255), line_type, cv2.LINE_AA)
                        if results.pose_landmarks:
                            mp_drawing.draw_landmarks(temp_image_final, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                      mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                                      mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
                        cv2.imshow('Sit-to-Stand Analysis', temp_image_final)
                        cv2.waitKey(3000) 
                        cap.release(); cv2.destroyAllWindows(); sys.exit(0)
                else: 
                    if stability_counter_start_time is not None: 
                        pass 
                    stability_counter_start_time = None 
                    val_stable_stand_timer_display = "N/A"


            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image_display, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
        else: 
            display_status_main = "No Person Detected"
            if overall_event_timer_start is not None:
                print(f"[{time.strftime('%H:%M:%S')}] Person lost during active event. Resetting sequence.")
            current_state = STATE_WAITING_FOR_STABLE_SIT
            overall_event_timer_start = None; stability_counter_start_time = None
            prev_hip_y_norm_for_velocity = None; sway_data_points_for_event = []

        # --- Display Info Panel ---
        is_final_display_moment = False 
        if current_state == STATE_ATTEMPTING_STAND_SEQUENCE:
             if combined_condition_for_stable_stand and stability_counter_start_time is not None:
                if (current_time - stability_counter_start_time) >= STABILITY_CONFIRMATION_STANDING_SEC:
                    is_final_display_moment = True
        
        if not is_final_display_moment:
            panel_y_start = 20; line_h = 25
            def draw_text_with_comparison(label, val_str, condition_met, y_pos, is_crit=True):
                color = font_color_default
                if is_crit: color = font_color_pass if condition_met else font_color_fail if val_str != "N/A" else font_color_default
                else: color = font_color_pass if condition_met else font_color_default if val_str == "N/A" else font_color_default
                cv2.putText(image_display, f"{label}: {val_str}", (10, y_pos), font, font_scale, color, line_type, cv2.LINE_AA)

            cv2.putText(image_display, f"Status: {display_status_main}", (10, panel_y_start), font, 0.7, (0,255,255), line_type, cv2.LINE_AA)
            panel_y_start += int(line_h * 1.2)

            # Show overall event timer if it has started
            if overall_event_timer_start:
                current_event_time = current_time - overall_event_timer_start
                cv2.putText(image_display, f"Event Time: {current_event_time:.2f}s", (10, panel_y_start), font, font_scale, font_color_default, line_type, cv2.LINE_AA)
                panel_y_start += line_h

            if person_detected: 
                is_crit_display = (current_state == STATE_ATTEMPTING_STAND_SEQUENCE or current_state == STATE_MONITORING_SIT_AND_TIMING)
                
                cond_hip_knee_stand = False
                if temp_hip_y_norm is not None and temp_knee_y_norm is not None: cond_hip_knee_stand = temp_hip_y_norm < temp_knee_y_norm * HIPS_ABOVE_KNEES_FACTOR_STANDING
                draw_text_with_comparison(f"H<K*{HIPS_ABOVE_KNEES_FACTOR_STANDING:.2f}", f"{val_hip_y_n}<{val_knee_y_n}*{HIPS_ABOVE_KNEES_FACTOR_STANDING:.2f}", cond_hip_knee_stand, panel_y_start, is_crit_display); panel_y_start += line_h
                
                cond_knee_angle_stand = False
                if isinstance(temp_avg_knee_angle, (int, float)) and temp_avg_knee_angle != 0.0 : cond_knee_angle_stand = temp_avg_knee_angle > STANDING_KNEE_ANGLE_THRESH_MIN
                draw_text_with_comparison(f"KAng>{STANDING_KNEE_ANGLE_THRESH_MIN}", val_avg_knee_angle, cond_knee_angle_stand, panel_y_start, is_crit_display); panel_y_start += line_h
                
                cond_torso_angle_stand = False
                if isinstance(temp_torso_angle, (int, float)) and temp_torso_angle != 0.0: cond_torso_angle_stand = temp_torso_angle < STANDING_TORSO_ANGLE_THRESH_MAX_DEVIATION
                draw_text_with_comparison(f"TAng<{STANDING_TORSO_ANGLE_THRESH_MAX_DEVIATION}", val_torso_angle, cond_torso_angle_stand, panel_y_start, is_crit_display); panel_y_start += line_h
                
                draw_text_with_comparison("StdPose (Geom)", str(is_currently_standing_pose), is_currently_standing_pose, panel_y_start, is_crit_display); panel_y_start += line_h
                draw_text_with_comparison(f"|HVel|<{VELOCITY_THRESHOLD_STABLE_STAND:.2f}", val_hip_vy_n, velocity_condition_met_for_stable_stand, panel_y_start, is_crit_display); panel_y_start += line_h
                draw_text_with_comparison("PoseOK&VelOK", str(combined_condition_for_stable_stand), combined_condition_for_stable_stand, panel_y_start, is_crit_display); panel_y_start += line_h
                cv2.putText(image_display, f"StdStbTmr: {val_stable_stand_timer_display}", (10, panel_y_start), font, font_scale, font_color_default, line_type, cv2.LINE_AA); panel_y_start += line_h

                if current_state == STATE_WAITING_FOR_STABLE_SIT or current_state == STATE_MONITORING_SIT_AND_TIMING:
                     panel_y_start += line_h 
                     cv2.putText(image_display, "--- Sitting Debug ---", (10, panel_y_start), font, 0.5, (200,200,0), 1, cv2.LINE_AA); panel_y_start += int(line_h*0.8)
                     # ... (Sitting debug info can be added here if needed)

        cv2.imshow('Sit-to-Stand Analysis', image_display)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()