import cv2
import mediapipe as mp
import numpy as np
import os

def analyze_romberg_outstretch(video_path, 
                               show_video=False,
                               sway_tolerance=0.07, 
                               arm_height_threshold=0.15,
                               arm_drop_threshold=0.20):
    """
    Analyzes a Romberg test where the timer starts when arms are raised
    and stops on shoulder sway or if arms are dropped.

    Args:
        video_path (str): The full path to the video file.
        show_video (bool, optional): If True, displays the video with visual analysis.
        sway_tolerance (float): Allowed horizontal sway for shoulders (fraction of width).
        arm_height_threshold (float): Max vertical distance between wrist and shoulder to
                                      start the test (fraction of torso height).
        arm_drop_threshold (float): Max vertical distance between wrist and hip to be
                                    considered a failure (fraction of torso height).

    Returns:
        list: A list containing [duration_held_str, was_failure_detected_str].
    """
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video: {video_path}")
        return ["0.00", "True"]

    # --- State Machine Definitions ---
    STATE_WAITING_FOR_ARMS_UP = 0
    STATE_HOLDING_POSE = 1
    STATE_FAILED = 2
    current_state = STATE_WAITING_FOR_ARMS_UP
    
    # --- State and Result Variables ---
    hold_start_time = None
    correct_posture_time = 0.0
    bad_posture_detected = False
    failure_reason = ""
    
    # This will be set once the hold begins
    reference_shoulder_x = None

    while True:
        ret, frame = cap.read()
        if not ret: 
            if current_state == STATE_HOLDING_POSE:
                end_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                correct_posture_time = end_time - hold_start_time
            break

        current_video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            # Get all necessary landmarks
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

            # Ensure all primary landmarks are visible
            all_visible = (left_shoulder.visibility > 0.7 and right_shoulder.visibility > 0.7 and
                           left_wrist.visibility > 0.7 and right_wrist.visibility > 0.7 and
                           left_hip.visibility > 0.7 and right_hip.visibility > 0.7)

            if all_visible:
                # Calculate reference heights for thresholds
                avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
                avg_hip_y = (left_hip.y + right_hip.y) / 2
                torso_height = abs(avg_hip_y - avg_shoulder_y)

                # --- State Machine Logic ---
                if current_state == STATE_WAITING_FOR_ARMS_UP:
                    # Check if both arms are raised to shoulder height
                    left_arm_up = abs(left_wrist.y - avg_shoulder_y) < (torso_height * arm_height_threshold)
                    right_arm_up = abs(right_wrist.y - avg_shoulder_y) < (torso_height * arm_height_threshold)

                    if left_arm_up and right_arm_up:
                        # --- START THE TEST ---
                        current_state = STATE_HOLDING_POSE
                        hold_start_time = current_video_time
                        # Set the reference for sway detection at this exact moment
                        reference_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2

                elif current_state == STATE_HOLDING_POSE:
                    # --- CHECK FOR FAILURE CONDITIONS ---
                    correct_posture_time = current_video_time - hold_start_time
                    
                    # 1. Check for Shoulder Sway
                    current_avg_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
                    if abs(current_avg_shoulder_x - reference_shoulder_x) > sway_tolerance:
                        bad_posture_detected = True
                        failure_reason = "Shoulder Sway"
                    
                    # 2. Check for Arm Drop (relative to hip)
                    left_arm_down = abs(left_wrist.y - left_hip.y) < (torso_height * arm_drop_threshold)
                    right_arm_down = abs(right_wrist.y - right_hip.y) < (torso_height * arm_drop_threshold)
                    if left_arm_down or right_arm_down:
                        bad_posture_detected = True
                        failure_reason = "Arms Dropped"
                    
                    if bad_posture_detected:
                        current_state = STATE_FAILED
                        break

        # --- VISUALIZATION ---
        if show_video:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (5, 5), (400, 70), (20, 20, 20), -1)
            alpha = 0.7
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            
            state_map = {
                STATE_WAITING_FOR_ARMS_UP: ("Waiting for Arms Up", (0, 255, 255)),
                STATE_HOLDING_POSE: ("Holding Pose", (0, 255, 0)),
                STATE_FAILED: (f"FAILED - {failure_reason}", (0, 0, 255))
            }
            status_text, color = state_map.get(current_state)

            cv2.putText(frame, f"Status: {status_text}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Hold Time: {correct_posture_time:.2f}s", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Romberg Test Analysis", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    pose.close()
    if show_video:
        cv2.destroyAllWindows()
    
    print(f"\n--- Analysis Complete ---")
    if bad_posture_detected:
        print(f"Test failed at {correct_posture_time:.2f}s due to: {failure_reason}")
    else:
        print(f"Test held successfully for {correct_posture_time:.2f}s")

    return [str(round(correct_posture_time, 2)), str(bad_posture_detected)]