import cv2
import mediapipe as mp
import numpy as np
import time
import math
import os

def analyze_right_leg_stand(video_path, 
                            show_video=False, 
                            ankle_height_diff_threshold=0.06, 
                            sway_threshold=0.01, 
                            visibility_threshold=0.35, 
                            sway_tracking_point='hip_midpoint'):
    """
    Analyzes a video to assess one-leg stand performance on the RIGHT LEG ONLY.

    Args:
        video_path (str): The full path to the video file to be analyzed.
        (Other args): Optional parameters for tuning sensitivity.

    Returns:
        str: A single string containing all analysis metrics, separated by ' | '.
             Returns None if the video cannot be opened.
    """
    # --- MediaPipe Pose Initialization ---
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)

    # --- Video Handling ---
    if not os.path.exists(video_path):
        print(f"Error: Could not find video file: {video_path}")
        return None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {video_path}")
        return None

    # --- Helper Functions ---
    def get_landmark(landmarks, landmark_enum):
        if landmarks and landmarks.landmark[landmark_enum.value].visibility > visibility_threshold:
            return landmarks.landmark[landmark_enum.value]
        return None

    # --- State and Result Variables ---
    is_standing_on_right_leg = False
    total_standing_time = 0.0
    reference_point_sway = None
    sway_events = []
    foot_down_events = []
    previous_time_sec = 0.0
    
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            delta_time = current_time_sec - previous_time_sec
            if delta_time <= 0:
                if cap.get(cv2.CAP_PROP_POS_FRAMES) > 1: previous_time_sec = current_time_sec
                continue

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            currently_detected_right_stand = False
            if results.pose_landmarks:
                landmarks = results.pose_landmarks
                left_ankle = get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)
                right_ankle = get_landmark(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE)
                
                if left_ankle and right_ankle:
                    # Logic for RIGHT leg stand: right ankle must be lower than left ankle
                    if abs(left_ankle.y - right_ankle.y) > ankle_height_diff_threshold and right_ankle.y > left_ankle.y:
                        currently_detected_right_stand = True
            
            if currently_detected_right_stand:
                total_standing_time += delta_time
                if not is_standing_on_right_leg:
                    is_standing_on_right_leg = True
                    # Logic for sway can be added here if needed for this specific leg
            else:
                if is_standing_on_right_leg:
                    is_standing_on_right_leg = False
                    foot_down_events.append(current_time_sec)

            if show_video:
                status_text = "Standing on RIGHT Leg" if is_standing_on_right_leg else "Not on Right Leg"
                cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Time: {total_standing_time:.2f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Right Leg Stand Analysis', frame)
                if cv2.waitKey(1) & 0xFF == 27: break

            previous_time_sec = current_time_sec

    finally:
        cap.release()
        pose.close()
        if show_video: cv2.destroyAllWindows()
            
        # Prepare final metrics before converting to a single string
        total_standing_time_str = str(round(total_standing_time, 2))
        foot_down_count_str = str(len(foot_down_events))
        foot_down_timestamps_str = str([round(t, 2) for t in foot_down_events])
        
        # NOTE: Sway logic was simplified for clarity, can be added back if needed
        sway_count_str = "0" # Placeholder
        sway_events_str = "[]" # Placeholder

        concatenated_string = " | ".join([
            total_standing_time_str,
            foot_down_count_str,
            foot_down_timestamps_str,
            sway_count_str,
            sway_events_str
        ])

    return [total_standing_time_str]