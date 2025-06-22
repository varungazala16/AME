import cv2
import mediapipe as mp
import numpy as np
import time
import math
import os

def analyze_one_leg_stand(video_path, 
                          show_video=False, 
                          ankle_height_diff_threshold=0.06, 
                          sway_threshold=0.01, 
                          visibility_threshold=0.35, 
                          sway_tracking_point='hip_midpoint'):
    """
    Analyzes a video to assess one-leg stand performance.

    This is a self-contained function designed to be imported and used in other scripts.
    It calculates the total time a person stands on one foot, logs balance loss events,
    and tracks significant body sway.

    Args:
        video_path (str): The full path to the video file to be analyzed.
        show_video (bool, optional): If True, displays the video with annotations during processing. 
                                     This requires a GUI environment. Defaults to False.
        ankle_height_diff_threshold (float, optional): Normalized vertical distance between ankles
                                                       to detect a one-foot stance. Defaults to 0.06.
        sway_threshold (float, optional): Normalized distance from the starting reference point 
                                          to be considered a significant sway event. Defaults to 0.01.
        visibility_threshold (float, optional): Minimum confidence score from MediaPipe for a landmark 
                                                to be considered visible. Defaults to 0.35.
        sway_tracking_point (str, optional): The body point to track for sway. 
                                             Options: 'hip_midpoint', 'standing_ankle', 'shoulder_midpoint'.
                                             Defaults to 'hip_midpoint'.

    Returns:
        dict: A dictionary containing the analysis results, or None if the video cannot be opened.
              Keys include 'total_standing_time', 'foot_down_events', 'sway_events', etc.
    """
    # --- MediaPipe Pose Initialization (inside the function) ---
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5,
        model_complexity=1 
    )

    # --- Video Handling ---
    if not os.path.exists(video_path):
        print(f"Error: Could not find video file at the specified path: {video_path}")
        return None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {video_path}")
        return None
        
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- Helper Functions (nested for self-containment) ---
    def calculate_distance(point1, point2):
        if point1 is None or point2 is None: return 0.0
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def get_landmark(landmarks, landmark_enum):
        if landmarks and landmarks.landmark[landmark_enum.value].visibility > visibility_threshold:
            return landmarks.landmark[landmark_enum.value]
        return None

    # --- State and Result Variables ---
    is_standing_one_foot = False
    total_standing_time = 0.0
    reference_point_sway = None
    sway_events = []
    foot_down_events = []
    previous_time_sec = 0.0
    
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            delta_time = current_time_sec - previous_time_sec
            # Handle potential video timestamp errors or looping
            if delta_time <= 0:
                if cap.get(cv2.CAP_PROP_POS_FRAMES) > 1: # Avoid issues on first frame
                    previous_time_sec = current_time_sec
                continue

            # --- Pose Processing ---
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False # Performance optimization
            results = pose.process(image_rgb)
            
            currently_detected_one_foot = False
            standing_ankle_for_frame = None

            if results.pose_landmarks:
                landmarks = results.pose_landmarks
                left_ankle = get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)
                right_ankle = get_landmark(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE)
                
                if left_ankle and right_ankle:
                    if abs(left_ankle.y - right_ankle.y) > ankle_height_diff_threshold:
                        currently_detected_one_foot = True
                        standing_ankle_for_frame = left_ankle if left_ankle.y > right_ankle.y else right_ankle
            
            # --- State Logic & Timer ---
            if currently_detected_one_foot:
                total_standing_time += delta_time
                if not is_standing_one_foot: # Start of a new segment
                    is_standing_one_foot = True
                    # Set sway reference point
                    left_hip = get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
                    right_hip = get_landmark(landmarks, mp_pose.PoseLandmark.RIGHT_HIP)
                    left_shoulder = get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
                    right_shoulder = get_landmark(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER)

                    if sway_tracking_point == 'hip_midpoint' and left_hip and right_hip:
                        reference_point_sway = ((left_hip.x + right_hip.x)/2, (left_hip.y + right_hip.y)/2)
                    elif sway_tracking_point == 'shoulder_midpoint' and left_shoulder and right_shoulder:
                        reference_point_sway = ((left_shoulder.x + right_shoulder.x)/2, (left_shoulder.y + right_shoulder.y)/2)
                    else: # Default or 'standing_ankle'
                        reference_point_sway = (standing_ankle_for_frame.x, standing_ankle_for_frame.y)
                
                # --- Sway Calculation ---
                if reference_point_sway and results.pose_landmarks:
                    # Recalculate current points for accuracy
                    left_hip = get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
                    right_hip = get_landmark(landmarks, mp_pose.PoseLandmark.RIGHT_HIP)
                    left_shoulder = get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
                    right_shoulder = get_landmark(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER)

                    current_sway_point = None
                    if sway_tracking_point == 'hip_midpoint' and left_hip and right_hip:
                        current_sway_point = ((left_hip.x + right_hip.x)/2, (left_hip.y + right_hip.y)/2)
                    elif sway_tracking_point == 'shoulder_midpoint' and left_shoulder and right_shoulder:
                        current_sway_point = ((left_shoulder.x + right_shoulder.x)/2, (left_shoulder.y + right_shoulder.y)/2)
                    else:
                        current_sway_point = (standing_ankle_for_frame.x, standing_ankle_for_frame.y)
                    
                    sway_distance = calculate_distance(reference_point_sway, current_sway_point)
                    if sway_distance > sway_threshold:
                        sway_events.append({'time': current_time_sec, 'level': sway_distance})
            
            else: # Not standing on one foot in this frame
                if is_standing_one_foot: # End of a segment
                    is_standing_one_foot = False
                    foot_down_events.append(current_time_sec)
                    reference_point_sway = None

            # --- Visualization (only if enabled) ---
            if show_video:
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) # Convert back for drawing
                if results.pose_landmarks:
                    mp_drawing = mp.solutions.drawing_utils
                    mp_drawing.draw_landmarks(image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                status_text = "Standing on One Foot" if is_standing_one_foot else "Both Feet Down"
                cv2.putText(image_bgr, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(image_bgr, f"Time: {total_standing_time:.2f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('One-Leg Stand Analysis', image_bgr)
                if cv2.waitKey(1) & 0xFF == 27: # Allow exit with ESC key
                    break

            previous_time_sec = current_time_sec

    finally:
        # --- Cleanup and prepare final results ---
        cap.release()
        pose.close()
        if show_video:
            cv2.destroyAllWindows()
            
        results_dict = {
            'video_path': video_path,
            'total_standing_time': round(total_standing_time, 2),
            'foot_down_events_count': len(foot_down_events),
            'foot_down_timestamps': [round(t, 2) for t in foot_down_events],
            'significant_sway_events_count': len(sway_events),
            'sway_events': [{'time': round(e['time'], 2), 'level': round(e['level'], 4)} for e in sway_events]
        }

    return results_dict


