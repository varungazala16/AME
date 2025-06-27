import cv2
import mediapipe as mp
import numpy as np
import os
import math

def analyze_right_leg_stand(video_path, 
                            show_video=False, 
                            ankle_height_diff_threshold=0.06, 
                            sway_threshold=0.01, 
                            visibility_threshold=0.35, 
                            sway_tracking_point='hip_midpoint'):
    """
    Analyzes a video to assess one-leg stand performance on the RIGHT LEG ONLY.
    The timer accumulates time only when the person is correctly standing on one leg,
    pausing when the foot is down and resuming when it's lifted again.

    Args:
        video_path (str): The full path to the video file to be analyzed.
        show_video (bool): If True, displays the video with rich visual analysis.
        (Other args): Optional parameters for tuning sensitivity.

    Returns:
        list: A list containing the total cumulative standing time as a string, e.g., ['25.43'].
              Returns None if the video cannot be opened.
    """
    # --- MediaPipe Pose Initialization ---
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {video_path}")
        return None

    # --- Helper Functions ---
    def get_landmark(landmarks, landmark_enum):
        if landmarks and landmark_enum.value < len(landmarks.landmark) and landmarks.landmark[landmark_enum.value].visibility > visibility_threshold:
            return landmarks.landmark[landmark_enum.value]
        return None

    # --- State and Result Variables ---
    is_standing_on_right_leg = False
    total_standing_time = 0.0
    foot_down_events = []
    previous_time_sec = 0.0
    
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            delta_time = current_time_sec - previous_time_sec
            if delta_time <= 0:
                if cap.get(cv2.CAP_PROP_POS_FRAMES) > 1:
                    previous_time_sec = current_time_sec
                continue

            h, w, _ = frame.shape
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            currently_detected_right_stand = False
            left_ankle, right_ankle = None, None # Initialize for the frame

            if results.pose_landmarks:
                landmarks = results.pose_landmarks
                left_ankle = get_landmark(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)
                right_ankle = get_landmark(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE)
                
                if left_ankle and right_ankle:
                    # Logic for RIGHT leg stand: right ankle must be lower than left ankle.
                    # In image coordinates, a higher position has a lower y-value.
                    # So, left_ankle.y should be significantly smaller than right_ankle.y.
                    if (right_ankle.y - left_ankle.y) > ankle_height_diff_threshold:
                        currently_detected_right_stand = True
            
            # --- This is the key logic for the cumulative timer ---
            if currently_detected_right_stand:
                # If standing correctly, add the frame's duration to the total time.
                total_standing_time += delta_time
                is_standing_on_right_leg = True
            else:
                # If not standing correctly (foot is down or landmarks not visible)
                # check if this is the *first frame* of the foot-down event.
                if is_standing_on_right_leg:
                    # The state changed from standing to not-standing, so record a "foot down" event.
                    foot_down_events.append(current_time_sec)
                # The timer is NOT incremented.
                is_standing_on_right_leg = False

            # --- VISUALIZATION ---
            if show_video:
                # Draw the pose skeleton
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
                
                # Highlight key landmarks and draw reference line
                if left_ankle and right_ankle:
                    # Right Ankle (Supporting) - Magenta
                    cv2.circle(frame, (int(right_ankle.x * w), int(right_ankle.y * h)), 10, (255, 0, 255), -1)
                    # Left Ankle (Raised) - Cyan
                    cv2.circle(frame, (int(left_ankle.x * w), int(left_ankle.y * h)), 10, (255, 255, 0), -1)
                    # Draw a reference "ground" line based on the supporting ankle
                    cv2.line(frame, (0, int(right_ankle.y * h)), (w, int(right_ankle.y * h)), (0, 255, 255), 2)
                
                # Create a status panel
                overlay = frame.copy()
                cv2.rectangle(overlay, (5, 5), (350, 100), (20, 20, 20), -1)
                alpha = 0.7
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                
                # Display Status Text
                status_text = "Standing on RIGHT Leg" if is_standing_on_right_leg else "Both Feet Down"
                status_color = (0, 255, 0) if is_standing_on_right_leg else (0, 255, 255)
                cv2.putText(frame, f"Status: {status_text}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                # Display Cumulative Timer
                cv2.putText(frame, f"Time: {total_standing_time:.2f}s", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display Foot Down Count
                cv2.putText(frame, f"Foot Downs: {len(foot_down_events)}", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow('Right Leg Stand Analysis', frame)
                if cv2.waitKey(1) & 0xFF == 27: break # Press ESC to quit

            previous_time_sec = current_time_sec

    finally:
        cap.release()
        pose.close()
        if show_video: cv2.destroyAllWindows()
            
        total_standing_time_str = str(round(total_standing_time, 2))
        foot_down_count = len(foot_down_events)

        print(f"\n--- Analysis Complete for {os.path.basename(video_path)} ---")
        print(f">>> Total Cumulative Stand Time: {total_standing_time_str} seconds")
        print(f">>> Number of Times Foot Touched Down: {foot_down_count}")

    # The function is expected to return only the time string in a list
    return [total_standing_time_str]