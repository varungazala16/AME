import cv2
import mediapipe as mp
import numpy as np
import time
import os

def analyze_left_foot_taps(video_path, 
                           show_video=False, 
                           raise_threshold=0.00028, 
                           drop_threshold=0.00028, 
                           heel_grounded_threshold=0.0005, 
                           heel_invalidation_threshold=0.002):
    """
    Analyzes a video to count the number of LEFT foot taps with a grounded heel.

    This function is self-contained and designed to be imported. It processes a video
    to detect the motion of a LEFT foot tap and returns the total count.

    Args:
        video_path (str): The full path to the video file to be analyzed.
        show_video (bool, optional): If True, displays the video with annotations. Defaults to False.
        (Other args): Thresholds for sensitivity tuning.

    Returns:
        str: The total count of detected left foot taps as a string.
             Returns None if the video cannot be opened.
    """
    # --- MediaPipe Pose Initialization ---
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(model_complexity=0, min_detection_confidence=0.7, min_tracking_confidence=0.7)
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open the video file '{video_path}' for processing.")
        return None

    # --- State and Result Variables for LEFT foot ---
    tap_count = 0
    index_tap_in_progress = False
    previous_left_foot_index_y = None
    previous_left_heel_y = None
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            target_width = 480  # A good default for pose estimation
            h_orig, w_orig, _ = frame.shape
            scale = target_width / w_orig
            new_h, new_w = int(h_orig * scale), int(w_orig * scale)
            frame_resized = cv2.resize(frame, (new_w, new_h))
            frame_resized = cv2.resize(frame, (new_w, new_h))
            
            h, w = frame_resized.shape[:2]
            if h == 0 or w == 0:
                cap.release()
                pose.close()
                return None

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            momentary_heel_is_grounded = False

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                try:
                    # *** USE LEFT FOOT LANDMARKS ***
                    left_foot_index_lm = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
                    left_heel_lm = landmarks[mp_pose.PoseLandmark.LEFT_HEEL]

                    if left_foot_index_lm.visibility < 0.6 or left_heel_lm.visibility < 0.6:
                        previous_left_foot_index_y, previous_left_heel_y, index_tap_in_progress = None, None, False
                    else:
                        current_left_foot_index_y = int(left_foot_index_lm.y * h)
                        current_left_heel_y = int(left_heel_lm.y * h)

                        if previous_left_heel_y is not None:
                            if abs(current_left_heel_y - previous_left_heel_y) <= (heel_grounded_threshold * h):
                                momentary_heel_is_grounded = True
                        
                        if previous_left_foot_index_y is not None:
                            index_movement_y = current_left_foot_index_y - previous_left_foot_index_y
                            if not index_tap_in_progress and momentary_heel_is_grounded and index_movement_y < (-raise_threshold * h):
                                index_tap_in_progress = True
                            elif index_tap_in_progress and index_movement_y > (drop_threshold * h):
                                tap_count += 1
                                index_tap_in_progress = False
                        
                        if index_tap_in_progress and previous_left_heel_y is not None:
                            if abs(current_left_heel_y - previous_left_heel_y) > (heel_invalidation_threshold * h):
                                index_tap_in_progress = False

                        previous_left_foot_index_y = current_left_foot_index_y
                        previous_left_heel_y = current_left_heel_y
                
                except (IndexError, AttributeError):
                     previous_left_foot_index_y, previous_left_heel_y, index_tap_in_progress = None, None, False
            
            else:
                previous_left_foot_index_y, previous_left_heel_y, index_tap_in_progress = None, None, False

            if show_video:
                cv2.putText(frame, f"LEFT Foot Taps: {tap_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.imshow("Left Foot Tap Analysis", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        cap.release()
        pose.close()
        if show_video: cv2.destroyAllWindows()

    return [str(tap_count)]