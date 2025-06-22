import cv2
import mediapipe as mp
import numpy as np
import time
import os

def analyze_foot_taps(video_path, 
                      show_video=False, 
                      print_debug_logs=False,
                      raise_threshold=0.0005, 
                      drop_threshold=0.0005, 
                      heel_grounded_threshold=0.0015, 
                      heel_invalidation_threshold=0.008):
    """
    Analyzes a video to count the number of foot taps with a grounded heel.

    This function is self-contained and designed to be imported. It processes a video
    frame by frame to detect the specific motion of a foot tap (raising the forefoot
    while the heel remains stable) and returns the total count.

    Args:
        video_path (str): The full path to the video file to be analyzed.
        show_video (bool, optional): If True, displays the video with annotations during 
                                     processing. Requires a GUI. Defaults to False.
        print_debug_logs (bool, optional): If True, prints detailed frame-by-frame logic 
                                           to the console for debugging. Defaults to False.
        raise_threshold (float, optional): Normalized Y-movement required for the foot index 
                                           to be considered 'raising'. Defaults to 0.0005.
        drop_threshold (float, optional): Normalized Y-movement required for the foot index 
                                          to be considered 'dropping' to count a tap. Defaults to 0.0005.
        heel_grounded_threshold (float, optional): Maximum allowed Y-movement for the heel to be 
                                                   considered 'grounded'. Defaults to 0.0015.
        heel_invalidation_threshold (float, optional): Y-movement threshold for the heel that will
                                                       invalidate a tap-in-progress. Defaults to 0.008.

    Returns:
        dict: A dictionary containing the analysis results, primarily the 'tap_count',
              or None if the video cannot be opened.
    """
    # --- MediaPipe Pose Initialization ---
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(model_complexity=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

    # --- Video Handling ---
    if not os.path.exists(video_path):
        print(f"ERROR: Could not find the video file: '{video_path}'")
        return None
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open the video file '{video_path}' for processing.")
        return None

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if h == 0 or w == 0:
        print(f"ERROR: Video file '{video_path}' has invalid dimensions (0 height or width).")
        cap.release()
        pose.close()
        return None

    # --- State and Result Variables ---
    tap_count = 0
    index_tap_in_progress = False
    previous_right_foot_index_y = None
    previous_right_heel_y = None
    

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # --- MediaPipe Processing ---
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            momentary_heel_is_grounded = False

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                try:
                    right_foot_index_lm = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
                    right_heel_lm = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL]

                    if right_foot_index_lm.visibility < 0.6 or right_heel_lm.visibility < 0.6:
                        # Reset state if landmarks are lost
                        previous_right_foot_index_y = None
                        previous_right_heel_y = None
                        index_tap_in_progress = False
                    else:
                        current_right_foot_index_y = int(right_foot_index_lm.y * h)
                        current_right_heel_y = int(right_heel_lm.y * h)

                        # Check Heel Grounded Status
                        if previous_right_heel_y is not None:
                            if abs(current_right_heel_y - previous_right_heel_y) <= (heel_grounded_threshold * h):
                                momentary_heel_is_grounded = True
                        
                        # Process Tap Logic
                        if previous_right_foot_index_y is not None:
                            index_movement_y = current_right_foot_index_y - previous_right_foot_index_y
                            
                            # START TAP
                            if not index_tap_in_progress and momentary_heel_is_grounded and \
                               index_movement_y < (-raise_threshold * h):
                                index_tap_in_progress = True
                            
                            # COUNT TAP
                            elif index_tap_in_progress and index_movement_y > (drop_threshold * h):
                                tap_count += 1
                                index_tap_in_progress = False
                        
                        # INVALIDATE TAP if heel moves too much
                        if index_tap_in_progress and previous_right_heel_y is not None:
                            if abs(current_right_heel_y - previous_right_heel_y) > (heel_invalidation_threshold * h):
                                index_tap_in_progress = False

                        # Update Previous Positions for the next frame
                        previous_right_foot_index_y = current_right_foot_index_y
                        previous_right_heel_y = current_right_heel_y
                
                except (IndexError, AttributeError):
                     # Reset state on landmark access error
                     previous_right_foot_index_y, previous_right_heel_y, index_tap_in_progress = None, None, False
            
            else: # No landmarks detected, reset state
                previous_right_foot_index_y, previous_right_heel_y, index_tap_in_progress = None, None, False

            # --- Visualization (only if enabled) ---
            if show_video:
                status_text = []
                if momentary_heel_is_grounded: status_text.append("Heel OK")
                if index_tap_in_progress: status_text.append("Index Tapping")
                
                cv2.putText(frame, f"Taps: {tap_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, " | ".join(status_text), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                cv2.imshow("Foot Tap Analysis", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        # --- Cleanup and prepare final results ---
        cap.release()
        pose.close()
        if show_video:
            cv2.destroyAllWindows()
            
        

    return tap_count