import cv2
import mediapipe as mp
import numpy as np
import time
import os

def analyze_right_foot_taps(video_path, 
                            show_video=False, 
                            raise_threshold=0.0002, 
                            drop_threshold=0.0002, 
                            heel_grounded_threshold=0.0005, 
                            heel_invalidation_threshold=0.002):
    """
    Analyzes a video to count the number of RIGHT foot taps with a grounded heel.

    This function is self-contained and designed to be imported. It processes a video
    to detect the motion of a RIGHT foot tap and returns the total count.

    Args:
        video_path (str): The full path to the video file to be analyzed.
        show_video (bool, optional): If True, displays the video with annotations.
                                     This is very useful for tuning thresholds. Defaults to False.
        raise_threshold (float): Sensitivity for detecting the upward foot movement.
        drop_threshold (float): Sensitivity for detecting the downward foot movement that completes a tap.
        heel_grounded_threshold (float): Max vertical movement allowed for the heel to be considered "grounded".
        heel_invalidation_threshold (float): If the heel moves more than this during a tap, the tap is cancelled.

    Returns:
        list: A list containing the total count of detected right foot taps as a string, e.g., ['15'].
              Returns None if the video cannot be opened.
    """
    # --- MediaPipe Pose Initialization ---
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open the video file '{video_path}' for processing.")
        return None

    # --- State and Result Variables for RIGHT foot ---
    tap_count = 0
    index_tap_in_progress = False
    previous_right_foot_index_y = None
    previous_right_heel_y = None
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # --- Frame Resizing for Efficiency ---
            # Process on a smaller frame for performance, but display this smaller frame.
            target_width = 640  # A slightly larger size for better viewing
            h_orig, w_orig, _ = frame.shape
            if w_orig == 0: continue
            scale = target_width / w_orig
            new_h, new_w = int(h_orig * scale), int(w_orig * scale)
            frame_proc = cv2.resize(frame, (new_w, new_h))
            
            h, w = frame_proc.shape[:2]
            if h == 0 or w == 0: continue

            # Convert to RGB and process with MediaPipe
            image_rgb = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False # Performance optimization
            results = pose.process(image_rgb)
            image_rgb.flags.writeable = True # Re-enable writing
            
            momentary_heel_is_grounded = False

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                try:
                    # *** USE RIGHT FOOT LANDMARKS ***
                    right_foot_index_lm = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
                    right_heel_lm = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL]

                    # Reset state if landmarks are not visible
                    if right_foot_index_lm.visibility < 0.6 or right_heel_lm.visibility < 0.6:
                        previous_right_foot_index_y, previous_right_heel_y, index_tap_in_progress = None, None, False
                    else:
                        current_right_foot_index_y = int(right_foot_index_lm.y * h)
                        current_right_heel_y = int(right_heel_lm.y * h)

                        # Check if heel is stable enough to be "grounded"
                        if previous_right_heel_y is not None:
                            if abs(current_right_heel_y - previous_right_heel_y) <= (heel_grounded_threshold * h):
                                momentary_heel_is_grounded = True
                        
                        # --- Main Tap Detection Logic ---
                        if previous_right_foot_index_y is not None:
                            index_movement_y = current_right_foot_index_y - previous_right_foot_index_y
                            # 1. Start a tap: Foot raises while heel is grounded
                            if not index_tap_in_progress and momentary_heel_is_grounded and index_movement_y < (-raise_threshold * h):
                                index_tap_in_progress = True
                            # 2. Complete a tap: Foot drops back down
                            elif index_tap_in_progress and index_movement_y > (drop_threshold * h):
                                tap_count += 1
                                index_tap_in_progress = False
                        
                        # 3. Invalidate a tap: If heel moves too much while foot is raised
                        if index_tap_in_progress and previous_right_heel_y is not None:
                            if abs(current_right_heel_y - previous_right_heel_y) > (heel_invalidation_threshold * h):
                                index_tap_in_progress = False

                        previous_right_foot_index_y = current_right_foot_index_y
                        previous_right_heel_y = current_right_heel_y
                
                except (IndexError, AttributeError):
                     # Reset state if landmarks are not found in this frame
                     previous_right_foot_index_y, previous_right_heel_y, index_tap_in_progress = None, None, False
            else:
                # Reset state if no pose is detected at all
                previous_right_foot_index_y, previous_right_heel_y, index_tap_in_progress = None, None, False

            # --- VISUALIZATION LOGIC ---
            if show_video:
                # Draw the skeleton
                mp_drawing.draw_landmarks(frame_proc, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
                
                # Highlight the key landmarks if visible
                if results.pose_landmarks:
                    if landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].visibility > 0.6:
                        heel_coords = (int(landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].x * w), int(landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].y * h))
                        cv2.circle(frame_proc, heel_coords, 8, (255, 0, 255), -1) # Magenta heel
                    if landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].visibility > 0.6:
                        index_coords = (int(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x * w), int(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * h))
                        cv2.circle(frame_proc, index_coords, 8, (255, 255, 0), -1) # Cyan foot index
                
                # Create a status panel
                overlay = frame_proc.copy()
                cv2.rectangle(overlay, (5, 5), (320, 100), (20, 20, 20), -1)
                alpha = 0.7
                frame_proc = cv2.addWeighted(overlay, alpha, frame_proc, 1 - alpha, 0)
                
                # Display Tap Count
                cv2.putText(frame_proc, f"RIGHT Foot Taps: {tap_count}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Display Heel Status
                heel_status_text = f"Heel Grounded: {momentary_heel_is_grounded}"
                heel_color = (0, 255, 0) if momentary_heel_is_grounded else (0, 0, 255)
                cv2.putText(frame_proc, heel_status_text, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, heel_color, 2)

                # Display Tap Progress
                tap_status_text = f"Tap In Progress: {index_tap_in_progress}"
                tap_color = (100, 255, 100) if index_tap_in_progress else (200, 200, 200)
                cv2.putText(frame_proc, tap_status_text, (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, tap_color, 2)

                # Show the final frame
                cv2.imshow("Right Foot Tap Analysis", frame_proc)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        cap.release()
        pose.close()
        if show_video: cv2.destroyAllWindows()

    print(f"\n--- Analysis Complete for {os.path.basename(video_path)} ---")
    print(f">>> Total RIGHT Foot Taps Detected: {tap_count} <<<")
    return [str(tap_count)]