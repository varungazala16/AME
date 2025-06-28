import cv2
import mediapipe as mp
import numpy as np
import time
import os

def analyze_right_foot_taps(video_path, show_video=False):
    """Wrapper function to count right foot movements with jitter elimination."""
    return _analyze_foot_taps_robust(video_path, side='right', show_video=show_video)

def analyze_left_foot_taps(video_path, show_video=False):
    """Wrapper function to count left foot movements with jitter elimination."""
    return _analyze_foot_taps_robust(video_path, side='left', show_video=show_video)

def _analyze_foot_taps_robust(video_path, 
                              side,
                              show_video=False, 
                              jitter_threshold=0.0015,
                              raise_threshold=0.004, 
                              drop_threshold=0.004):
    """
    Robustly analyzes foot taps based ONLY on toe movement, using a two-threshold
    system to eliminate jitter while the foot is static.

    Args:
        video_path (str): The path to the video file.
        side (str): 'left' or 'right'.
        show_video (bool): If True, displays the analysis window.
        jitter_threshold (float): Movement below this is ignored as noise for the toe.
        raise_threshold (float): Movement must be larger than this to start a tap. Must be > jitter_threshold.
        drop_threshold (float): Movement must be larger than this to complete a tap.
    """
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open the video file '{video_path}' for processing.")
        return None

    # --- Landmark and Variable Setup based on side ---
    foot_index_enum = mp_pose.PoseLandmark.LEFT_FOOT_INDEX if side == 'left' else mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
    
    tap_count = 0
    tap_in_progress = False
    previous_foot_index_y = None
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            target_width = 640
            h_orig, w_orig, _ = frame.shape
            if w_orig == 0: continue
            scale = target_width / w_orig
            new_h, new_w = int(h_orig * scale), int(w_orig * scale)
            frame_proc = cv2.resize(frame, (new_w, new_h))
            
            h, w = frame_proc.shape[:2]
            if h == 0 or w == 0: continue

            image_rgb = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            toe_is_stable = True

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                try:
                    foot_index_lm = landmarks[foot_index_enum.value]

                    if foot_index_lm.visibility < 0.6:
                        previous_foot_index_y, tap_in_progress = None, False
                    else:
                        current_foot_index_y = int(foot_index_lm.y * h)

                        if previous_foot_index_y is not None:
                            # --- JITTER-PROOF LOGIC (Simplified) ---
                            index_movement_y = current_foot_index_y - previous_foot_index_y
                            
                            jitter_px = jitter_threshold * h
                            raise_px = raise_threshold * h
                            drop_px = drop_threshold * h
                            
                            # STEP 1: Check if the toe is just jittering (inside the dead zone).
                            if abs(index_movement_y) < jitter_px:
                                toe_is_stable = True
                            else:
                                toe_is_stable = False
                                # STEP 2: If the toe is moving intentionally, check for a tap.
                                # Condition to START a tap: Large upward movement.
                                if not tap_in_progress and index_movement_y < -raise_px:
                                    tap_in_progress = True
                                # Condition to COMPLETE a tap: Large downward movement.
                                elif tap_in_progress and index_movement_y > drop_px:
                                    tap_count += 1
                                    tap_in_progress = False

                        previous_foot_index_y = current_foot_index_y
                
                except (IndexError, AttributeError):
                     previous_foot_index_y, tap_in_progress = None, False
            else:
                previous_foot_index_y, tap_in_progress = None, False

            if show_video:
                mp_drawing.draw_landmarks(frame_proc, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                if results.pose_landmarks:
                    if landmarks[foot_index_enum.value].visibility > 0.6:
                        index_coords = (int(landmarks[foot_index_enum.value].x * w), int(landmarks[foot_index_enum.value].y * h))
                        cv2.circle(frame_proc, index_coords, 8, (255, 255, 0), -1) # Cyan foot index
                
                overlay = frame_proc.copy()
                cv2.rectangle(overlay, (5, 5), (320, 100), (20, 20, 20), -1)
                alpha = 0.7
                frame_proc = cv2.addWeighted(overlay, alpha, frame_proc, 1 - alpha, 0)
                
                cv2.putText(frame_proc, f"{side.upper()} Foot Taps: {tap_count}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                toe_color = (0, 255, 0) if toe_is_stable else (0, 165, 255)
                cv2.putText(frame_proc, f"Toe Stable: {toe_is_stable}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, toe_color, 2)

                tap_color = (100, 255, 100) if tap_in_progress else (200, 200, 200)
                cv2.putText(frame_proc, f"Tap In Progress: {tap_in_progress}", (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tap_color, 2)

                cv2.imshow(f"{side.capitalize()} Foot Tap Analysis", frame_proc)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

    finally:
        cap.release()
        pose.close()
        if show_video: cv2.destroyAllWindows()

    print(f"\n--- Analysis Complete for {os.path.basename(video_path)} ({side.upper()}) ---")
    print(f">>> Total {side.upper()} Foot Taps Detected: {tap_count} <<<")
    return [str(tap_count)]

