import cv2
import mediapipe as mp
import numpy as np

def analyze_romberg_outstretch(video_path, tolerance_shoulder=0.1):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video: {video_path}")
        return ["0.00", "True"]

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-2:
        fps = 30.0
    # frame_duration_sec is no longer needed for the main timer

    correct_posture_time = 0.0
    bad_posture_detected = False

    # --- Grace Period Logic ---
    GRACE_PERIOD_SEC = 3.0
    is_in_grace_period = True
    grace_period_start_time = None
    grace_period_left_shoulders = []
    grace_period_right_shoulders = []
    
    reference_left_shoulder = None
    reference_right_shoulder = None
    
    # *** NEW: Timer for the actual stable hold period ***
    stable_hold_start_time = None

    while True:
        ret, frame = cap.read()
        if not ret: 
            # If video ends while holding a stable pose, calculate final time
            if stable_hold_start_time is not None and not bad_posture_detected:
                # Use the last known video time as the end time
                end_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                correct_posture_time = end_time - stable_hold_start_time
            break

        current_video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

            if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5:
                if is_in_grace_period:
                    if grace_period_start_time is None:
                        grace_period_start_time = current_video_time
                    
                    grace_period_left_shoulders.append((left_shoulder.x, left_shoulder.y))
                    grace_period_right_shoulders.append((right_shoulder.x, right_shoulder.y))
                    
                    if current_video_time - grace_period_start_time >= GRACE_PERIOD_SEC:
                        if grace_period_left_shoulders:
                            avg_left = np.mean(grace_period_left_shoulders, axis=0)
                            avg_right = np.mean(grace_period_right_shoulders, axis=0)
                            reference_left_shoulder = type('obj', (object,), {'x': avg_left[0], 'y': avg_left[1]})
                            reference_right_shoulder = type('obj', (object,), {'x': avg_right[0], 'y': avg_right[1]})
                            
                            # *** TIMER FIX: Record the start time of the stable hold ***
                            stable_hold_start_time = current_video_time
                            print(f"DEBUG: Grace period ended. Stable reference set. True timer started at {stable_hold_start_time:.2f}s.")
                        
                        is_in_grace_period = False

                else: # Analysis phase
                    if reference_left_shoulder:
                        left_moved = abs(left_shoulder.x - reference_left_shoulder.x) > tolerance_shoulder
                        right_moved = abs(right_shoulder.x - reference_right_shoulder.x) > tolerance_shoulder

                        if left_moved or right_moved:
                            bad_posture_detected = True
                            # *** TIMER FIX: Calculate duration from start time to now ***
                            if stable_hold_start_time is not None:
                                correct_posture_time = current_video_time - stable_hold_start_time
                            print(f"DEBUG: Posture deviation detected at {current_video_time:.2f}s. Final hold time: {correct_posture_time:.2f}s.")
                            break

    cap.release()
    pose.close()
    
    return [str(round(correct_posture_time, 2)), str(bad_posture_detected)]