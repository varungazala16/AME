import cv2
import mediapipe as mp

def count_stomps_left(video_path):
    """
    Counts left foot stomps in a video without displaying a window.
    This is a wrapper for the core analysis function.
    """
    return _analyze_stomps_headless(video_path, side='left')

def count_stomps_right(video_path):
    """
    Counts right foot stomps in a video without displaying a window.
    This is a wrapper for the core analysis function.
    """
    return _analyze_stomps_headless(video_path, side='right')

def _analyze_stomps_headless(video_path, side):
    """
    Core analysis function to count stomps. It contains the final working logic
    with jitter elimination but without any video display code.
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file: {video_path}")
        return ['0']

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    heel_landmark_enum = mp_pose.PoseLandmark.LEFT_HEEL if side == 'left' else mp_pose.PoseLandmark.RIGHT_HEEL

    # --- THRESHOLD AND WINDOW SETTINGS ---
    # These are the tuned values from the version that worked correctly.
    STOMP_THRESHOLD_RATIO = 0.01
    WINDOW_SIZE = 5

    heel_history = []
    stomp_count = 0
    stomp_in_progress = False

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame to get landmarks
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                heel_landmark = results.pose_landmarks.landmark[heel_landmark_enum.value]
                
                # Only process if the heel is clearly visible
                if heel_landmark.visibility > 0.65:
                    heel_y = int(heel_landmark.y * h)
                    heel_history.append(heel_y)

                    if len(heel_history) > WINDOW_SIZE:
                        heel_history.pop(0)

                    if len(heel_history) == WINDOW_SIZE:
                        # --- THE WORKING STOMP COUNTING LOGIC ---
                        stomp_threshold_px = STOMP_THRESHOLD_RATIO * h
                        
                        y_diff = heel_history[0] - heel_y

                        # Condition to START a stomp: Foot was down, and now there's a large upward movement.
                        if not stomp_in_progress and y_diff > stomp_threshold_px:
                            stomp_in_progress = True

                        # Condition to COMPLETE a stomp: Foot was up, and now there's a large downward movement.
                        elif stomp_in_progress and -y_diff > stomp_threshold_px:
                            stomp_count += 1
                            stomp_in_progress = False
                else:
                    # If foot is not visible, reset the state
                    stomp_in_progress = False
    finally:
        # Clean up resources
        cap.release()
        pose.close()
            
    return [str(stomp_count)]
