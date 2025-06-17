import cv2
import mediapipe as mp
import time

def analyze_romberg_outstretch(video_path, tolerance_shoulder=0.1):
    # MediaPipe setup
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video: {video_path}")
        return 0, True

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-2:
        fps = 30

    correct_posture_time = 0
    bad_posture_detected = False

    reference_left_shoulder = None
    reference_right_shoulder = None

    prev_time = time.time()
    start_time = prev_time

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

            # Set initial reference posture
            if reference_left_shoulder is None or reference_right_shoulder is None:
                reference_left_shoulder = left_shoulder
                reference_right_shoulder = right_shoulder

            # Compare current shoulder position with reference
            left_moved = abs(left_shoulder.x - reference_left_shoulder.x) > tolerance_shoulder or \
                         abs(left_shoulder.y - reference_left_shoulder.y) > tolerance_shoulder
            right_moved = abs(right_shoulder.x - reference_right_shoulder.x) > tolerance_shoulder or \
                          abs(right_shoulder.y - reference_right_shoulder.y) > tolerance_shoulder

            if left_moved or right_moved:
                bad_posture_detected = True
                break
            else:
                correct_posture_time += current_time - prev_time

        prev_time = current_time

    cap.release()
    return [str(round(correct_posture_time, 2)), str(bad_posture_detected)]
