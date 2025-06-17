import cv2
import mediapipe as mp

def count_stomps_left(video_path, start_time_sec=0.0, end_time_sec=None):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file: {video_path}")
        return 0

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-2:
        fps = 30

    heel_history = []
    stomp_count = 0
    stomp_in_progress = False
    RAISE_THRESHOLD = 0.75
    DROP_THRESHOLD = 0.75
    WINDOW_SIZE = 4

    # Seek to start time in milliseconds
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time_sec * 1000)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if current_time_sec < start_time_sec:
            continue
        if end_time_sec is not None and current_time_sec > end_time_sec:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL]
            heel_y = int(heel.y * h)

            heel_history.append(heel_y)
            if len(heel_history) > WINDOW_SIZE:
                heel_history.pop(0)

            if len(heel_history) == WINDOW_SIZE:
                if not stomp_in_progress and heel_history[0] - heel_y > RAISE_THRESHOLD:
                    stomp_in_progress = True
                elif stomp_in_progress and heel_y - heel_history[0] > DROP_THRESHOLD:
                    stomp_count += 1
                    stomp_in_progress = False

    cap.release()
    return stomp_count

def count_stomps_right(video_path, start_time_sec=0.0, end_time_sec=None):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file: {video_path}")
        return 0

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-2:
        fps = 30

    heel_history = []
    stomp_count = 0
    stomp_in_progress = False
    RAISE_THRESHOLD = 0.75
    DROP_THRESHOLD = 0.75
    WINDOW_SIZE = 4

    # Seek to start time in milliseconds
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time_sec * 1000)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if current_time_sec < start_time_sec:
            continue
        if end_time_sec is not None and current_time_sec > end_time_sec:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            heel = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL]
            heel_y = int(heel.y * h)

            heel_history.append(heel_y)
            if len(heel_history) > WINDOW_SIZE:
                heel_history.pop(0)

            if len(heel_history) == WINDOW_SIZE:
                if not stomp_in_progress and heel_history[0] - heel_y > RAISE_THRESHOLD:
                    stomp_in_progress = True
                elif stomp_in_progress and heel_y - heel_history[0] > DROP_THRESHOLD:
                    stomp_count += 1
                    stomp_in_progress = False

    cap.release()
    return stomp_count

def count_stomps(video_path):
    stomps_left = count_stomps_left(video_path,4.0,14.0)
    stomps_right = count_stomps_right(video_path,23.0,34.0)
    return [str(stomps_left), str(stomps_right)]