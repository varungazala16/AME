import cv2
import mediapipe as mp

def analyze_sit_to_stand(video_path):
    mp_pose = mp.solutions.pose
    pose_estimator = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    STATE_UNKNOWN = -1
    STATE_SITTING = 0
    STATE_STANDING = 1
    STATE_TRANSITIONING = 2

    current_state = STATE_UNKNOWN
    last_state = STATE_UNKNOWN
    state_buffer = []
    BUFFER_LEN = 10

    stand_up_times = []
    sit_down_times = []

    frame_count = 0

    def classify_pose(landmarks):
        try:
            hip_y = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2
            knee_y = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y) / 2
            ankle_y = (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y) / 2
            shoulder_y = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2

            hip_knee_diff = hip_y - knee_y
            shoulder_hip_dist = abs(shoulder_y - hip_y)
            hip_ankle_dist = abs(hip_y - ankle_y)

            if hip_ankle_dist < shoulder_hip_dist * 1.3:
                return STATE_SITTING
            elif hip_knee_diff < -0.05:
                return STATE_STANDING
            else:
                if hip_knee_diff >= -0.05:
                    return STATE_SITTING
                else:
                    return STATE_TRANSITIONING
        except:
            return STATE_UNKNOWN

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return None

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose_estimator.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        frame_state = STATE_UNKNOWN
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            frame_state = classify_pose(results.pose_landmarks.landmark)
        else:
            frame_state = STATE_UNKNOWN

        state_buffer.append(frame_state)
        if len(state_buffer) > BUFFER_LEN:
            state_buffer.pop(0)

        if len(state_buffer) == BUFFER_LEN:
            valid_states = [s for s in state_buffer if s != STATE_UNKNOWN and s != STATE_TRANSITIONING]
            if not valid_states:
                stable_state = STATE_UNKNOWN
            else:
                stable_state = max(set(valid_states), key=valid_states.count)

            if stable_state != current_state:
                if current_state == STATE_SITTING and stable_state == STATE_STANDING:
                    stand_up_times.append(current_time_sec)
                elif current_state == STATE_STANDING and stable_state == STATE_SITTING:
                    sit_down_times.append(current_time_sec)

                last_state = current_state
                current_state = stable_state

        cv2.imshow('Pose Estimation', image)
        if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()
    pose_estimator.close()

    elapsed_time = None
    if stand_up_times and sit_down_times:
        elapsed_time = abs(stand_up_times[0] - sit_down_times[-1])
        print(f"Stand up Timestamp : {stand_up_times[0]:.2f}")
        print(f"Sit down Timestamp : {sit_down_times[-1]:.2f}")
        print(f"Time Elapsed: {elapsed_time:.2f} seconds")
    else:
        if not stand_up_times:
            print("No stand ups recorded.")
        if not sit_down_times:
            print("No sit downs recorded.")

    return [str(elapsed_time)]
