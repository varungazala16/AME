import cv2
import mediapipe as mp

# Parameters
DIST_THRESHOLD = 0.07  # Adjust as needed for your video
MIN_FRAMES_BETWEEN_TAPS = 5  # Debounce

def calc_distance(lm1, lm2):
    return ((lm1.x - lm2.x) ** 2 + (lm1.y - lm2.y) ** 2) ** 0.5

def count_right_hand_taps(video_path):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    tap_count = 0
    last_tap_frame = -MIN_FRAMES_BETWEEN_TAPS
    frame_idx = 0
    tap_active = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB for mediapipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                if label.lower() == "right":
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    dist = calc_distance(thumb_tip, index_tip)
                    
                    # Detect tap event
                    if dist < DIST_THRESHOLD and not tap_active and (frame_idx - last_tap_frame) > MIN_FRAMES_BETWEEN_TAPS:
                        tap_count += 1
                        tap_active = True
                        last_tap_frame = frame_idx
                    elif dist >= DIST_THRESHOLD:
                        tap_active = False

        frame_idx += 1

    cap.release()
    hands.close()
    return [str(tap_count)]
