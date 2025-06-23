import cv2
import mediapipe as mp
import numpy as np
import os
import subprocess
import whisper
import csv
import re
from word2number import w2n
import json

# ======= CONFIGURATION =======
SHOW_VIDEO = False
AUDIO_FILE = "audio.wav"

# ======= 1. TUG Video Pose Analysis =======
def analyze_tug_from_video(video_path: str, show_video: bool = True):
    mp_pose = mp.solutions.pose
    pose_estimator = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    STATE_UNKNOWN = -1
    STATE_SITTING = 0
    STATE_STANDING = 1
    STATE_TRANSITIONING = 2

    CONFIRMATION_WINDOW_FRAMES = 15
    HIP_Y_STABILITY_THRESHOLD = 0.02
    HIP_X_MOVEMENT_THRESHOLD = 0.015
    BUFFER_LEN = 10

    def classify_pose(landmarks):
        try:
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
            right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            hip_y = (left_hip.y + right_hip.y) / 2
            knee_y = (left_knee.y + right_knee.y) / 2
            ankle_y = (left_ankle.y + right_ankle.y) / 2
            shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
            shoulder_hip_dist = abs(shoulder_y - hip_y)
            if shoulder_hip_dist < 1e-6: shoulder_hip_dist = 1.0
            hip_ankle_dist = abs(hip_y - ankle_y)
            hip_knee_diff = hip_y - knee_y
            if hip_ankle_dist < shoulder_hip_dist * 1.3:
                return STATE_SITTING
            elif hip_knee_diff < -0.05:
                if hip_ankle_dist > shoulder_hip_dist * 0.8:
                    return STATE_STANDING
                else:
                    return STATE_SITTING
            else:
                if hip_knee_diff < 0.02:
                    return STATE_SITTING
                return STATE_TRANSITIONING
        except Exception:
            return STATE_UNKNOWN

    def get_avg_hip_coords(landmarks):
        if not landmarks: return None, None
        try:
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            if left_hip.visibility > 0.5 and right_hip.visibility > 0.5:
                return (left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2
            elif left_hip.visibility > 0.5:
                return left_hip.x, left_hip.y
            elif right_hip.visibility > 0.5:
                return right_hip.x, right_hip.y
            else:
                return None, None
        except Exception:
            return None, None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return {'status': 'error', 'message': f'Could not open video file {video_path}'}
    print(f"\n--- Starting TUG Analysis on: {os.path.basename(video_path)} ---")
    current_state = STATE_UNKNOWN
    state_buffer = []
    first_confirmed_stand_up_time = None
    is_confirming_stand = False
    candidate_stand_up_time = None
    hip_y_history_for_confirmation = []
    hip_x_history_for_confirmation = []
    tug_timer_running = False
    final_tug_time = None

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose_estimator.process(image_rgb)
        frame_state = STATE_UNKNOWN
        current_hip_x, current_hip_y = None, None

        if results.pose_landmarks:
            frame_state = classify_pose(results.pose_landmarks.landmark)
            current_hip_x, current_hip_y = get_avg_hip_coords(results.pose_landmarks.landmark)

        state_buffer.append(frame_state)
        if len(state_buffer) > BUFFER_LEN:
            state_buffer.pop(0)
        stable_state = current_state
        if len(state_buffer) == BUFFER_LEN:
            valid_states = [s for s in state_buffer if s in [STATE_SITTING, STATE_STANDING]]
            if valid_states:
                stable_state = max(set(valid_states), key=valid_states.count)

        if stable_state != current_state:
            if current_state == STATE_SITTING and stable_state == STATE_STANDING and not is_confirming_stand and first_confirmed_stand_up_time is None:
                is_confirming_stand = True
                candidate_stand_up_time = current_time_sec
                hip_y_history_for_confirmation.clear()
                hip_x_history_for_confirmation.clear()
            elif current_state == STATE_STANDING and stable_state == STATE_SITTING:
                if tug_timer_running:
                    final_tug_time = current_time_sec - first_confirmed_stand_up_time
                    tug_timer_running = False
                if is_confirming_stand:
                    is_confirming_stand = False
            current_state = stable_state

        if is_confirming_stand and current_state == STATE_STANDING:
            if current_hip_x is not None and current_hip_y is not None:
                hip_y_history_for_confirmation.append(current_hip_y)
                hip_x_history_for_confirmation.append(current_hip_x)
                if len(hip_y_history_for_confirmation) == CONFIRMATION_WINDOW_FRAMES:
                    y_range = np.max(hip_y_history_for_confirmation) - np.min(hip_y_history_for_confirmation)
                    x_range = np.max(hip_x_history_for_confirmation) - np.min(hip_x_history_for_confirmation)
                    if y_range < HIP_Y_STABILITY_THRESHOLD and x_range > HIP_X_MOVEMENT_THRESHOLD:
                        if first_confirmed_stand_up_time is None:
                            first_confirmed_stand_up_time = candidate_stand_up_time
                            tug_timer_running = True
                            print(f"Time: {first_confirmed_stand_up_time:.2f}s - SUCCESSFUL STAND-UP CONFIRMED. TUG TIMER STARTED.")
                        is_confirming_stand = False
                    hip_y_history_for_confirmation.pop(0)
                    hip_x_history_for_confirmation.pop(0)
        elif is_confirming_stand and current_state != STATE_STANDING:
            is_confirming_stand = False

        if show_video:
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imshow('TUG Analysis', frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break

    cap.release()
    pose_estimator.close()
    if show_video:
        cv2.destroyAllWindows()
    print("\n--- Analysis Complete ---")
    if final_tug_time:
        end_time = first_confirmed_stand_up_time + final_tug_time
        print(f">>> TOTAL TUG DURATION: {final_tug_time:.2f} seconds <<<")
        return [
            str(round(first_confirmed_stand_up_time, 2)),
            str(round(final_tug_time, 2)),
            "success",
            str(round(end_time, 2))
        ]
    elif first_confirmed_stand_up_time:
        print("Test was not completed (person did not sit back down before video ended).")
        return [
            str(round(first_confirmed_stand_up_time, 2)),
            "NA",
            "incomplete",
            "NA"
        ]
    else:
        print("No confirmed stand-up was detected. TUG test did not start.")
        return ["NA", "NA", "failure", "NA"]

# ======= 2. Audio Extraction and Whisper Speech Analysis =======
def extract_audio_and_analyze_speech(video_path,audio_file=AUDIO_FILE):
    
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-ac", "1", "-ar", "16000", audio_file
        ], check=True)

        # --- Load Whisper model & transcribe with better segmentation ---
        model = whisper.load_model("small")
        print("Transcribing with Whisper…")
        options = {
            "beam_size": 5,
            "best_of": 5,
            "word_timestamps": True,
            "verbose": True,
        }
        res = model.transcribe(audio_file, **options)
        numbers = []
        number_words_pattern = re.compile(
            r'\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion)\b',
            re.IGNORECASE
        )
        

        if "words" in res and len(res["words"]) > 0:
            print("Using word-level timestamps:")
            for word_info in res["words"]:
                word = word_info.get("word", "").strip()
                start_time = word_info.get("start", 0)
                if not word:
                    continue
                if number_words_pattern.match(word):
                    try:
                        num = w2n.word_to_num(word)
                        numbers.append((start_time, num))
                    except Exception:
                        pass
                elif word.isdigit():
                    num = int(word)
                    numbers.append((start_time, num))
                elif "-" in word:
                    try:
                        num = w2n.word_to_num(word.replace("-", " "))
                        numbers.append((start_time, num))
                    except Exception:
                        pass
        else:
            print("\nFallback: Estimating timestamps from segments")
            for i, seg in enumerate(res["segments"]):
                start_time = seg["start"]
                end_time = seg["end"]
                text = seg["text"].strip()
                duration = end_time - start_time
                words = text.split()
                if not words:
                    continue
                for j, word in enumerate(words):
                    est_time = start_time + (duration * j / len(words))
                    clean_word = word.strip('.,!?;:"\'').lower()
                    if j < len(words) - 1:
                        next_word = words[j+1].strip('.,!?;:"\'').lower()
                        compound = clean_word + " " + next_word
                        try:
                            if (number_words_pattern.match(clean_word) and 
                                number_words_pattern.match(next_word)):
                                num = w2n.word_to_num(compound)
                                numbers.append((est_time, num))
                                continue
                        except Exception:
                            pass
                    if number_words_pattern.match(clean_word):
                        try:
                            num = w2n.word_to_num(clean_word)
                            numbers.append((est_time, num))
                        except Exception:
                            pass
                    elif clean_word.isdigit():
                        num = int(clean_word)
                        numbers.append((est_time, num))

        # --- Pattern extraction and CSV writing (same as original code) ---
        print("\nIdentifying numbers in sequence:")
        numbers_dict = {}
        for ts, num in numbers:
            ts_rounded = round(ts, 2)
            numbers_dict[ts_rounded] = num

        countdown_candidates = []
        final_countdown = []
        for ts, num in sorted(numbers):
            if num in [1, 2, 3]:
                countdown_candidates.append((ts, num))

        if len(countdown_candidates) >= 3:
            countdown_candidates.sort(key=lambda x: x[0])
            countdown_dict = {}
            for ts, num in countdown_candidates:
                if num not in countdown_dict:
                    countdown_dict[num] = ts
            if len(countdown_dict) == 3:
                for num in [3, 2, 1]:
                    final_countdown.append((countdown_dict[num], num))
            else:
                for num in sorted(countdown_dict.keys(), reverse=True):
                    final_countdown.append((countdown_dict[num], num))

        expected_values = [100, 93, 86, 79, 72, 65, 58, 51, 44, 37, 30, 23, 16, 9, 2]
        main_pattern = []
        last_timestamp = 0

        def find_best_match(expected, numbers_list, last_ts=0, min_time_diff=0.5):
            best_match = None
            best_timestamp = None
            best_diff = float('inf')
            numbers_list.sort(key=lambda x: x[0])
            for timestamp, num in numbers_list:
                if timestamp <= last_ts:
                    continue
                diff = abs(num - expected)
                if diff < best_diff and diff <= 10:
                    best_diff = diff
                    best_timestamp = timestamp
                    best_match = num
            return best_timestamp, best_match, best_diff

        ts_100, match_100, diff_100 = find_best_match(100, numbers)
        if ts_100 is not None:
            main_pattern.append((ts_100, 100))
            last_timestamp = ts_100
            for expected in expected_values[1:]:
                ts, match, diff = find_best_match(expected, numbers, last_timestamp)
                if ts is not None:
                    main_pattern.append((ts, expected))
                    last_timestamp = ts

        if final_countdown and main_pattern:
            if main_pattern[0][0] <= final_countdown[-1][0]:
                first_main_ts = max(main_pattern[0][0], final_countdown[-1][0] + 0.3)
                new_main_pattern = [(first_main_ts, main_pattern[0][1])]
                for i in range(1, len(main_pattern)):
                    ts = first_main_ts + (i * 0.8)
                    new_main_pattern.append((ts, main_pattern[i][1]))
                main_pattern = new_main_pattern

        tug_result = analyze_tug_from_video(video_path, show_video=SHOW_VIDEO)

        return [
    tug_result,
    [num for ts, num in main_pattern]
]
    finally:
        if os.path.exists("audio.wav"):
            os.remove("audio.wav")
