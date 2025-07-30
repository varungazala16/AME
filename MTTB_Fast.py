import cv2
import mediapipe as mp
import os

def analyze_marching_beats(video_path, show_video=True):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file: {video_path}")
        return ['0']

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    # --- THRESHOLD AND WINDOW SETTINGS ---
    STOMP_THRESHOLD_RATIO = 0.015
    WINDOW_SIZE = 7
    VISIBILITY_THRESHOLD = 0.7

    # --- STATE MANAGEMENT FOR BOTH FEET ---
    stomp_states = {
        'left': {'history': [], 'in_progress': False},
        'right': {'history': [], 'in_progress': False}
    }
    
    # This list will still store individual stomp events for accurate timing and visualization
    stomp_events = []
    
    # Map sides to their corresponding MediaPipe landmarks
    foot_landmarks = {
        'left': mp_pose.PoseLandmark.LEFT_HEEL,
        'right': mp_pose.PoseLandmark.RIGHT_HEEL
    }
    
    # ### NEW: VARIABLES FOR INTERVAL-BASED COUNTING ###
    INTERVAL_DURATION = 2.38  # seconds
    interval_start_time = 5.5
    current_interval_beats = 0
    # This list will store the counts for each completed interval
    interval_data = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if interval_start_time < 5.5:
                continue
            if interval_start_time > 24.5:
                break

            # ### NEW: CHECK IF A NEW INTERVAL HAS STARTED ###
            if current_time_sec >= interval_start_time + INTERVAL_DURATION:
                # Store the result of the completed interval
                interval_data.append(current_interval_beats)
                
                # Print the result for the interval that just ended
                end_time_marker = interval_start_time + INTERVAL_DURATION
                print(f"Interval [{interval_start_time:.1f}s - {end_time_marker:.1f}s]: {current_interval_beats} marches")

                # Reset for the next interval
                interval_start_time += INTERVAL_DURATION
                current_interval_beats = 0


            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            # --- LOOP THROUGH BOTH FEET TO PROCESS THEM ---
            for side, landmark_enum in foot_landmarks.items():
                state = stomp_states[side] # Get the current state for this foot

                if results.pose_landmarks:
                    heel_landmark = results.pose_landmarks.landmark[landmark_enum.value]
                    
                    if heel_landmark.visibility > VISIBILITY_THRESHOLD:
                        heel_y = int(heel_landmark.y * h)
                        state['history'].append(heel_y)

                        if len(state['history']) > WINDOW_SIZE:
                            state['history'].pop(0)

                        if len(state['history']) == WINDOW_SIZE:
                            stomp_threshold_px = STOMP_THRESHOLD_RATIO * h
                            y_diff = state['history'][0] - heel_y

                            # Condition to START a stomp: Upward movement detected
                            if not state['in_progress'] and y_diff > stomp_threshold_px:
                                state['in_progress'] = True

                            # Condition to COMPLETE a stomp: Downward movement detected after starting
                            elif state['in_progress'] and -y_diff > stomp_threshold_px:
                                stomp_events.append({'time': current_time_sec, 'foot': side})
                                # ### MODIFIED: Increment the count for the CURRENT interval ###
                                current_interval_beats += 1
                                state['in_progress'] = False
                    else:
                        state['in_progress'] = False
            
            # --- VISUALIZATION (if enabled) ---
            if show_video:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # ### MODIFIED: Update status panel for new display format ###
                total_beats = len(stomp_events)

                # Create a larger status panel for two lines of text
                overlay = frame.copy()
                cv2.rectangle(overlay, (5, 5), (320, 70), (20, 20, 20), -1)
                frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

                # Display the current interval count and the total count
                cv2.putText(frame, f"Interval Marches: {current_interval_beats}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255, 255), 2)
                cv2.putText(frame, f"Total Marches: {total_beats}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Flash a circle on the foot that just stomped for visual feedback
                if stomp_events and (current_time_sec - stomp_events[-1]['time'] < 0.2):
                    last_stomp_foot = stomp_events[-1]['foot']
                    if results.pose_landmarks:
                        heel_lm = results.pose_landmarks.landmark[foot_landmarks[last_stomp_foot].value]
                        if heel_lm.visibility > VISIBILITY_THRESHOLD:
                            heel_px = (int(heel_lm.x * w), int(heel_lm.y * h))
                            cv2.circle(frame, heel_px, 30, (0, 0, 255), 3)

                cv2.imshow('Marching Beat Analysis', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        cap.release()
        pose.close()
        if show_video:
            cv2.destroyAllWindows()

    # ### NEW: Finalize the last interval after the loop ends ###
    # This captures the marches from the final, potentially shorter, interval
    interval_data.append(current_interval_beats)
    end_time_marker = interval_start_time + INTERVAL_DURATION
    print(f"Interval [{interval_start_time:.1f}s - {end_time_marker:.1f}s]: {current_interval_beats} marches")


    # ### MODIFIED: FINAL CONSOLE OUTPUT AND RETURN VALUE ---
    total_beats = sum(interval_data)
    rythm_beats = 0
    for i in interval_data:
        rythm_beats+=min(4,i)
    print("\n--- Marching Beat Analysis Complete ---")
    print("\n--- Summary per 3-second interval ---")
    
    start_time_marker = 5.5
    for i, count in enumerate(interval_data):
        end_time_marker = start_time_marker + INTERVAL_DURATION
        print(f"Interval {i+1} ({start_time_marker:.1f}s - {end_time_marker:.1f}s): {count} marches")
        start_time_marker += INTERVAL_DURATION

    print(f"\nCUMULATIVE MARCHES DETECTED: {total_beats}")
    print(f"\nCUMULATIVE MARCHES DETECTED IN RYTHM: {rythm_beats}")
    
    # Prepare the return value as a list of strings
    # First, the count for each interval, then the final cumulative count
    results_to_return = [str(count) for count in interval_data]
    results_to_return.append(str(total_beats))

    return rythm_beats