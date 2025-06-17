import cv2
import mediapipe as mp
import numpy as np
import time
import os

# --- Configuration ---
RECORD_DURATION = 15  # Seconds to record
OUTPUT_FILENAME = "foot_tap_recording.mp4" # Name of the video file to save
WEBCAM_INDEX = 0 # Usually 0 for the default webcam

# Processing settings
# *** SET THIS TO TRUE TO SEE THE VIDEO DURING PROCESSING ***
SHOW_VIDEO_DURING_PROCESSING = True
PRINT_DEBUG_LOGS = False # Set to True for detailed frame-by-frame logic prints

# --- Auto-determine Output Path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
video_output_path = os.path.join(script_dir, OUTPUT_FILENAME)
print(f"INFO: Recording will be saved to: {video_output_path}")

# --- Recording Phase ---
print(f"\n--- Preparing to Record for {RECORD_DURATION} seconds ---")
cap_record = cv2.VideoCapture(WEBCAM_INDEX)

if not cap_record.isOpened():
    print(f"ERROR: Cannot open webcam (index {WEBCAM_INDEX}).")
    exit()

frame_width = int(cap_record.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap_record.get(cv2.CAP_PROP_FRAME_HEIGHT))
record_fps = cap_record.get(cv2.CAP_PROP_FPS)
if record_fps <= 0 or record_fps > 100:
    print(f"WARN: Webcam FPS reported as {record_fps}. Using default 30 FPS for recording.")
    record_fps = 30.0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_record = cv2.VideoWriter(video_output_path, fourcc, record_fps, (frame_width, frame_height))

if not out_record.isOpened():
    print(f"ERROR: Could not open VideoWriter for path '{video_output_path}'. Check permissions/codec.")
    cap_record.release()
    exit()

print(f"INFO: Webcam opened ({frame_width}x{frame_height} @ {record_fps:.2f} FPS).")
print(f"*** STARTING RECORDING for {RECORD_DURATION} seconds... Press 'q' in window to stop early. ***")

record_start_time = time.time()
frames_recorded = 0
recording_aborted = False

# Create window for recording preview
cv2.namedWindow('Recording Preview (Press Q to Stop Early)', cv2.WINDOW_AUTOSIZE)


while True:
    elapsed_time = time.time() - record_start_time
    if elapsed_time >= RECORD_DURATION:
        break

    ret, frame_record = cap_record.read()
    if not ret:
        print("WARN: Could not read frame from webcam during recording.")
        break

    out_record.write(frame_record)
    frames_recorded += 1

    remaining_time = RECORD_DURATION - elapsed_time
    cv2.putText(frame_record, f"REC {remaining_time:.1f}s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Recording Preview (Press Q to Stop Early)', frame_record)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nINFO: Recording stopped early by user.")
        recording_aborted = True
        break

cap_record.release()
out_record.release()
cv2.destroyWindow('Recording Preview (Press Q to Stop Early)') # Close only the preview window

if frames_recorded > 0:
     print(f"\n--- Recording {'Finished' if not recording_aborted else 'Aborted'}. {frames_recorded} frames saved to {OUTPUT_FILENAME} ---")
else:
    print("\nERROR: No frames were recorded. Processing cannot proceed.")
    exit()


# --- Processing Phase ---
print(f"\n--- Starting Processing of {OUTPUT_FILENAME} ---")
time.sleep(0.5)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

tap_count = 0
index_tap_in_progress = False

RAISE_THRESHOLD_INDEX = 0.0005
DROP_THRESHOLD_INDEX = 0.0005
HEEL_GROUNDED_THRESHOLD_Y = 0.0015
HEEL_INVALIDATION_THRESHOLD_Y = 0.008

cap_process = cv2.VideoCapture(video_output_path)
if not cap_process.isOpened():
    print(f"ERROR: Could not open the recorded video file '{video_output_path}' for processing.")
    exit()

h = int(cap_process.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = int(cap_process.get(cv2.CAP_PROP_FRAME_WIDTH))
total_frames = int(cap_process.get(cv2.CAP_PROP_FRAME_COUNT))
process_fps = cap_process.get(cv2.CAP_PROP_FPS)

if h == 0 or w == 0 or total_frames < 1:
     print(f"ERROR: Recorded video file '{video_output_path}' seems invalid or empty.")
     cap_process.release()
     exit()

print(f"File Properties: {w}x{h}, FPS: {process_fps:.2f}, Total Frames: {total_frames}")
print(f"Using Thresholds (px): IdxRaise={-RAISE_THRESHOLD_INDEX * h:.2f}, IdxDrop={DROP_THRESHOLD_INDEX * h:.2f}, HeelInit={HEEL_GROUNDED_THRESHOLD_Y * h:.2f}, HeelInvalid={HEEL_INVALIDATION_THRESHOLD_Y * h:.2f}")
print("Processing...")
if SHOW_VIDEO_DURING_PROCESSING:
    print("INFO: Displaying video frames during processing. Press 'q' in the 'Offline Processing' window to stop.")

previous_right_foot_index_y = None
previous_right_heel_y = None
frame_number = 0
start_process_time = time.time()

# Create window for processing view if needed
if SHOW_VIDEO_DURING_PROCESSING:
    cv2.namedWindow("Offline Processing", cv2.WINDOW_AUTOSIZE)

# --- Processing Loop ---
while cap_process.isOpened():
    ret, frame = cap_process.read()
    if not ret:
        if frame_number < total_frames -1 :
             print(f"\nWarning: Processing ended unexpectedly at frame {frame_number} (expected {total_frames}).")
        else:
             print(f"\nFinished processing all {frame_number} frames.")
        break

    frame_number += 1

    # --- Progress Update (Console) ---
    if frame_number % 100 == 0 or frame_number == 1:
        elapsed_time = time.time() - start_process_time
        est_total_time = (elapsed_time / frame_number) * total_frames if frame_number > 0 else 0
        est_remaining = est_total_time - elapsed_time
        print(f"  Processing frame {frame_number}/{total_frames}... (Est. remaining: {est_remaining:.0f}s)", end='\r')

    # --- MediaPipe Processing ---
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    momentary_heel_is_grounded = False

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        try:
            right_foot_index_lm = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
            right_heel_lm = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL]

            if right_foot_index_lm.visibility < 0.6 or right_heel_lm.visibility < 0.6:
                # State Reset on low visibility
                if previous_right_foot_index_y is not None or previous_right_heel_y is not None:
                    if PRINT_DEBUG_LOGS: print(f"\nFrame {frame_number}: Landmarks low visibility/lost. Resetting state.")
                previous_right_foot_index_y = None
                previous_right_heel_y = None
                index_tap_in_progress = False
            else:
                # Get Coordinates
                current_right_foot_index_y = int(right_foot_index_lm.y * h)
                current_right_heel_y = int(right_heel_lm.y * h)

                # Check Heel Grounded Status
                if previous_right_heel_y is not None:
                    heel_movement_y = abs(current_right_heel_y - previous_right_heel_y)
                    if heel_movement_y <= HEEL_GROUNDED_THRESHOLD_Y * h:
                        momentary_heel_is_grounded = True
                    # Optional Debug Print for Heel Movement
                    # if PRINT_DEBUG_LOGS and heel_movement_y > HEEL_GROUNDED_THRESHOLD_Y * h:
                    #      print(f"\nFrame {frame_number}: Heel moved {heel_movement_y:.1f}px (Thresh: {HEEL_GROUNDED_THRESHOLD_Y*h:.1f}). Grounded: {momentary_heel_is_grounded}")
                else:
                    momentary_heel_is_grounded = False

                # Process Tap Logic
                if previous_right_foot_index_y is not None:
                    index_movement_y = current_right_foot_index_y - previous_right_foot_index_y
                    # START TAP
                    if not index_tap_in_progress and momentary_heel_is_grounded and \
                       index_movement_y < -RAISE_THRESHOLD_INDEX * h:
                        index_tap_in_progress = True
                        if PRINT_DEBUG_LOGS: print(f"\nFrame {frame_number}: RAISE Start. Heel OK. IdxMov:{index_movement_y:.1f}")
                    # COUNT TAP
                    elif index_tap_in_progress and index_movement_y > DROP_THRESHOLD_INDEX * h:
                        tap_count += 1
                        print(f"\n  -> 👣 Specific Foot Tap #{tap_count} detected at Frame ~{frame_number}") # Print detected taps
                        index_tap_in_progress = False
                        if PRINT_DEBUG_LOGS: print(f"\nFrame {frame_number}: DROP Counted. IdxMov:{index_movement_y:.1f}")

                # INVALIDATE TAP
                if index_tap_in_progress and previous_right_heel_y is not None:
                     heel_movement_y = abs(current_right_heel_y - previous_right_heel_y)
                     if heel_movement_y > HEEL_INVALIDATION_THRESHOLD_Y * h:
                         if PRINT_DEBUG_LOGS: print(f"\nFrame {frame_number}: Tap Invalidated - Heel moved too much ({heel_movement_y:.1f}px > {HEEL_INVALIDATION_THRESHOLD_Y*h:.1f}px)")
                         index_tap_in_progress = False

                # Update Previous Positions
                previous_right_foot_index_y = current_right_foot_index_y
                previous_right_heel_y = current_right_heel_y

                # --- Draw on frame IF display is enabled ---
                if SHOW_VIDEO_DURING_PROCESSING:
                    # Draw Markers
                    cv2.circle(frame, (int(right_foot_index_lm.x * w), current_right_foot_index_y), 6, (255, 0, 0), -1) # Blue Index
                    cv2.circle(frame, (int(right_heel_lm.x * w), current_right_heel_y), 6, (0, 255, 255), -1)    # Yellow Heel

        except (IndexError, AttributeError) as e:
             # Error Handling for Landmarks
             if PRINT_DEBUG_LOGS: print(f"\nFrame {frame_number}: Landmark access error - {e}")
             previous_right_foot_index_y = None
             previous_right_heel_y = None
             index_tap_in_progress = False
    else: # No landmarks detected in frame
        if previous_right_foot_index_y is not None or previous_right_heel_y is not None:
             if PRINT_DEBUG_LOGS: print(f"\nFrame {frame_number}: No landmarks detected. Resetting state.")
        previous_right_foot_index_y = None
        previous_right_heel_y = None
        index_tap_in_progress = False

    # --- Display Frame (if enabled) ---
    if SHOW_VIDEO_DURING_PROCESSING:
        # Draw Count and Status Text (always draw these if displaying)
        cv2.putText(frame, f"Frame: {frame_number}/{total_frames}", (w - 250, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
        cv2.putText(frame, f"Taps: {tap_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        status_text = []
        if momentary_heel_is_grounded: status_text.append("Heel OK")
        if index_tap_in_progress: status_text.append("Index Tapping")
        cv2.putText(frame, " | ".join(status_text), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        cv2.imshow("Offline Processing", frame)
        # Use waitKey(1) - Crucial to allow the window to refresh without pausing indefinitely
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nProcessing aborted by user.")
            break

# --- Cleanup and Final Report ---
print(' ' * 80, end='\r') # Clear the progress line

end_process_time = time.time()
cap_process.release() # Release the processing video capture
cv2.destroyAllWindows() # Destroy all OpenCV windows (including processing one if shown)

print("\n--- Analysis Complete ---")
print(f"Processed video: {OUTPUT_FILENAME}")
print(f"Total Specific Foot Taps detected: {tap_count}")
print(f"Processing took: {end_process_time - start_process_time:.2f} seconds")
