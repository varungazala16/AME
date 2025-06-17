import cv2
import mediapipe as mp
import numpy as np
import time
import math

# --- Configuration ---
VIDEO_SOURCE = 0 # Use 0 for webcam, or provide video file path
# VIDEO_SOURCE = 0 # Example for webcam

# MediaPipe Pose Initialization
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5,
    model_complexity=1 
)

ANKLE_HEIGHT_DIFF_THRESHOLD = 0.06 # <--- TUNABLE

SWAY_THRESHOLD = 0.01 # <--- TUNABLE

VISIBILITY_THRESHOLD = 0.35 # <--- TUNABLE

# Point to track for sway (options: 'hip_midpoint', 'standing_ankle', 'shoulder_midpoint')
SWAY_TRACKING_POINT = 'hip_midpoint'
# --- --- --- --- ---

# --- State Variables ---
is_standing_one_foot = False
current_stand_segment_start_time = None
total_standing_time = 0.0 
reference_point_sway = None 
reference_standing_ankle_pos = None # 

sway_events = []        
foot_down_events = []   

frame_count = 0
start_process_time = time.time()
previous_time_sec = 0.0 

# --- Video Handling ---
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"Error: Could not open video source: {VIDEO_SOURCE}")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# --- Helper Functions ---
def calculate_distance(point1, point2):
    """Calculates Euclidean distance between two points (landmarks or tuples)."""
    if point1 is None or point2 is None:
        return 0.0 # Or handle as an error/invalid state if preferred
    try:
        if hasattr(point1, 'x') and hasattr(point2, 'x'): # If they are MediaPipe landmarks
            return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
        elif isinstance(point1, (tuple, list)) and isinstance(point2, (tuple, list)): # If they are tuples (x, y)
            return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        else:
             return 0.0
    except AttributeError:
        return 0.0


def get_landmark_coords(landmarks, landmark_enum):
    """Safely get landmark coordinates, checking visibility."""
    if landmarks and landmarks.landmark[landmark_enum.value].visibility > VISIBILITY_THRESHOLD:
        lm = landmarks.landmark[landmark_enum.value]
        return lm
    return None

# --- Main Processing Loop ---
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("End of video or cannot receive frame.")
        break

    frame_count += 1
    current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC)/1000 # Time within the video in seconds
    delta_time = current_time_sec - previous_time_sec # Time elapsed since last frame

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False # Optimize: mark as non-writeable before processing

    # Process the image and detect pose
    results = pose.process(image_rgb)

    # Convert back to BGR for drawing
    image_rgb.flags.writeable = True # Make writeable again for drawing
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    status_text = "Status: Initializing"
    timer_text = f"Total Stand Time: {total_standing_time:.2f}s" # Always show accumulated time
    sway_text = ""
    currently_detected_one_foot = False # Reset detection for the current frame

    if results.pose_landmarks:
        landmarks = results.pose_landmarks

        # Get relevant landmarks with visibility check
        left_ankle = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE)
        right_ankle = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE)
        left_hip = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
        right_hip = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_HIP)
        left_shoulder = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
        right_shoulder = get_landmark_coords(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER)

        standing_ankle = None
        lifted_ankle = None

        # Check if ankles are visible before comparing
        if left_ankle and right_ankle:
            # Check vertical distance between ankles
            ankle_y_diff = abs(left_ankle.y - right_ankle.y)

            if ankle_y_diff > ANKLE_HEIGHT_DIFF_THRESHOLD:
                currently_detected_one_foot = True
                # Determine which ankle is lower (standing)
                if left_ankle.y > right_ankle.y: # Y increases downwards
                    standing_ankle = left_ankle
                    lifted_ankle = right_ankle
                    status_text = "Status: Standing on Left Foot"
                else:
                    standing_ankle = right_ankle
                    lifted_ankle = left_ankle
                    status_text = "Status: Standing on Right Foot"
            else:
                 status_text = "Status: Both Feet Approx. Level"
        else:
            status_text = "Status: Ankles Not Clearly Visible"


        # --- State Logic & Continuous Timer ---
        if currently_detected_one_foot:
            # Add the time duration of this frame to the total
            total_standing_time += delta_time

            if not is_standing_one_foot:
                # Transition: START of a one-foot stand segment
                print(f"[{current_time_sec:.2f}s] Started one-foot stand segment.")
                is_standing_one_foot = True
                current_stand_segment_start_time = current_time_sec # Log start for this segment

                # Set reference point for sway calculation FOR THIS SEGMENT
                reference_point_sway = None # Reset just in case
                if standing_ankle: # Ensure we know which ankle is standing
                    reference_standing_ankle_pos = (standing_ankle.x, standing_ankle.y)
                    if SWAY_TRACKING_POINT == 'hip_midpoint' and left_hip and right_hip:
                        reference_point_sway = ((left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2)
                    elif SWAY_TRACKING_POINT == 'standing_ankle':
                         reference_point_sway = reference_standing_ankle_pos
                    elif SWAY_TRACKING_POINT == 'shoulder_midpoint' and left_shoulder and right_shoulder:
                         reference_point_sway = ((left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2)
                    else: # Default or if points missing for chosen method
                        reference_point_sway = reference_standing_ankle_pos # Fallback

            # --- Sway Calculation (runs whenever standing on one foot) ---
            if is_standing_one_foot and reference_point_sway: # Need a valid reference point
                current_sway_point = None
                # Calculate current position of the tracked point
                if SWAY_TRACKING_POINT == 'hip_midpoint' and left_hip and right_hip:
                     current_sway_point = ((left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2)
                elif SWAY_TRACKING_POINT == 'standing_ankle' and standing_ankle: # Use the current standing ankle pos
                     current_sway_point = (standing_ankle.x, standing_ankle.y)
                elif SWAY_TRACKING_POINT == 'shoulder_midpoint' and left_shoulder and right_shoulder:
                     current_sway_point = ((left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2)
                # Add more tracking point options if needed

                if current_sway_point: # Ensure current point is valid
                    sway_distance = calculate_distance(reference_point_sway, current_sway_point)
                    sway_text = f"Sway Level: {sway_distance:.4f}" # Display current sway

                    if sway_distance > SWAY_THRESHOLD:
                        sway_text += " (Significant Sway!)"
                        sway_events.append((current_time_sec, sway_distance))
                        # Optional: print(f"[{current_time_sec:.2f}s] Sway detected! Level: {sway_distance:.4f}")

                        # --- Draw Sway Visual ---
                        ref_px = (int(reference_point_sway[0] * frame_width), int(reference_point_sway[1] * frame_height))
                        curr_px = (int(current_sway_point[0] * frame_width), int(current_sway_point[1] * frame_height))
                        cv2.circle(image_bgr, ref_px, 5, (255, 255, 0), -1) # Cyan reference
                        cv2.line(image_bgr, ref_px, curr_px, (0, 255, 255), 2) # Yellow line
                        cv2.circle(image_bgr, curr_px, 5, (0, 255, 255), -1) # Yellow current
                else:
                    sway_text = "Sway: Track Point Lost"

            # (No need to update timer text here, it uses the main total_standing_time)

        else: # Not currently detected standing on one foot (or landmarks lost)
            if is_standing_one_foot:
                # Transition: END of a one-foot stand segment
                print(f"[{current_time_sec:.2f}s] Ended one-foot stand segment (foot down or landmarks lost).")
                is_standing_one_foot = False
                foot_down_events.append(current_time_sec) # Log the time the stand ended
                # Reset segment-specific variables
                current_stand_segment_start_time = None
                reference_point_sway = None
                reference_standing_ankle_pos = None
                # total_standing_time continues accumulating from previous frames

            # Update status text if landmarks were detected but feet were level
            if not status_text.startswith("Status: Ankles"): # Avoid overwriting visibility message
                 if left_ankle and right_ankle: # Check if person is detected at all
                     status_text = "Status: Both Feet Approx. Level"
                 else:
                     status_text = "Status: Person Not Detected / Key Landmarks Lost"


        # Draw the pose annotation on the image.
        mp_drawing.draw_landmarks(
            image_bgr,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

    else:
        # No pose landmarks detected in this frame
        status_text = "Status: No Person Detected"
        if is_standing_one_foot:
             # If standing state was active but landmarks lost completely
            print(f"[{current_time_sec:.2f}s] Ended one-foot stand segment (PERSON LOST).")
            is_standing_one_foot = False
            foot_down_events.append(current_time_sec) # Log the time the stand ended
            current_stand_segment_start_time = None
            reference_point_sway = None
            reference_standing_ankle_pos = None

    # --- Display Info ---
    y_pos = 30
    cv2.putText(image_bgr, status_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    y_pos += 25
    cv2.putText(image_bgr, timer_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 200, 50), 2, cv2.LINE_AA) # Green timer
    if is_standing_one_foot and sway_text:
        y_pos += 25
        sway_color = (0, 165, 255) if "Significant" in sway_text else (255, 255, 255) # Orange for significant sway
        cv2.putText(image_bgr, sway_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, sway_color, 2, cv2.LINE_AA)

    # --- Show Frame ---
    # Consider resizing if the video is very large, for display purposes
    # display_frame = cv2.resize(image_bgr, (int(frame_width * 0.75), int(frame_height*0.75)))
    cv2.imshow('MediaPipe Pose Analysis', image_bgr) # Show original size

    # --- Update Previous Time ---
    previous_time_sec = current_time_sec

    # Exit condition
    if cv2.waitKey(5) & 0xFF == 27: # Press ESC to exit
        print("[INFO] ESC key pressed. Exiting...")
        # Log if standing when exiting (though total time is already accumulated)
        if is_standing_one_foot:
            print(f"[EXIT] Was in one-foot stance state at exit time {current_time_sec:.2f}s.")
            foot_down_events.append(current_time_sec) # Record final segment end
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
pose.close()

end_process_time = time.time()
processing_duration = end_process_time - start_process_time

# --- Final Report ---
print("\n--- Analysis Summary ---")
print(f"Video Source: {VIDEO_SOURCE}")
print(f"Total video time analyzed: {current_time_sec:.2f} seconds ({frame_count} frames)")
print(f"Total processing time: {processing_duration:.2f} seconds")
print(f"\nTOTAL CUMULATIVE time standing on one foot: {total_standing_time:.2f} seconds")

print(f"\nOne-Foot Stand Segment END Times (Foot Down / Lost) (Timestamp in seconds):")
if foot_down_events:
    print([f"{t:.2f}" for t in foot_down_events])
else:
    print("No one-foot stand segments detected or ended.")

print(f"\nSignificant Sway Events (> {SWAY_THRESHOLD:.3f} normalized distance) (Timestamp, Sway Level):")
if sway_events:
    # Basic filtering for display: Group events within a small time window (e.g., 0.2s)
    # and show the maximum sway in that window.
    filtered_sway = {}
    grouping_interval = 0.2 # seconds
    for t, level in sway_events:
        time_key = round(t / grouping_interval) * grouping_interval
        filtered_sway[time_key] = max(filtered_sway.get(time_key, 0), level)

    if filtered_sway:
        print("Timestamp (s) ~ | Max Sway Level")
        print("----------------|---------------")
        # Sort by timestamp before printing
        for t_group, max_level in sorted(filtered_sway.items()):
             print(f"{t_group:15.2f} | {max_level:.4f}")
        print(f"(Grouped within {grouping_interval}s intervals)")
    else:
        # This case shouldn't happen if sway_events is not empty, but good to have
        print("No significant sway events recorded above the threshold.")

else:
    print("No significant sway events recorded.")

print("\n--- End of Summary ---")