import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class FingerTapDetector:
    def __init__(self, video_path):
        self.video_path = video_path
        
        # Detection parameters
        self.velocity_threshold = 0.020         # Lower to detect more subtle finger movements
        self.direction_change_threshold = 0.4   # Slightly relaxed for easier direction change detection
        self.min_frames_between_taps = 4        # Reduce debounce time to catch faster taps
        self.history_length = 15                # Slightly extended history for smoother velocity analysis
 # Number of frames to track for movement analysis
        
        # State tracking
        self.position_history = deque(maxlen=self.history_length)
        self.velocity_history = deque(maxlen=self.history_length)
        self.tap_locations = []            # Store frame numbers where taps occur
        self.tap_count = 0
        self.last_tap_frame = -self.min_frames_between_taps  # Initialize to allow immediate first tap
        
        # For visualization
        self.distance_buffer = []          # Store distances for the whole video
        self.velocity_buffer = []          # Store velocities for the whole video
        self.frame_indices = []            # Store frame numbers for plotting
        
        # Debug flags
        self.debug = True
        self.plot_results = True
    
    def analyze_velocity_pattern(self, positions, frame_idx):
        """Analyze velocity patterns to detect taps based on characteristic finger movements"""
        if len(positions) < 3:
            return False  # Need at least 3 frames for analysis
        
        # Calculate recent velocities (changes in distance)
        recent_velocities = [positions[i] - positions[i-1] for i in range(1, len(positions))]
        
        # Check if we have enough history to detect pattern
        if len(recent_velocities) < 3:
            return False
            
        # A tap typically involves:
        # 1. Decreasing distance (negative velocity) - fingers coming together
        # 2. Followed by increasing distance (positive velocity) - fingers moving apart
        
        # Find where velocity changes from negative to positive (change in direction)
        direction_changes = []
        for i in range(1, len(recent_velocities)):
            if recent_velocities[i-1] < 0 and recent_velocities[i] > 0:
                direction_changes.append(i)
        
        # Find significant velocity changes (magnitude of change)
        significant_changes = []
        for i in range(1, len(recent_velocities)):
            if abs(recent_velocities[i] - recent_velocities[i-1]) > self.velocity_threshold:
                significant_changes.append(i)
        
        # Detect tap pattern - intersection of direction and significant changes
        # with debounce to prevent multiple detections
        for change_idx in direction_changes:
            if change_idx in significant_changes:
                if frame_idx - self.last_tap_frame >= self.min_frames_between_taps:
                    self.last_tap_frame = frame_idx
                    return True
                    
        return False
    
    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("❌ Error: Could not open video.")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 240:
            fps = 30  # Default if invalid
            
            # Initialize MediaPipe Hands
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        ) as hands:
            frame_idx = 0
            
            while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_idx += 1
                    time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    if time_sec <= 0:
                        time_sec = frame_idx / fps
                    
                    # Process image
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb)
                    
                    thumb_x, thumb_y = None, None
                    index_x, index_y = None, None
                    distance = None
                    velocity = None
                    is_tap = False
                    
                    # Draw hand landmarks and extract fingertip positions
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            # Draw hand skeleton
                            mp_drawing.draw_landmarks(
                                frame,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style()
                            )
                            
                            # Get thumb and index finger tip positions
                            thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                            index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                            
                            # Ensure valid coordinates
                            thumb_x = max(0.0, min(1.0, thumb.x))
                            thumb_y = max(0.0, min(1.0, thumb.y))
                            index_x = max(0.0, min(1.0, index.x))
                            index_y = max(0.0, min(1.0, index.y))
                            
                            # Calculate 2D Euclidean distance
                            distance = np.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2)
                            
                            # Add to position history
                            self.position_history.append(distance)
                            
                            # Calculate velocity if we have enough history
                            if len(self.position_history) >= 2:
                                velocity = self.position_history[-1] - self.position_history[-2]
                                self.velocity_history.append(velocity)
                            
                            # Detect tap based on movement pattern
                            if len(self.position_history) == self.history_length:
                                is_tap = self.analyze_velocity_pattern(list(self.position_history), frame_idx)
                                if is_tap:
                                    self.tap_count += 1
                                    self.tap_locations.append(frame_idx)
                                    print(f"Tap #{self.tap_count} detected at frame {frame_idx}, time {time_sec:.2f}s")
                            
                            # Draw fingertip markers with color based on tap state
                            tip_color = (0, 255, 0) if is_tap else (0, 0, 255)
                            tp = (int(thumb_x * width), int(thumb_y * height))
                            ip = (int(index_x * width), int(index_y * height))
                            cv2.circle(frame, tp, 12, tip_color, -1)
                            cv2.circle(frame, ip, 12, tip_color, -1)
                            cv2.line(frame, tp, ip, tip_color, 3)
                    
                    # Save data for plotting
                    if distance is not None:
                        self.distance_buffer.append(distance)
                        if velocity is not None:
                            self.velocity_buffer.append(velocity)
                        else:
                            self.velocity_buffer.append(0)
                        self.frame_indices.append(frame_idx)
                    
                    # Draw information on frame
                    cv2.putText(frame, f"Taps: {self.tap_count}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    if distance is not None:
                        cv2.putText(frame, f"Distance: {distance:.3f}", (10, height - 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    if velocity is not None:
                        vel_color = (0, 0, 255) if velocity < 0 else (0, 255, 0)
                        cv2.putText(frame, f"Velocity: {velocity:.3f}", (10, height - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, vel_color, 2)
                    
                    # Progress display
                    if frame_idx % int(fps) == 0:
                        print(f"Processed {frame_idx} frames, taps={self.tap_count}")
        
        # Close video resources
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"Final taps: {self.tap_count}")
        return self.tap_count

def count_taps(video_path):
    detector = FingerTapDetector(video_path)
    tap_count = detector.process_video()
    print(f"Detection complete. Found {tap_count} finger taps.")
    return [str(tap_count)]