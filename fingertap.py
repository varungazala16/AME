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
        
        # Improved detection parameters
        self.velocity_threshold = 0.08          # Increased to reduce false positives
        self.min_tap_velocity = 0.15            # Minimum velocity change for a tap
        self.min_frames_between_taps = 8        # Increased debounce time
        self.history_length = 10                # Reduced for more responsive detection
        self.distance_threshold = 0.05          # Minimum distance change for valid tap
        
        # State tracking
        self.position_history = deque(maxlen=self.history_length)
        self.velocity_history = deque(maxlen=self.history_length)
        self.tap_locations = []
        self.tap_count = 0
        self.last_tap_frame = -self.min_frames_between_taps
        
        # For better tap detection
        self.previous_distance = None
        self.tap_state = "idle"  # idle, approaching, separating
        self.min_distance_in_sequence = float('inf')
        self.max_distance_in_sequence = 0
        
        # For visualization
        self.distance_buffer = []
        self.velocity_buffer = []
        self.frame_indices = []
        
        # Debug flags
        self.debug = True
        self.plot_results = True
    
    def detect_tap_improved(self, current_distance, frame_idx):
        """Improved tap detection using state machine approach"""
        if self.previous_distance is None:
            self.previous_distance = current_distance
            return False
        
        # Calculate velocity (change in distance)
        velocity = current_distance - self.previous_distance
        self.velocity_history.append(velocity)
        
        # Check debounce
        if frame_idx - self.last_tap_frame < self.min_frames_between_taps:
            self.previous_distance = current_distance
            return False
        
        # State machine for tap detection
        tap_detected = False
        
        if self.tap_state == "idle":
            # Look for fingers starting to come together (negative velocity)
            if velocity < -self.velocity_threshold:
                self.tap_state = "approaching"
                self.min_distance_in_sequence = current_distance
                self.max_distance_in_sequence = current_distance
                
        elif self.tap_state == "approaching":
            # Update minimum distance
            self.min_distance_in_sequence = min(self.min_distance_in_sequence, current_distance)
            
            # Check if fingers start separating (positive velocity)
            if velocity > self.velocity_threshold:
                # Validate this was a significant approach
                distance_change = self.max_distance_in_sequence - self.min_distance_in_sequence
                if distance_change > self.distance_threshold:
                    self.tap_state = "separating"
                    self.max_distance_in_sequence = current_distance
                else:
                    # Not significant enough, reset
                    self.tap_state = "idle"
                    
        elif self.tap_state == "separating":
            # Update maximum distance
            self.max_distance_in_sequence = max(self.max_distance_in_sequence, current_distance)
            
            # Check if separation is complete (velocity becomes small or negative)
            if abs(velocity) < self.velocity_threshold * 0.5:
                # Validate the complete tap sequence
                total_distance_change = self.max_distance_in_sequence - self.min_distance_in_sequence
                if total_distance_change > self.min_tap_velocity:
                    tap_detected = True
                    self.last_tap_frame = frame_idx
                    self.tap_count += 1
                    self.tap_locations.append(frame_idx)
                
                # Reset state
                self.tap_state = "idle"
                self.min_distance_in_sequence = float('inf')
                self.max_distance_in_sequence = 0
        
        self.previous_distance = current_distance
        return tap_detected
    
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
        
        # Initialize MediaPipe Hands with stricter parameters
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.5,  # Increased for better stability
            min_tracking_confidence=0.5    # Increased for better stability
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
                        
                        # Calculate 2D Euclidean distance (fixed formula)
                        distance = np.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2)
                        
                        # Add to position history
                        self.position_history.append(distance)
                        
                        # Detect tap using improved method
                        is_tap = self.detect_tap_improved(distance, frame_idx)
                        
                        if is_tap:
                            print(f"Tap #{self.tap_count} detected at frame {frame_idx}, time {time_sec:.2f}s")
                        
                        # Calculate current velocity for display
                        if len(self.velocity_history) > 0:
                            velocity = self.velocity_history[-1]
                        
                        # Draw fingertip markers with color based on tap state
                        if is_tap:
                            tip_color = (0, 255, 0)  # Green for tap
                        elif self.tap_state == "approaching":
                            tip_color = (0, 255, 255)  # Yellow for approaching
                        elif self.tap_state == "separating":
                            tip_color = (255, 0, 255)  # Magenta for separating
                        else:
                            tip_color = (0, 0, 255)  # Red for idle
                            
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
                
                cv2.putText(frame, f"State: {self.tap_state}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                if distance is not None:
                    cv2.putText(frame, f"Distance: {distance:.3f}", (10, height - 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if velocity is not None:
                    vel_color = (0, 0, 255) if velocity < 0 else (0, 255, 0)
                    cv2.putText(frame, f"Velocity: {velocity:.3f}", (10, height - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, vel_color, 2)
                
                # Show frame (optional - comment out for faster processing)
                # cv2.imshow('Finger Tap Detection', frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                
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