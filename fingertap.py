import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from scipy.signal import find_peaks  # NEW

def estimate_tap_speed(distances, fps):
    """Estimate average tap interval from initial distance curve."""
    arr = np.array(distances)
    inverted = -arr
    # Find local minima (candidate taps), minimum 0.10s apart
    peaks, _ = find_peaks(inverted, distance=int(0.10 * fps))
    if len(peaks) < 2:
        return None
    intervals = np.diff(peaks) / fps  # seconds
    median_interval = np.median(intervals)
    # Debounce set to 70% of median tap interval, at least 100ms
    return max(0.10, 0.7 * median_interval)

class FingerTapDetector:
    def __init__(self, video_path):
        self.video_path = video_path
        
        # State tracking
        self.history_length = 10
        self.distance_history = deque(maxlen=self.history_length)
        self.tap_count = 0
        self.tap_locations = []
        self.last_tap_frame = -1  # Will be set after calibration
        
        # For analysis
        self.all_distances = []
        self.frame_count = 0
        
        # Adaptive parameters
        self.distance_threshold = None
        self.min_distance_seen = float('inf')
        self.max_distance_seen = 0
        
        # Debounce, will be set dynamically after calibration
        self.min_frames_between_taps = 3  # placeholder, real value after calibration
        self.in_tap = False  # Tap state fla3

    def detect_tap(self, current_distance, frame_idx):
        # Update min/max for adaptive threshold
        self.min_distance_seen = min(self.min_distance_seen, current_distance)
        self.max_distance_seen = max(self.max_distance_seen, current_distance)

        if frame_idx > 30:
            distance_range = self.max_distance_seen - self.min_distance_seen
            self.distance_threshold = self.min_distance_seen + (distance_range * 0.3)
        
        self.distance_history.append(current_distance)
        if len(self.distance_history) < 5 or self.distance_threshold is None:
            return False

        fingers_close = current_distance < self.distance_threshold

        # -- Tap state logic --
        is_tap = False
        # Only count a tap if we just transitioned from apart -> close
        if fingers_close and not self.in_tap and (frame_idx - self.last_tap_frame) >= self.min_frames_between_taps:
            is_tap = True
            self.in_tap = True
            self.last_tap_frame = frame_idx
        elif not fingers_close and self.in_tap:
            # Reset tap state when fingers are apart
            self.in_tap = False

        return is_tap

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("❌ Error: Could not open video.")
            return 0
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 240:
            fps = 30
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"🎬 Processing video: {width}x{height}, {fps:.1f} FPS, {total_frames} frames")
        
        # Initialize MediaPipe
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3
        ) as hands:
            
            # --- Calibration Phase ---
            calibration_seconds = 2.5
            calibration_frames = int(calibration_seconds * fps)
            calibration_distances = []
            frame_idx = 0
            while frame_idx < calibration_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    current_distance = np.sqrt((thumb.x - index.x)**2 + (thumb.y - index.y)**2)
                    calibration_distances.append(current_distance)
            
            # --- Estimate Tap Speed & Set Debounce ---
            min_seconds_between_taps = 0.18  # fallback
            if len(calibration_distances) > 8:
                adaptive_debounce = estimate_tap_speed(calibration_distances, fps)
                if adaptive_debounce:
                    min_seconds_between_taps = adaptive_debounce
            self.min_frames_between_taps = max(1, int(min_seconds_between_taps * fps))
            print(f"🛠️ Adaptive debounce: {self.min_frames_between_taps} frames ({min_seconds_between_taps:.3f}s)")
            
            # Init last_tap_frame for new timeline
            self.last_tap_frame = frame_idx - self.min_frames_between_taps
            
            # --- Main Detection Loop ---
            frames_with_hands = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_idx += 1
                time_sec = frame_idx / fps
                
                # Process frame
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                
                current_distance = None
                is_tap_frame = False
                
                if results.multi_hand_landmarks:
                    frames_with_hands += 1
                    
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw hand landmarks
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                        
                        # Get thumb and index finger tips
                        thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                        index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        
                        # Calculate distance between fingertips
                        current_distance = np.sqrt((thumb.x - index.x)**2 + (thumb.y - index.y)**2)
                        self.all_distances.append(current_distance)
                        
                        # Detect tap
                        is_tap_frame = self.detect_tap(current_distance, frame_idx)
                        
                        if is_tap_frame:
                            self.tap_count += 1
                            self.tap_locations.append(frame_idx)
                            print(f"✅ TAP #{self.tap_count} detected at frame {frame_idx} (time: {time_sec:.2f}s)")
                        
                        # Draw fingertips
                        thumb_pixel = (int(thumb.x * width), int(thumb.y * height))
                        index_pixel = (int(index.x * width), int(index.y * height))
                        
                        # Color based on state
                        if is_tap_frame:
                            color = (0, 255, 0)  # Green for tap
                            cv2.circle(frame, thumb_pixel, 15, color, 3)
                            cv2.circle(frame, index_pixel, 15, color, 3)
                            cv2.putText(frame, "TAP!", (thumb_pixel[0] - 30, thumb_pixel[1] - 40),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                        elif current_distance < self.distance_threshold if self.distance_threshold else False:
                            color = (0, 255, 255)  # Yellow for close
                            cv2.circle(frame, thumb_pixel, 10, color, -1)
                            cv2.circle(frame, index_pixel, 10, color, -1)
                        else:
                            color = (0, 0, 255)  # Red for apart
                            cv2.circle(frame, thumb_pixel, 8, color, -1)
                            cv2.circle(frame, index_pixel, 8, color, -1)
                        
                        # Draw line between fingers
                        cv2.line(frame, thumb_pixel, index_pixel, color, 2)
                
                # Display information
                cv2.putText(frame, f"TAPS: {self.tap_count}", (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                
                if current_distance is not None:
                    cv2.putText(frame, f"Distance: {current_distance:.4f}", (10, height - 80),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if self.distance_threshold is not None:
                    cv2.putText(frame, f"Threshold: {self.distance_threshold:.4f}", (10, height - 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    # Show state
                    if current_distance is not None:
                        state = "CLOSE" if current_distance < self.distance_threshold else "APART"
                        state_color = (0, 255, 255) if state == "CLOSE" else (0, 0, 255)
                        cv2.putText(frame, f"State: {state}", (10, height - 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
                
                # Progress update
                if frame_idx % (int(fps) * 2) == 0:  # Every 2 seconds
                    progress = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
                    print(f"⏳ Progress: {progress:.1f}% - Frame {frame_idx}/{total_frames} - Taps: {self.tap_count}")
                
                # Optional: Show video (uncomment to see real-time processing)
                # cv2.imshow('Finger Tap Detection', frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Print summary
        print(f"\n📊 DETECTION SUMMARY:")
        print(f"🎯 Total taps detected: {self.tap_count}")
        print(f"📹 Total frames processed: {frame_idx}")
        print(f"👋 Frames with hands detected: {frames_with_hands}")
        print(f"📍 Tap frame locations: {self.tap_locations}")
        
        if self.all_distances:
            print(f"📏 Distance range: {min(self.all_distances):.4f} - {max(self.all_distances):.4f}")
            print(f"🎚️ Threshold used: {self.distance_threshold:.4f}" if self.distance_threshold else "❌ No threshold calculated")
        
        if frames_with_hands == 0:
            print("⚠️  WARNING: No hands detected in video! Check video quality and hand visibility.")
        elif self.tap_count == 0:
            print("⚠️  No taps detected. Try adjusting hand position or tap more distinctly.")
        
        return self.tap_count

def count_taps(video_path):
    """Count finger taps in a video file"""
    print(f"🚀 Starting finger tap detection...")
    print(f"📁 Video file: {video_path}")
    
    try:
        detector = FingerTapDetector(video_path)
        tap_count = detector.process_video()
        
        print(f"\n✅ FINAL RESULT: {tap_count} finger taps detected")
        return [str(tap_count)]
    
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        return ["0"]
