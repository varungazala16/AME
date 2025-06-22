import cv2
import mediapipe as mp
import numpy as np
from collections import deque

class FingerTapDetector:
    def __init__(self, video_path):
        self.video_path = video_path
        
        # Simple, reliable parameters
        self.history_length = 10
        self.min_frames_between_taps = 4
        
        # State tracking
        self.distance_history = deque(maxlen=self.history_length)
        self.tap_count = 0
        self.tap_locations = []
        self.last_tap_frame = -self.min_frames_between_taps
        
        # For analysis
        self.all_distances = []
        self.frame_count = 0
        
        # Adaptive parameters
        self.distance_threshold = None
        self.min_distance_seen = float('inf')
        self.max_distance_seen = 0
        
    def detect_tap(self, current_distance, frame_idx):
        """Simple and effective tap detection"""
        
        # Update min/max for adaptive threshold
        self.min_distance_seen = min(self.min_distance_seen, current_distance)
        self.max_distance_seen = max(self.max_distance_seen, current_distance)
        
        # Calculate adaptive threshold after seeing some data
        if frame_idx > 30:  # After 30 frames
            distance_range = self.max_distance_seen - self.min_distance_seen
            # Threshold is 30% above minimum distance
            self.distance_threshold = self.min_distance_seen + (distance_range * 0.3)
        
        # Add current distance to history
        self.distance_history.append(current_distance)
        
        # Need at least 5 frames for detection
        if len(self.distance_history) < 5:
            return False
            
        # Simple tap detection logic:
        # 1. Current distance is below threshold (fingers close)
        # 2. Was above threshold recently (fingers were apart)
        # 3. Minimum time has passed since last tap
        
        if self.distance_threshold is None:
            return False
            
        # Check if we're currently in a "tap" state (fingers close)
        fingers_close = current_distance < self.distance_threshold
        
        # Check if fingers were apart recently
        recent_distances = list(self.distance_history)[-5:]  # Last 5 frames
        was_apart_recently = any(d > self.distance_threshold for d in recent_distances[:-1])
        
        # Detect tap: fingers were apart, now they're close
        is_tap = (fingers_close and 
                 was_apart_recently and 
                 (frame_idx - self.last_tap_frame) >= self.min_frames_between_taps)
        
        if is_tap:
            self.last_tap_frame = frame_idx
            
        return is_tap
    
    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("❌ Error: Could not open video.")
            return 0

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 240:
            fps = 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"🎬 Processing video at {fps:.1f} FPS, total frames: {total_frames}")

        # Initialize MediaPipe
        mp_hands = mp.solutions.hands

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3
        ) as hands:

            frame_idx = 0
            frames_with_hands = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                # Convert to RGB and process
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    frames_with_hands += 1

                    for hand_landmarks in results.multi_hand_landmarks:
                        # Extract thumb and index tips
                        thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                        index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                        # Compute distance
                        current_distance = np.sqrt((thumb.x - index.x)**2 + (thumb.y - index.y)**2)
                        self.all_distances.append(current_distance)

                        # Detect tap
                        if self.detect_tap(current_distance, frame_idx):
                            self.tap_count += 1
                            self.tap_locations.append(frame_idx)
                            print(f"✅ TAP #{self.tap_count} at frame {frame_idx} ({time_sec:.2f}s)")

                # Optional: print every few seconds
                if frame_idx % int(fps * 2) == 0:
                    print(f"⏳ Frame {frame_idx} | Taps: {self.tap_count}")

        cap.release()

        # Summary
        print("\n📊 ANALYSIS SUMMARY")
        print(f"🎯 Taps detected: {self.tap_count}")
        print(f"📍 Tap frames: {self.tap_locations}")
        print(f"📏 Distance range: {min(self.all_distances):.4f} - {max(self.all_distances):.4f}" if self.all_distances else "No distance data.")
        print(f"🎚️ Threshold used: {self.distance_threshold:.4f}" if self.distance_threshold else "❌ No threshold set")
        if frames_with_hands == 0:
            print("⚠️ No hands detected in video.")

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
        return ["0","0"]