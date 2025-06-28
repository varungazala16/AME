import os
# UNCOMMENT THE LINE BELOW IF RUNNING ON A CLOUD CPU INSTANCE
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import cv2
import statistics
import mediapipe as mp
import math
from collections import deque
from tqdm import tqdm

# --- Configuration Constants (kept at module level for easy access) ---
CONTACT_THR             = 0.022
SEPARATION_THR          = 0.060
REQ_CONTACT_FRAMES      = 7
REQ_SEPARATION_FRAMES   = 4
DIST_SMOOTH_WINDOW      = 5
VIS_THR                 = 0.6

mp_pose = mp.solutions.pose

# --- Core State Machine Class ---
class FootFSM:
    def __init__(self):
        self.contact_frames     = 0
        self.separation_frames  = REQ_SEPARATION_FRAMES
        self.in_contact         = False

    def update(self, gap):
        if gap is None:
            self.contact_frames    = 0
            self.separation_frames = 0
            return 0
        if gap < CONTACT_THR:
            self.contact_frames   += 1
            self.separation_frames = 0
        elif gap > SEPARATION_THR:
            self.separation_frames += 1
            self.contact_frames    = 0
        else:
            self.contact_frames = self.separation_frames = 0
        stepped = 0
        if (not self.in_contact) and self.contact_frames >= REQ_CONTACT_FRAMES:
            stepped = 1
            self.in_contact = True
        if self.in_contact and self.separation_frames >= REQ_SEPARATION_FRAMES:
            self.in_contact = False
        return stepped

# --- Function signature reverted to original (no display parameters) ---
def analyze_tandem_gait(video_path: str, show_progress: bool = False) -> int:
    
    def lmk_dist(lmk1, lmk2):
        if (lmk1.visibility < VIS_THR) or (lmk2.visibility < VIS_THR):
            return None
        return math.hypot(lmk1.x - lmk2.x, lmk1.y - lmk2.y)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video or URL: {video_path}")
        return 0

    left_fsm, right_fsm = FootFSM(), FootFSM()
    left_buf, right_buf = deque(maxlen=DIST_SMOOTH_WINDOW), deque(maxlen=DIST_SMOOTH_WINDOW)
    step_count = 0

    with mp_pose.Pose(
            static_image_mode=False, model_complexity=1,
            enable_segmentation=False, min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_iterator = range(total_frames) if total_frames > 0 else iter(int, 1)
        if show_progress:
            frame_iterator = tqdm(frame_iterator, desc="Processing Tandem Gait")

        for _ in frame_iterator:
            ret, frame = cap.read()
            if not ret: break

            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                d_left  = lmk_dist(lm[mp_pose.PoseLandmark.LEFT_HEEL], lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX])
                d_right = lmk_dist(lm[mp_pose.PoseLandmark.RIGHT_HEEL], lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX])
                
                left_buf.append(d_left)
                right_buf.append(d_right)
                
                gap_left  = statistics.median(left_buf)  if None not in left_buf  else None
                gap_right = statistics.median(right_buf) if None not in right_buf else None
                
                step_count += left_fsm.update(gap_left)
                step_count += right_fsm.update(gap_right)
            
            # All video display logic has been removed from here.

    cap.release()
    return [str(step_count)]

