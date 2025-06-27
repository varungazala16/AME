import cv2
import statistics
import mediapipe as mp
import math
from collections import deque
from tqdm import tqdm

# --- Configuration Constants (kept at module level for easy access) ---
CONTACT_THR             = 0.022   # "touching" if median-gap < this
SEPARATION_THR          = 0.060   # "apart"   if median-gap > this
REQ_CONTACT_FRAMES      = 7       # consecutive "touching" frames
REQ_SEPARATION_FRAMES   = 4       # consecutive "apart"   frames
DIST_SMOOTH_WINDOW      = 5       # frames in median filter
VIS_THR                 = 0.6 

mp_pose = mp.solutions.pose

# --- Core State Machine Class (kept at module level) ---
class FootFSM:
    """
    Finite-state machine that decides when to add a step
    using debounced contact / separation windows.
    """
    def __init__(self):
        self.contact_frames     = 0
        self.separation_frames  = REQ_SEPARATION_FRAMES
        self.in_contact         = False

    def update(self, gap):
        """
        gap : current smoothed distance (None → landmark missing)
        Returns 1 if a *new* step is recognized this frame, else 0.
        """
        if gap is None:
            self.contact_frames    = 0
            self.separation_frames = 0
            return 0

        # Check contact condition
        if gap < CONTACT_THR:
            self.contact_frames   += 1
            self.separation_frames = 0
        elif gap > SEPARATION_THR:
            self.separation_frames += 1
            self.contact_frames    = 0
        else:                             
            self.contact_frames = self.separation_frames = 0

        # Debounced transition logic
        stepped = 0
        if (not self.in_contact) and self.contact_frames >= REQ_CONTACT_FRAMES:
            stepped = 1
            self.in_contact = True
        if self.in_contact and self.separation_frames >= REQ_SEPARATION_FRAMES:
            self.in_contact = False
        return stepped

def analyze_tandem_gait(video_path: str, show_progress: bool = False) -> int:
    
    def lmk_dist(lmk1, lmk2):
        """Return normalized heel-toe distance or None if either landmark is invisible."""
        if (lmk1.visibility < VIS_THR) or (lmk2.visibility < VIS_THR):
            return None
        return math.hypot(lmk1.x - lmk2.x, lmk1.y - lmk2.y)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video: {video_path}")
        return 0

    left_fsm, right_fsm = FootFSM(), FootFSM()
    left_buf, right_buf = deque(maxlen=DIST_SMOOTH_WINDOW), deque(maxlen=DIST_SMOOTH_WINDOW)
    step_count = 0

    with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Set up the iterator, with an optional progress bar
        frame_iterator = range(total_frames)
        if show_progress:
            frame_iterator = tqdm(frame_iterator, desc="Processing Tandem Gait")

        for _ in frame_iterator:
            ret, frame = cap.read()
            if not ret: break

            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                # Calculate the heel-to-opposite-toe distances
                d_left  = lmk_dist(lm[mp_pose.PoseLandmark.LEFT_HEEL],
                                   lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX])
                d_right = lmk_dist(lm[mp_pose.PoseLandmark.RIGHT_HEEL],
                                   lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX])

                # Add distances to their respective buffers for smoothing
                left_buf.append(d_left)
                right_buf.append(d_right)

                # Get the smoothed (median) gap values
                gap_left  = statistics.median(left_buf)  if None not in left_buf  else None
                gap_right = statistics.median(right_buf) if None not in right_buf else None

                # Update state machines and increment step count if a new step is detected
                step_count += left_fsm.update(gap_left)
                step_count += right_fsm.update(gap_right)

    cap.release()
    return [str(step_count)]