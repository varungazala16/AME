import cv2
import gdown
import keras
import numpy as np
import pathlib
import requests
import tensorflow as tf
import zipfile
from typing import Tuple

# Assume your 'utils' folder is correctly set up
from utils.configurations import KEYPOINT_DICT as kp_names
from utils.predictor import preprocess, get_prediction

# -----------------------------------------------------------------------------
# Updated Google Drive download function
# -----------------------------------------------------------------------------

def download_and_extract_gdrive_zip(gdrive_url: str, output_dir: str) -> Tuple[pathlib.Path, bool]:
    """
    Download a ZIP from Google Drive, extract it, and find the actual model directory,
    even if it's nested.
    """
    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    final_model_dir = out_dir / "model"

    # Check if the model is already present and correctly structured
    if (final_model_dir / "saved_model.pb").exists():
        print("Model already present and structured correctly.")
        return final_model_dir, True

    # --- Download Logic (remains the same) ---
    zip_path = out_dir / "model.zip"
    if not zip_path.exists(): # Only download if the zip isn't already there
        print("Downloading model from Google Drive using gdown…")
        try:
            gdown.download(url=gdrive_url, output=str(zip_path), quiet=False)
        except Exception as e:
            print(f"ERROR: gdown failed to download the file. Reason: {e}")
            return None, False

    if not zip_path.exists() or zip_path.stat().st_size == 0:
        print("ERROR: Download did not create a valid file.")
        return None, False

    # --- Extraction Logic (remains the same) ---
    print("Download complete. Extracting ZIP archive...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
    except zipfile.BadZipFile:
        print("ERROR: The downloaded file is not a valid ZIP archive.")
        return None, False
    finally:
        # We can clean up the zip file now
        if zip_path.exists():
            zip_path.unlink()

    # *** CHANGE 2: Verify the nested structure and return the correct path ***
    if (final_model_dir / "saved_model.pb").exists():
        print(f"Model extracted successfully into: {final_model_dir}")
        return final_model_dir, True
    else:
        print(f"ERROR: saved_model.pb not found in the expected nested path: {final_model_dir}")
        return None, False

# -----------------------------------------------------------------------------
# Model + scoring logic
# -----------------------------------------------------------------------------

def build_interpreter(model_dir: str):
    """Load a TensorFlow SavedModel via TFSMLayer."""
    return keras.layers.TFSMLayer(model_dir, call_endpoint="serving_default")

def marching_score(
    video_path: str,
    trial_speed: str,
    *,
    is_child: bool = False,
    model_path: str = "modelMTTB", # This is now just the parent/cache directory
    gdrive_model_url: str = "https://drive.google.com/uc?export=download&id=1teO7SqOphCIU8FMSyTol-Z2wxyCrRQau",
    show: bool = False,
) -> list:
    """Compute the marching score for a video."""

    # 1) Ensure model is present
    if gdrive_model_url:
        # 'actual_model_path' will now be '.../modelMTTB/model'
        actual_model_path, ok = download_and_extract_gdrive_zip(gdrive_model_url, model_path)
        if not ok:
            raise RuntimeError("Model download or extraction failed. See messages above.")
    else:
        # If not downloading, assume the nested structure exists
        actual_model_path = pathlib.Path(model_path) / "model"
        if not (actual_model_path / "saved_model.pb").exists():
            raise FileNotFoundError(f"SavedModel not found in expected path: {actual_model_path}")

    # *** CHANGE 3: Load the interpreter from the correct, nested path ***
    print(f"Loading model from: {actual_model_path}")
    interpreter = build_interpreter(str(actual_model_path))

    # 3) Video handling (The rest of the function is unchanged)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    threshold = 15 if is_child else 10
    score = 0
    time_intervals = {
        ("f", True): (5, 9),
        ("f", False): (3, 24),
        ("s", True): (5, 12),
        ("s", False): (3, 23),
    }
    try:
        start_time, end_time = time_intervals[(trial_speed.lower(), is_child)]
    except KeyError as err:
        raise ValueError("trial_speed must be 's' or 'f'") from err

    prev_left_y = prev_right_y = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        tensor, _, _ = preprocess(frame[..., ::-1], input_size=(512, 512))
        fh, fw, _ = frame.shape
        kpts, *_ = get_prediction(tensor, interpreter, from_class=1)
        kpts = kpts[..., ::-1] * np.array([fw, fh])

        left_y = kpts[0, kp_names["left_ankle"]][1]
        right_y = kpts[0, kp_names["right_ankle"]][1]
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        if start_time <= t <= end_time:
            if prev_left_y is not None and (prev_left_y - left_y > threshold) and (left_y - prev_left_y < -threshold):
                score += 1
            if prev_right_y is not None and (prev_right_y - right_y > threshold) and (right_y - prev_right_y < -threshold):
                score += 1

        prev_left_y, prev_right_y = left_y, right_y

        if show:
            for kp in kpts[0]:
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if show:
        cv2.destroyAllWindows()
        
    return [str(score)]