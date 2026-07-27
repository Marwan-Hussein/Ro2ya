import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import glob
import math
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import threading
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
MODEL_PATH = "holistic_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"

progress_lock = threading.Lock()
completed_count = 0

def ensure_model_exists():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading MediaPipe Holistic model bundle to {MODEL_PATH}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download completed.")

def get_landmarks(landmarks_field):
    if not landmarks_field:
        return []
    first = landmarks_field[0]
    # Check if first is a sub-list/container and not a single landmark object (which has attribute 'x')
    if hasattr(first, '__len__') and not hasattr(first, 'x'):
        return first
    return landmarks_field

def extract_landmarks_from_video(video_path):
    ensure_model_exists()
    cap = cv2.VideoCapture(video_path)
    frames_data = []
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO
    )
    with vision.HolisticLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = landmarker.detect_for_video(mp_image, timestamp_ms)
            row = np.full((75, 3), np.nan)

            pose_lms = get_landmarks(results.pose_landmarks)
            for i, lm in enumerate(pose_lms):
                if i < 33: row[i] = [lm.x, lm.y, lm.z]

            left_lms = get_landmarks(results.left_hand_landmarks)
            for i, lm in enumerate(left_lms):
                if i < 21: row[33 + i] = [lm.x, lm.y, lm.z]

            right_lms = get_landmarks(results.right_hand_landmarks)
            for i, lm in enumerate(right_lms):
                if i < 21: row[54 + i] = [lm.x, lm.y, lm.z]

            frames_data.append(row)
    cap.release()
    return np.array(frames_data)

def interpolate_and_normalize(frames_array):
    if frames_array is None or len(frames_array) == 0: return None
    shape = frames_array.shape
    df = pd.DataFrame(frames_array.reshape(shape[0], -1))
    df.interpolate(method='linear', limit_direction='forward', inplace=True)
    frames_array = df.to_numpy().reshape(shape)
    if np.all(np.isnan(frames_array)): return None

    left_shoulder, right_shoulder = frames_array[:, 11, :], frames_array[:, 12, :]
    mid_shoulder = (left_shoulder + right_shoulder) / 2.0
    centered_frames = frames_array - mid_shoulder[:, np.newaxis, :]

    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder, axis=1)
    shoulder_width[shoulder_width == 0] = 1e-6
    normalized_frames = centered_frames / shoulder_width[:, np.newaxis, np.newaxis]
    return np.nan_to_num(normalized_frames, nan=0.0)

def process_video_pipeline(video_path, output_root, base_path, total_total):
    global completed_count
    try:
        rel_path = os.path.relpath(video_path, base_path)
        output_path = os.path.join(output_root, os.path.splitext(rel_path)[0] + ".npy")

        raw_frames = extract_landmarks_from_video(video_path)
        if len(raw_frames) > 0:
            normalized_frames = interpolate_and_normalize(raw_frames)
            if normalized_frames is not None:
                final_features = normalized_frames.reshape(normalized_frames.shape[0], -1)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                np.save(output_path, final_features)

        with progress_lock:
            completed_count += 1
            if completed_count % 10 == 0:
                print(f"Progress: {completed_count}/{total_total} files processed...")

    except Exception as e:
        print(f"Error processing {video_path}: {e}")

def process_entire_dataset(base_path, output_root, max_workers=2):
    global completed_count
    completed_count = 0
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.MP4']
    all_videos = []
    for ext in video_extensions:
        all_videos.extend(glob.glob(os.path.join(base_path, '**', ext), recursive=True))

    total_vids = len(all_videos)
    if total_vids == 0:
        print("No videos found. Check your base_path.")
        return

    print(f"Found {total_vids} videos. Starting parallel processing with {max_workers} workers...")
    worker_func = partial(process_video_pipeline, output_root=output_root, base_path=base_path, total_total=total_vids)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(worker_func, all_videos)

    print(f"Processing Complete. Final count: {completed_count}/{total_vids}")

if __name__ == "__main__":
    base_path = r"d:\projects\Ro2ya\Ai\gestures_RO2YA"
    output_root = r"d:\projects\Ro2ya\Ai\processed_features"
    process_entire_dataset(base_path, output_root, max_workers=4)