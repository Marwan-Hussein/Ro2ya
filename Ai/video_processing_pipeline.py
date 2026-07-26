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

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "holistic_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"

def ensure_model_exists():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading MediaPipe Holistic model bundle to {MODEL_PATH}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download completed.")

def extract_landmarks_from_video(video_path):
    """
    Extracts 75 landmarks from each frame of the video using MediaPipe Tasks API.
    Pose (33), Left Hand (21), Right Hand (21)
    Returns: numpy array of shape (num_frames, 75, 3)
    """
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
            if not ret:
                break
                
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            
            # Convert BGR to RGB and wrap into mp.Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detect landmarks for current video frame
            results = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            row = np.full((75, 3), np.nan)
            
            # Extract Pose (33 landmarks)
            if results.pose_landmarks and len(results.pose_landmarks) > 0:
                for i, lm in enumerate(results.pose_landmarks[0]):
                    row[i] = [lm.x, lm.y, lm.z]
                    
            # Extract Left Hand (indices 33 to 53)
            if results.left_hand_landmarks and len(results.left_hand_landmarks) > 0:
                for i, lm in enumerate(results.left_hand_landmarks[0]):
                    row[33 + i] = [lm.x, lm.y, lm.z]
                    
            # Extract Right Hand (indices 54 to 74)
            if results.right_hand_landmarks and len(results.right_hand_landmarks) > 0:
                for i, lm in enumerate(results.right_hand_landmarks[0]):
                    row[54 + i] = [lm.x, lm.y, lm.z]
                    
            frames_data.append(row)
            
    cap.release()
    return np.array(frames_data)


def interpolate_and_normalize(frames_array):
    """
    Fixes the 5 major flaws (temporal interpolation, global anchoring to mid-shoulder, global scaling by shoulder width).
    frames_array: shape (num_frames, 75, 3)
    Returns: normalized frames_array of shape (num_frames, 75, 3)
    """
    if frames_array is None or len(frames_array) == 0:
        return None

    shape = frames_array.shape
    
    # Issue 4 Fix: We extract directly with np.nan now, so no exact zero-mask check is needed.

    # --- TEMPORAL INTERPOLATION ---
    df = pd.DataFrame(frames_array.reshape(shape[0], -1))
    
    # Issue 2 Fix: Use limit_direction='forward' to prevent data leakage from the future during real-time tracking
    df.interpolate(method='linear', limit_direction='forward', inplace=True)
    
    # Issue 1 Fix: Do NOT fillna(0.0) here. Leave missing hands as NaN so they 
    # don't get shifted by mid-shoulder subtraction into "ghost hands".
    frames_array = df.to_numpy().reshape(shape)

    if np.all(np.isnan(frames_array)):
        return None

    # --- GLOBAL ANCHOR ---
    left_shoulder = frames_array[:, 11, :]
    right_shoulder = frames_array[:, 12, :]
    mid_shoulder = (left_shoulder + right_shoulder) / 2.0
    
    # Center only the valid landmarks (NaNs remain NaN)
    centered_frames = frames_array - mid_shoulder[:, np.newaxis, :]

    # --- GLOBAL SCALING ---
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder, axis=1)
    # Prevent division by zero
    shoulder_width[shoulder_width == 0] = 1e-6
    normalized_frames = centered_frames / shoulder_width[:, np.newaxis, np.newaxis]

    # Now that centering/scaling is done, we can safely fill the un-interpolated missing points with 0.0
    normalized_frames = np.nan_to_num(normalized_frames, nan=0.0)

    return normalized_frames


def handedness_normalization(hand_landmarks, is_left_hand=False):
    """
    Method 1: Handedness Normalization at Inference
    Centers hand landmarks around the wrist and flips X if it is a left hand,
    converting it into a 'right hand' coordinate space.
    hand_landmarks: shape (num_frames, 21, 3)
    Returns: normalized hand of shape (num_frames, 21, 3)
    """
    # Center around the wrist (index 0)
    wrist = hand_landmarks[:, 0, :]
    centered_hand = hand_landmarks - wrist[:, np.newaxis, :]
    
    if is_left_hand:
        # Flip X coordinates
        centered_hand[:, :, 0] *= -1
        
    return centered_hand


def synthetic_hand_augmentation(hand_landmarks):
    """
    Method 2: Synthetic Data Augmentation
    Takes a hand and creates a synthetic opposite-hand sample by flipping X 
    relative to the palm center (approximated by MCP of middle finger, index 9).
    hand_landmarks: shape (num_frames, 21, 3)
    Returns: synthetic_hand of shape (num_frames, 21, 3)
    """
    palm_center = hand_landmarks[:, 9, :]
    centered_hand = hand_landmarks - palm_center[:, np.newaxis, :]
    
    # Flip X
    centered_hand[:, :, 0] *= -1
    
    # Move back to original palm center position
    synthetic_hand = centered_hand + palm_center[:, np.newaxis, :]
    return synthetic_hand


def extract_angle_distance_features(hand_landmarks):
    """
    Method 3: Angle / Distance-Based Features
    Extracts rotation/translation invariant features from the hand.
    hand_landmarks: shape (num_frames, 21, 3)
    Returns: features array of shape (num_frames, num_features)
    """
    num_frames = hand_landmarks.shape[0]
    features = []
    
    # Indices for the 5 fingers (MCP, PIP, DIP, TIP)
    fingers = [
        [1, 2, 3, 4],     # Thumb
        [5, 6, 7, 8],     # Index
        [9, 10, 11, 12],  # Middle
        [13, 14, 15, 16], # Ring
        [17, 18, 19, 20]  # Pinky
    ]
    
    for i in range(num_frames):
        frame_lms = hand_landmarks[i]
        frame_features = []
        
        # 1. Euclidean distances between fingertips (4, 8, 12, 16, 20) and wrist (0)
        wrist = frame_lms[0]
        mcp_middle = frame_lms[9]
        palm_scale = np.linalg.norm(mcp_middle - wrist)
        if palm_scale == 0:
            palm_scale = 1e-6
            
        tips = [4, 8, 12, 16, 20]
        for tip in tips:
            # Issue 3 Fix: Normalize fingertip distances by palm scale instead of relying on global shoulder width
            dist = np.linalg.norm(frame_lms[tip] - wrist) / palm_scale
            frame_features.append(dist)
            
        # 2. Joint angles between adjacent fingers (e.g. angle at PIP joint)
        for finger in fingers:
            for j in range(len(finger) - 2):
                p1 = frame_lms[finger[j]]
                p2 = frame_lms[finger[j+1]]
                p3 = frame_lms[finger[j+2]]
                
                v1 = p1 - p2
                v2 = p3 - p2
                
                # Cosine angle
                norm_v1 = np.linalg.norm(v1)
                norm_v2 = np.linalg.norm(v2)
                if norm_v1 == 0 or norm_v2 == 0:
                    angle = 0.0
                else:
                    cosine_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
                    # Clip to valid range to avoid numerical issues with arccos
                    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
                    angle = np.arccos(cosine_angle)
                
                frame_features.append(angle)
                
        features.append(frame_features)
        
    return np.array(features)


def process_video_pipeline(video_path, output_path, feature_method="raw"):
    """
    Main pipeline function:
    1. Extracts landmarks
    2. Interpolates and normalizes to mid-shoulder
    3. Applies hand specific feature extraction based on method selected:
       - 'raw': just returns the (num_frames, 225) normalized array
       - 'invariant': applies Method 3 to both hands and concatenates
    """
    raw_frames = extract_landmarks_from_video(video_path)
    if raw_frames is None or raw_frames.size == 0 or len(raw_frames) == 0:
        print(f"[SKIP] No frames extracted from: {video_path}")
        return
        
    normalized_frames = interpolate_and_normalize(raw_frames)
    if normalized_frames is None or np.all(normalized_frames == 0):
        print(f"[SKIP] No valid landmarks in: {video_path}")
        return
        
    if feature_method == "raw":
        # Flatten (num_frames, 75, 3) to (num_frames, 225)
        final_features = normalized_frames.reshape(normalized_frames.shape[0], -1)
        
    elif feature_method == "invariant":
        pose_features = normalized_frames[:, :33, :].reshape(normalized_frames.shape[0], -1)
        left_hand = normalized_frames[:, 33:54, :]
        right_hand = normalized_frames[:, 54:75, :]

        lh_features = extract_angle_distance_features(left_hand)
        rh_features = extract_angle_distance_features(right_hand)

        final_features = np.concatenate((pose_features, lh_features, rh_features), axis=1)
        
    else:
        raise ValueError(f"Unknown feature method: {feature_method}")

    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    np.save(output_path, final_features)
    print(f"[SUCCESS] {os.path.basename(video_path)} -> {output_path} | Shape: {final_features.shape}")


def run_batch_processing(base_dir, target_folders, output_dir, feature_method="raw"):
    """
    Locates video files from target subdirectories and processes them sequentially.
    """
    video_extensions = ('*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm')
    processed_count = 0

    print(f"\n--- Starting Batch Processing ---")
    print(f"Base Directory: {base_dir}")
    print(f"Folders Targeted: {target_folders}\n")

    for folder in target_folders:
        folder_path = os.path.join(base_dir, folder)

        if not os.path.exists(folder_path):
            print(f"Warning: Directory '{folder_path}' does not exist.")
            continue

        video_files = []
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(folder_path, ext)))

        print(f"\nFound {len(video_files)} video(s) in folder '{folder}'")

        for video_path in video_files:
            file_name = os.path.basename(video_path)
            save_name = os.path.splitext(file_name)[0] + ".npy"
            save_path = os.path.join(output_dir, folder, save_name)

            process_video_pipeline(video_path, save_path, feature_method=feature_method)
            processed_count += 1

    print(f"\nBatch processing complete! Total processed: {processed_count}")


def compile_dataset(processed_dir, target_folders, max_sequence_length=60):
    """
    Loads saved .npy files, pads/truncates sequences, and builds X and y arrays.
    """
    X_list, y_list = [], []
    label_map = {folder: i for i, folder in enumerate(target_folders)}

    for folder, label in label_map.items():
        folder_path = os.path.join(processed_dir, folder)
        npy_files = glob.glob(os.path.join(folder_path, "*.npy"))

        for file in npy_files:
            features = np.load(file)
            num_frames, feature_dim = features.shape

            # Zero-padding or truncating
            if num_frames < max_sequence_length:
                padding = np.zeros((max_sequence_length - num_frames, feature_dim))
                padded_features = np.vstack([features, padding])
            else:
                padded_features = features[:max_sequence_length, :]

            X_list.append(padded_features)
            y_list.append(label)

    X = np.array(X_list)
    y = np.array(y_list)

    print(f"\n--- Dataset Summary ---")
    print(f"X shape: {X.shape} (Samples, Sequence Length, Features)")
    print(f"y shape: {y.shape}")
    print(f"Label map: {label_map}")
    return X, y, label_map

if __name__ == "__main__":
    # Example usage:
    # process_video_pipeline("dataset/videos/sample.mp4", "dataset/features/sample.npy", feature_method="raw")
    pass
