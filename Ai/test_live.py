import os
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import urllib.request
from collections import deque
import onnxruntime as ort
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

# Paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), "holistic_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
ONNX_MODEL_PATH = os.path.join(os.path.dirname(__file__), "gesture_classifier.onnx")

def ensure_model_exists():
    """Downloads MediaPipe Holistic Landmarker bundle if missing."""
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading MediaPipe Holistic model bundle to {MODEL_PATH}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download completed.")

def get_landmarks(landmarks_field):
    """Safely extracts landmarks from the mediapipe result."""
    if not landmarks_field:
        return []
    first = landmarks_field[0]
    if hasattr(first, '__len__') and not hasattr(first, 'x'):
        return first
    return landmarks_field

def interpolate_and_normalize(frames_array: np.ndarray) -> np.ndarray:
    """Interpolates missing landmarks and centers/scales coordinates using shoulders."""
    if frames_array is None or len(frames_array) == 0:
        return None

    shape = frames_array.shape

    # Linear interpolation both forward and backward to handle missing leading frames
    df = pd.DataFrame(frames_array.reshape(shape[0], -1))
    df.interpolate(method='linear', limit_direction='both', inplace=True)
    frames_array = df.to_numpy().reshape(shape)

    if np.all(np.isnan(frames_array)):
        return None

    # Global Center on Mid-Shoulder
    left_shoulder = frames_array[:, 11, :]
    right_shoulder = frames_array[:, 12, :]
    mid_shoulder = (left_shoulder + right_shoulder) / 2.0
    centered_frames = frames_array - mid_shoulder[:, np.newaxis, :]

    # Global Scale by Shoulder Width
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder, axis=1)
    shoulder_width[shoulder_width == 0] = 1e-6
    normalized_frames = centered_frames / shoulder_width[:, np.newaxis, np.newaxis]

    return np.nan_to_num(normalized_frames, nan=0.0)

def extract_angle_distance_features(hand_landmarks: np.ndarray) -> np.ndarray:
    """Extracts scale-invariant angles and fingertip distances for hand landmarks."""
    num_frames = hand_landmarks.shape[0]
    features = []

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

        wrist = frame_lms[0]
        mcp_middle = frame_lms[9]
        palm_scale = np.linalg.norm(mcp_middle - wrist)
        if palm_scale == 0:
            palm_scale = 1e-6

        # Fingertip relative distances
        tips = [4, 8, 12, 16, 20]
        for tip in tips:
            dist = np.linalg.norm(frame_lms[tip] - wrist) / palm_scale
            frame_features.append(dist)

        # Joint angles
        for finger in fingers:
            for j in range(len(finger) - 2):
                p1, p2, p3 = frame_lms[finger[j]], frame_lms[finger[j+1]], frame_lms[finger[j+2]]
                v1, v2 = p1 - p2, p3 - p2

                norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if norm_v1 == 0 or norm_v2 == 0:
                    angle = 0.0
                else:
                    cosine_angle = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
                    angle = np.arccos(cosine_angle)

                frame_features.append(angle)

        features.append(frame_features)

    return np.array(features)

def main():
    ensure_model_exists()
    
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"Error: ONNX model not found at {ONNX_MODEL_PATH}")
        return
        
    print("Loading ONNX model...")
    session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
    
    # Class mapping based on TARGET_FOLDERS in the training script
    classes = ["back_button", "forward_button"]
    
    # Setup MediaPipe Holistic Landmarker for IMAGE mode (better for simple looping than VIDEO mode)
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE
    )
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
        
    # Match the 720p 30fps training format for temporal and spatial consistency
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    seq_len = 60
    frames_buffer = deque(maxlen=seq_len)
    
    # Cooldown setup for triggering actions
    last_action_time = 0
    cooldown = 2.0  # Seconds to wait before another action can be triggered
    
    print("Starting webcam... Press 'q' to quit.")
    
    with vision.HolisticLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Mirror the frame for intuitive interactions
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detect landmarks
            results = landmarker.detect(mp_image)
            
            # Extract landmarks for current frame (75x3)
            row = np.full((75, 3), np.nan)
            if results:
                # Pose (0..32)
                pose_lms = get_landmarks(results.pose_landmarks)
                for i, lm in enumerate(pose_lms):
                    if i < 33:
                        row[i] = [lm.x, lm.y, lm.z]
                        
                # Left Hand (33..53)
                left_lms = get_landmarks(results.left_hand_landmarks)
                for i, lm in enumerate(left_lms):
                    if i < 21:
                        row[33 + i] = [lm.x, lm.y, lm.z]
                        
                # Right Hand (54..74)
                right_lms = get_landmarks(results.right_hand_landmarks)
                for i, lm in enumerate(right_lms):
                    if i < 21:
                        row[54 + i] = [lm.x, lm.y, lm.z]
                        
            # Always append the row (could be all NaNs if nothing detected, our preprocessing handles NaNs)
            frames_buffer.append(row)
            
            detected_gesture = "None"
            confidence = 0.0
            
            # Once we have enough frames, run inference
            if len(frames_buffer) == seq_len:
                raw_frames = np.array(frames_buffer)
                normalized_frames = interpolate_and_normalize(raw_frames)
                
                # If we had valid landmarks in this window
                if normalized_frames is not None and not np.all(normalized_frames == 0):
                    # Extract invariant features
                    pose_features = normalized_frames[:, :33, :].reshape(normalized_frames.shape[0], -1)
                    lh_features = extract_angle_distance_features(normalized_frames[:, 33:54, :])
                    rh_features = extract_angle_distance_features(normalized_frames[:, 54:75, :])
                    final_features = np.concatenate((pose_features, lh_features, rh_features), axis=1)
                    
                    # Prepare ONNX Inputs
                    input_data = final_features.astype(np.float32)[np.newaxis, ...]  # Shape: (1, 60, feat_dim)
                    mask_data = np.ones((1, seq_len), dtype=bool)                    # Shape: (1, 60)
                    
                    onnx_inputs = {
                        'input': input_data,
                        'mask': mask_data
                    }
                    
                    # Run Model
                    logits = session.run(['output'], onnx_inputs)[0]
                    
                    # Softmax
                    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
                    
                    pred_class = np.argmax(probs, axis=1)[0]
                    confidence = probs[0][pred_class]
                    
                    # Threshold for triggering
                    if confidence > 0.85:
                        detected_gesture = classes[pred_class]
                        
                        current_time = time.time()
                        if current_time - last_action_time > cooldown:
                            print(f"\n---> ACTION DETECTED: {detected_gesture} (Confidence: {confidence:.2f})")
                            last_action_time = current_time
                            
                            # --- UNCOMMENT BELOW TO ACTUALLY TRIGGER BROWSER ACTIONS ---
                            # try:
                            #     import pyautogui
                            #     if detected_gesture == "back_button":
                            #         pyautogui.hotkey('browserback')
                            #     elif detected_gesture == "forward_button":
                            #         pyautogui.hotkey('browserforward')
                            # except ImportError:
                            #     print("pyautogui not installed. Run `pip install pyautogui` to enable actual keyboard control.")
                            # -----------------------------------------------------------

            # -----------------------------------
            # Draw UI on the frame
            # -----------------------------------
            cv2.putText(frame, f"Gesture: {detected_gesture}", (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, f"Conf: {confidence:.2f}", (10, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                        
            time_since_action = time.time() - last_action_time
            if time_since_action < cooldown:
                # Show cooldown bar or text
                cv2.putText(frame, f"COOLDOWN: {cooldown - time_since_action:.1f}s", (10, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            
            # Add some instructions
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("Live Gesture Recognition", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
