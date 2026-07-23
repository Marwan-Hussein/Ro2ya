import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class MediaPipeHandTracker:
    """
    A class to run the modern MediaPipe Tasks Hand Landmarker on input images.
    Uses the modern Google MediaPipe Tasks API.
    """
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    MODEL_FILE = "hand_landmarker.task"

    def __init__(self, num_hands: int = 2, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        """
        Initializes the MediaPipe Hand Landmarker.
        Downloads the model file if it is not present in the current directory.
        """
        # Ensure model file is downloaded
        self._ensure_model_exists()

        # Configure MediaPipe Hand Landmarker options
        base_options = python.BaseOptions(model_asset_path=self.MODEL_FILE)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence
        )
        
        print(f"[MediaPipeHandTracker] Initializing Hand Landmarker...")
        self.detector = vision.HandLandmarker.create_from_options(options)

        # Joint connections for drawing the skeleton
        self.HAND_CONNECTIONS = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle
            (5, 9), (9, 10), (10, 11), (11, 12),
            # Ring
            (9, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]

    def _ensure_model_exists(self):
        """Downloads the hand_landmarker.task model file if it does not exist."""
        if not os.path.exists(self.MODEL_FILE):
            print(f"Model file '{self.MODEL_FILE}' not found. Downloading from Google...")
            try:
                # Use a user-agent to avoid potential blockings
                req = urllib.request.Request(
                    self.MODEL_URL, 
                    headers={'User-Agent': 'Mozilla/5.0'}
                )
                with urllib.request.urlopen(req) as response, open(self.MODEL_FILE, 'wb') as out_file:
                    out_file.write(response.read())
                print("Download completed successfully.")
            except Exception as e:
                raise RuntimeError(f"Failed to download the MediaPipe model file: {e}")

    def predict(self, image: np.ndarray):
        """
        Processes an input BGR image and extracts hand landmarks.
        
        Args:
            image (np.ndarray): BGR image (OpenCV format).
            
        Returns:
            list: A list of dicts, where each dict represents a hand and contains:
                - 'landmarks': numpy array of shape (21, 3) containing normalized (x, y, z) coordinates.
                - 'handedness': string ('Left' or 'Right').
                - 'score': float (confidence score).
        """
        if image is None or image.size == 0:
            return []

        # MediaPipe expects RGB format
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # Run inference
        result = self.detector.detect(mp_image)
        
        hands = []
        if result.hand_landmarks:
            for idx, landmarks in enumerate(result.hand_landmarks):
                # Extract coordinates
                coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
                
                # Get handedness label (Left/Right)
                handedness_label = result.handedness[idx][0].category_name
                score = result.handedness[idx][0].score
                
                hands.append({
                    'landmarks': coords,
                    'handedness': handedness_label,
                    'score': score
                })
                
        return hands

    def draw_landmarks(self, image: np.ndarray, hands: list):
        """
        Draws the detected hand landmarks and skeleton connection lines on the image.
        
        Args:
            image (np.ndarray): The BGR image to draw on.
            hands (list): List of detected hands returned by predict().
            
        Returns:
            np.ndarray: The image with drawn landmarks.
        """
        annotated_image = image.copy()
        h, w, _ = annotated_image.shape

        for hand in hands:
            landmarks = hand['landmarks']
            
            # Convert normalized coordinates to pixel coordinates
            pixel_coords = []
            for lm in landmarks:
                px = int(lm[0] * w)
                py = int(lm[1] * h)
                pixel_coords.append((px, py))

            # Draw connection lines
            # Use different colors for Left (greenish) and Right (blueish) hands
            is_left = hand['handedness'] == 'Left'
            line_color = (0, 255, 100) if is_left else (255, 100, 0)
            kpt_color = (0, 200, 255)

            for connection in self.HAND_CONNECTIONS:
                start_idx, end_idx = connection
                cv2.line(annotated_image, pixel_coords[start_idx], pixel_coords[end_idx], line_color, 2)

            # Draw landmark joints
            for px, py in pixel_coords:
                cv2.circle(annotated_image, (px, py), 4, kpt_color, -1)
                cv2.circle(annotated_image, (px, py), 5, (0, 0, 0), 1)

        return annotated_image

    def close(self):
        """Closes the detector."""
        self.detector.close()
