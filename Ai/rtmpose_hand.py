import cv2
import numpy as np
import onnxruntime as ort
from rtmlib import Hand, draw_skeleton

class RTMPoseHandTracker:
    """
    A class to run RTMPose hand landmark estimation on input images.
    Uses rtmlib and onnxruntime under the hood for fast, lightweight inference.
    """
    def __init__(self, device: str = None, backend: str = 'onnxruntime', to_openpose: bool = False):
        """
        Initializes the RTMPose hand tracking pipeline.
        
        Args:
            device (str): 'cpu' or 'cuda'. If None, automatically detects if CUDA is available.
            backend (str): Inference backend (default: 'onnxruntime').
            to_openpose (bool): Maps landmarks to standard 21-keypoint OpenPose format if True.
        """
        if device is None:
            # Auto-detect CUDA availability
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available_providers:
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = device
            
        print(f"[RTMPoseHandTracker] Initializing on device: {self.device}")
        
        # Initialize rtmlib Hand solution (handles detection + pose estimation)
        self.hand_model = Hand(
            to_openpose=to_openpose,
            backend=backend,
            device=self.device
        )
        self.to_openpose = to_openpose

    def predict(self, image: np.ndarray):
        """
        Processes an input BGR image and extracts hand landmarks.
        
        Args:
            image (np.ndarray): BGR image (OpenCV format).
            
        Returns:
            tuple: (keypoints, scores)
                - keypoints: numpy array of shape (num_hands, 21, 2) containing (x, y) coordinates.
                - scores: numpy array of shape (num_hands, 21) containing confidence scores.
        """
        if image is None or image.size == 0:
            return np.zeros((0, 21, 2), dtype=np.float32), np.zeros((0, 21), dtype=np.float32)
            
        keypoints, scores = self.hand_model(image)
        return keypoints, scores

    def draw_landmarks(self, image: np.ndarray, keypoints: np.ndarray, scores: np.ndarray, threshold: float = 0.4):
        """
        Draws the detected hand landmarks and skeleton connection lines on the image.
        
        Args:
            image (np.ndarray): The image to draw on.
            keypoints (np.ndarray): Hand keypoints from predict().
            scores (np.ndarray): Keypoint confidence scores from predict().
            threshold (float): Confidence threshold for rendering keypoints.
            
        Returns:
            np.ndarray: The image with drawn landmarks.
        """
        if keypoints is None or len(keypoints) == 0:
            return image.copy()
            
        return draw_skeleton(
            image,
            keypoints,
            scores,
            openpose_skeleton=self.to_openpose,
            kpt_thr=threshold
        )
