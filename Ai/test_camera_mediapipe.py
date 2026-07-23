import cv2
import time
from mediapipe_hand import MediaPipeHandTracker

def main():
    # Initialize the tracker
    tracker = MediaPipeHandTracker()
    
    # Open default camera
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
        
    print("Camera opened successfully. Press 'q' or 'Esc' to exit.")
    
    # Variables for calculating FPS
    prev_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
            
        # Flip the frame horizontally for a natural mirror effect
        frame = cv2.flip(frame, 1)
        
        # Run inference to get hand landmarks
        hands = tracker.predict(frame)
        
        # Draw the hand landmarks on the frame
        annotated_frame = tracker.draw_landmarks(frame, hands)
        
        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if prev_time > 0 else 0.0
        prev_time = curr_time
        
        # Display the number of detected hands and their labels
        num_hands = len(hands)
        labels = [f"{h['handedness']} ({h['score']:.2f})" for h in hands]
        labels_str = ", ".join(labels) if labels else "None"
        
        cv2.putText(annotated_frame, f"Hands: {num_hands} [{labels_str}]", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                    
        # Overlay the FPS on the top-right corner
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (frame.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                    
        # Display the frame
        cv2.imshow("MediaPipe Hand Tracking", annotated_frame)
        
        # Handle keystrokes (wait for 1 ms)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or Esc
            break
            
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    tracker.close()
    print("Resources released. Exiting.")

if __name__ == "__main__":
    main()
