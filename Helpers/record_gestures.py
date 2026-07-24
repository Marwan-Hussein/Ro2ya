"""
Available Gestures
------------------
left-click
right-click

brightness-up
brightness-down

zoom-in
zoom-out

back-button
forward-button

cursor-up
cursor-down
cursor-left
cursor-right

tab-next
tab-previous

scroll-up
scroll-down
scroll-left
scroll-right

volume-up
volume-down

drag-drop
screen-shot
sleep-mode
"""

from pathlib import Path
import re

import cv2

TESTER_NAME = "marwan"
GESTURE_NAME = "scroll-left"


WINDOW_NAME = "Camera Recorder"
OUTPUT_DIR = Path(__file__).resolve().parent / f"recordings/{GESTURE_NAME}"
FILENAME_PATTERN = re.compile(rf"^(\d+)_{TESTER_NAME}\.mp4$")
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_RATE = 30.0


class CameraRecorder:
    def __init__(self, camera_index=0):
        self.capture = cv2.VideoCapture(camera_index)
        if not self.capture.isOpened():
            raise RuntimeError("Could not open camera.")

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.capture.set(cv2.CAP_PROP_FPS, FRAME_RATE)

        self.writer = None
        self.current_file = None
        self.is_recording = False

    def next_output_file(self):
        OUTPUT_DIR.mkdir(exist_ok=True)
        existing_ids = []

        for path in OUTPUT_DIR.iterdir():
            match = FILENAME_PATTERN.match(path.name)
            if match:
                existing_ids.append(int(match.group(1)))

        next_id = max(existing_ids, default=0) + 1
        return OUTPUT_DIR / f"{next_id}_{TESTER_NAME}.mp4"

    def toggle_recording(self, frame):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording(frame)

    def start_recording(self, frame):
        height, width = frame.shape[:2]

        self.current_file = self.next_output_file()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(self.current_file), fourcc, FRAME_RATE, (width, height)
        )

        if not self.writer.isOpened():
            self.writer = None
            self.current_file = None
            raise RuntimeError("Could not create video file.")

        self.is_recording = True
        print(f"Recording started: {self.current_file}")

    def stop_recording(self):
        if self.writer is not None:
            self.writer.release()

        print(f"Recording saved: {self.current_file}")
        self.writer = None
        self.current_file = None
        self.is_recording = False

    def release(self):
        if self.is_recording:
            self.stop_recording()
        self.capture.release()


def draw_status(frame, is_recording):
    if is_recording:
        cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
        cv2.putText(
            frame, "REC", (48, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2
        )

    cv2.putText(
        frame,
        "Space: start/stop recording | Q or Esc: quit",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )


def main():
    recorder = CameraRecorder()

    cv2.namedWindow(WINDOW_NAME)

    try:
        while True:
            ok, frame = recorder.capture.read()
            if not ok:
                print("Could not read from camera.")
                break

            if recorder.is_recording and recorder.writer is not None:
                recorder.writer.write(frame)

            draw_status(frame, recorder.is_recording)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                recorder.toggle_recording(frame)
    finally:
        recorder.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
