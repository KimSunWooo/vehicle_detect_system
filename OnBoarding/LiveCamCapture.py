from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path

import cv2


# --------------------------------------------------
# Project Path
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]


# --------------------------------------------------
# Config
# --------------------------------------------------

CAMERA_DEVICE = "/dev/video2"

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
CAMERA_FPS = 30

CAPTURE_DIR = PROJECT_DIR / "captures"


# --------------------------------------------------
# Camera
# --------------------------------------------------

class LatestFrameCamera:

    def __init__(
        self,
        device: str,
        width: int,
        height: int,
        fps: int,
    ):
        self.cap = cv2.VideoCapture(
            device,
            cv2.CAP_V4L2,
        )

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera: {device}"
            )

        # 최신 프레임 위주로 가져오기 위해 버퍼 최소화
        self.cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1,
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            width,
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            height,
        )

        self.cap.set(
            cv2.CAP_PROP_FPS,
            fps,
        )

        self.frame = None
        self.running = True
        self.lock = threading.Lock()

        self.thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
        )

        self.thread.start()

    def _capture_loop(self):

        while self.running:

            ret, frame = self.cap.read()

            if not ret:
                time.sleep(0.01)
                continue

            with self.lock:
                self.frame = frame

    def read(self):

        with self.lock:

            if self.frame is None:
                return None

            return self.frame.copy()

    def release(self):

        self.running = False

        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

        self.cap.release()


# --------------------------------------------------
# Save Capture
# --------------------------------------------------

def save_capture(frame) -> Path:

    CAPTURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    save_path = (
        CAPTURE_DIR
        / f"capture_{timestamp}.jpg"
    )

    success = cv2.imwrite(
        str(save_path),
        frame,
    )

    if not success:
        raise RuntimeError(
            f"Failed to save image: {save_path}"
        )

    return save_path


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    camera = None

    try:

        camera = LatestFrameCamera(
            CAMERA_DEVICE,
            FRAME_WIDTH,
            FRAME_HEIGHT,
            CAMERA_FPS,
        )

        print("========================================")
        print("           Live Camera Capture")
        print("========================================")
        print(f"Camera : {CAMERA_DEVICE}")
        print(f"Size   : {FRAME_WIDTH}x{FRAME_HEIGHT}")
        print()
        print("[s] Save current frame")
        print("[q] Quit")
        print()

        while True:

            frame = camera.read()

            if frame is None:
                time.sleep(0.001)
                continue

            display_frame = frame.copy()

            cv2.putText(
                display_frame,
                "S: Capture | Q: Quit",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "Live Camera Capture",
                display_frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("s"):

                save_path = save_capture(frame)

                print(
                    f"[CAPTURED] {save_path}"
                )

    finally:

        if camera is not None:
            camera.release()

        cv2.destroyAllWindows()

        print()
        print("Camera stopped.")


if __name__ == "__main__":
    main()
