from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests


# --------------------------------------------------
# Project Path
# --------------------------------------------------

# 이 파일을 기존과 동일하게
# root/OnBoarding/entrance_cam/ 아래에 두는 것을 기준으로 함.
PROJECT_DIR = Path(__file__).resolve().parents[2]


# --------------------------------------------------
# ESP32-CAM Config
# --------------------------------------------------

ESP32_CAM_IP = "192.168.0.2"

CAPTURE_URL = f"http://{ESP32_CAM_IP}/capture"
STREAM_URL = f"http://{ESP32_CAM_IP}:81/stream"

CAPTURE_DIR = (
    PROJECT_DIR
    / "captures"
    / "entrance"
)

VERTICAL_FLIP = True
HORIZONTAL_FLIP = True

REQUEST_TIMEOUT = 5.0


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def apply_flip(frame: np.ndarray) -> np.ndarray:
    """기존 entrance_cam.py와 동일한 화면 반전 설정."""

    if VERTICAL_FLIP and HORIZONTAL_FLIP:
        return cv2.flip(frame, -1)

    if VERTICAL_FLIP:
        return cv2.flip(frame, 0)

    if HORIZONTAL_FLIP:
        return cv2.flip(frame, 1)

    return frame


def decode_jpeg(image_bytes: bytes) -> np.ndarray:
    """ESP32-CAM /capture 응답을 OpenCV 이미지로 변환."""

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    if frame is None:
        raise RuntimeError(
            "ESP32-CAM 이미지를 OpenCV frame으로 변환하지 못했습니다."
        )

    return frame


def save_capture(frame: np.ndarray) -> Path:
    """캡처 이미지를 captures/entrance 폴더에 저장."""

    CAPTURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    image_path = (
        CAPTURE_DIR
        / f"entrance_{timestamp}.jpg"
    )

    if not cv2.imwrite(
        str(image_path),
        frame,
    ):
        raise RuntimeError(
            f"이미지 저장 실패: {image_path}"
        )

    return image_path


def capture_image() -> Path:
    """
    S 키 입력 시 ESP32-CAM의 /capture에서
    현재 프레임 한 장을 받아 저장한다.
    """

    response = requests.get(
        CAPTURE_URL,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    if not response.content:
        raise RuntimeError(
            "ESP32-CAM에서 빈 이미지가 반환되었습니다."
        )

    frame = decode_jpeg(response.content)
    frame = apply_flip(frame)
    image_path = save_capture(frame)

    print(
        f"[Capture] Saved: {image_path} "
        f"({frame.shape[1]}x{frame.shape[0]})"
    )

    return image_path


# --------------------------------------------------
# Live Stream
# --------------------------------------------------

def run() -> None:
    print()
    print("========================================")
    print(" ESP32-CAM Capture Only")
    print("----------------------------------------")
    print(" S     : Capture")
    print(" Q/ESC : Quit")
    print("========================================")
    print()

    cap = cv2.VideoCapture(STREAM_URL)

    if not cap.isOpened():
        raise RuntimeError(
            f"ESP32-CAM 스트림 연결 실패: {STREAM_URL}"
        )

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    try:
        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                print(
                    "[ESP32-CAM] Stream frame receive failed"
                )
                continue

            display_frame = apply_flip(frame.copy())

            cv2.putText(
                display_frame,
                "S: Capture | Q/ESC: Quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "ESP32 Entrance Camera",
                display_frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                print(
                    "[ESP32-CAM] Capture mode stopped"
                )
                break

            if key == ord("s"):
                try:
                    capture_image()

                except requests.RequestException as exc:
                    print(
                        "[Capture] Request failed: "
                        f"{exc}"
                    )

                except Exception as exc:
                    print(
                        "[Capture] Failed: "
                        f"{type(exc).__name__}: {exc}"
                    )

    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    run()


if __name__ == "__main__":
    main()