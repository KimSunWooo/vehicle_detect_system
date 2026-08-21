from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# --------------------------------------------------
# Project Path
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


# --------------------------------------------------
# Pipeline
# --------------------------------------------------

from pipeline.detect_pipeline import (
    load_pipeline,
    process_frame,
)


# --------------------------------------------------
# Config
# --------------------------------------------------

CAMERA_DEVICE = "/dev/video2"

FRAME_WIDTH = 640
FRAME_HEIGHT = 450
CAMERA_FPS = 30

VEHICLE_CONF = 0.20
PLATE_CONF = 0.15

# AI_Pipeline 동작 범위
ROI_X1 = 220
ROI_Y1 = 240
ROI_X2 = 410
ROI_Y2 = 440

# --------------------------------------------------
# Trigger frame enhancement
# --------------------------------------------------

ENABLE_IMAGE_ENHANCEMENT = True

# 노이즈 제거
DENOISE_DIAMETER = 5
DENOISE_SIGMA_COLOR = 35
DENOISE_SIGMA_SPACE = 35

# CLAHE
CLAHE_CLIP_LIMIT = 1.8
CLAHE_TILE_GRID_SIZE = (8, 8)

# 약한 샤프닝
SHARPEN_SIGMA = 1.0
SHARPEN_AMOUNT = 1.25
SHARPEN_BLUR_WEIGHT = -0.25

# 너무 흐린 프레임인지 참고용으로 출력
SHARPNESS_WARNING_THRESHOLD = 60.0


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

        actual_width = self.cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
        actual_height = self.cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
        actual_fps = self.cap.get(
            cv2.CAP_PROP_FPS
        )

        print(
            "[Camera] Actual: "
            f"{actual_width:.0f}x{actual_height:.0f}"
            f" @ {actual_fps:.1f} FPS"
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

            if not ret or frame is None:
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
            self.thread.join(
                timeout=1.0
            )

        self.cap.release()


# --------------------------------------------------
# Image Quality Helpers
# --------------------------------------------------

def calculate_sharpness(
    frame: np.ndarray,
) -> float:
    """
    Laplacian variance 기반의 간단한 선명도 지표.
    값이 높을수록 일반적으로 edge가 선명하다.
    """

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    return float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )


def enhance_trigger_frame(
    frame: np.ndarray,
) -> np.ndarray:
    """
    트리거 순간 확보한 프레임에만 적용한다.

    순서:
        1. Bilateral Filter
        2. LAB L-channel CLAHE
        3. 약한 Unsharp Mask

    전체 영상을 계속 필터링하지 않고
    AI에 넘길 한 장에만 적용한다.
    """

    if not ENABLE_IMAGE_ENHANCEMENT:
        return frame.copy()

    denoised = cv2.bilateralFilter(
        frame,
        d=DENOISE_DIAMETER,
        sigmaColor=DENOISE_SIGMA_COLOR,
        sigmaSpace=DENOISE_SIGMA_SPACE,
    )

    lab = cv2.cvtColor(
        denoised,
        cv2.COLOR_BGR2LAB,
    )

    l_channel, a_channel, b_channel = (
        cv2.split(lab)
    )

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE,
    )

    enhanced_l = clahe.apply(
        l_channel
    )

    enhanced_lab = cv2.merge(
        (
            enhanced_l,
            a_channel,
            b_channel,
        )
    )

    enhanced = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR,
    )

    blurred = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        SHARPEN_SIGMA,
    )

    sharpened = cv2.addWeighted(
        enhanced,
        SHARPEN_AMOUNT,
        blurred,
        SHARPEN_BLUR_WEIGHT,
        0,
    )

    return sharpened


# --------------------------------------------------
# Entrance Trigger
# --------------------------------------------------

def handle_vehicle_trigger(
    frame: np.ndarray,
    vehicle_detector,
    plate_detector,
    vehicle_classifier,
    plate_ocr,
) -> dict[str, Any]:
    """
    S 키 또는 실제 초음파 센서 신호가 들어왔을 때
    호출할 실제 진입 함수.

    저장하지 않고:
        현재 프레임 복사
        -> 품질 측정
        -> 전처리
        -> process_frame()
    """

    print()
    print(
        "========================================"
    )
    print(
        "[Entrance] Vehicle trigger received"
    )
    print(
        "========================================"
    )

    snapshot = frame.copy()

    raw_sharpness = calculate_sharpness(
        snapshot
    )

    print(
        "[Entrance] Raw sharpness: "
        f"{raw_sharpness:.1f}"
    )

    if (
        raw_sharpness
        < SHARPNESS_WARNING_THRESHOLD
    ):
        print(
            "[Entrance] WARNING: "
            "Captured frame may be blurry"
        )

    processed = enhance_trigger_frame(
        snapshot
    )

    processed_sharpness = (
        calculate_sharpness(
            processed
        )
    )

    print(
        "[Entrance] Processed sharpness: "
        f"{processed_sharpness:.1f}"
    )

    print(
        "[Entrance] AI pipeline start"
    )

    start = time.perf_counter()

    annotated, result = process_frame(
        processed,
        vehicle_detector,
        plate_detector,
        vehicle_classifier,
        plate_ocr,
    )

    elapsed_ms = (
        time.perf_counter()
        - start
    ) * 1000.0

    print(
        "[Entrance] AI pipeline complete "
        f"({elapsed_ms:.1f} ms)"
    )

    simple_result = simplify_result(
        result
    )   

    print(
        "[Entrance] Result: "
        f"{simple_result}"
    )

    print(
        "========================================"
    )
    print()

    cv2.imshow(
        "Entrance AI Result",
        annotated,
    )

    return result

def simplify_result(result: dict) -> list[dict]:

    vehicles = result.get(
        "vehicles",
        [],
    )

    fallback_plates = result.get(
        "fallback_plates",
        [],
    )

    results = []

    # fallback OCR 결과 추출
    fallback_numbers = []

    for plate in fallback_plates:

        ocr = plate.get(
            "ocr",
            {},
        )

        text = ocr.get(
            "text",
            "",
        )

        accepted = ocr.get(
            "accepted",
            False,
        )

        if accepted and text:
            fallback_numbers.append(text)

    # 차량별 결과 생성
    for index, vehicle in enumerate(vehicles):

        # -----------------------------
        # Classification
        # -----------------------------

        classification = vehicle.get(
            "classification",
            {},
        )

        model = classification.get(
            "model",
            None,
        )

        # -----------------------------
        # Vehicle OCR
        # -----------------------------

        ocr_data = None

        plates = vehicle.get(
            "plates",
            [],
        )

        for plate in plates:

            ocr = plate.get(
                "ocr",
                {},
            )

            text = ocr.get(
                "text",
                "",
            )

            accepted = ocr.get(
                "accepted",
                False,
            )

            if accepted and text:
                ocr_data = text
                break

        # -----------------------------
        # Fallback OCR
        # -----------------------------

        fallback_ocr = (
            fallback_numbers[index]
            if index < len(fallback_numbers)
            else None
        )

        # -----------------------------
        # Result
        # -----------------------------

        results.append(
            {
                "index": index + 1,
                "classification": {
                    "model": model,
                },
                "ocr_data": ocr_data,
                "fallback_ocr": fallback_ocr,
            }
        )

    return results


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    camera = None

    print(
        "[Entrance] Loading AI pipeline..."
    )

    (
        vehicle_detector,
        plate_detector,
        vehicle_classifier,
        plate_ocr,
    ) = load_pipeline(
        vehicle_conf=VEHICLE_CONF,
        plate_conf=PLATE_CONF,
    )

    print(
        "[Entrance] AI pipeline loaded"
    )

    try:

        camera = LatestFrameCamera(
            CAMERA_DEVICE,
            FRAME_WIDTH,
            FRAME_HEIGHT,
            CAMERA_FPS,
        )

        print()
        print(
            "========================================"
        )
        print(
            "      Entrance Camera Detection"
        )
        print(
            "========================================"
        )
        print(
            f"Camera : {CAMERA_DEVICE}"
        )
        print(
            f"Size   : "
            f"{FRAME_WIDTH}x{FRAME_HEIGHT}"
        )
        print()
        print(
            "[s] Vehicle trigger / run pipeline"
        )
        print(
            "[q] Quit"
        )
        print()

        while True:

            frame = camera.read()

            if frame is None:
                time.sleep(0.001)
                continue

            display_frame = frame.copy()

            cv2.putText(
                display_frame,
                (
                    "S: Vehicle Trigger | "
                    "Q: Quit"
                ),
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "Entrance Camera",
                display_frame,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):
                break

            if key == ord("s"):

                roi = frame[
                    ROI_Y1:ROI_Y2,
                    ROI_X1:ROI_X2
                ]             

                try:
                    if roi.size == 0:
                        print("[Entrance] Invalid ROI")
                        continue

                    handle_vehicle_trigger(
                        frame,
                        vehicle_detector,
                        plate_detector,
                        vehicle_classifier,
                        plate_ocr,
                    )

                except Exception as exc:

                    print(
                        "[Entrance] "
                        "Trigger processing failed: "
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    )

    finally:

        if camera is not None:
            camera.release()

        cv2.destroyAllWindows()

        print()
        print(
            "Camera stopped."
        )


if __name__ == "__main__":
    main()