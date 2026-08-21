from __future__ import annotations

import sys
import threading
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------
# Project Path
# --------------------------------------------------

# root/
# ├── ocr_adapter/
# │   └── ocr_adapter.py
# ├── pipeline/
# │   └── detect_pipeline.py
# └── models/
#
PROJECT_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


# --------------------------------------------------
# AI Pipeline
# --------------------------------------------------

from pipeline.detect_pipeline import (
    load_pipeline,
    process_frame,
)


# --------------------------------------------------
# Config
# --------------------------------------------------

VEHICLE_CONF = 0.20
PLATE_CONF = 0.15


# --------------------------------------------------
# Pipeline Singleton
# --------------------------------------------------
#
# 요청이 들어올 때마다 모델을 load하면 매우 느리기 때문에
# AI 서버 실행 중 한 번만 load한다.
# --------------------------------------------------

_pipeline = None
_pipeline_lock = threading.Lock()


def get_pipeline():
    global _pipeline

    if _pipeline is None:

        with _pipeline_lock:

            # 다른 thread가 먼저 초기화했을 수도 있으므로 재확인
            if _pipeline is None:

                print("[OCR Adapter] Loading AI pipeline...")

                _pipeline = load_pipeline(
                    vehicle_conf=VEHICLE_CONF,
                    plate_conf=PLATE_CONF,
                )

                print("[OCR Adapter] AI pipeline loaded")

    return _pipeline


# --------------------------------------------------
# Image Decode
# --------------------------------------------------

def decode_image(image_data) -> np.ndarray:
    """
    서버에서 받은 image_data를 OpenCV BGR 이미지로 변환한다.

    지원:
        - np.ndarray
        - bytes
        - bytearray
        - memoryview
    """

    # 이미 OpenCV 이미지라면 그대로 사용
    if isinstance(image_data, np.ndarray):

        if image_data.size == 0:
            raise ValueError("empty image")

        return image_data.copy()

    # HTTP / multipart 등으로 받은 binary 이미지
    if isinstance(
        image_data,
        (
            bytes,
            bytearray,
            memoryview,
        ),
    ):

        image_array = np.frombuffer(
            image_data,
            dtype=np.uint8,
        )

        frame = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR,
        )

        if frame is None:
            raise ValueError(
                "failed to decode image_data"
            )

        return frame

    raise TypeError(
        "image_data must be "
        "np.ndarray, bytes, bytearray or memoryview"
    )


# --------------------------------------------------
# Result Simplification
# --------------------------------------------------

def simplify_result(result: dict) -> list[dict]:
    """
    detect_pipeline의 복잡한 결과를
    메인 서버에서 필요한 정보만 남긴다.

    반환 형태:

    [
        {
            "index": 1,
            "classification": {
                "model": "Tesla_Roadster"
            },
            "ocr_data": "78소1234",
            "fallback_ocr": None
        }
    ]
    """

    vehicles = result.get(
        "vehicles",
        [],
    )

    fallback_plates = result.get(
        "fallback_plates",
        [],
    )

    results = []

    # ------------------------------------------
    # Fallback OCR
    # ------------------------------------------

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

    # ------------------------------------------
    # Vehicle Results
    # ------------------------------------------

    for index, vehicle in enumerate(vehicles):

        # 차량 분류
        classification = vehicle.get(
            "classification",
            {},
        )

        model = classification.get(
            "model",
        )

        # -----------------------------
        # OCR
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
# AI Recognition
# --------------------------------------------------

def run_ai_pipeline(image_data) -> list[dict]:
    """
    image_data 한 장을 받아서 전체 AI Pipeline을 실행한다.

    Vehicle Detection
        ↓
    Vehicle Classification
        ↓
    Plate Detection
        ↓
    OCR
    """

    frame = decode_image(
        image_data
    )

    (
        vehicle_detector,
        plate_detector,
        vehicle_classifier,
        plate_ocr,
    ) = get_pipeline()

    _, result = process_frame(
        frame,
        vehicle_detector,
        plate_detector,
        vehicle_classifier,
        plate_ocr,
    )

    return simplify_result(
        result
    )


# --------------------------------------------------
# Entry
# --------------------------------------------------

def recognize_entry(image_data):
    """
    입차 이미지 분석.

    차량 종류 + 차량 번호판을 분석한다.
    """

    return run_ai_pipeline(
        image_data
    )


# --------------------------------------------------
# Exit
# --------------------------------------------------

def recognize_exit(image_data):
    """
    출차 이미지 분석.

    현재는 입차와 동일한 AI Pipeline을 사용한다.
    필요하면 추후 번호판 OCR만 수행하도록 분리할 수 있다.
    """

    return run_ai_pipeline(
        image_data
    )


# --------------------------------------------------
# OCR Handler
# --------------------------------------------------

def ocr_handler(
    direction,
    image_data,
):
    """
    AI 서버 통신 코드에서 호출하는 진입점.
    """

    if direction == "entry":
        return recognize_entry(
            image_data
        )

    if direction == "exit":
        return recognize_exit(
            image_data
        )

    raise ValueError(
        f"invalid direction: {direction}"
    )