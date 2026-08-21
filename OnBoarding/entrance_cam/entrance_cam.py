"""
entrance_cam_android.py

Android 스마트폰(IP Webcam / MJPEG stream)을
주차장 입구 카메라로 사용하는 모듈.

실제 운영:
    초음파 센서 신호
        -> handle_vehicle_trigger()
        -> Android 카메라 스트림의 최신 프레임 사용
        -> 이미지 저장
        -> 기존 detect_pipeline.process_frame()
        -> 결과 반환

테스트:
    S     : 초음파 센서 신호를 흉내내는 테스트 트리거
    Q/ESC : 종료

중요:
    AI 모델은 프로그램 시작 시 load_pipeline()으로 한 번만 로드하고,
    각 트리거마다 재사용한다.

권장 Android 앱:
    IP Webcam 계열 MJPEG 서버

예시 스트림 주소:
    http://192.168.0.15:8080/video
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


# --------------------------------------------------
# Project Path
# --------------------------------------------------

# 현재 파일:
# root/OnBoarding/entrance_cam/entrance_cam_android.py
#
# 프로젝트 루트:
# root/
PROJECT_DIR = Path(__file__).resolve().parents[2]

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

# Android 스마트폰과 Ubuntu PC는 같은 Wi-Fi에 연결해야 한다.
#
# IP Webcam 예:
#   앱 화면에 http://192.168.0.15:8080 이 표시되면
#   일반적인 MJPEG 스트림 주소는 아래와 같다.
ANDROID_CAM_IP = "192.168.0.213"
ANDROID_CAM_PORT = 8080

STREAM_URL = (
    f"http://{ANDROID_CAM_IP}:"
    f"{ANDROID_CAM_PORT}/video"
)

CAPTURE_DIR = (
    PROJECT_DIR
    / "captures"
    / "entrance"
)

# 갤럭시를 가로 방향으로 정상 고정했다면 보통 둘 다 False 권장.
VERTICAL_FLIP = False
HORIZONTAL_FLIP = False

# 카메라 스트림 reconnect 설정
RECONNECT_DELAY = 1.0
MAX_RECONNECT_ATTEMPTS = 5

# OpenCV 내부 버퍼를 작게 유지해서 최신 프레임에 가깝게 사용
CAMERA_BUFFER_SIZE = 1

# 연결 직후 오래된/불안정한 프레임을 버리기 위한 warm-up
CAMERA_WARMUP_FRAMES = 5

# 트리거 발생 시 최신 프레임을 얻기 위해 몇 장 버릴지 설정
TRIGGER_FLUSH_FRAMES = 2


# --------------------------------------------------
# Image Enhancement
# --------------------------------------------------

# 스마트폰 카메라는 ESP32-CAM보다 품질이 높으므로
# 기본값은 False를 권장한다.
#
# 번호판 OCR 정확도를 실제 비교하면서 필요할 때만 True로 변경.
ENABLE_IMAGE_ENHANCEMENT = False

# 전체 프레임 업스케일은 스마트폰 1080p 입력에서는 보통 불필요.
ENABLE_UPSCALE = False
UPSCALE_FACTOR = 1.5

# CLAHE 설정
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Bilateral Filter 설정
DENOISE_DIAMETER = 5
DENOISE_SIGMA_COLOR = 50
DENOISE_SIGMA_SPACE = 50

# Unsharp Mask 설정
SHARPEN_AMOUNT = 1.5
SHARPEN_BLUR_WEIGHT = -0.5
SHARPEN_SIGMA = 1.0

# Livecam_tracked와 같은 기본 threshold
VEHICLE_CONF = 0.20
PLATE_CONF = 0.15


# --------------------------------------------------
# Runtime Pipeline Objects
# --------------------------------------------------

vehicle_detector = None
plate_detector = None
vehicle_classifier = None
plate_ocr = None

camera: Optional[cv2.VideoCapture] = None


# --------------------------------------------------
# Pipeline Initialization
# --------------------------------------------------

def initialize_pipeline() -> None:
    """
    detect_pipeline.load_pipeline()을 이용해
    모델 4개를 한 번만 초기화한다.
    """

    global vehicle_detector
    global plate_detector
    global vehicle_classifier
    global plate_ocr

    print()
    print("============================================")
    print("[EntranceCam] Loading AI pipeline...")
    print("============================================")

    (
        vehicle_detector,
        plate_detector,
        vehicle_classifier,
        plate_ocr,
    ) = load_pipeline(
        vehicle_conf=VEHICLE_CONF,
        plate_conf=PLATE_CONF,
    )

    print()
    print("[EntranceCam] AI pipeline loaded")
    print()


def _ensure_pipeline_loaded() -> None:
    if any(
        model is None
        for model in (
            vehicle_detector,
            plate_detector,
            vehicle_classifier,
            plate_ocr,
        )
    ):
        raise RuntimeError(
            "AI pipeline이 초기화되지 않았습니다. "
            "initialize_pipeline()을 먼저 실행하세요."
        )


# --------------------------------------------------
# Image Enhancement
# --------------------------------------------------

def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """
    스마트폰 프레임에 선택적으로 후처리를 적용한다.

    처리 순서:
        1. Bilateral Filter
        2. CLAHE
        3. Unsharp Mask
        4. 선택적 업스케일

    스마트폰 원본 화질이 충분한 경우
    ENABLE_IMAGE_ENHANCEMENT=False를 권장한다.
    """

    if not ENABLE_IMAGE_ENHANCEMENT:
        return frame

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

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE,
    )

    enhanced_l = clahe.apply(l_channel)

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

    if ENABLE_UPSCALE and UPSCALE_FACTOR != 1.0:
        sharpened = cv2.resize(
            sharpened,
            None,
            fx=UPSCALE_FACTOR,
            fy=UPSCALE_FACTOR,
            interpolation=cv2.INTER_CUBIC,
        )

    return sharpened


# --------------------------------------------------
# Camera Helpers
# --------------------------------------------------

def _apply_flip(frame: np.ndarray) -> np.ndarray:
    if VERTICAL_FLIP and HORIZONTAL_FLIP:
        return cv2.flip(frame, -1)

    if VERTICAL_FLIP:
        return cv2.flip(frame, 0)

    if HORIZONTAL_FLIP:
        return cv2.flip(frame, 1)

    return frame


def _save_capture(
    frame: np.ndarray,
    save_dir: Optional[Path | str] = None,
) -> Path:

    target_dir = (
        Path(save_dir)
        if save_dir
        else CAPTURE_DIR
    )

    target_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    image_path = (
        target_dir
        / f"entrance_{timestamp}.jpg"
    )

    saved = cv2.imwrite(
        str(image_path),
        frame,
    )

    if not saved:
        raise RuntimeError(
            "캡처 이미지를 저장하지 못했습니다: "
            f"{image_path}"
        )

    return image_path


def open_camera() -> cv2.VideoCapture:
    """
    Android IP Webcam MJPEG 스트림에 연결한다.
    """

    global camera

    if camera is not None and camera.isOpened():
        return camera

    print(
        "[EntranceCam] Connecting Android camera: "
        f"{STREAM_URL}"
    )

    cap = cv2.VideoCapture(STREAM_URL)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            "Android 카메라 스트림 연결 실패: "
            f"{STREAM_URL}"
        )

    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        CAMERA_BUFFER_SIZE,
    )

    # 연결 직후 프레임 안정화
    for _ in range(CAMERA_WARMUP_FRAMES):
        cap.read()

    camera = cap

    print(
        "[EntranceCam] Android camera connected"
    )

    return camera


def close_camera() -> None:
    global camera

    if camera is not None:
        camera.release()
        camera = None

    print(
        "[EntranceCam] Android camera released"
    )


def reconnect_camera() -> cv2.VideoCapture:
    """
    스트림이 끊어진 경우 카메라 연결을 다시 시도한다.
    """

    global camera

    close_camera()

    last_error: Optional[Exception] = None

    for attempt in range(
        1,
        MAX_RECONNECT_ATTEMPTS + 1,
    ):
        print(
            "[EntranceCam] Reconnect attempt "
            f"{attempt}/{MAX_RECONNECT_ATTEMPTS}"
        )

        try:
            return open_camera()

        except Exception as exc:
            last_error = exc
            time.sleep(RECONNECT_DELAY)

    raise RuntimeError(
        "Android 카메라 재연결 실패"
    ) from last_error


def read_camera_frame(
    flush_frames: int = 0,
) -> np.ndarray:
    """
    현재 Android 카메라 스트림에서 프레임 한 장을 가져온다.

    flush_frames:
        버퍼에 남아 있는 이전 프레임을 버리고
        가능한 최신 프레임을 사용할 때 사용.
    """

    cap = open_camera()

    for _ in range(flush_frames):
        ret, _ = cap.read()

        if not ret:
            cap = reconnect_camera()
            break

    ret, frame = cap.read()

    if not ret or frame is None:
        print(
            "[EntranceCam] Frame read failed. "
            "Trying reconnect..."
        )

        cap = reconnect_camera()
        ret, frame = cap.read()

    if not ret or frame is None:
        raise RuntimeError(
            "Android 카메라 프레임을 가져오지 못했습니다."
        )

    return frame


def capture_entrance_image(
    save_image: bool = True,
    save_dir: Optional[Path | str] = None,
) -> tuple[np.ndarray, Optional[Path]]:
    """
    차량 트리거 시점에 Android 스트림의 최신 프레임을
    정지 이미지처럼 가져온다.

    ESP32-CAM의 /capture 요청을 대체하는 함수이다.
    """

    frame = read_camera_frame(
        flush_frames=TRIGGER_FLUSH_FRAMES,
    )

    frame = _apply_flip(frame)

    frame = enhance_frame(frame)

    image_path: Optional[Path] = None

    if save_image:
        image_path = _save_capture(
            frame,
            save_dir,
        )

        print(
            "[EntranceCam] "
            f"Saved: {image_path}"
        )

    print(
        "[EntranceCam] Capture complete: "
        f"{frame.shape[1]}x{frame.shape[0]}"
        f" | enhancement="
        f"{'ON' if ENABLE_IMAGE_ENHANCEMENT else 'OFF'}"
    )

    return (
        frame,
        image_path,
    )


# --------------------------------------------------
# Pipeline
# --------------------------------------------------

def run_pipeline(
    frame: np.ndarray,
) -> tuple[Any, dict]:
    """
    process_frame()에 모델 4개를 모두 넘긴다.
    """

    _ensure_pipeline_loaded()

    annotated, result = process_frame(
        frame,
        vehicle_detector,
        plate_detector,
        vehicle_classifier,
        plate_ocr,
    )

    return (
        annotated,
        result,
    )


# --------------------------------------------------
# Vehicle Trigger
# --------------------------------------------------

def handle_vehicle_trigger() -> dict:
    """
    차량 감지 이벤트의 실제 진입점.

    테스트:
        S -> handle_vehicle_trigger()

    실제 운영:
        초음파 센서
            -> handle_vehicle_trigger()
    """

    print()
    print("============================================")
    print("[Entrance] Vehicle trigger received")
    print("============================================")

    # 1. Android 카메라의 현재 프레임 캡처
    frame, image_path = capture_entrance_image()

    # 2. AI Pipeline 실행
    print(
        "[Entrance] AI pipeline start"
    )

    annotated, result = run_pipeline(frame)

    print(
        "[Entrance] AI pipeline complete"
    )

    print(
        "[Entrance] "
        f"Image : {image_path}"
    )

    print(
        "[Entrance] "
        f"Result: {result}"
    )

    print("============================================")
    print()

    # 테스트 시 결과 화면 확인
    cv2.imshow(
        "Entrance AI Result",
        annotated,
    )

    return result


# --------------------------------------------------
# Test Mode
# --------------------------------------------------

def test_mode() -> None:
    """
    Android 스마트폰 스트림 확인용 테스트 모드.

    S:
        차량 감지 이벤트 테스트.

    Q / ESC:
        종료.

    실제 운영에서는 스트림 창이 필요 없으며
    초음파 센서 코드에서
    handle_vehicle_trigger()만 호출하면 된다.
    """

    print()
    print("============================================")
    print(" Android Entrance Camera Test Mode")
    print("--------------------------------------------")
    print(f" Stream: {STREAM_URL}")
    print(" S     : Vehicle trigger test")
    print(" Q/ESC : Quit")
    print("============================================")
    print()

    open_camera()

    try:
        while True:
            try:
                frame = read_camera_frame()

            except Exception as exc:
                print(
                    "[EntranceCam] "
                    "Stream frame receive failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue

            display_frame = _apply_flip(
                frame.copy()
            )

            cv2.putText(
                display_frame,
                (
                    "Android Entrance Cam | "
                    "S: Trigger | Q/ESC: Quit"
                ),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "Android Entrance Camera - TEST",
                display_frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (
                ord("q"),
                27,
            ):
                print(
                    "[EntranceCam] "
                    "Test mode stopped"
                )
                break

            if key == ord("s"):
                try:
                    handle_vehicle_trigger()

                except Exception as exc:
                    print(
                        "[EntranceCam] "
                        "Trigger processing failed: "
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    )

    finally:
        close_camera()
        cv2.destroyAllWindows()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:
    # 프로그램 시작 시 pipeline을 한 번만 로드
    initialize_pipeline()

    # Android 카메라 연결
    open_camera()

    # 테스트 모드
    test_mode()


if __name__ == "__main__":
    main()