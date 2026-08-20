from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import cv2


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

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
CAMERA_FPS = 30

# AI 실행 간격
AI_INTERVAL = 3.0

VEHICLE_CONF = 0.20
PLATE_CONF = 0.15

# 실제 파일 저장 대신 메모리에서 JPEG encode/decode
JPEG_QUALITY = 95


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
            self.thread.join(
                timeout=1.0
            )

        self.cap.release()


# --------------------------------------------------
# Snapshot
# --------------------------------------------------

def make_jpeg_snapshot(frame):

    success, encoded = cv2.imencode(
        ".jpg",
        frame,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            JPEG_QUALITY,
        ],
    )

    if not success:
        raise RuntimeError(
            "Failed to encode snapshot as JPEG"
        )

    snapshot = cv2.imdecode(
        encoded,
        cv2.IMREAD_COLOR,
    )

    if snapshot is None:
        raise RuntimeError(
            "Failed to decode JPEG snapshot"
        )

    return snapshot


# --------------------------------------------------
# AI Worker
# --------------------------------------------------

class AIWorker:

    def __init__(
        self,
        camera: LatestFrameCamera,
        vehicle_detector,
        plate_detector,
        vehicle_classifier,
        plate_ocr,
    ):

        self.camera = camera

        self.vehicle_detector = vehicle_detector
        self.plate_detector = plate_detector
        self.vehicle_classifier = vehicle_classifier
        self.plate_ocr = plate_ocr

        self.result = {
            "vehicles": [],
            "fallback_plates": [],
        }

        self.pipeline_ms = 0.0
        self.last_run_timestamp = 0.0

        self.running = True
        self.lock = threading.Lock()

        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
        )

        self.thread.start()

    def _run(self):

        while self.running:

            # --------------------------------------------------
            # 1. AI 실행 간격 대기
            # --------------------------------------------------

            now = time.monotonic()

            if (
                now
                - self.last_run_timestamp
                < AI_INTERVAL
            ):
                time.sleep(0.02)
                continue

            # --------------------------------------------------
            # 2. 최신 Camera Frame 가져오기
            # --------------------------------------------------

            frame = self.camera.read()

            if frame is None:
                time.sleep(0.02)
                continue

            self.last_run_timestamp = now

            print()
            print(
                "========================================"
            )
            print("[AI] Snapshot captured")

            # --------------------------------------------------
            # 3. 현재 Frame을 JPEG 이미지 형태로 변환
            #
            # Camera ndarray를 바로 YOLO에 넣지 않고,
            # 실제 저장 이미지와 유사한 encode/decode 경로 사용
            # --------------------------------------------------

            try:

                snapshot = make_jpeg_snapshot(
                    frame
                )

            except Exception as exc:

                print(
                    "[SNAPSHOT ERROR] "
                    f"{type(exc).__name__}: {exc}"
                )

                continue

            # --------------------------------------------------
            # 4. AI Pipeline 실행
            # --------------------------------------------------

            start = time.perf_counter()

            try:

                _, result = process_frame(
                    snapshot,
                    self.vehicle_detector,
                    self.plate_detector,
                    self.vehicle_classifier,
                    self.plate_ocr,
                )

                pipeline_ms = (
                    time.perf_counter()
                    - start
                ) * 1000.0

                vehicle_count = len(
                    result.get(
                        "vehicles",
                        [],
                    )
                )

                fallback_count = len(
                    result.get(
                        "fallback_plates",
                        [],
                    )
                )

                print(
                    "[AI] Pipeline completed"
                )

                print(
                    f"     Vehicles : "
                    f"{vehicle_count}"
                )

                print(
                    f"     Fallback plates : "
                    f"{fallback_count}"
                )

                print(
                    f"     Time : "
                    f"{pipeline_ms:.1f} ms"
                )

                print(
                    "========================================"
                )

                # ----------------------------------------------
                # 전체 pipeline이 정상 종료했을 때만
                # 마지막 결과를 갱신
                # ----------------------------------------------

                with self.lock:

                    self.result = result
                    self.pipeline_ms = pipeline_ms

            except Exception as exc:

                print(
                    "[AI PIPELINE ERROR] "
                    f"{type(exc).__name__}: {exc}"
                )

                print(
                    "========================================"
                )

                # 실패하더라도 기존 성공 결과는 유지

    def get_result(self):

        with self.lock:

            return (
                self.result,
                self.pipeline_ms,
            )

    def release(self):

        self.running = False

        if self.thread.is_alive():
            self.thread.join(
                timeout=2.0
            )


# --------------------------------------------------
# Draw Detection
# --------------------------------------------------

def draw_detection(
    frame,
    result: dict,
):

    # --------------------------------------------------
    # Vehicle Results
    # --------------------------------------------------

    vehicles = result.get(
        "vehicles",
        [],
    )

    for vehicle in vehicles:

        vx1, vy1, vx2, vy2 = (
            vehicle["bbox"]
        )

        detection = vehicle.get(
            "detection",
            {},
        )

        classification = vehicle.get(
            "classification",
            {},
        )

        detection_conf = detection.get(
            "confidence",
            0.0,
        )

        model_name = classification.get(
            "model",
            detection.get(
                "class_name",
                "vehicle",
            ),
        )

        model_conf = classification.get(
            "confidence",
            detection_conf,
        )

        # BLUE
        cv2.rectangle(
            frame,
            (vx1, vy1),
            (vx2, vy2),
            (255, 0, 0),
            3,
        )

        vehicle_label = (
            f"{model_name} "
            f"{model_conf:.2f}"
        )

        cv2.putText(
            frame,
            vehicle_label,
            (
                vx1,
                max(
                    30,
                    vy1 - 10,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

        # --------------------------------------------------
        # Plate Results
        # --------------------------------------------------

        for plate in vehicle.get(
            "plates",
            [],
        ):

            ax1, ay1, ax2, ay2 = (
                plate[
                    "bbox_in_image"
                ]
            )

            ocr = plate.get(
                "ocr",
                {},
            )

            plate_text = (
                ocr.get("text")
                or "PLATE"
            )

            cv2.rectangle(
                frame,
                (ax1, ay1),
                (ax2, ay2),
                (0, 0, 255),
                3,
            )

            cv2.putText(
                frame,
                plate_text,
                (
                    ax1,
                    max(
                        30,
                        ay1 - 10,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

    # --------------------------------------------------
    # Fallback Plate Results
    # --------------------------------------------------

    for plate in result.get(
        "fallback_plates",
        [],
    ):

        px1, py1, px2, py2 = (
            plate[
                "bbox_in_image"
            ]
        )

        ocr = plate.get(
            "ocr",
            {},
        )

        plate_text = (
            ocr.get("text")
            or "FALLBACK PLATE"
        )

        # ORANGE
        cv2.rectangle(
            frame,
            (px1, py1),
            (px2, py2),
            (0, 165, 255),
            2,
        )

        cv2.putText(
            frame,
            plate_text,
            (
                px1,
                max(
                    30,
                    py1 - 10,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )

    return frame


# --------------------------------------------------
# Status Overlay
# --------------------------------------------------

def draw_status(
    frame,
    display_fps: float,
    pipeline_ms: float,
    result: dict,
):

    vehicle_count = len(
        result.get(
            "vehicles",
            [],
        )
    )

    cv2.putText(
        frame,
        f"Display FPS: {display_fps:.1f}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"AI: {pipeline_ms:.1f} ms",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"Vehicles: {vehicle_count}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"AI Interval: {AI_INTERVAL:.1f}s",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return frame


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    camera = None
    ai_worker = None

    try:

        # --------------------------------------------------
        # 1. Load AI Pipeline
        # --------------------------------------------------

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
        print(
            "========================================"
        )
        print(
            "Parking AI Live Snapshot Detection"
        )
        print(
            "========================================"
        )

        try:

            print(
                "[Vehicle Model]"
            )

            print(
                "  names   :",
                vehicle_detector.model.names,
            )

            print(
                "  classes :",
                vehicle_detector.classes,
            )

            print(
                "  conf    :",
                vehicle_detector.conf,
            )

        except Exception:
            pass

        print()
        print(
            f"Camera       : {CAMERA_DEVICE}"
        )

        print(
            f"Resolution   : "
            f"{FRAME_WIDTH}x{FRAME_HEIGHT}"
        )

        print(
            f"AI interval  : "
            f"{AI_INTERVAL:.1f} sec"
        )

        print(
            "[s] Save current live frame"
        )

        print(
            "[q] Quit"
        )

        print()

        # --------------------------------------------------
        # 2. Camera
        # --------------------------------------------------

        camera = LatestFrameCamera(
            CAMERA_DEVICE,
            FRAME_WIDTH,
            FRAME_HEIGHT,
            CAMERA_FPS,
        )

        # --------------------------------------------------
        # 3. AI Worker
        # --------------------------------------------------

        ai_worker = AIWorker(
            camera,
            vehicle_detector,
            plate_detector,
            vehicle_classifier,
            plate_ocr,
        )

        # --------------------------------------------------
        # FPS
        # --------------------------------------------------

        previous_time = (
            time.perf_counter()
        )

        smoothed_fps = 0.0

        # --------------------------------------------------
        # Main Display Loop
        #
        # 중요:
        # AI 처리 여부와 관계없이 최신 camera frame을
        # 계속 화면에 표시한다.
        # --------------------------------------------------

        while True:

            current_frame = (
                camera.read()
            )

            if current_frame is None:

                time.sleep(0.001)
                continue

            # --------------------------------------------------
            # 마지막 성공 AI Result
            # --------------------------------------------------

            result, pipeline_ms = (
                ai_worker.get_result()
            )

            # --------------------------------------------------
            # 반드시 최신 Camera Frame을 표시
            #
            # AI가 OCR에서 오래 걸려도
            # Live 화면은 절대 AI frame에 묶이지 않는다.
            # --------------------------------------------------

            display_frame = (
                current_frame.copy()
            )

            # --------------------------------------------------
            # 마지막 AI 결과를 최신 frame에 Overlay
            #
            # 고정 카메라 환경이므로 이전 bbox를
            # 현재 frame에 재사용한다.
            # --------------------------------------------------

            display_frame = (
                draw_detection(
                    display_frame,
                    result,
                )
            )

            # --------------------------------------------------
            # Display FPS
            # --------------------------------------------------

            current_time = (
                time.perf_counter()
            )

            delta = (
                current_time
                - previous_time
            )

            previous_time = (
                current_time
            )

            if delta > 0:

                current_fps = (
                    1.0
                    / delta
                )

            else:

                current_fps = 0.0

            if smoothed_fps == 0:

                smoothed_fps = (
                    current_fps
                )

            else:

                smoothed_fps = (
                    smoothed_fps
                    * 0.9
                    + current_fps
                    * 0.1
                )

            display_frame = (
                draw_status(
                    display_frame,
                    smoothed_fps,
                    pipeline_ms,
                    result,
                )
            )

            # --------------------------------------------------
            # Display
            # --------------------------------------------------

            cv2.imshow(
                "Parking AI Live",
                display_frame,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            # --------------------------------------------------
            # Quit
            # --------------------------------------------------

            if key == ord("q"):
                break

            # --------------------------------------------------
            # Current Live Frame Save
            #
            # AI frame이 아니라 반드시 현재 live frame 저장
            # --------------------------------------------------

            if key == ord("s"):

                save_path = (
                    PROJECT_DIR
                    / "webcam_test.jpg"
                )

                cv2.imwrite(
                    str(save_path),
                    current_frame,
                )

                print(
                    f"[SAVED] {save_path}"
                )

    finally:

        if ai_worker is not None:
            ai_worker.release()

        if camera is not None:
            camera.release()

        cv2.destroyAllWindows()

        print()
        print(
            "Camera stopped."
        )


if __name__ == "__main__":
    main()
