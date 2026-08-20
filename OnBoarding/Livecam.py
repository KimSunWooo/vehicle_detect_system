from __future__ import annotations

import sys
import time
import threading
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

AI_INTERVAL = 3.0  # AI Pipeline 실행 간격 (초)

VEHICLE_CONF = 0.20
PLATE_CONF = 0.15


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
            "vehicles": []
        }

        self.pipeline_ms = 0.0

        self.running = True

        self.lock = threading.Lock()


        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
        )

        self.thread.start()


    def _run(self):

        last_run_time = 0.0

        while self.running:

            current_time = time.monotonic()

            # 마지막 AI 실행 후 3초가 지나지 않았다면 대기
            if current_time - last_run_time < AI_INTERVAL:
                time.sleep(0.01)
                continue

            frame = self.camera.read()

            # ==========================================
            # DEBUG 1
            # VehicleDetector만 직접 실행
            # ==========================================

            debug_frame = frame.copy()

            vehicles_direct = self.vehicle_detector.detect(
                debug_frame
            )

            print()
            print("========== DIRECT DETECTOR ==========")
            print(
                f"Detected: {len(vehicles_direct)}"
            )

            for vehicle in vehicles_direct:
                print(
                    f"class={vehicle.class_name}, "
                    f"conf={vehicle.confidence:.3f}, "
                    f"bbox={vehicle.bbox}"
                )


            # ==========================================
            # DEBUG 2
            # 동일 프레임을 파일로 저장
            # ==========================================

            cv2.imwrite(
                str(PROJECT_DIR / "debug_live_frame.jpg"),
                debug_frame,
            )

            if frame is None:
                time.sleep(0.01)
                continue

            # 실제 AI Pipeline 실행 시작 시각
            last_run_time = current_time

            start = time.perf_counter()

            frame = self.camera.read()

            if frame is None:
                time.sleep(0.01)
                continue


            # -------------------------------
            # DEBUG
            # -------------------------------

            vehicles = self.vehicle_detector.detect(frame)

            print(
                "[LIVE DETECTOR]",
                len(vehicles),
            )

            for vehicle in vehicles:
                print(
                    vehicle.class_name,
                    vehicle.confidence,
                    vehicle.bbox,
                )


            start = time.perf_counter()

            try:

                _, result = process_frame(
                    frame,
                    self.vehicle_detector,
                    self.plate_detector,
                    self.vehicle_classifier,
                    self.plate_ocr,
                )

                print(
                    "PIPELINE VEHICLES:",
                    len(result.get("vehicles", []))
                )

                pipeline_ms = (
                    time.perf_counter()
                    - start
                ) * 1000.0

                with self.lock:

                    self.result = result
                    self.pipeline_ms = pipeline_ms

            except Exception as exc:

                print(
                    f"[AI ERROR] {exc}"
                )


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

    vehicles = result.get(
        "vehicles",
        [],
    )


    for vehicle in vehicles:

        # --------------------------------------------------
        # Vehicle
        # --------------------------------------------------

        vx1, vy1, vx2, vy2 = (
            vehicle["bbox"]
        )


        classification = (
            vehicle["classification"]
        )


        model_name = (
            classification["model"]
        )

        model_conf = (
            classification["confidence"]
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
                max(30, vy1 - 10),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )


        # --------------------------------------------------
        # Plates
        # --------------------------------------------------

        for plate in vehicle.get(
            "plates",
            [],
        ):

            ax1, ay1, ax2, ay2 = (
                plate["bbox_in_image"]
            )


            ocr = plate["ocr"]

            plate_text = (
                ocr["text"]
                if ocr["text"]
                else "PLATE"
            )


            # RED
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
                    max(30, ay1 - 10),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )


    return frame


# --------------------------------------------------
# Console
# --------------------------------------------------

def print_status(
    result: dict,
    display_fps: float,
    pipeline_ms: float,
):

    print(
        "\033[2J\033[H",
        end="",
    )


    print(
        "========================================"
    )

    print(
        "       Parking AI Live Detection"
    )

    print(
        "========================================"
    )

    print(
        f"Display FPS : {display_fps:.1f}"
    )

    print(
        f"AI Pipeline : {pipeline_ms:.1f} ms"
    )


    vehicles = result.get(
        "vehicles",
        [],
    )


    print(
        f"Vehicles    : {len(vehicles)}"
    )

    print()


    if not vehicles:

        print(
            "[WAITING] No vehicle detected"
        )

        return


    for index, vehicle in enumerate(
        vehicles
    ):

        detection = vehicle[
            "detection"
        ]

        classification = vehicle[
            "classification"
        ]


        print(
            f"[Vehicle {index:02d}]"
        )

        print(
            f"  Type       : "
            f"{detection['class_name']}"
        )

        print(
            f"  Detection  : "
            f"{detection['confidence']:.3f}"
        )

        print(
            f"  Model      : "
            f"{classification['model']}"
        )

        print(
            f"  Class Conf : "
            f"{classification['confidence']:.3f}"
        )


        plates = vehicle.get(
            "plates",
            [],
        )


        if not plates:

            print(
                "  Plate      : NOT DETECTED"
            )


        for plate_index, plate in enumerate(
            plates
        ):

            ocr = plate["ocr"]


            print(
                f"  [Plate {plate_index:02d}]"
            )

            print(
                f"    Detect : "
                f"{plate['detection_confidence']:.3f}"
            )

            print(
                f"    Number : "
                f"{ocr['text'] or 'UNKNOWN'}"
            )

            print(
                f"    OCR    : "
                f"{ocr['confidence']:.3f}"
            )


        print()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    camera = None
    ai_worker = None


    try:

        # --------------------------------------------------
        # 1. Load AI
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
        # Display FPS
        # --------------------------------------------------

        previous_time = (
            time.perf_counter()
        )

        smoothed_fps = 0.0

        console_update_time = 0.0


        # --------------------------------------------------
        # Main Display Loop
        # --------------------------------------------------

        while True:

            frame = camera.read()


            if frame is None:

                time.sleep(
                    0.001
                )

                continue


            # --------------------------------------------------
            # 최신 AI Result
            # --------------------------------------------------

            result, pipeline_ms = (
                ai_worker.get_result()
            )


            # --------------------------------------------------
            # Detection Overlay
            # --------------------------------------------------

            display_frame = (
                draw_detection(
                    frame,
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
                    1.0 / delta
                )

            else:

                current_fps = 0.0


            if smoothed_fps == 0:

                smoothed_fps = (
                    current_fps
                )

            else:

                smoothed_fps = (
                    smoothed_fps * 0.9
                    + current_fps * 0.1
                )


            # --------------------------------------------------
            # Performance Overlay
            # --------------------------------------------------

            cv2.putText(
                display_frame,
                f"Display FPS: {smoothed_fps:.1f}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )


            cv2.putText(
                display_frame,
                f"AI: {pipeline_ms:.1f} ms",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )


            # --------------------------------------------------
            # Console
            #
            # 너무 자주 갱신하면 Terminal 자체가 부하가 됨
            # 0.5초마다 갱신
            # --------------------------------------------------

            if (
                current_time
                - console_update_time
                >= 0.5
            ):

                print_status(
                    result,
                    smoothed_fps,
                    pipeline_ms,
                )

                console_update_time = (
                    current_time
                )


            # --------------------------------------------------
            # Display
            # --------------------------------------------------

            cv2.imshow(
                "Parking AI Live",
                display_frame,
            )


            # --------------------------------------------------
            # Keyboard
            # --------------------------------------------------

            key = (
                cv2.waitKey(1)
                & 0xFF
            )


            if key == ord("q"):
                break


            if key == ord("s"):

                save_path = (
                    PROJECT_DIR
                    / "webcam_test.jpg"
                )

                cv2.imwrite(
                    str(save_path),
                    frame,
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