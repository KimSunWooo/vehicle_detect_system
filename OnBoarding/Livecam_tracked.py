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
    detect_vehicles_fast,
    analyze_vehicle,
    detect_fallback_plates,
)


# --------------------------------------------------
# Config
# --------------------------------------------------

CAMERA_DEVICE = "/dev/video2"

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
CAMERA_FPS = 30

# 빠른 차량 detection 주기
DETECTION_INTERVAL = 0.5

# 동일 차량 OCR 재시도 간격 / 최대 샘플 수
DEEP_ANALYSIS_RETRY_INTERVAL = 1.0
MAX_OCR_SAMPLES = 3

VEHICLE_CONF = 0.20
PLATE_CONF = 0.15

# --------------------------------------------------
# Vehicle Tracking
# --------------------------------------------------
# 0.5초 단위 detection bbox끼리 IoU로 같은 차량을 매칭한다.
TRACK_IOU_THRESHOLD = 0.20

# 연속 detection miss 허용 횟수.
# 0.5초 * 4 = 약 2초 동안 일시적인 miss를 허용한다.
MAX_TRACK_MISSES = 4

# 최근 차량이 있었는데 연속으로 이 횟수만큼 차량 detection이 비면
# full-frame 번호판 fallback을 1회 시도한다.
FALLBACK_AFTER_MISSES = 3
FALLBACK_COOLDOWN = 3.0

# Camera reconnect 설정
MAX_CAMERA_READ_FAILURES = 30
CAMERA_RECONNECT_DELAY = 1.0
CAMERA_RECONNECT_ATTEMPTS = 5

# 실제 Camera FPS 계산 시 최근 구간의 smoothing 비율
CAMERA_FPS_SMOOTHING = 0.90


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
        self.device = device
        self.width = width
        self.height = height
        self.requested_fps = fps

        self.cap = None

        self.frame = None
        self.frame_id = 0

        self.running = True
        self.lock = threading.Lock()

        self.camera_fps = 0.0
        self.last_capture_time = None

        self.consecutive_failures = 0

        self._open_camera()

        self.thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
        )

        self.thread.start()

    def _open_camera(self):
        """
        V4L2 카메라를 열고 요청 해상도/FPS를 적용한다.
        실제 적용된 값도 로그로 출력한다.
        """

        cap = cv2.VideoCapture(
            self.device,
            cv2.CAP_V4L2,
        )

        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                f"Could not open camera: {self.device}"
            )

        cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1,
        )

        cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            self.width,
        )

        cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            self.height,
        )

        cap.set(
            cv2.CAP_PROP_FPS,
            self.requested_fps,
        )

        actual_width = cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )

        actual_height = cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )

        actual_fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        print(
            "[Camera] Opened "
            f"{self.device}"
        )

        print(
            "[Camera] Requested: "
            f"{self.width}x{self.height}"
            f" @ {self.requested_fps} FPS"
        )

        print(
            "[Camera] Actual   : "
            f"{actual_width:.0f}x{actual_height:.0f}"
            f" @ {actual_fps:.1f} FPS"
        )

        with self.lock:
            old_cap = self.cap
            self.cap = cap

        if old_cap is not None:
            old_cap.release()

    def _reconnect_camera(self) -> bool:
        """
        연속 프레임 읽기 실패 시 카메라를 다시 연다.
        """

        print(
            "[Camera] "
            "Read failure threshold reached. "
            "Trying reconnect..."
        )

        for attempt in range(
            1,
            CAMERA_RECONNECT_ATTEMPTS + 1,
        ):
            if not self.running:
                return False

            print(
                "[Camera] Reconnect attempt "
                f"{attempt}/"
                f"{CAMERA_RECONNECT_ATTEMPTS}"
            )

            try:
                self._open_camera()

                self.consecutive_failures = 0

                print(
                    "[Camera] Reconnected successfully"
                )

                return True

            except Exception as exc:
                print(
                    "[Camera] Reconnect failed: "
                    f"{type(exc).__name__}: {exc}"
                )

                time.sleep(
                    CAMERA_RECONNECT_DELAY
                )

        print(
            "[Camera] "
            "Reconnect attempts exhausted"
        )

        return False

    def _capture_loop(self):

        while self.running:

            with self.lock:
                cap = self.cap

            if cap is None:
                time.sleep(0.05)
                continue

            ret, frame = cap.read()

            if not ret or frame is None:

                self.consecutive_failures += 1

                if (
                    self.consecutive_failures
                    >= MAX_CAMERA_READ_FAILURES
                ):
                    success = (
                        self._reconnect_camera()
                    )

                    if not success:
                        time.sleep(
                            CAMERA_RECONNECT_DELAY
                        )

                else:
                    time.sleep(0.01)

                continue

            self.consecutive_failures = 0

            now = time.perf_counter()

            if self.last_capture_time is not None:

                delta = (
                    now
                    - self.last_capture_time
                )

                if delta > 0:
                    instant_fps = (
                        1.0 / delta
                    )

                    if self.camera_fps == 0.0:
                        self.camera_fps = (
                            instant_fps
                        )

                    else:
                        self.camera_fps = (
                            self.camera_fps
                            * CAMERA_FPS_SMOOTHING
                            + instant_fps
                            * (
                                1.0
                                - CAMERA_FPS_SMOOTHING
                            )
                        )

            self.last_capture_time = now

            with self.lock:
                self.frame = frame
                self.frame_id += 1

    def read(self):

        with self.lock:

            if self.frame is None:
                return None

            return self.frame.copy()

    def read_with_meta(self):

        with self.lock:

            if self.frame is None:
                return None, 0, self.camera_fps

            return (
                self.frame.copy(),
                self.frame_id,
                self.camera_fps,
            )

    def get_camera_fps(self) -> float:

        with self.lock:
            return self.camera_fps

    def release(self):

        self.running = False

        if self.thread.is_alive():
            self.thread.join(
                timeout=2.0
            )

        with self.lock:
            cap = self.cap
            self.cap = None

        if cap is not None:
            cap.release()


# --------------------------------------------------
# Snapshot
# --------------------------------------------------
#
# 기존 JPEG encode -> decode 과정은 제거했다.
# AI에는 최신 OpenCV ndarray frame.copy()를 그대로 전달한다.

def simplify_result(result: dict) -> list[dict]:

    vehicles = result.get(
        "vehicles",
        [],
    )

    fallback_plates = result.get(
        "fallback_plates",
        [],
    )

    # fallback OCR 목록
    fallback_numbers = []

    for plate in fallback_plates:

        ocr = plate.get(
            "ocr",
            {},
        )

        text = (
            ocr.get("text")
            or ""
        ).strip()

        if text:
            fallback_numbers.append(text)

    simplified = []

    # 차량별 정보
    for index, vehicle in enumerate(
        vehicles,
        start=1,
    ):

        # -------------------------
        # 차량 모델
        # -------------------------

        classification = vehicle.get(
            "classification",
            {},
        )

        model = classification.get(
            "model",
        )

        # -------------------------
        # OCR
        # -------------------------

        ocr_data = None

        plates = vehicle.get(
            "plates",
            [],
        )

        if plates:

            # OCR confidence가 가장 높은 번호판 사용
            best_plate = max(
                plates,
                key=lambda p: float(
                    p.get(
                        "ocr",
                        {},
                    ).get(
                        "confidence",
                        0.0,
                    )
                ),
            )

            ocr = best_plate.get(
                "ocr",
                {},
            )

            text = (
                ocr.get("text")
                or ""
            ).strip()

            if text:
                ocr_data = text

        # -------------------------
        # Fallback OCR
        # -------------------------

        fallback_ocr = (
            []
        )

        simplified.append(
            {
                "index": index,
                "classification": {
                    "model": model,
                },
                "ocr_data": ocr_data,
                "fallback_ocr": fallback_ocr,
            }
        )

    return simplified

# --------------------------------------------------
# Vehicle Tracking Helpers
# --------------------------------------------------

def calculate_iou(box_a, box_b):
    """두 bbox의 IoU(Intersection over Union)를 계산한다."""

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


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

        self.tracks = []
        self.next_track_id = 1

        self.pipeline_ms = 0.0
        self.last_detection_timestamp = 0.0
        self.last_fallback_timestamp = 0.0
        self.consecutive_empty_detections = 0

        self.running = True
        self.lock = threading.Lock()

        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
        )
        self.thread.start()

    @staticmethod
    def _copy_bbox(bbox):
        return [int(v) for v in bbox]

    def _match_detections(self, detections):
        """0.5초 단위 bbox를 기존 track과 IoU 매칭한다."""
        unmatched_track_indexes = set(range(len(self.tracks)))
        matches = []
        unmatched_detections = []

        for detection in detections:
            best_index = None
            best_iou = 0.0

            for track_index in unmatched_track_indexes:
                track = self.tracks[track_index]
                iou = calculate_iou(
                    track["bbox"],
                    detection.bbox,
                )
                if iou > best_iou:
                    best_iou = iou
                    best_index = track_index

            if (
                best_index is not None
                and best_iou >= TRACK_IOU_THRESHOLD
            ):
                matches.append((best_index, detection))
                unmatched_track_indexes.remove(best_index)
            else:
                unmatched_detections.append(detection)

        return (
            matches,
            list(unmatched_track_indexes),
            unmatched_detections,
        )

    def _collect_ocr_sample(self, track, vehicle_record):
        """상세 분석 결과에서 가장 신뢰도 높은 번호판 OCR 1개를 누적한다."""
        plates = vehicle_record.get("plates", [])
        if not plates:
            return

        best_plate = max(
            plates,
            key=lambda p: (
                float(p.get("ocr", {}).get("confidence", 0.0)),
                float(p.get("detection_confidence", 0.0)),
            ),
        )

        ocr = best_plate.get("ocr", {})
        text = (ocr.get("text") or "").strip()

        if not text:
            return

        track["ocr_samples"].append(
            {
                "text": text,
                "confidence": float(ocr.get("confidence", 0.0)),
                "accepted": bool(ocr.get("accepted", False)),
                "valid_format": bool(ocr.get("valid_format", False)),
            }
        )

    @staticmethod
    def _vote_ocr(samples):
        """동일 문자열 빈도 우선, confidence 합을 보조 기준으로 사용한다."""
        if not samples:
            return None

        grouped = {}
        for sample in samples:
            text = sample["text"]
            bucket = grouped.setdefault(
                text,
                {
                    "count": 0,
                    "confidence_sum": 0.0,
                    "confidence_max": 0.0,
                    "valid_count": 0,
                    "accepted_count": 0,
                },
            )
            bucket["count"] += 1
            bucket["confidence_sum"] += sample["confidence"]
            bucket["confidence_max"] = max(
                bucket["confidence_max"],
                sample["confidence"],
            )
            bucket["valid_count"] += int(sample["valid_format"])
            bucket["accepted_count"] += int(sample["accepted"])

        best_text, stats = max(
            grouped.items(),
            key=lambda item: (
                item[1]["count"],
                item[1]["valid_count"],
                item[1]["accepted_count"],
                item[1]["confidence_sum"],
                item[1]["confidence_max"],
            ),
        )

        return {
            "text": best_text,
            "confidence": stats["confidence_sum"] / stats["count"],
            "vote_count": stats["count"],
            "sample_count": len(samples),
        }

    def _apply_track_bbox_to_analysis(self, track):
        """최신 vehicle bbox에 맞춰 plate 원본 좌표를 다시 계산한다."""
        analysis = track.get("analysis")
        if not analysis:
            return None

        vx1, vy1, vx2, vy2 = track["bbox"]
        analysis["bbox"] = list(track["bbox"])
        analysis["detection"] = dict(analysis.get("detection", {}))
        analysis["detection"]["confidence"] = track["confidence"]

        for plate in analysis.get("plates", []):
            rel = plate.get("bbox_in_vehicle")
            if not rel:
                continue
            px1, py1, px2, py2 = rel
            plate["bbox_in_image"] = [
                vx1 + px1,
                vy1 + py1,
                vx1 + px2,
                vy1 + py2,
            ]

        voted = self._vote_ocr(track["ocr_samples"])
        if voted and analysis.get("plates"):
            # 표시용 대표 plate OCR에 voting 결과를 반영한다.
            best_plate = max(
                analysis["plates"],
                key=lambda p: float(
                    p.get("ocr", {}).get("confidence", 0.0)
                ),
            )
            best_plate["ocr"] = dict(best_plate.get("ocr", {}))
            best_plate["ocr"]["text"] = voted["text"]
            best_plate["ocr"]["confidence"] = voted["confidence"]
            best_plate["ocr"]["vote_count"] = voted["vote_count"]
            best_plate["ocr"]["sample_count"] = voted["sample_count"]

        return analysis

    def _deep_analyze_track(self, track, detection, now):
        """최초 또는 OCR 재시도 시 상세 분석. 차종은 최초 결과를 cache한다."""
        cached_classification = None
        if track.get("analysis"):
            cached_classification = track["analysis"].get("classification")

        start = time.perf_counter()
        vehicle_record = analyze_vehicle(
            detection,
            self.plate_detector,
            self.vehicle_classifier,
            self.plate_ocr,
            cached_classification=cached_classification,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        track["analysis"] = vehicle_record
        track["last_deep_analysis"] = now
        track["deep_analysis_count"] += 1
        self._collect_ocr_sample(track, vehicle_record)

        return elapsed_ms

    def _should_retry_deep_analysis(self, track, now):
        if track["deep_analysis_count"] >= MAX_OCR_SAMPLES:
            return False

        if (
            now - track["last_deep_analysis"]
            < DEEP_ANALYSIS_RETRY_INTERVAL
        ):
            return False

        voted = self._vote_ocr(track["ocr_samples"])

        # 같은 문자열이 두 번 이상 나오면 충분히 안정화된 것으로 간주.
        if voted and voted["vote_count"] >= 2:
            return False

        return True

    def _build_result(self, fallback_plates=None):
        vehicles = []
        for track in self.tracks:
            analysis = self._apply_track_bbox_to_analysis(track)
            if analysis is not None:
                vehicles.append(analysis)

        return {
            "vehicles": vehicles,
            "fallback_plates": [],
        }

    def _run(self):
        while self.running:
            now = time.monotonic()

            if (
                now - self.last_detection_timestamp
                < DETECTION_INTERVAL
            ):
                time.sleep(0.01)
                continue

            frame = self.camera.read()
            if frame is None:
                time.sleep(0.02)
                continue

            self.last_detection_timestamp = now

            cycle_start = time.perf_counter()

            try:
                detections = detect_vehicles_fast(
                    frame,
                    self.vehicle_detector,
                )

                if detections:
                    self.consecutive_empty_detections = 0
                else:
                    self.consecutive_empty_detections += 1

                (
                    matches,
                    unmatched_track_indexes,
                    unmatched_detections,
                ) = self._match_detections(detections)

                deep_ms_total = 0.0

                # 기존 track 갱신
                for track_index, detection in matches:
                    track = self.tracks[track_index]
                    track["bbox"] = self._copy_bbox(detection.bbox)
                    track["confidence"] = float(detection.confidence)
                    track["miss_count"] = 0

                    if self._should_retry_deep_analysis(track, now):
                        deep_ms_total += self._deep_analyze_track(
                            track,
                            detection,
                            now,
                        )

                # 이번 detection에서 놓친 track
                for track_index in unmatched_track_indexes:
                    self.tracks[track_index]["miss_count"] += 1

                # 새 차량 -> 즉시 상세 분석
                for detection in unmatched_detections:
                    track = {
                        "track_id": self.next_track_id,
                        "bbox": self._copy_bbox(detection.bbox),
                        "confidence": float(detection.confidence),
                        "miss_count": 0,
                        "analysis": None,
                        "ocr_samples": [],
                        "deep_analysis_count": 0,
                        "last_deep_analysis": 0.0,
                    }
                    self.next_track_id += 1

                    deep_ms_total += self._deep_analyze_track(
                        track,
                        detection,
                        now,
                    )
                    self.tracks.append(track)

                    print(
                        f"[TRACK] New vehicle #{track['track_id']} "
                        f"bbox={track['bbox']}"
                    )

                # stale track 제거
                alive_tracks = []
                for track in self.tracks:
                    if track["miss_count"] <= MAX_TRACK_MISSES:
                        alive_tracks.append(track)
                    else:
                        print(
                            f"[TRACK] Removed vehicle #{track['track_id']} "
                            f"after {track['miss_count']} misses"
                        )
                self.tracks = alive_tracks

                fallback_plates = []

                # 최근 차량 track이 남아 있는데 detection이 연속 miss인 경우에만 fallback.
                should_fallback = (
                    bool(self.tracks)
                    and not detections
                    and self.consecutive_empty_detections >= FALLBACK_AFTER_MISSES
                    and now - self.last_fallback_timestamp >= FALLBACK_COOLDOWN
                )

                if should_fallback:
                    print(
                        "[FALLBACK] Consecutive vehicle misses. "
                        "Running full-frame plate detection once."
                    )
                    fallback_plates = detect_fallback_plates(
                        frame,
                        self.plate_detector,
                        self.plate_ocr,
                    )
                    self.last_fallback_timestamp = now

                cycle_ms = (time.perf_counter() - cycle_start) * 1000.0

                with self.lock:
                    self.result = self._build_result(
                        fallback_plates
                    )

                    self.pipeline_ms = cycle_ms

                    current_result = self.result


                # ------------------------------------------
                # Terminal Result
                # ------------------------------------------

                simple_results = simplify_result(
                    current_result
                )

                for vehicle in simple_results:
                    print(vehicle)

            except Exception as exc:
                print(
                    "[AI PIPELINE ERROR] "
                    f"{type(exc).__name__}: {exc}"
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
            self.thread.join(timeout=2.0)
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def put_korean_text(
    frame,
    text,
    position,
    font_path="/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    font_size=24,
    color=(0, 0, 255),
):
    """
    OpenCV 이미지에 한글 텍스트를 출력한다.

    color는 OpenCV 기준 BGR.
    """

    # BGR -> RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    pil_image = Image.fromarray(
        rgb_frame
    )

    draw = ImageDraw.Draw(
        pil_image
    )

    font = ImageFont.truetype(
        font_path,
        font_size,
    )

    # BGR -> RGB
    rgb_color = (
        color[2],
        color[1],
        color[0],
    )

    draw.text(
        position,
        text,
        font=font,
        fill=rgb_color,
    )

    # RGB -> BGR
    result = cv2.cvtColor(
        np.array(pil_image),
        cv2.COLOR_RGB2BGR,
    )

    return result

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

        # 차량 모델명은 영문이므로 기존 OpenCV putText 사용
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

            bbox = plate.get(
                "bbox_in_image"
            )

            if not bbox:
                continue

            ax1, ay1, ax2, ay2 = bbox

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

            frame = put_korean_text(
                frame,
                plate_text,
                (
                    ax1,
                    max(
                        0,
                        ay1 - 32,
                    ),
                ),
                font_size=26,
                color=(0, 0, 255),
            )

    # --------------------------------------------------
    # Fallback Plate Results
    # --------------------------------------------------

    for plate in result.get(
        "fallback_plates",
        [],
    ):

        bbox = plate.get(
            "bbox_in_image"
        )

        if not bbox:
            continue

        px1, py1, px2, py2 = bbox

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

        frame = put_korean_text(
            frame,
            plate_text,
            (
                px1,
                max(
                    0,
                    py1 - 30,
                ),
            ),
            font_size=24,
            color=(0, 165, 255),
        )

    return frame


# --------------------------------------------------
# Status Overlay
# --------------------------------------------------

def draw_status(
    frame,
    display_fps: float,
    camera_fps: float,
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
        f"Camera FPS: {camera_fps:.1f}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"Display FPS: {display_fps:.1f}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"AI: {pipeline_ms:.1f} ms",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"Vehicles: {vehicle_count}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"Detect Interval: {DETECTION_INTERVAL:.1f}s",
        (20, 150),
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
            "Parking AI Live Snapshot Detection - Stable"
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
            f"Detect interval: "
            f"{DETECTION_INTERVAL:.1f} sec"
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

            (
                current_frame,
                current_frame_id,
                camera_fps,
            ) = camera.read_with_meta()

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
                    camera_fps,
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