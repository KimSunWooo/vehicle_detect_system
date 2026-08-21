from __future__ import annotations

# --------------------------------------------------
# PaddleOCR / oneDNN 설정
# 반드시 PaddleOCR import 전에 실행
# --------------------------------------------------

import os

os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_use_onednn", "0")


import argparse
import json
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO


# --------------------------------------------------
# Project path
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


# --------------------------------------------------
# Project imports
# --------------------------------------------------

from config import (
    ANNOTATED_DIR,
    IMAGE_EXTENSIONS,
    INPUT_DIR,
    PLATE_CONF,
    PLATE_CROP_DIR,
    PLATE_MODEL,
    PLATE_OCR_CONF,
    VEHICLE_CLASSES,
    VEHICLE_CLASSIFIER_CONF,
    VEHICLE_CLASSIFIER_MODEL,
    VEHICLE_CONF,
    VEHICLE_CROP_DIR,
    VEHICLE_MODEL,
    ensure_directories,
)

from detector.plate_detector import PlateDetector  # noqa: E402
from detector.vehicle_detector import VehicleDetector  # noqa: E402

# OCR 추가
from ocr.plate_ocr import PlateOCR  # noqa: E402


# --------------------------------------------------
# Image iterator
# --------------------------------------------------

def iter_images(source: Path) -> list[Path]:

    if source.is_file():
        return (
            [source]
            if source.suffix.lower() in IMAGE_EXTENSIONS
            else []
        )

    if source.is_dir():
        return sorted(
            p
            for p in source.rglob("*")
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        )

    raise FileNotFoundError(
        f"Source not found: {source}"
    )


# --------------------------------------------------
# Vehicle classification
# --------------------------------------------------

def classify_vehicle(
    vehicle_crop,
    vehicle_classifier: YOLO,
) -> dict:

    results = vehicle_classifier(
        vehicle_crop,
        verbose=False,
    )

    result = results[0]

    if result.probs is None:
        return {
            "model": "UNKNOWN",
            "raw_model": "UNKNOWN",
            "confidence": 0.0,
            "top3": [],
        }

    # --------------------------------------------------
    # Top 1
    # --------------------------------------------------

    top1_id = int(
        result.probs.top1
    )

    top1_conf = float(
        result.probs.top1conf.item()
    )

    top1_name = (
        result.names[top1_id]
    )

    if top1_conf < VEHICLE_CLASSIFIER_CONF:
        final_name = "UNKNOWN"
    else:
        final_name = top1_name

    # --------------------------------------------------
    # Top 3
    # --------------------------------------------------

    probabilities = (
        result.probs.data.cpu()
    )

    top_count = min(
        3,
        len(probabilities),
    )

    top_values, top_indices = (
        probabilities.topk(
            top_count
        )
    )

    top3 = []

    for class_id, confidence in zip(
        top_indices.tolist(),
        top_values.tolist(),
    ):

        top3.append(
            {
                "model": (
                    result.names[class_id]
                ),
                "confidence": (
                    float(confidence)
                ),
            }
        )

    return {
        "model": final_name,
        "raw_model": top1_name,
        "confidence": top1_conf,
        "top3": top3,
    }


# --------------------------------------------------
# Process image
# --------------------------------------------------

def process_image(
    image_path: Path,
    vehicle_detector: VehicleDetector,
    plate_detector: PlateDetector,
    vehicle_classifier: YOLO,
    plate_ocr: PlateOCR,
) -> dict:

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    annotated = image.copy()

    # --------------------------------------------------
    # 1. Vehicle Detection
    # --------------------------------------------------

    vehicles = (
        vehicle_detector.detect(
            image
        )
    )

    record = {
        "image": str(image_path),
        "vehicles": [],
    }

    stem = image_path.stem


    # --------------------------------------------------
    # Process Vehicles
    # --------------------------------------------------

    for vehicle_index, vehicle in enumerate(
        vehicles
    ):

        # --------------------------------------------------
        # Vehicle Crop 저장
        # --------------------------------------------------

        vehicle_name = (
            f"{stem}_vehicle_"
            f"{vehicle_index:02d}.jpg"
        )

        vehicle_path = (
            VEHICLE_CROP_DIR
            / vehicle_name
        )

        cv2.imwrite(
            str(vehicle_path),
            vehicle.crop,
        )


        # --------------------------------------------------
        # 2. Vehicle Classification
        # --------------------------------------------------

        # 현재 classifier는 승용차 중심이므로
        # car에 대해서만 세부 차종 분류
        if vehicle.class_name == "vehicle":

            classification = (
                classify_vehicle(
                    vehicle.crop,
                    vehicle_classifier,
                )
            )

        else:

            classification = {
                "model": vehicle.class_name,
                "raw_model": vehicle.class_name,
                "confidence": (
                    vehicle.confidence
                ),
                "top3": [],
            }


        vehicle_model_name = (
            classification["model"]
        )

        vehicle_model_conf = (
            classification["confidence"]
        )


        # --------------------------------------------------
        # Vehicle bbox
        # --------------------------------------------------

        vx1, vy1, vx2, vy2 = (
            vehicle.bbox
        )


        # --------------------------------------------------
        # Vehicle Bounding Box
        # --------------------------------------------------

        cv2.rectangle(
            annotated,
            (vx1, vy1),
            (vx2, vy2),
            (0, 255, 0),
            2,
        )


        # --------------------------------------------------
        # Vehicle Label
        # --------------------------------------------------

        vehicle_label = (
            f"{vehicle_model_name} "
            f"{vehicle_model_conf:.2f}"
        )

        cv2.putText(
            annotated,
            vehicle_label,
            (
                vx1,
                max(
                    20,
                    vy1 - 8,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


        # --------------------------------------------------
        # 3. Plate Detection
        # --------------------------------------------------

        plates = (
            plate_detector.detect(
                vehicle.crop
            )
        )


        # --------------------------------------------------
        # Vehicle Record
        # --------------------------------------------------

        vehicle_record = {

            "bbox": list(
                vehicle.bbox
            ),

            "detection": {

                "confidence": (
                    vehicle.confidence
                ),

                "class_id": (
                    vehicle.class_id
                ),

                "class_name": (
                    vehicle.class_name
                ),
            },

            "classification": {

                "model": (
                    classification[
                        "model"
                    ]
                ),

                "raw_model": (
                    classification[
                        "raw_model"
                    ]
                ),

                "confidence": (
                    classification[
                        "confidence"
                    ]
                ),

                "top3": (
                    classification[
                        "top3"
                    ]
                ),
            },

            "crop": str(
                vehicle_path
            ),

            "plates": [],
        }


        # --------------------------------------------------
        # Process Plates
        # --------------------------------------------------

        for plate_index, plate in enumerate(
            plates
        ):

            plate_name = (
                f"{stem}"
                f"_vehicle_"
                f"{vehicle_index:02d}"
                f"_plate_"
                f"{plate_index:02d}"
                f".jpg"
            )

            plate_path = (
                PLATE_CROP_DIR
                / plate_name
            )


            # --------------------------------------------------
            # Plate crop 저장
            # --------------------------------------------------

            cv2.imwrite(
                str(plate_path),
                plate.crop,
            )


            # --------------------------------------------------
            # 4. OCR
            # --------------------------------------------------

            try:

                ocr_result = (
                    plate_ocr.recognize(
                        plate.crop
                    )
                )

            except Exception as exc:

                print(
                    f"      OCR FAIL "
                    f"vehicle={vehicle_index} "
                    f"plate={plate_index}: "
                    f"{exc}"
                )

                ocr_result = {
                    "text": "",
                    "raw_text": "",
                    "confidence": 0.0,
                    "accepted": False,
                    "valid_format": False,
                    "best_variant": None,
                    "candidates": [],
                }


            # --------------------------------------------------
            # Plate bbox
            # --------------------------------------------------

            px1, py1, px2, py2 = (
                plate.bbox
            )

            ax1 = vx1 + px1
            ay1 = vy1 + py1
            ax2 = vx1 + px2
            ay2 = vy1 + py2


            # --------------------------------------------------
            # Plate Bounding Box
            # --------------------------------------------------

            cv2.rectangle(
                annotated,
                (ax1, ay1),
                (ax2, ay2),
                (0, 0, 255),
                2,
            )


            # --------------------------------------------------
            # Plate label
            # --------------------------------------------------

            plate_text = (
                ocr_result["text"]
            )

            ocr_conf = (
                ocr_result[
                    "confidence"
                ]
            )

            if plate_text:

                plate_label = (
                    f"{plate_text} "
                    f"{ocr_conf:.2f}"
                )

            else:

                plate_label = (
                    f"plate "
                    f"{plate.confidence:.2f}"
                )


            cv2.putText(
                annotated,
                plate_label,
                (
                    ax1,
                    max(
                        20,
                        ay1 - 8,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )


            # --------------------------------------------------
            # Plate Record
            # --------------------------------------------------

            vehicle_record[
                "plates"
            ].append(
                {

                    "bbox_in_vehicle": (
                        list(
                            plate.bbox
                        )
                    ),

                    "bbox_in_image": [
                        ax1,
                        ay1,
                        ax2,
                        ay2,
                    ],

                    "detection_confidence": (
                        plate.confidence
                    ),

                    "crop": str(
                        plate_path
                    ),

                    "ocr": {

                        "text": (
                            ocr_result[
                                "text"
                            ]
                        ),

                        "raw_text": (
                            ocr_result[
                                "raw_text"
                            ]
                        ),

                        "confidence": (
                            ocr_result[
                                "confidence"
                            ]
                        ),

                        "accepted": (
                            ocr_result[
                                "accepted"
                            ]
                        ),

                        "valid_format": (
                            ocr_result[
                                "valid_format"
                            ]
                        ),

                        "best_variant": (
                            ocr_result[
                                "best_variant"
                            ]
                        ),

                        "candidates": (
                            ocr_result[
                                "candidates"
                            ]
                        ),
                    },
                }
            )


        # --------------------------------------------------
        # Vehicle 추가
        # --------------------------------------------------

        record[
            "vehicles"
        ].append(
            vehicle_record
        )


    # --------------------------------------------------
    # Annotated 이미지 저장
    # --------------------------------------------------

    annotated_path = (
        ANNOTATED_DIR
        / f"{image_path.stem}"
          "_annotated.jpg"
    )

    cv2.imwrite(
        str(annotated_path),
        annotated,
    )

    record[
        "annotated"
    ] = str(
        annotated_path
    )

    return record


# --------------------------------------------------
# Arguments
# --------------------------------------------------

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Vehicle Detection "
            "-> Vehicle Classification "
            "-> Plate Detection "
            "-> OCR Pipeline"
        )
    )

    parser.add_argument(
        "--source",
        type=Path,
        default=INPUT_DIR,
        help=(
            "Image file or directory "
            "(default: input/)"
        ),
    )

    parser.add_argument(
        "--vehicle-conf",
        type=float,
        default=VEHICLE_CONF,
    )

    parser.add_argument(
        "--plate-conf",
        type=float,
        default=PLATE_CONF,
    )

    parser.add_argument(
        "--classifier-conf",
        type=float,
        default=(
            VEHICLE_CLASSIFIER_CONF
        ),
    )

    parser.add_argument(
        "--ocr-conf",
        type=float,
        default=(
            PLATE_OCR_CONF
        ),
    )

    return parser

def _empty_ocr_result() -> dict:
    return {
        "text": "",
        "raw_text": "",
        "confidence": 0.0,
        "accepted": False,
        "valid_format": False,
        "best_variant": None,
        "candidates": [],
    }


def detect_vehicles_fast(
    image,
    vehicle_detector: VehicleDetector,
):
    """Live camera용 lightweight vehicle detection.

    Classification / plate detection / OCR을 수행하지 않고 차량 detector만 실행한다.
    """
    return vehicle_detector.detect(image)


def analyze_vehicle(
    vehicle,
    plate_detector: PlateDetector,
    vehicle_classifier: YOLO,
    plate_ocr: PlateOCR,
    cached_classification: dict | None = None,
) -> dict:
    """검출된 차량 1대에 대해 상세 분석을 수행한다.

    cached_classification이 있으면 차종 classifier를 다시 실행하지 않는다.
    번호판 detection + OCR은 현재 vehicle crop 기준으로 수행한다.
    """
    if cached_classification is not None:
        classification = cached_classification
    elif vehicle.class_name == "vehicle":
        classification = classify_vehicle(
            vehicle.crop,
            vehicle_classifier,
        )
    else:
        classification = {
            "model": vehicle.class_name,
            "raw_model": vehicle.class_name,
            "confidence": vehicle.confidence,
            "top3": [],
        }

    vx1, vy1, vx2, vy2 = vehicle.bbox

    vehicle_record = {
        "bbox": list(vehicle.bbox),
        "detection": {
            "confidence": vehicle.confidence,
            "class_id": vehicle.class_id,
            "class_name": vehicle.class_name,
        },
        "classification": classification,
        "plates": [],
    }

    plates = plate_detector.detect(vehicle.crop)

    for plate_index, plate in enumerate(plates):
        try:
            ocr_result = plate_ocr.recognize(plate.crop)
        except Exception as exc:
            print(
                f"OCR FAIL plate={plate_index}: "
                f"{type(exc).__name__}: {exc}"
            )
            ocr_result = _empty_ocr_result()

        px1, py1, px2, py2 = plate.bbox

        vehicle_record["plates"].append(
            {
                "bbox_in_vehicle": list(plate.bbox),
                "bbox_in_image": [
                    vx1 + px1,
                    vy1 + py1,
                    vx1 + px2,
                    vy1 + py2,
                ],
                "detection_confidence": plate.confidence,
                "ocr": ocr_result,
            }
        )

    return vehicle_record


def detect_fallback_plates(
    image,
    plate_detector: PlateDetector,
    plate_ocr: PlateOCR,
) -> list[dict]:
    """차량 detector miss가 의심될 때만 호출하는 full-frame fallback."""
    fallback_records = []

    plates = plate_detector.detect(image)

    for plate_index, plate in enumerate(plates):
        try:
            ocr_result = plate_ocr.recognize(plate.crop)
        except Exception as exc:
            print(
                f"FALLBACK OCR FAIL plate={plate_index}: "
                f"{type(exc).__name__}: {exc}"
            )
            ocr_result = _empty_ocr_result()

        px1, py1, px2, py2 = plate.bbox

        fallback_records.append(
            {
                "bbox_in_image": [px1, py1, px2, py2],
                "detection_confidence": plate.confidence,
                "source": "full_frame_fallback",
                "ocr": ocr_result,
            }
        )

    return fallback_records


def process_frame(
    image,
    vehicle_detector: VehicleDetector,
    plate_detector: PlateDetector,
    vehicle_classifier: YOLO,
    plate_ocr: PlateOCR,
) -> tuple:
    """기존 snapshot/entrance camera 호환용 전체 파이프라인.

    Live camera에서는 detect_vehicles_fast()와 analyze_vehicle()를 분리 호출하는 것을 권장한다.
    """
    annotated = image.copy()
    vehicles = detect_vehicles_fast(image, vehicle_detector)

    print(f"[DEBUG] vehicles detected: {len(vehicles)}")

    record = {
        "vehicles": [],
        "fallback_plates": [],
    }

    for vehicle in vehicles:
        vehicle_record = analyze_vehicle(
            vehicle,
            plate_detector,
            vehicle_classifier,
            plate_ocr,
        )
        record["vehicles"].append(vehicle_record)

        vx1, vy1, vx2, vy2 = vehicle_record["bbox"]
        classification = vehicle_record["classification"]

        cv2.rectangle(
            annotated,
            (vx1, vy1),
            (vx2, vy2),
            (255, 0, 0),
            2,
        )

        cv2.putText(
            annotated,
            f"{classification['model']} {classification['confidence']:.2f}",
            (vx1, max(20, vy1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

        for plate in vehicle_record["plates"]:
            ax1, ay1, ax2, ay2 = plate["bbox_in_image"]
            ocr_result = plate["ocr"]
            plate_text = ocr_result.get("text", "")
            ocr_conf = float(ocr_result.get("confidence", 0.0))

            cv2.rectangle(
                annotated,
                (ax1, ay1),
                (ax2, ay2),
                (0, 0, 255),
                2,
            )

            label = (
                f"{plate_text} {ocr_conf:.2f}"
                if plate_text
                else f"plate {plate['detection_confidence']:.2f}"
            )

            cv2.putText(
                annotated,
                label,
                (ax1, max(20, ay1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

    # Snapshot/entrance 경로는 기존 동작을 보존한다.
    if not vehicles:
        print(
            "[FALLBACK] No vehicle detected. "
            "Running plate detection on full frame."
        )
        fallback_records = detect_fallback_plates(
            image,
            plate_detector,
            plate_ocr,
        )
        record["fallback_plates"] = fallback_records

        for plate in fallback_records:
            px1, py1, px2, py2 = plate["bbox_in_image"]
            ocr_result = plate["ocr"]
            plate_text = ocr_result.get("text", "")
            ocr_conf = float(ocr_result.get("confidence", 0.0))

            cv2.rectangle(
                annotated,
                (px1, py1),
                (px2, py2),
                (0, 165, 255),
                2,
            )

            label = (
                f"FALLBACK {plate_text} {ocr_conf:.2f}"
                if plate_text
                else f"FALLBACK PLATE {plate['detection_confidence']:.2f}"
            )

            cv2.putText(
                annotated,
                label,
                (px1, max(20, py1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )

    return annotated, record

def load_pipeline(
    vehicle_conf: float = VEHICLE_CONF,
    plate_conf: float = PLATE_CONF,
    ocr_conf: float = PLATE_OCR_CONF,
):

    ensure_directories()

    # --------------------------------------------------
    # Classifier model 확인
    # --------------------------------------------------

    if not VEHICLE_CLASSIFIER_MODEL.exists():
        raise FileNotFoundError(
            "Vehicle classifier model not found:\n"
            f"{VEHICLE_CLASSIFIER_MODEL}"
        )


    # --------------------------------------------------
    # 1. Vehicle Detector
    # --------------------------------------------------

    print(
        "[1/4] "
        "Loading vehicle detector:"
    )

    print(
        f"      {VEHICLE_MODEL}"
    )

    vehicle_detector = VehicleDetector(
        VEHICLE_MODEL,
        conf=vehicle_conf,
        classes=VEHICLE_CLASSES,
    )


    # --------------------------------------------------
    # 2. Vehicle Classifier
    # --------------------------------------------------

    print(
        "[2/4] "
        "Loading vehicle classifier:"
    )

    print(
        f"      {VEHICLE_CLASSIFIER_MODEL}"
    )

    vehicle_classifier = YOLO(
        VEHICLE_CLASSIFIER_MODEL
    )


    # --------------------------------------------------
    # 3. Plate Detector
    # --------------------------------------------------

    print(
        "[3/4] "
        "Loading plate detector:"
    )

    print(
        f"      {PLATE_MODEL}"
    )

    plate_detector = PlateDetector(
        PLATE_MODEL,
        conf=plate_conf,
    )


    # --------------------------------------------------
    # 4. OCR
    # --------------------------------------------------

    print(
        "[4/4] "
        "Loading Korean plate OCR"
    )

    plate_ocr = PlateOCR(
        min_confidence=ocr_conf,
        scale=3,
    )


    print("Pipeline loaded.")


    return (
        vehicle_detector,
        plate_detector,
        vehicle_classifier,
        plate_ocr,
    )

# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:

    global VEHICLE_CLASSIFIER_CONF


    # --------------------------------------------------
    # Arguments
    # --------------------------------------------------

    args = (
        build_parser()
        .parse_args()
    )


    VEHICLE_CLASSIFIER_CONF = (
        args.classifier_conf
    )


    # --------------------------------------------------
    # Pipeline Load
    # --------------------------------------------------

    (
        vehicle_detector,
        plate_detector,
        vehicle_classifier,
        plate_ocr,
    ) = load_pipeline(
        vehicle_conf=args.vehicle_conf,
        plate_conf=args.plate_conf,
        ocr_conf=args.ocr_conf,
    )


    # --------------------------------------------------
    # Input
    # --------------------------------------------------

    images = iter_images(
        args.source
    )


    if not images:

        print(
            "No input images found: "
            f"{args.source}"
        )

        return


    # --------------------------------------------------
    # Pipeline
    # --------------------------------------------------

    print(
        f"Processing "
        f"{len(images)} image(s)"
    )


    all_results = []


    for image_path in images:

        try:

            result = process_image(
                image_path,
                vehicle_detector,
                plate_detector,
                vehicle_classifier,
                plate_ocr,
            )


            all_results.append(
                result
            )


            # --------------------------------------------------
            # Console Result
            # --------------------------------------------------

            print()

            print(
                f"  OK "
                f"{image_path.name}"
            )


            for index, vehicle in enumerate(
                result["vehicles"]
            ):

                cls = (
                    vehicle[
                        "classification"
                    ]
                )


                print(
                    f"     Vehicle "
                    f"{index:02d}"
                )


                print(
                    f"       model: "
                    f"{cls['model']} "
                    f"("
                    f"{cls['confidence']:.3f}"
                    f")"
                )


                for plate_index, plate in enumerate(
                    vehicle["plates"]
                ):

                    ocr_data = (
                        plate["ocr"]
                    )


                    print(
                        f"       plate "
                        f"{plate_index:02d}: "
                        f"{ocr_data['text'] or 'UNKNOWN'} "
                        f"("
                        f"{ocr_data['confidence']:.3f}"
                        f")"
                    )


        except Exception as exc:

            print(
                f"  FAIL "
                f"{image_path.name}: "
                f"{exc}"
            )


    # --------------------------------------------------
    # JSON 저장
    # --------------------------------------------------

    result_path = (
        PROJECT_DIR
        / "output"
        / "results.json"
    )


    result_path.write_text(
        json.dumps(
            all_results,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


    print()

    print(
        f"Done. Results: "
        f"{result_path}"
    )


if __name__ == "__main__":
    main()