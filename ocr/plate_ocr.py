from __future__ import annotations

import os

# Must be set before importing Paddle/PaddleOCR
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_use_onednn", "0")

import argparse
import re
from pathlib import Path

import cv2
from paddleocr import PaddleOCR

DEFAULT_MIN_CONFIDENCE = 0.50

KOREAN_PLATE_PATTERN = re.compile(
    r"^(?:[가-힣]{1,2})?\d{2,3}[가-힣]\d{4}$"
)


def create_ocr() -> PaddleOCR:
    return PaddleOCR(
        lang="korean",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
    )


def normalize_plate_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^0-9가-힣]", "", text)
    return text


def is_valid_korean_plate(text: str) -> bool:
    return bool(KOREAN_PLATE_PATTERN.fullmatch(text))


def build_preprocessing_variants(
    image,
    scale: int = 3,
) -> dict:
    variants = {
        "raw": image,
    }

    # 3배 확대
    upscaled = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )

    variants["upscaled"] = upscaled

    # grayscale
    gray = cv2.cvtColor(
        upscaled,
        cv2.COLOR_BGR2GRAY,
    )

    # PaddleOCR가 3채널 이미지를 기대하므로
    # grayscale을 다시 BGR 3채널로 변환
    gray_bgr = cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2BGR,
    )

    variants["gray"] = gray_bgr

    # contrast
    contrast = cv2.equalizeHist(gray)

    # 이것도 3채널로 변환
    contrast_bgr = cv2.cvtColor(
        contrast,
        cv2.COLOR_GRAY2BGR,
    )

    variants["contrast"] = contrast_bgr

    return variants


def extract_best_result(ocr_result) -> tuple[str, float]:
    if not ocr_result:
        return "", 0.0

    result = ocr_result[0]
    texts = result.get("rec_texts", [])
    scores = result.get("rec_scores", [])

    if not texts or not scores:
        return "", 0.0

    best_index = max(
        range(len(scores)),
        key=lambda i: float(scores[i]),
    )

    text = normalize_plate_text(str(texts[best_index]))
    confidence = float(scores[best_index])

    return text, confidence


class PlateOCR:
    def __init__(
        self,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        scale: int = 3,
    ) -> None:
        self.min_confidence = min_confidence
        self.scale = scale

        print("[PlateOCR] Loading PaddleOCR...")
        self.ocr = create_ocr()
        print("[PlateOCR] Ready")

    def _predict_image(self, image) -> tuple[str, float]:
        result = self.ocr.predict(image)
        return extract_best_result(result)

    def recognize(
        self,
        image_or_path,
        save_debug_dir: Path | None = None,
    ) -> dict:
        if isinstance(image_or_path, (str, Path)):
            image_path = Path(image_or_path)
            image = cv2.imread(str(image_path))

            if image is None:
                raise FileNotFoundError(
                    f"번호판 이미지를 읽을 수 없습니다: {image_path}"
                )
        else:
            image_path = None
            image = image_or_path

            if image is None:
                raise ValueError("번호판 이미지가 None 입니다.")

        variants = build_preprocessing_variants(
            image,
            scale=self.scale,
        )

        candidates = []

        for variant_name, variant_image in variants.items():
            text, confidence = self._predict_image(variant_image)

            candidates.append(
                {
                    "variant": variant_name,
                    "text": text,
                    "confidence": confidence,
                    "valid_format": is_valid_korean_plate(text),
                }
            )

        valid_candidates = [
            c for c in candidates if c["valid_format"]
        ]

        pool = valid_candidates if valid_candidates else candidates

        if pool:
            best = max(pool, key=lambda item: item["confidence"])
        else:
            best = {
                "variant": None,
                "text": "",
                "confidence": 0.0,
                "valid_format": False,
            }

        accepted = best["confidence"] >= self.min_confidence
        final_text = best["text"] if accepted else ""

        if save_debug_dir is not None:
            save_debug_dir = Path(save_debug_dir)
            save_debug_dir.mkdir(parents=True, exist_ok=True)

            stem = image_path.stem if image_path is not None else "plate"

            for variant_name, variant_image in variants.items():
                output_path = (
                    save_debug_dir
                    / f"{stem}_{variant_name}.jpg"
                )
                cv2.imwrite(str(output_path), variant_image)

        return {
            "text": final_text,
            "raw_text": best["text"],
            "confidence": best["confidence"],
            "accepted": accepted,
            "valid_format": best["valid_format"],
            "best_variant": best["variant"],
            "candidates": candidates,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Korean license plate OCR using PaddleOCR"
    )

    parser.add_argument(
        "image",
        type=Path,
        help="번호판 crop 이미지 경로",
    )

    parser.add_argument(
        "--min-conf",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        help="최소 OCR confidence",
    )

    parser.add_argument(
        "--scale",
        type=int,
        default=3,
        help="번호판 확대 배율",
    )

    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="전처리 이미지를 저장할 debug 폴더",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    plate_ocr = PlateOCR(
        min_confidence=args.min_conf,
        scale=args.scale,
    )

    result = plate_ocr.recognize(
        args.image,
        save_debug_dir=args.debug_dir,
    )

    print()
    print("=" * 60)
    print("Plate OCR Result")
    print("=" * 60)

    print(f"Final text      : {result['text'] or 'UNKNOWN'}")
    print(f"Raw text        : {result['raw_text']}")
    print(f"Confidence      : {result['confidence']:.3f}")
    print(f"Valid format    : {result['valid_format']}")
    print(f"Accepted        : {result['accepted']}")
    print(f"Best preprocess : {result['best_variant']}")

    print()
    print("Candidates")

    for candidate in result["candidates"]:
        print(
            f"  {candidate['variant']:<10} "
            f"{candidate['text']:<15} "
            f"conf={candidate['confidence']:.3f} "
            f"valid={candidate['valid_format']}"
        )


if __name__ == "__main__":
    main()
