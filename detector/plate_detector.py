from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


@dataclass
class PlateDetection:
    bbox: tuple[int, int, int, int]
    confidence: float
    crop: np.ndarray


class PlateDetector:
    """EasyKoreanLpDetector's YOLOv5 license-plate model wrapper."""

    def __init__(self, model_path: str | Path, conf: float = 0.25) -> None:
        self.model_path = Path(model_path)
        self.conf = conf

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Plate model not found: {self.model_path}\n"
                "Run: python scripts/download_plate_model.py"
            )

        self.model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path=str(self.model_path),
            trust_repo=True,
        )
        self.model.conf = conf

    def detect(self, image_bgr: np.ndarray) -> list[PlateDetection]:
        # Original EasyKoreanLpDetector feeds PIL images into the YOLOv5 model.
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.model(Image.fromarray(image_rgb))
        rows = results.xyxy[0]

        detections: list[PlateDetection] = []
        h, w = image_bgr.shape[:2]

        for row in rows:
            values = row.detach().cpu().tolist()
            x1, y1, x2, y2 = [int(v) for v in values[:4]]
            confidence = float(values[4])

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = image_bgr[y1:y2, x1:x2].copy()
            detections.append(
                PlateDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    crop=crop,
                )
            )

        return detections
