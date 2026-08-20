from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO


@dataclass
class VehicleDetection:
    bbox: tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str
    crop: np.ndarray


class VehicleDetector:
    def __init__(
        self,
        model_path: str | Path,
        conf: float = 0.25,
        classes: list[int] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.conf = conf
        self.classes = [0] if classes is None else classes

        # If a local YOLO weight is absent, use its filename so Ultralytics can
        # download an official pretrained model automatically.
        model_source = str(self.model_path if self.model_path.exists() else self.model_path.name)
        self.model = YOLO(model_source)

    def detect(self, image_bgr: np.ndarray) -> list[VehicleDetection]:
        result = self.model.predict(
            source=image_bgr,
            conf=self.conf,
            classes=self.classes,
            verbose=False,
            
        )[0]

        detections: list[VehicleDetection] = []
        h, w = image_bgr.shape[:2]

        for box in result.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            class_name = str(self.model.names.get(class_id, class_id))
            crop = image_bgr[y1:y2, x1:x2].copy()

            detections.append(
                VehicleDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                    crop=crop,
                )
            )

        return detections
