from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent


# --------------------------------------------------
# Input / output
# --------------------------------------------------

INPUT_DIR = PROJECT_DIR / "input"
OUTPUT_DIR = PROJECT_DIR / "output"

VEHICLE_CROP_DIR = OUTPUT_DIR / "vehicle_crops"
PLATE_CROP_DIR = OUTPUT_DIR / "plate_crops"
ANNOTATED_DIR = OUTPUT_DIR / "annotated"


# --------------------------------------------------
# Models
# --------------------------------------------------

MODEL_DIR = PROJECT_DIR / "models" / "runtime"

VEHICLE_MODEL = MODEL_DIR / "vehicle_detector_add_enterance.pt"
VEHICLE_CLASSIFIER_MODEL = MODEL_DIR / "vehicle_classifier.pt"
PLATE_MODEL = MODEL_DIR / "plate_detector.pt"


# --------------------------------------------------
# Detection settings
# --------------------------------------------------

# COCO vehicle classes
# car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASSES = [0]

VEHICLE_CONF = 0.25
PLATE_CONF = 0.25


# --------------------------------------------------
# Classification / OCR settings
# --------------------------------------------------

VEHICLE_CLASSIFIER_CONF = 0.50
PLATE_OCR_CONF = 0.50


# --------------------------------------------------
# Supported image extensions
# --------------------------------------------------

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# --------------------------------------------------
# Directory creation
# --------------------------------------------------

def ensure_directories() -> None:
    for directory in (
        INPUT_DIR,
        OUTPUT_DIR,
        VEHICLE_CROP_DIR,
        PLATE_CROP_DIR,
        ANNOTATED_DIR,
        MODEL_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )
