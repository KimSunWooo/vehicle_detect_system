from pathlib import Path
import time

from ocr_adapter.ocr_adapter import ocr_handler


# --------------------------------------------------
# Config
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
CAPTURE_DIR = ROOT_DIR / "capture"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    if not CAPTURE_DIR.exists():
        raise FileNotFoundError(
            f"Capture directory not found: {CAPTURE_DIR}"
        )

    # capture 폴더 내 이미지 전체 검색
    image_paths = sorted(
        [
            path
            for path in CAPTURE_DIR.iterdir()
            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            )
        ]
    )

    if not image_paths:
        print(
            f"No images found in: {CAPTURE_DIR}"
        )
        return

    print()
    print("========================================")
    print("       OCR Adapter Batch Test")
    print("========================================")
    print(f"Directory : {CAPTURE_DIR}")
    print(f"Images    : {len(image_paths)}")
    print("========================================")
    print()

    success_count = 0
    fail_count = 0
    vehicle_count = 0

    # --------------------------------------------------
    # 전체 이미지 테스트
    # --------------------------------------------------

    for image_index, image_path in enumerate(
        image_paths,
        start=1,
    ):

        print()
        print(
            f"[{image_index}/{len(image_paths)}] "
            f"{image_path.name}"
        )

        try:

            image_data = image_path.read_bytes()

            start = time.perf_counter()

            result = ocr_handler(
                "entry",
                image_data,
            )

            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000

            print(
                f"Processing Time: "
                f"{elapsed_ms:.1f} ms"
            )

            # 차량 없음
            if not result:

                print("Vehicle: NOT DETECTED")

                success_count += 1
                continue

            # 차량 있음
            print(
                f"Detected Vehicles: {len(result)}"
            )

            for vehicle in result:

                vehicle_count += 1

                print(
                    f"  Vehicle #{vehicle['index']}"
                )

                print(
                    "    Model        :",
                    vehicle[
                        "classification"
                    ]["model"],
                )

                print(
                    "    OCR          :",
                    vehicle["ocr_data"],
                )

                print(
                    "    Fallback OCR :",
                    vehicle["fallback_ocr"],
                )

            success_count += 1

        except Exception as exc:

            fail_count += 1

            print(
                "ERROR:",
                f"{type(exc).__name__}: {exc}",
            )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print()
    print()
    print("========================================")
    print("              SUMMARY")
    print("========================================")

    print(
        f"Total Images     : {len(image_paths)}"
    )

    print(
        f"Success          : {success_count}"
    )

    print(
        f"Failed           : {fail_count}"
    )

    print(
        f"Detected Vehicles: {vehicle_count}"
    )

    print("========================================")


if __name__ == "__main__":
    main()