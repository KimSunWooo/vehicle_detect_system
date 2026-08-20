from pathlib import Path

from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_DIR
    / "runs"
    / "detect"
    / "runs"
    / "vehicle"
    / "webcam_finetune-2"
    / "weights"
    / "best.pt"
)

DATASET_YAML = (
    PROJECT_DIR
    / "vehicle_dataset"
    / "vehicle.yaml"
)


def main():

    print("========================================")
    print("YOLO Vehicle Additional Fine-Tuning")
    print("========================================")
    print(f"Model   : {MODEL_PATH}")
    print(f"Dataset : {DATASET_YAML}")
    print()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Existing best.pt not found: {MODEL_PATH}"
        )

    if not DATASET_YAML.exists():
        raise FileNotFoundError(
            f"Dataset yaml not found: {DATASET_YAML}"
        )

    # --------------------------------------------------
    # 기존 학습 모델 불러오기
    # --------------------------------------------------

    model = YOLO(
        str(MODEL_PATH)
    )

    # --------------------------------------------------
    # 추가 Fine-Tuning
    # --------------------------------------------------

    model.train(
        data=str(DATASET_YAML),

        # 기존 학습된 모델에서 추가 학습이므로
        # 처음보다 epoch를 적게 설정
        epochs=30,

        patience=12,

        imgsz=640,

        # 기존 학습 지식을 크게 훼손하지 않도록
        # learning rate 낮춤
        lr0=0.0005,

        # 이미 외부에서 augmentation을 많이 만들어놨으므로
        # YOLO 내부 augmentation은 비교적 약하게 설정
        mosaic=0.3,
        mixup=0.0,
        copy_paste=0.0,

        # 결과 저장 경로
        project=str(
            PROJECT_DIR
            / "runs"
            / "vehicle"
        ),

        name="webcam_finetune_long_distance",

        # 기존 폴더가 있더라도 자동으로
        # -2, -3 식으로 생성
        exist_ok=False,
    )


if __name__ == "__main__":
    main()