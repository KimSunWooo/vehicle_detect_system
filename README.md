# Camera Recognition System

차량 이미지를 입력받아 **차량 검출 → 차종 분류 → 번호판 검출 → OCR**을 수행하는 차량 인식 파이프라인입니다.

최종적으로 이미지에서 차량의 **세부 차종과 차량번호**를 추출하는 것을 목표로 합니다.

## Pipeline

```text
Input Image
     │
     ▼
Vehicle Detection
(YOLO26n)
     │
     ▼
Vehicle Crop
     │
     ├───────────────┐
     ▼               ▼
Vehicle          License Plate
Classification     Detection
(YOLO11n-cls)     (lp_det.pt)
     │               │
     │               ▼
     │            Plate Crop
     │               │
     │               ▼
     │            PaddleOCR
     │               │
     └───────┬───────┘
             ▼
        Final Result
     차종 + 차량번호
```

## 주요 기능

### 1. 차량 검출

Ultralytics YOLO26n을 이용하여 이미지에서 차량을 검출합니다.

COCO 기준 다음 차량 클래스를 대상으로 합니다.

```text
car
motorcycle
bus
truck
```

검출된 차량은 별도의 `vehicle crop` 이미지로 생성됩니다.

### 2. 세부 차종 분류

검출된 차량 이미지를 YOLO11n Classification 모델에 입력하여 세부 차종을 판별합니다.

예:

```text
Hyundai_Avante
Hyundai_Ioniq 5
Kia_EV6
Jeep_Renegade
BMW_5 Series
...
```

차종 분류 데이터는 Car-1000 계열 데이터를 기반으로 구성했습니다.

초기 데이터셋은 다음과 같습니다.

| Dataset | Images |
|---|---:|
| Train | 36,182 |
| Validation | 4,418 |
| Test | 4,745 |

초기 245개 클래스로 학습했으며 Test 성능은:

```text
Top-1 Accuracy : 91.7%
Top-5 Accuracy : 99.3%
```

였습니다.

기존 데이터에 `Hyundai Ioniq 5`가 존재하지 않아 별도로 약 160장의 이미지를 추가하여 246개 클래스로 재학습했습니다.

재학습 모델의 Test 성능:

```text
Top-1 Accuracy : 92.2%
Top-5 Accuracy : 99.3%
```

PHEV, AMG, GTE, Recharge 등 동일 기본 차량의 파생 모델은 프로젝트 목적상 하나의 차량 모델로 통합했습니다.

### 3. 번호판 검출

차량 전체 이미지가 아닌 **Vehicle Crop 내부에서 번호판을 다시 검출**합니다.

사용 모델:

```text
EasyKoreanLpDetector
lp_det.pt
```

번호판 검출 결과는 `plate crop`으로 저장하고 원본 이미지 좌표로 변환하여 Bounding Box도 표시합니다.

### 4. 번호판 OCR

검출된 번호판 이미지에 PaddleOCR을 적용하여 실제 차량번호를 추출합니다.

OCR 안정성을 높이기 위해 다음 전처리 결과를 비교합니다.

```text
raw
upscaled
gray
contrast
```

각 결과의 confidence와 한국 번호판 형식 유효성을 비교하여 최종 결과를 선택합니다.

테스트 예:

```text
Final text      : 245우9315
Confidence      : 0.998
Valid format    : True
Best preprocess : contrast
```

## 전체 결과

파이프라인 실행 후 차량별로 다음 정보를 얻을 수 있습니다.

```json
{
  "vehicle_type": "car",
  "classification": {
    "model": "Hyundai_Ioniq 5",
    "confidence": 0.94
  },
  "plates": [
    {
      "ocr": {
        "text": "245우9315",
        "confidence": 0.998
      }
    }
  ]
}
```

결과는 `output/results.json`에 저장됩니다.

## 실행

`input/` 디렉터리에 테스트할 이미지를 넣고:

```bash
python pipeline/detect_pipeline.py
```

또는 특정 이미지만 처리할 수 있습니다.

```bash
python pipeline/detect_pipeline.py \
  --source input/test_image.jpg
```

결과:

```text
output/
├── vehicle_crops/
├── plate_crops/
├── annotated/
└── results.json
```

## Project Structure

```text
camera_recognize_system/
├── config.py
├── detector/
│   ├── vehicle_detector.py
│   └── plate_detector.py
├── ocr/
│   └── plate_ocr.py
├── pipeline/
│   └── detect_pipeline.py
├── models/
│   └── runtime/
│       ├── vehicle_detector.pt
│       ├── vehicle_classifier.pt
│       └── plate_detector.pt
├── input/
├── output/
├── requirements.txt
└── README.md
```

학습용 데이터와 학습 과정에서 생성된 파일은 `tools/`에 별도로 관리하며 운영 파이프라인에서는 사용하지 않습니다.

## 개발 과정에서 해결한 주요 문제

차종 데이터에는 AMG, PHEV, GTE 등 동일 차량의 파생 클래스가 분리되어 있어 기본 차량 모델을 기준으로 클래스를 통합했습니다.

또한 기존 데이터셋에 Ioniq 5가 존재하지 않아 다른 전기차로 오분류되는 문제가 있었고, Ioniq 5 데이터를 별도로 추가하여 재학습했습니다.

OCR에서는 PaddleOCR 실행 환경 문제와 grayscale 이미지의 채널 문제를 해결했으며, 여러 이미지 전처리 결과 중 가장 신뢰도가 높은 OCR 결과를 선택하도록 개선했습니다.

## 현재 한계

현재 **번호판 검출 및 OCR은 정상적으로 전체 파이프라인에 연결되어 있지만, 세부 차종 분류는 추가 개선이 필요합니다.**

Car-1000 Test에서는 Top-1 약 92.2%의 성능을 보이지만 실제 촬영 환경에서는 학습 데이터와 촬영 조건의 차이로 오분류가 발생할 수 있습니다.

특히 다음과 같은 경우 정확도가 떨어질 수 있습니다.

- 학습 이미지가 부족한 차종
- 학습 데이터에 존재하지 않는 차량
- 신형 차량
- 유사한 디자인의 차량
- 야간/역광 환경
- 특이한 촬영 각도
- 모형 차량

실제로 추가 학습 이후 전체 Test 정확도는 향상되었지만 기존 모델에서 정상 인식되던 일부 외부 이미지가 오분류되는 현상도 확인했습니다.

따라서 현재 단계에서는 **차량번호를 주요 식별 정보로 사용하고 세부 차종은 보조 정보로 사용하는 것이 적절합니다.**

## 향후 개선 방향

향후에는 실제 운영 카메라 환경에서 촬영한 차량 데이터를 확보하여 차종 분류 모델을 추가 학습하고, 클래스별 데이터 불균형을 개선할 예정입니다.

또한 학습되지 않은 차량을 기존 클래스 중 하나로 강제 분류하지 않도록 confidence 기반 `Unknown` 처리도 추가할 필요가 있습니다.

최종적으로는 다음 구조로 확장하는 것을 목표로 합니다.

```text
External Camera
      ↓
Image Capture
      ↓
Recognition API
      ↓
Vehicle Detection
      ↓
Vehicle Classification
      ↓
Plate Detection
      ↓
OCR
      ↓
차종 + 차량번호
      ↓
Database
```

이를 통해 외부 카메라에서 전달받은 이미지를 자동으로 분석하고 차량번호와 차종 정보를 DB에 저장하거나 조회할 수 있는 형태로 확장할 예정입니다.

## Model Attribution

License plate detection model:

```text
gyupro/EasyKoreanLpDetector
lp_det.pt
YOLOv5 based
```

실제 배포 및 모델 재배포 전 upstream repository의 라이선스와 사용 조건을 확인해야 합니다.
