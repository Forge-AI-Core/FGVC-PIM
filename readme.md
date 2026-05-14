# 세밀한 시각 분류를 위한 플러그인 모듈 (FGVC-PIM)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-novel-plug-in-module-for-fine-grained-1/fine-grained-image-classification-on-cub-200)](https://paperswithcode.com/sota/fine-grained-image-classification-on-cub-200?p=a-novel-plug-in-module-for-fine-grained-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-novel-plug-in-module-for-fine-grained-1/fine-grained-image-classification-on-nabirds)](https://paperswithcode.com/sota/fine-grained-image-classification-on-nabirds?p=a-novel-plug-in-module-for-fine-grained-1)

**논문**: [https://arxiv.org/abs/2202.03822](https://arxiv.org/abs/2202.03822)  
**원본 저장소**: [chou141253/FGVC-PIM](https://github.com/chou141253/FGVC-PIM)

CNN 또는 Transformer 백본에 통합 가능한 플러그인 모듈로, 판별력 높은 영역을 강조하여 분류 성능을 향상시킵니다. CUB200-2011에서 **92.77%**, NABirds에서 **92.83%** 달성.

![framework](./imgs/0001.png)

---

# 👁️ 프로젝트 목표

## 🚀 Phase 1: 순차적 전이학습 (Sequential Transfer Learning)
PIM 모델의 일반화 능력과 지식 전이 능력을 다양한 세밀 분류 도메인에서 검증합니다.

**전이학습 파이프라인:**
```text
ImageNet 사전학습 (1,000 클래스)
        ↓
CUB-200 파인튜닝 (200 클래스)
        ↓ [체크포인트: CUB 사전학습 모델]
Aircraft 파인튜닝 (100 클래스)
        ↓ [체크포인트: Aircraft 사전학습 모델]
Stanford Cars 파인튜닝 (196 클래스)
```

---

## 1. 환경 설정

### 1.1. 의존성 설치
```zsh
# uv로 환경 동기화
uv sync

# 가상환경 활성화
source .venv/bin/activate
```

> [!NOTE]
> **macOS 지원**: 이 프로젝트는 macOS(Apple Silicon)에서 **추론 전용**으로 최적화되어 있습니다. 학습은 안정성과 성능을 위해 NVIDIA GPU 환경을 권장합니다.

### 1.2. 데이터셋 및 사전학습 가중치

**데이터셋 다운로드:**
- **CUB-200-2011**: [다운로드](https://drive.google.com/drive/folders/15brdvEQZMWW2CJVEZ70Bx28ULaGwfaIr)
- **FGVC-Aircraft**: [다운로드](https://drive.google.com/drive/folders/1iKTP2H-Tb8sanzqHcEKPhYe3YtfcjbN6)
- **Stanford Cars**: [다운로드](https://drive.google.com/drive/folders/1fnB_L1fnx3kTugqt2DkqdeX06K2XZCIu)

**사전학습 가중치 위치:**
- CUB-200-2011: `pretrained/cub200/best.pt`
- FGVC-Aircraft: `pretrained/aircraft/best.pt`
- Stanford Cars: `pretrained/cars/best.pt`

### 1.3. 데이터 전처리 - FGVC-Aircraft
원본 FGVC-Aircraft 데이터셋은 10,000장의 이미지가 하나의 폴더에 섞여 있습니다. PyTorch의 `ImageFolder` 구조(클래스별 하위 폴더)로 변환하려면:

1. 압축 해제 후 `./datas/fgvc-aircraft-2013b/` 위치에 폴더가 있어야 합니다.
2. 전처리 스크립트 실행:
```zsh
python preprocess/prep_aircraft.py
```
*스크립트 실행 후 이미지가 `./datas/FGVC-Aircraft/train/`, `val/`, `test/` 폴더로 자동 정리됩니다. 완료 후 원본 `fgvc-aircraft-2013b` 폴더는 삭제해도 됩니다.*

### 1.4. 데이터 전처리 - Stanford Cars
Stanford Cars 데이터셋은 MATLAB `.mat` 파일로 어노테이션이 제공됩니다.

1. `./datas/Stanford-Cars/` 경로에 `cars_train/`, `cars_test/`, `devkit/` 폴더가 있어야 합니다.
2. 전처리 스크립트 실행:
```zsh
python preprocess/prep_cars.py
```
*스크립트 실행 후 이미지가 `./datas/Stanford-Cars/train/`, `test/` 폴더로 정리됩니다.*

---

## 2. 🏋️ 학습

```zsh
# CUB-200-2011 학습
time TORCH_HOME=/workspace/projects/FGVC-PIM/.cache python main.py --c ./configs/CUB200_SwinT.yaml

# FGVC-Aircraft 학습
time TORCH_HOME=/workspace/projects/FGVC-PIM/.cache python main.py --c ./configs/Aircraft_SwinT.yaml

# Stanford Cars 학습
time TORCH_HOME=/workspace/projects/FGVC-PIM/.cache python main.py --c ./configs/Cars_SwinT.yaml
```

- **설정 변경**: `./configs/` 폴더의 YAML 파일 수정
- **체크포인트 저장 위치**: `./records/{project_name}/{exp_name}/backup/`
- **혼합 정밀도(AMP)**: `use_amp: True` 설정 시 학습 시간 단축 (예: 5시간 → 3시간)

---

## 3. 📊 평가 및 추론

### 방법 A: 빠른 정확도 확인 (`main.py`)
```zsh
# CUB-200-2011
python main.py --c ./configs/CUB200_SwinT_Pre.yaml

# FGVC-Aircraft
python main.py --c ./configs/Aircraft_SwinT_Pre.yaml

# Stanford Cars
python main.py --c ./configs/Cars_SwinT_Pre.yaml
```
- **조건**: YAML에서 `train_root: ~` 설정
- **출력**: `eval_results.txt`

### 방법 B: 상세 분석 (`infer.py`)
```zsh
# CUB-200-2011
python infer.py --c ./configs/CUB200_SwinT_Pre.yaml

# FGVC-Aircraft
python infer.py --c ./configs/Aircraft_SwinT_Pre.yaml

# Stanford Cars
python infer.py --c ./configs/Cars_SwinT_Pre.yaml
```
- **출력 파일:**
  1. `infer_results.txt`: 지표 요약
  2. `infer_result.xlsx`: 이미지별 상세 예측 결과
  3. `infer_cm.png`: 혼동 행렬(Confusion Matrix) 시각화

#### 💡 비교표
| 기능 | `main.py` (빠른 확인) | `infer.py` (상세 분석) |
| :--- | :--- | :--- |
| **Excel 리포트** | ❌ 없음 | ✅ 있음 |
| **혼동 행렬** | ❌ 없음 | ✅ 있음 |
| **출력 파일명** | `eval_results.txt` | `infer_results.txt` |

---

## 4. 🔥 히트맵 시각화
Grad-CAM 히트맵으로 모델이 어느 영역을 보는지 확인할 수 있습니다.
```zsh
# CUB-200-2011
python heat.py --c ./configs/CUB200_SwinT_Pre.yaml --img ./vis/001.jpg --save_img ./vis/001/

# FGVC-Aircraft
python heat.py --c ./configs/Aircraft_SwinT_Pre.yaml --img ./vis/aircraft_sample.jpg --save_img ./vis/aircraft_out/

# Stanford Cars
python heat.py --c ./configs/Cars_SwinT_Pre.yaml --img ./vis/car_sample.jpg --save_img ./vis/car_out/
```

| 원본 이미지 | Attention 히트맵 | 합성 이미지 |
| :---: | :---: | :---: |
| ![Original](./vis/001/rbg_img.jpg) | ![Heatmap](./vis/001/heatmap.jpg) | ![Mixed](./vis/001/mix.jpg) |

---

## 5. 📈 학습 출력 및 지표

학습 완료 후 `./records/{project_name}/{exp_name}/`에 저장되는 파일:

| 파일 | 설명 |
| :--- | :--- |
| `backup/best.pt` | 최고 성능 체크포인트 (**combiner-top-1** 기준으로 저장) |
| `backup/last.pt` | 마지막 에포크 체크포인트 |
| `eval_results.txt` | 평가 결과 요약 (정확도, Precision, Recall, F1) |
| `train_metrics.png` | 에포크별 Train ACC / Precision / Recall / F1 그래프 |
| `eval_metrics.png` | 에포크별 Eval ACC / Precision / Recall / F1 그래프 |

### 터미널 출력 형식
```
Train | ACC: 85.123% | Precision: 84.900% | Recall: 85.100% | F1-Score: 85.000%
Eval  | ACC: 88.456% (88.456%) | Precision: 88.200% | Recall: 88.400% | F1-Score: 88.300%
```
- **Eval ACC 왼쪽**: 지금까지 최고 Eval 정확도 (best.pt 저장 기준)
- **Eval ACC 괄호 안**: 현재 에포크 combiner-top-1 정확도
- 모든 지표: sklearn macro 평균

---

## 🍏 macOS(Apple Silicon)에서 학습

PyTorch MPS 백엔드를 통해 macOS에서도 학습이 가능합니다.

### macOS 주요 제약사항 및 해결책
- **멀티프로세싱 제한**: macOS는 프로세스 생성에 제한이 있어 데드락 발생 가능 → YAML에서 `num_workers: 0` 설정
- **통합 메모리 최적화**: `batch_size: 4`, `update_freq: 8`로 설정하여 실질적 배치 크기 32 유지
- **절전 모드 방지**: 장시간 학습 중 시스템 절전으로 중단될 수 있음 → `caffeinate -ids` 명령어 사용

### macOS 실행 명령어
```zsh
caffeinate -ids time python main.py --c ./configs/Aircraft_SwinT_Mac.yaml
```

---

## 🔧 원본 대비 주요 변경사항

[원본 저장소](https://github.com/chou141253/FGVC-PIM)에서 변경된 내용:

| 항목 | 변경 내용 |
| :--- | :--- |
| **best.pt 저장 기준** | `highest-5`(전체 레이어 앙상블)에서 `combiner-top-1`(실제 배포 지표)으로 변경 |
| **평가 지표 추가** | sklearn을 통한 macro Precision / Recall / F1-Score 추가 (combiner 기준) |
| **학습 지표 추가** | 에포크별 macro Precision / Recall / F1-Score 누적 계산 추가 |
| **지표 그래프 저장** | 학습 완료 시 `train_metrics.png`, `eval_metrics.png` 자동 저장 |
| **eval.py — `select_` / `drop_` 제거** | `evaluate()` 내부에서 `select_*` / `drop_*` 출력에 대한 top-k 정확도 계산 블록 삭제. S=2048 토큰 기준으로 배치당 32,768개 샘플을 처리하던 병목이었음. |
| **eval.py — `_average_top_k_result` 제거** | `evaluate()` 내부에서 `_average_top_k_result` 호출 제거 (Python 루프 병목). `highest-1` ~ `highest-5` 지표는 더 이상 계산하지 않음. 터미널 출력 및 그래프에서도 `Highest-5 ACC` 항목 제거. |
| **다중 디바이스 지원** | `get_device()` 유틸 추가 — CUDA, Apple MPS, CPU 자동 감지 |
| **Aircraft 설정 파일** | `configs/Aircraft_SwinT.yaml` 추가 (100클래스 항공기 파인튜닝용) |

---

## 🛠️ 커스텀 모델 지원

PIM 모듈을 원하는 백본에 직접 통합할 수 있습니다:

- **모델 빌더**: [`models/builder.py`](./models/builder.py) 참고 (Swin-T, ResNet 등 백본 등록 방법)
- **튜토리얼**: [**`how_to_build_pim_model.ipynb`**](./how_to_build_pim_model.ipynb) 노트북에서 단계별 가이드 확인

---
