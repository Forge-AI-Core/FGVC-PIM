# A Novel Plug-in Module for Fine-grained Visual Classification

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-novel-plug-in-module-for-fine-grained-1/fine-grained-image-classification-on-cub-200)](https://paperswithcode.com/sota/fine-grained-image-classification-on-cub-200?p=a-novel-plug-in-module-for-fine-grained-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-novel-plug-in-module-for-fine-grained-1/fine-grained-image-classification-on-nabirds)](https://paperswithcode.com/sota/fine-grained-image-classification-on-nabirds?p=a-novel-plug-in-module-for-fine-grained-1)

**Paper URL**: [https://arxiv.org/abs/2202.03822](https://arxiv.org/abs/2202.03822)  
**Original Repository**: [chou141253/FGVC-PIM](https://github.com/chou141253/FGVC-PIM)

We propose a novel plug-in module that can be integrated into common backbones (CNN or Transformer) to provide strongly discriminative regions. Experimental results show significant improvements, reaching **92.77%** on CUB200-2011 and **92.83%** on NABirds.

![framework](./imgs/0001.png)


---

# 👁️ Eye-Opening Project

## 🚀 Phase 1 Objective: Sequential Transfer Learning
The primary goal of Phase 1 is to verify the generalizability and knowledge transfer capability of the PIM model across diverse fine-grained domains.

**Transfer Learning Pipeline:**
```text
ImageNet Pretraining (1,000 classes)
        ↓
CUB-200 Fine-tuning (200 classes) 
        ↓ [Checkpoint: CUB Pretrained Model]
Aircraft Fine-tuning (100 classes)
        ↓ [Checkpoint: Aircraft Pretrained Model]
Stanford Cars Fine-tuning (196 classes)
```

---

## 1. Setup & Installation

### 1.1. Install Dependencies
```zsh
# Synchronize environment using uv
uv sync

# Activate virtual environment
source .venv/bin/activate
```

> [!NOTE]
> **macOS Support**: This project is optimized for **Inference only** on macOS (Apple Silicon). Training is recommended on NVIDIA GPU environments for stability and performance.

### 1.2. Datasets & Pretrained Models
*   **CUB-200-2011**: [Download Link](https://drive.google.com/drive/folders/15brdvEQZMWW2CJVEZ70Bx28ULaGwfaIr)
*   **NA-Birds**: [Download Link](https://dl.allaboutbirds.org/nabirds)

#### 📥 Download Pretrained Weights
*   [SwinT_CUB200_Pretrained](https://drive.google.com/drive/folders/15brdvEQZMWW2CJVEZ70Bx28ULaGwfaIr)
    *   Place the weight file at `pretrained/best.pt`.

### 1.3. Data Preprocessing (FGVC-Aircraft)
For the FGVC-Aircraft dataset, the downloaded images are initially mixed in a single directory. To use the original PyTorch `ImageDataset` logic (which expects images organized into class-specific subfolders), run the provided preprocessing script:

```zsh
python preprocess/prep_aircraft.py
```
*This script will read the CSV files and automatically organize the 10,000 images into `datas/FGVC-Aircraft/train/`, `val/`, and `test/` folders based on their aircraft class. After running this, the original `fgvc-aircraft-2013b` folder and CSV files can be safely deleted to save space.*

---

## 2. 🏋️ Training
If you want to train the model from scratch:
```zsh
python main.py --c ./configs/CUB200_SwinT.yaml
```
- **Configuration**: Modify YAML files in `./configs/`.
- **Checkpointing**: Models are saved in `./records/{project_name}/{exp_name}/backup/`.
- **Mixed Precision**: Set `use_amp: True` to reduce training time (e.g., from 5h to 3h).

---

## 3. 📊 Evaluation & Inference

### Option A: Quick Accuracy Check (via `main.py`)
```zsh
python main.py --c ./configs/CUB200_SwinT_Pre.yaml
time TORCH_HOME=./.cache python main.py --c ./configs/Aircraft_SwinT.yaml
```
*   **Condition**: Set `train_root: ~` in your YAML.
*   **Output**: `eval_results.txt`.

### Option B: Detailed Analysis (via `infer.py`)
```zsh
python infer.py --c ./configs/CUB200_SwinT_Pre.yaml
```
*   **Outputs**:
    1.  `infer_results.txt`: Summary of metrics.
    2.  `infer_result.xlsx`: **Detailed per-image logs.**
    3.  `infer_cm.png`: **Confusion Matrix Heatmap.**

#### 💡 Comparison Table
| Feature | `main.py` (Quick) | `infer.py` (Detailed) |
| :--- | :--- | :--- |
| **Excel Report** | ❌ No | ✅ Yes |
| **Confusion Matrix**| ❌ No | ✅ Yes |
| **Output Name** | `eval_results.txt` | `infer_results.txt` |

---

## 4. 🔥 HeatMap Visualization
Generate Grad-CAM heatmaps to see where the model is looking.
```zsh
python heat.py --c ./configs/CUB200_SwinT_Pre.yaml --img ./vis/001.jpg --save_img ./vis/001/
```

### Examples:
| Original Image | Attention Heatmap | Blended (Mix) |
| :---: | :---: | :---: |
| ![Original](./vis/001/rbg_img.jpg) | ![Heatmap](./vis/001/heatmap.jpg) | ![Mixed](./vis/001/mix.jpg) |

---

## 5. 📈 Training Output & Metrics

After training finishes, the following files are saved to `./records/{project_name}/{exp_name}/`:

| File | Description |
| :--- | :--- |
| `backup/best.pt` | Best checkpoint (saved based on **combiner-top-1** accuracy) |
| `backup/last.pt` | Latest epoch checkpoint |
| `eval_results.txt` | Evaluation summary (accuracy, precision, recall, F1) |
| `train_metrics.png` | Per-epoch train ACC / Precision / Recall / F1-Score graph |
| `eval_metrics.png` | Per-epoch eval ACC / Precision / Recall / F1-Score graph |

### Terminal Log Format
```
Train | ACC: 85.123% (85.123%) | Precision: 84.900% | Recall: 85.100% | F1-Score: 85.000%
Eval  | ACC: 88.456% (88.456%) | Precision: 88.200% | Recall: 88.400% | F1-Score: 88.300%
```
- **ACC (left)**: max(current epoch ACC, historical best ACC)
- **ACC (right, in parentheses)**: current epoch combiner-top-1 accuracy
- All metrics: macro-averaged via sklearn

---

## 🔧 Modifications from Original

The following changes have been made from the [original repository](https://github.com/chou141253/FGVC-PIM):

| Area | Change |
| :--- | :--- |
| **best.pt criterion** | Changed from `highest-5` (average of all layer outputs) to `combiner-top-1` (single forward pass — the actual deployable metric) |
| **Eval metrics** | Added macro Precision / Recall / F1-Score via `sklearn` (combiner output) |
| **Train metrics** | Added per-epoch macro Precision / Recall / F1-Score accumulated across batches |
| **Metric graphs** | `save_metrics_plots()` saves `train_metrics.png` and `eval_metrics.png` at end of training |
| **eval.py — `select_` / `drop_` 제거** | `evaluate()` 내부에서 `select_*` / `drop_*` 출력에 대한 top-k 정확도 계산 블록 삭제. 원래 S=2048 토큰 기준으로 배치당 32,768개 샘플을 처리하던 병목이었음. |
| **eval.py — `_average_top_k_result` 제거** | `evaluate()` 내부에서 `_average_top_k_result` 호출 제거 (Python 루프 병목). `highest-1` ~ `highest-5` 지표는 더 이상 계산하지 않음. 터미널 출력 및 그래프에서도 `Highest-5 ACC` 항목 제거. |
| **Multi-device support** | `get_device()` util added — supports CUDA, Apple MPS, and CPU automatically |
| **FGVC-Aircraft config** | `configs/Aircraft_SwinT.yaml` added for 100-class aircraft fine-tuning |

---

## 🛠️ Custom Model Support

You can integrate the PIM module into your own custom backbones:

- **Model Builder**: Refer to [`models/builder.py`](./models/builder.py) to see how different backbones (Swin-T, ResNet, etc.) are registered and constructed.
- **Tutorial**: For a step-by-step guide on building your own PIM-enabled model, check out the [**`how_to_build_pim_model.ipynb`**](./how_to_build_pim_model.ipynb) notebook.

---

