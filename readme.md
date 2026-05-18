# A Novel Plug-in Module for Fine-grained Visual Classification

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-novel-plug-in-module-for-fine-grained-1/fine-grained-image-classification-on-cub-200)](https://paperswithcode.com/sota/fine-grained-image-classification-on-cub-200?p=a-novel-plug-in-module-for-fine-grained-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-novel-plug-in-module-for-fine-grained-1/fine-grained-image-classification-on-nabirds)](https://paperswithcode.com/sota/fine-grained-image-classification-on-nabirds?p=a-novel-plug-in-module-for-fine-grained-1)

**Paper URL**: [https://arxiv.org/abs/2202.03822](https://arxiv.org/abs/2202.03822)  
**Original Repository**: [chou141253/FGVC-PIM](https://github.com/chou141253/FGVC-PIM)

We propose a novel plug-in module that can be integrated into common backbones (CNN or Transformer) to provide strongly discriminative regions. Experimental results show significant improvements, reaching **92.77%** on CUB200-2011 and **92.83%** on NABirds.

![framework](./imgs/0001.png)


---

# 👁️ Eye-Opening Project

## 🚀 Objective: Domain-Specific Fine-Grained Recognition
The primary goal of this project is to verify the generalizability and feature extraction capabilities of the PIM model across diverse fine-grained domains by independently fine-tuning from a flagship pre-trained backbone.

**Independent Fine-Tuning Pipeline:**
```text
          ImageNet-22K Pretraining (Swin-Large Backbone)
            ↙                   ↓                   ↘
    CUB-200-2011          FGVC-Aircraft          Stanford Cars
   (200 classes)          (100 classes)          (196 classes)
         ↓                      ↓                      ↓
   [best.pt eval]         [best.pt eval]         [best.pt eval]
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

**Datasets:**
*   **CUB-200-2011**: [Download Link](https://drive.google.com/drive/folders/15brdvEQZMWW2CJVEZ70Bx28ULaGwfaIr)
*   **FGVC-Aircraft**: [Download Link](https://drive.google.com/drive/folders/1iKTP2H-Tb8sanzqHcEKPhYe3YtfcjbN6)
*   **Stanford Cars**: [Download Link](https://drive.google.com/drive/folders/1fnB_L1fnx3kTugqt2DkqdeX06K2XZCIu)

#### 📥 Download Pretrained Weights
*   **CUB-200-2011 Pretrained**: Place the weight file at `pretrained/cub200/best.pt`.
*   **FGVC-Aircraft Pretrained**: Place the weight file at `pretrained/aircraft/best.pt`.
*   **Stanford Cars Pretrained**: Place the weight file at `pretrained/cars/best.pt`.

### 1.3. Data Preprocessing (FGVC-Aircraft)
Unlike CUB-200, the raw FGVC-Aircraft dataset packages all 10,000 images into a single flat folder alongside CSV annotation files. To make it compatible with PyTorch's standard `ImageFolder` structure (class-specific subfolders), follow these steps:

1. Extract the downloaded archive so that the folder resides at `./datas/fgvc-aircraft-2013b/`.
2. Run the automated parsing script:
```zsh
python preprocess/prep_aircraft.py
```
*This script automatically splits and organizes the images into `./datas/FGVC-Aircraft/train/`, `val/`, and `test/` subdirectories based on their 100 variant classes. Once completed successfully, you can safely delete the raw `fgvc-aircraft-2013b` folder to free up storage space.*

### 1.4. Data Preprocessing (Stanford Cars)
The raw Stanford Cars dataset distributes images in flat directories (`cars_train` and `cars_test`) and requires parsing MATLAB `.mat` structure arrays to map images to their 196 specific model variants. To properly format the directories:

1. Ensure the dataset folder resides at `./datas/Stanford-Cars/` containing `cars_train/`, `cars_test/`, and `devkit/`.
2. Run the automated matrix parsing and folder formatting script:
```zsh
python preprocess/prep_cars.py
```
*This script copies the corrected test annotations containing ground truth labels, extracts bounding box metadata, and seamlessly arranges images into class subdirectories (`001` to `196`) under `./datas/Stanford-Cars/train/` and `test/`.*

---

## 2. 🏋️ Training

#### Training on Swin-T v1 backbone from scratch
If you want to train the models from scratch:
```zsh
# CUB-200-2011 Training
time TORCH_HOME=/workspace/projects/FGVC-PIM/.cache python main.py --c ./configs/CUB200_SwinT.yaml

# FGVC-Aircraft Training
time TORCH_HOME=/workspace/projects/FGVC-PIM/.cache python main.py --c ./configs/Aircraft_SwinT.yaml

# Stanford Cars Training
time TORCH_HOME=/workspace/projects/FGVC-PIM/.cache python main.py --c ./configs/Cars_SwinT.yaml
```
- **Configuration**: Modify YAML files in `./configs/`.
- **Checkpointing**: Models are saved in `./records/{project_name}/{exp_name}/backup/`.
- **Mixed Precision**: Set `use_amp: True` to reduce training time (e.g., from 5h to 3h).

---

#### Training on ConvNeXt-Tiny backbone from scratch
```zsh
# CUB-200-2011 Training
time TORCH_HOME=/workspace/projects/FGVC-PIM/.cache python main.py --c ./configs/CUB200_ConvNV1.yaml
```


## 3. 📊 Evaluation & Inference

### Option A: Quick Accuracy Check (via `main.py`)
```zsh
# CUB-200-2011 (Swin-T)
time TORCH_HOME=/workspace/projects/FGVC-PIM/.cache python main.py --c ./configs/CUB200_SwinT_Pre.yaml

# CUB-200-2011 (ConvNeXt-Tiny)
time TORCH_HOME=/workspace/projects/FGVC-PIM/.cache python main.py --c ./configs/CUB200_ConvNV1.yaml

# FGVC-Aircraft
time TORCH_HOME=/workspace/projects/FGVC-PIM/.cache python main.py --c ./configs/Aircraft_SwinT_Pre.yaml

# Stanford Cars
time TORCH_HOME=/workspace/projects/FGVC-PIM/.cache python main.py --c ./configs/Cars_SwinT_Pre.yaml
```
*   **Condition**: Set `train_root: ~` in your YAML. For configs without a dedicated `_Pre.yaml` (such as `CUB200_ConvNV1.yaml`), you must manually set `train_root: ~` and point `pretrained` to your checkpoint (e.g., `./records/CUB200-ConvNV1/ConvNV1-001/backup/best.pt`).
*   **Output**: `eval_results.txt`.

### Option B: Detailed Analysis (via `infer.py`)
```zsh
# CUB-200-2011 (Swin-T) Detailed Scoring & Excel/Confusion Matrix
python infer.py --c ./configs/CUB200_SwinT_Pre.yaml

# CUB-200-2011 (ConvNeXt-Tiny) Detailed Scoring & Excel/Confusion Matrix
python infer.py --c ./configs/CUB200_ConvNV1.yaml

# FGVC-Aircraft Detailed Scoring & Excel/Confusion Matrix
python infer.py --c ./configs/Aircraft_SwinT_Pre.yaml

# Stanford Cars Detailed Scoring & Excel/Confusion Matrix
python infer.py --c ./configs/Cars_SwinT_Pre.yaml
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
# CUB-200-2011 (Swin-T) Heatmap
python heat.py --c ./configs/CUB200_SwinT_Pre.yaml --img ./vis/001.jpg --save_img ./vis/001/

# CUB-200-2011 (ConvNeXt-Tiny) Heatmap
python heat.py --c ./configs/CUB200_ConvNV1.yaml --img ./vis/001.jpg --save_img ./vis/001/

# FGVC-Aircraft Heatmap
python heat.py --c ./configs/Aircraft_SwinT_Pre.yaml --img ./vis/aircraft_sample.jpg --save_img ./vis/aircraft_out/

# Stanford Cars Heatmap
python heat.py --c ./configs/Cars_SwinT_Pre.yaml --img ./vis/car_sample.jpg --save_img ./vis/car_out/
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
Train | ACC: 85.123% | Precision: 84.900% | Recall: 85.100% | F1-Score: 85.000%
Eval  | ACC: 88.456% (88.456%) | Precision: 88.200% | Recall: 88.400% | F1-Score: 88.300%
```
- **Eval ACC (left)**: best eval accuracy so far (basis for saving best.pt)
- **Eval ACC (right, in parentheses)**: current epoch combiner-top-1 accuracy
- All metrics: macro-averaged via sklearn

---

## 🍏 Training on macOS (Apple Silicon)

Training fine-grained models natively on macOS is fully supported using PyTorch's MPS (Metal Performance Shaders) backend. To ensure stability and avoid common system limits during hours-long training runs, use the tailored configuration file and command setup:

### Key macOS Constraints & Config Solutions
- **Multiprocessing Bottleneck**: macOS limits process spawning (`fork`), causing deadlocks with standard data loading. Set `num_workers: 0` in your YAML config.
- **Unified Memory Optimization**: Scale down the physical batch size to fit safely inside Mac Unified Memory (e.g., `batch_size: 4`), while increasing `update_freq: 8` (gradient accumulation) to preserve an effective batch size of 32.
- **Preventing Sleep Disruptions**: Long runs on Mac can be unexpectedly paused by system sleep or screensavers. Prefix the execution command with `caffeinate -ids`.

### macOS Execution Command
```zsh
caffeinate -ids time python main.py --c ./configs/Aircraft_SwinT_Mac.yaml
```

---

## 🔧 Modifications from Original

The following changes have been made from the [original repository](https://github.com/chou141253/FGVC-PIM):

| Area | Change |
| :--- | :--- |
| **best.pt criterion** | Changed from `highest-5` (average of all layer outputs) to `combiner-top-1` (single forward pass — the actual deployable metric) |
| **Eval metrics** | Added macro Precision / Recall / F1-Score via `sklearn` (combiner output) |
| **Train metrics** | Added per-epoch macro Precision / Recall / F1-Score accumulated across batches |
| **Metric graphs** | `save_metrics_plots()` saves `train_metrics.png` and `eval_metrics.png` at end of training |
| **eval.py — `select_` / `drop_` removed** | Removed top-k accuracy computation for `select_*` / `drop_*` outputs inside `evaluate()`. These were a major bottleneck — processing up to 32,768 samples per batch (S=2048 tokens). |
| **eval.py — `_average_top_k_result` removed** | Removed `_average_top_k_result` call from `evaluate()` (Python loop bottleneck). `highest-1` ~ `highest-5` metrics are no longer computed. Removed from terminal output and graphs as well. |
| **Multi-device support** | Added `get_device()` util — auto-detects CUDA, Apple MPS, and CPU |
| **FGVC-Aircraft config** | Added `configs/Aircraft_SwinT.yaml` for 100-class aircraft fine-tuning |
| **ConvNeXt-Tiny config** | Added `configs/CUB200_ConvNV1.yaml` for CUB-200-2011 ConvNeXt-Tiny fine-tuning |

---

## 🛠️ Custom Model Support

You can integrate the PIM module into your own custom backbones:

- **Model Builder**: Refer to [`models/builder.py`](./models/builder.py) to see how different backbones (Swin-T, ResNet, etc.) are registered and constructed.
- **Tutorial**: For a step-by-step guide on building your own PIM-enabled model, check out the [**`how_to_build_pim_model.ipynb`**](./how_to_build_pim_model.ipynb) notebook.

---
