# A Novel Plug-in Module for Fine-grained Visual Classification



**Paper URL**: [https://arxiv.org/abs/2202.03822](https://arxiv.org/abs/2202.03822)  
**Original Repository**: [chou141253/FGVC-PIM](https://github.com/chou141253/FGVC-PIM)

We propose a novel plug-in module that can be integrated into common backbones (CNN or Transformer) to provide strongly discriminative regions. Experimental results show significant improvements, reaching **92.77%** on CUB200-2011 and **92.83%** on NABirds.

![framework](./imgs/0001.png)


---

# 👁️ Eye-Opening Project

## 🚀 Objective: Domain-Specific Fine-Grained Recognition
The primary goal of this project is to verify the generalizability and feature extraction capabilities of the PIM model across diverse fine-grained domains by independently fine-tuning from a flagship pre-trained backbone.


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


---

## 2. 🏋️ Training

#### Training
If you want to train the models from scratch:
```zsh
# Bogonet Training (ConvNeXt)
time TORCH_HOME=/workspace/projects/FGVC-PIM/.cache python main.py --c ./configs/Bogonet_ConvNV1.yaml
```
- **Configuration**: Modify YAML files in `./configs/`.
- **Checkpointing**: Models are saved in `./records/{project_name}/{exp_name}/backup/`.
- **Mixed Precision**: Set `use_amp: True` to reduce training time (e.g., from 5h to 3h).

---



## 3. 📊 Evaluation & Inference

### Option A: Quick Accuracy Check (via `main.py`)
```zsh

# Bogonet (ConvNeXt-L)
time TORCH_HOME=/workspace/projects/FGVC-PIM/.cache python main.py --c ./configs/Bogonet_ConvNV1_Pre.yaml
```
*   **Condition**: Set `train_root: ~` in your YAML. All datasets have a dedicated `_Pre.yaml` config (e.g., `CUB200_ConvNV1_Pre.yaml`) which has `train_root` set to `~` and `pretrained` set to the respective pre-trained model path.
*   **Output**: `eval_results.txt`.

### Option B: Detailed Analysis (via `infer.py`)
```zsh
# Bogonet (ConvNeXt-L) Detailed Scoring & Excel/Confusion Matrix
python infer.py --c ./configs/Bogonet_ConvNV1_Pre.yaml
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
python heat.py --c ./configs/Public-Benchmark/CUB200_SwinT_Pre.yaml --img ./vis/001.jpg --save_img ./vis/001/

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
