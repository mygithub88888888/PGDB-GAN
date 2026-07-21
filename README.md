# PGDB-GAN: Physics-Guided Dual-Branch GAN for Low-Light Facial Enhancement

**Official PyTorch Implementation**

[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.3+-green)](https://developer.nvidia.com/cuda-toolkit)

**Authors:** Xianglong Yan, Jin Tao
**Affiliation:** School of Artificial Intelligence, Gansu University of Political Science and Law, Lanzhou, China
**Journal:** Applied Soft Computing (Elsevier)
**Paper:** [DOI pending]

---

## Abstract

We propose **PGDB-GAN**, a Physics-Guided Dual-Branch Generative Adversarial Network that establishes a dynamic synergy between physical illumination modeling and adversarial feature learning for low-light facial image enhancement. The framework integrates Retinex-based illumination decomposition as real-time spatial guidance for a dual-branch GAN, a face-perceptual distillation module with Gabor filter banks for multi-scale texture preservation, and structured pruning for efficient edge deployment (1.32M parameters, 3 ms inference at 720P).

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/mygithub88888888/PGDB-GAN.git
cd PGDB-GAN

# Install dependencies
pip install -r requirements.txt

# Run inference with a pre-trained checkpoint
python scripts/test.py \
    --data_path_test_low ./data/test_images/ \
    --model_test ./weights/LOL.pt \
    --save ./test_results/
```

---

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.10+ with CUDA 11.3+
- 16 GB GPU memory recommended (tested on NVIDIA RTX 4060 Ti)

```bash
pip install -r requirements.txt
```

### Hardware

| Component | Specification |
|:---|:---|
| GPU | NVIDIA RTX 4060 Ti (16 GB VRAM) |
| CPU | Intel Core i7-10700K @ 3.80 GHz |
| RAM | 16 GB |
| OS | Windows 11 / Ubuntu 20.04+ |
| CUDA | 11.3+ |
| cuDNN | 8.2+ |

---

## Repository Structure

```
PGDB-GAN/
├── src/                          # Core source code
│   ├── model.py                  # Base model: LD-Net, IE-Net, RD-Net, Finetunemodel
│   ├── model_gan.py              # GAN model: Generator (Enhancer), PatchGAN Discriminator
│   ├── model111.py               # Extended GAN model variant
│   ├── loss.py                   # Composite loss: enhancement, reconstruction, color, smoothness, TV
│   ├── loss_gan.py               # GAN loss: adversarial, perceptual, Gabor texture alignment
│   ├── gan_losses.py             # Additional GAN loss components
│   ├── distillation.py           # Face-aware knowledge distillation (Stage 2)
│   ├── pruning.py                # Gabor-driven structured pruning (Stage 3)
│   ├── generate_masks.py         # Face mask generation from bounding box annotations
│   ├── data_filter.py            # Data filtering based on face region ratio
│   ├── dataset.py                # Data loader for paired/unpaired image datasets
│   ├── dataset_gan.py            # GAN data loader with mask support
│   ├── gan_dataset.py            # Alternative GAN dataset loader
│   ├── gan_config.py             # GAN configuration
│   ├── gan_train.py              # GAN training loop
│   ├── utils.py                  # Utilities: Gabor filters, visualization, checkpoint management
│   └── utils_gan.py              # GAN utilities
├── scripts/                      # Training and evaluation scripts
│   ├── train_stage1.py           # Stage 1: Physical foundation + joint GAN training
│   ├── train_gan.py              # Alternative GAN training entry point
│   ├── test.py                   # Model inference and evaluation
│   └── test_gan.py               # GAN model testing
├── configs/                      # Configuration files
│   └── default.yaml              # Default training configuration
├── data/                         # Data directory (populated by user)
│   ├── data_choose/              # Pre-filtered training data
│   └── data_choose_denoise/      # Pre-filtered denoising data
├── weights/                      # Pre-trained model checkpoints
│   ├── LOL.pt                    # Pre-trained on LOL dataset
│   ├── L-Nikon.pt                # Pre-trained on DarkFace (Nikon subset)
│   ├── LSRW.pt                   # Pre-trained on DarkFace (LSRW subset)
│   ├── weights_3000.pt           # Stage 1 checkpoint (3,000 epochs)
│   └── weights_100000.pt         # Stage 2 checkpoint (100,000 iterations)
├── figures/                      # Paper figures
├── Visual comparison chart group/ # Qualitative comparison visualizations
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── LICENSE                       # MIT License
```

---

## Data Preparation

### Datasets

PGDB-GAN is evaluated on three benchmarks. For exact reproducibility, the data splits used in our experiments are specified below.

#### LOL (LOw-Light) Dataset

- **Source:** [Wei et al., BMVC 2018](https://daooshee.github.io/BMVC2018website/)
- **Resolution:** 600 x 400 pixels
- **Train/Test Split:** 485 training pairs / 15 testing pairs (standard public split)
- **Format:** Paired low-light / normal-light images in separate directories

```
data/LOL/
├── train/
│   ├── low/          # 485 low-light training images
│   └── high/         # 485 normal-light training images
└── test/
    ├── low/          # 15 low-light testing images
    └── high/         # 15 normal-light testing images (ground truth)
```

#### MIT-Adobe FiveK Dataset

- **Source:** [Bychkovsky et al., CVPR 2011](https://data.csail.mit.edu/graphics/fivek/)
- **Resolution:** Resized to 256 x 256 pixels for training and evaluation
- **Train/Test Split:** 4,500 training pairs / 500 testing pairs (Expert-C subset)
- **Format:** Paired RAW / Expert-C retouched images

```
data/FiveK/
├── train/
│   ├── input/        # 4,500 RAW training images
│   └── expert_c/     # 4,500 Expert-C retouched images
└── test/
    ├── input/        # 500 RAW testing images
    └── expert_c/     # 500 Expert-C retouched images
```

#### DarkFace Dataset

- **Source:** [Yang et al., TPAMI 2022](https://flyywh.github.io/CVPRW2019LowLight/)
- **Resolution:** Evaluated at 600 x 400 pixels
- **Split:** Subject-disjoint --- no individual appears in both training and testing sets
- **Format:** Unpaired low-light facial images with bounding box annotations

```
data/DarkFace/
├── train/
│   ├── image/        # Low-light training images
│   └── label/        # Bounding box annotations
└── test/
    ├── image/        # Low-light testing images
    └── label/        # Bounding box annotations
```

### Preprocessing

1. **Face Mask Generation:** Run `src/generate_masks.py` to construct binary face masks from bounding box annotations (used for DarkFace).
2. **Data Filtering:** Run `src/data_filter.py` to filter images based on face region ratio (IoU > 0.5 deduplication).
3. **Augmentation:** Training-time only --- random horizontal flipping and random resizing within [0.8, 1.2] scale range.
4. **Normalization:** All input images normalized to [0, 1]. No additional preprocessing during testing.

---

## Pre-trained Models

We provide pre-trained checkpoints for immediate inference. All weights are the structurally pruned student models (1.32M parameters).

| Checkpoint | Dataset | Size |
|:---|:---|:---|
| [`weights/LOL.pt`](weights/LOL.pt) | LOL (low-light paired) | 358 KB |
| [`weights/L-Nikon.pt`](weights/L-Nikon.pt) | DarkFace (Nikon subset) | 358 KB |
| [`weights/LSRW.pt`](weights/LSRW.pt) | DarkFace (LSRW subset) | 358 KB |

**Intermediate checkpoints** (for reproducing the full training pipeline):

| Checkpoint | Stage | Size | Description |
|:---|:---|:---|:---|
| [`weights/weights_3000.pt`](weights/weights_3000.pt) | Stage 1 | 358 KB | Physical foundation initialization (3,000 epochs) |
| [`weights/weights_100000.pt`](weights/weights_100000.pt) | Stage 2 | 3.01 MB | Joint GAN training (100,000 iterations) |

### Inference with Pre-trained Weights

```python
import torch
from src.model import Finetunemodel

# Load pre-trained checkpoint
model = Finetunemodel('weights/LOL.pt')
model = model.cuda()
model.eval()

# Run inference
with torch.no_grad():
    enhanced, denoised = model(input_tensor)  # input_tensor shape: [1, 3, H, W]
```

### Command-line Inference

```bash
# Inference on LOL test set
python scripts/test.py \
    --data_path_test_low ./data/LOL/test/low/ \
    --model_test ./weights/LOL.pt \
    --save ./test_results/LOL/

# Inference on DarkFace
python scripts/test.py \
    --data_path_test_low ./data/DarkFace/test/image/ \
    --model_test ./weights/L-Nikon.pt \
    --save ./test_results/DarkFace/
```

---

## Training

The model is trained in a three-stage protocol. All stages use a **fixed random seed of 2** for PyTorch, NumPy, and Python's built-in random module.

### Reproducibility Settings

| Component | Setting |
|:---|:---|
| PyTorch seed | `torch.manual_seed(2)` |
| CUDA seed | `torch.cuda.manual_seed(2)` |
| NumPy seed | `np.random.seed(2)` |
| Python random | `random.seed(2)` |
| cuDNN | `benchmark=True`, `deterministic=False` |

### Stage 1: Physical Foundation + Joint GAN Training

Trains the full PGDB-GAN pipeline end-to-end: LD-Net (denoising), IE-Net (illumination), RD-Net (reflectance), and the dual-branch GAN.

```bash
python scripts/train_stage1.py \
    --batch_size 16 \
    --epochs 3000 \
    --lr 0.0002 \
    --seed 2 \
    --gpu 0 \
    --save ./train_results/LOL_results
```

**Key hyperparameters:**

| Parameter | Value |
|:---|:---|
| Batch size | 16 |
| Epochs | 3,000 |
| Optimizer | Adam (beta1=0.5, beta2=0.999) |
| Learning rate | 2e-4 |
| Training patch size | 256 x 256 (random crop) |
| Random seed | 2 |

### Stage 2: Face-Aware Knowledge Distillation

Transfers texture-critical features from the teacher network to a lightweight student using Gabor-weighted feature alignment.

```bash
python scripts/train_gan.py \
    --batch_size 16 \
    --epochs 5001 \
    --lr 0.0001 \
    --seed 2 \
    --gpu 0 \
    --model_pretrain ./weights/weights_3000.pt \
    --mask_dir ./data/DarkFace/train/mask/ \
    --save ./train_results/GAN_results
```

**Key hyperparameters:**

| Parameter | Value |
|:---|:---|
| Iterations | 100,000 |
| Learning rate | 1e-4 |
| lambda_distill | 2.0 |
| lambda_tex (texture preservation) | 0.5 |
| Gabor orientations | 6 |
| Gabor frequency bands | 2 |

### Stage 3: Gabor-Driven Structured Pruning

Identifies and removes redundant channels based on Gabor activation sensitivity scores.

```bash
python src/pruning.py \
    --model_path ./weights/weights_100000.pt \
    --pruning_ratio 0.25 \
    --output_path ./weights/pgdb_gan_pruned.pt
```

**Result:** 5.08M to 1.32M parameters (74% reduction), 0.21 dB PSNR trade-off on LOL.

---

## Evaluation

### Image Quality Metrics

All quantitative metrics are computed using a unified evaluation script:

```bash
python scripts/test.py \
    --data_path_test_low <path_to_test_images> \
    --model_test <path_to_checkpoint> \
    --save <output_directory>
```

**Metrics computed:** PSNR, SSIM, LPIPS, FID, NIQE, NSR, MSE

**Metric computation protocols (DarkFace --- unpaired):**

| Metric | Protocol |
|:---|:---|
| LPIPS | AlexNet-based, enhanced output vs. original low-light input |
| FID | Inception-v3 (pool3), images resized to 299x299, reference = original low-light test set |
| NSR | Wavelet-domain MAD estimator (Daubechies-4, sigma = MAD/0.6745), NSR = (sigma_input - sigma_enhanced) / sigma_input x 100% |
| NIQE | Standard implementation with default parameters |

For paired datasets (LOL, MIT-Adobe FiveK), all reference-based metrics use the corresponding ground-truth normal-light images.

### Downstream Face Detection

```bash
# MTCNN face detection benchmark at multiple scales
python scripts/test.py \
    --data_path_test_low ./data/DarkFace/test/image/ \
    --model_test ./weights/L-Nikon.pt \
    --detection_scale 1.0 2.0 4.0 \
    --save ./test_results/detection/
```

---

## Training Configuration Summary

### Loss Function Weights

| Loss Term | Symbol | Weight | Purpose |
|:---|:---|:---|:---|
| Illumination consistency | lambda_light | 1.0 | Global brightness matching |
| Content reconstruction | lambda_content | 10.0 | L1 pixel-level fidelity |
| Face-aware distillation | lambda_distill | 2.0 | Gabor-weighted teacher-student alignment |
| Texture preservation | lambda_tex | 0.5 | Post-pruning texture retention |
| Adversarial | lambda_adv | 1.0 | Perceptual realism via PatchGAN |

### Physical Prior Constants

| Constant | Value | Description |
|:---|:---|:---|
| alpha | 0.5 | Brightness scaling coefficient (adaptive) |
| beta | 0.8--1.2 | Per-pixel adaptive adjustment ratio |
| gamma | 0.9 | Contrast adjustment coefficient |
| E | 0.7 | Empirical intensity threshold |

---

## Efficiency

| Metric | Value | Platform |
|:---|---:|:---|
| Parameters (pruned) | **1.320 M** | PyTorch |
| FLOPs | 74.200 G | PyTorch |
| Inference time (720P) | **3 ms** (0.003 s) | NVIDIA RTX 4060 Ti |
| Model size (pruned) | 1.32 MB | --- |
| Compression ratio | 74% (5.08M to 1.32M) | --- |

> **Note on cross-platform comparison:** Runtime values reported in the full manuscript comparison table are measured in each method's original implementation framework (PyTorch, TensorFlow, Theano, MATLAB). Direct cross-platform runtime comparisons should be interpreted with caution. The framework-agnostic metrics (parameter count and FLOPs) provide a more reliable basis for cross-method efficiency assessment.

---

## License

This project is released under the [MIT License](LICENSE).

## Citation

If you find this work useful in your research, please cite:

```bibtex
@article{yan2025pgdbgan,
  title     = {{PGDB-GAN}: A Dynamic Enhancement Method for Low-Light Facial Features
               through Synergy of Physical Illumination and Adversarial Learning},
  author    = {Yan, Xianglong and Tao, Jin},
  journal   = {Applied Soft Computing},
  year      = {2025},
  note      = {Under review}
}
```

## Contact

For questions or collaborations, please contact:
- Xianglong Yan: 17393335628@163.com
- Jin Tao: 359071039@qq.com
