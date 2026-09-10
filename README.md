# SRRA: Stable-Rank-Based Residual Adaptation for Generalizable Deepfake Detection

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-brightgreen.svg)](https://creativecommons.org/licenses/by-nc/4.0/) ![PyTorch](https://img.shields.io/badge/PyTorch-1.12-brightgreen) ![Python](https://img.shields.io/badge/Python-3.8+-brightgreen)

**ECCV 2026**

## 🎯 Overview

Official implementation of **SRRA: Stable-Rank-Based Residual Adaptation for Generalizable Deepfake Detection**.

SRRA is a parameter-efficient fine-tuning strategy for generalizable deepfake detection. It decomposes pretrained ViT attention weights into a frozen principal component and a trainable residual component, and adaptively determines the residual rank using the stable rank of each weight matrix. Two regularizers, **Residual Energy Constraint (REC)** and **Residual
Subspace Orthogonalization (RSO)**, are further introduced to suppress excessive perturbation and improve cross-domain generalization.

<p align="center">
  <img src="SRRA_overview.pdf" width="90%">
</p>

## Results

The model is trained on **FaceForensics++ (c23)** and evaluated on unseen datasets.

### Frame-level AUC

| Method | CDF | DFDC | DFDCP | DFD | Avg. |
|---|---:|---:|---:|---:|---:|
| SRRA | 0.911 | 0.856 | 0.890 | 0.947 | **0.901** |

### Video-level AUC

| Method | CDF | DFDC | DFDCP | DFD | Avg. |
|---|---:|---:|---:|---:|---:|
| SRRA | 0.968 | 0.889 | 0.923 | 0.980 | **0.940** |

## Repository Structure

```text
SRRA/
├── config/
│   ├── detector/
│   │   └── SRRA.yaml
│   └── train_config.yaml
├── dataset/
├── detectors/
│   └── SRRA.py
├── trainer/
├── train.py
├── test.py
└── demo.py
```

## Installation

```bash
git clone https://github.com/LHK-CodeLab/SRRA.git
cd SRRA

conda create -n srra python=3.10 -y
conda activate srra
```

Install PyTorch according to your CUDA version, then install the required packages:

```bash
pip install torchvision transformers opencv-python scikit-learn pyyaml tqdm pillow scipy matplotlib tensorboard loralib
```

## Pretrained CLIP

SRRA uses **OpenAI CLIP ViT-L/14** as the pretrained backbone.

Please download `openai/clip-vit-large-patch14` and replace the placeholder CLIP path in:

```text
detectors/SRRA.py
demo.py
```

For example:

```python
clip_model = CLIPModel.from_pretrained('/path/to/clip-vit-large-patch14')
```

## Dataset Preparation

The experiments follow the DeepfakeBench preprocessing protocol.

Default setting:

- Training set: `FaceForensics++` with `c23` compression
- Test sets: `Celeb-DF-v2`, `DFDC`, `DFDCP`, and `DeepFakeDetection`
- Input resolution: `224 x 224`
- Frames per video: `32`

Set your local dataset paths in:

```text
config/train_config.yaml
```

especially:

```yaml
dataset_root_rgb: '/path/to/rgb_data'
dataset_json_folder: '/path/to/dataset_json'
```

## Model Weights

Pretrained SRRA weights are available from Baidu Netdisk:

**Download:** https://pan.baidu.com/s/1m9Zk6aFuFmYCmpwN1ua4lg?pwd=md6y  
**Password:** `md6y`

After downloading, place the checkpoint at a convenient location, for example:

```text
weights/SRRA.pth
```

## Demo

Before running the demo, set the following paths in `demo.py`:

```python
image_path = '/path/to/image.png'
weights_path = '/path/to/SRRA.pth'
```

and set the local CLIP path.

Then run:

```bash
python demo.py
```

Example output:

```text
Real probability: 0.08
Fake probability: 0.92
Prediction: Fake
```

## Notes

Several local paths in the released scripts/configuration files are placeholders (`/`). Please replace them with paths in your own environment before training or evaluation.

## Citation

SRRA has been accepted by **ECCV 2026**. The paper is already listed in the ECCV 2026 proceedings table of contents. The official chapter-level DOI/BibTeX was not publicly indexed at the time this README was prepared, so this section will be updated once the final Springer metadata becomes available.

## Acknowledgements

This repository follows the data preprocessing and evaluation protocol of **DeepfakeBench**. We thank the authors of DeepfakeBench and related open-source deepfake detection projects for their contributions to the community.
