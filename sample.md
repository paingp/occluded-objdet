# 🔍 Occluded Object Detection with PyTorch

> Benchmarking and improving object detection under occlusion using Faster-RCNN, DETR, and a custom attention-augmented architecture — evaluated on the Dhaka Occluded Objects Dataset.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Models](#models)
- [Results](#results)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Training](#training)
- [Evaluation](#evaluation)
- [Key Findings](#key-findings)
- [Future Work](#future-work)
- [References](#references)

---

## Overview

This project investigates the challenge of **object detection under occlusion** — a common failure case for modern detectors where objects are partially or fully blocked by other objects. Three architectures are studied:

1. **Faster-RCNN** (fine-tuned baseline)
2. **DETR** (transformer-based baseline)
3. **Custom Attention-Augmented Faster-RCNN** (proposed model)

The proposed model incorporates **attention mechanisms into the CNN backbone** of Faster-RCNN to improve robustness against partial occlusion. All models are trained and evaluated on the [Dhaka Occluded Objects Dataset](#dataset) using standard COCO metrics.

---

## Models

### 1. Faster-RCNN (Fine-tuned)
- Pretrained on COCO, fine-tuned on the Dhaka Occluded Objects Dataset
- Two-stage detector: Region Proposal Network (RPN) + classification head
- Strong baseline for occluded detection benchmarking

### 2. DETR (Fine-tuned)
- Transformer-based end-to-end detector (no NMS or anchors)
- Fine-tuned on the Dhaka Occluded Objects Dataset
- Evaluated for architectural trade-offs vs. CNN-based approaches under occlusion

### 3. Custom Attention-Augmented Faster-RCNN *(Proposed)*
- Based on Faster-RCNN architecture
- Attention mechanisms (e.g., CBAM / Self-Attention — *replace with what you used*) integrated into the CNN backbone
- Designed to improve feature discrimination in occluded regions

---

## Results

All models evaluated using **COCO metrics** on the Dhaka Occluded Objects Dataset test split.

| Model | mAP@0.5 | mAP@0.5:0.95 | Notes |
|---|---|---|---|
| Faster-RCNN (fine-tuned) | **0.776** | *add value* | Strong two-stage baseline |
| DETR (fine-tuned) | 0.539 | *add value* | Transformer; slower convergence |
| Custom Attn-Faster-RCNN | *add value* | *add value* | Proposed; attention in backbone |

> 📌 *Fill in additional metrics (mAP@0.5:0.95, per-class AP, inference speed) as available.*

---

## Dataset

**Dhaka Occluded Objects Dataset**
- *Add a brief description of the dataset: number of images, classes, occlusion types, etc.*
- Source: [Link to dataset or paper](#)
- Splits used: Train / Val / Test — *add split sizes*

```
data/
├── train/
│   ├── images/
│   └── annotations/
├── val/
│   ├── images/
│   └── annotations/
└── test/
    ├── images/
    └── annotations/
```

---

## Project Structure

```
occluded-object-detection/
├── configs/                  # Model and training configs
├── data/                     # Dataset (not tracked by git)
├── models/
│   ├── faster_rcnn.py        # Fine-tuned Faster-RCNN
│   ├── detr.py               # Fine-tuned DETR
│   └── attention_rcnn.py     # Custom attention-augmented model
├── train.py                  # Training script
├── evaluate.py               # COCO evaluation script
├── utils/
│   ├── transforms.py
│   └── metrics.py
├── notebooks/                # Exploratory analysis & result visualizations
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.x
- CUDA (recommended)

### Installation

```bash
git clone https://github.com/your-username/occluded-object-detection.git
cd occluded-object-detection
pip install -r requirements.txt
```

### Download Dataset

```bash
# Add instructions for downloading/preparing the dataset
# e.g., gdown, wget, or manual steps
```

---

## Training

```bash
# Fine-tune Faster-RCNN
python train.py --model faster_rcnn --config configs/faster_rcnn.yaml

# Fine-tune DETR
python train.py --model detr --config configs/detr.yaml

# Train custom attention model
python train.py --model attention_rcnn --config configs/attention_rcnn.yaml
```

> *Update commands to match your actual CLI arguments.*

---

## Evaluation

```bash
python evaluate.py --model faster_rcnn --weights checkpoints/faster_rcnn_best.pth
```

Metrics reported:
- **mAP@0.5** and **mAP@0.5:0.95** (COCO standard)
- Per-class Average Precision
- *Optionally: inference time, FLOPs*

---

## Key Findings

- **Faster-RCNN** achieved the strongest overall mAP@0.5 (0.776), demonstrating that two-stage detectors remain highly competitive on occluded scenes when fine-tuned on domain-specific data.
- **DETR** (mAP@0.5: 0.539) underperformed relative to Faster-RCNN, likely due to its sensitivity to small and partially visible objects and the limited dataset size.
- **Attention mechanisms** incorporated into the backbone improved feature representation in occluded regions — *add your specific observations here*.
- *Add 1–2 more findings specific to your experiments.*

---

## Future Work

- [ ] Extend evaluation to additional occluded object benchmarks (e.g., MS-COCO partially occluded subset)
- [ ] Explore deformable attention and deformable convolutions for better spatial adaptability
- [ ] Investigate data augmentation strategies that simulate occlusion during training
- [ ] Compare inference speed vs. accuracy trade-offs across all three architectures

---

## References

- Ren et al. (2015). [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
- Carion et al. (2020). [End-to-End Object Detection with Transformers (DETR)](https://arxiv.org/abs/2005.12872)
- Woo et al. (2018). [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521) *(if applicable)*
- *Add the Dhaka Occluded Objects Dataset citation here*

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
