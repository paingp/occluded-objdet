# Occluded Object Detection with PyTorch
> Benchmarking and improving object detection under occlusion using Faster-RCNN, DETR, and a custom attention-augmented architecture — evaluated on the Dhaka Occluded Objects Dataset.

---

## What is Occlusion?
Occlusion refers to the scenario when an object of interest is partially hidden from our view. The image below shows an example where a bus is occluded by a trash can.
![Example Image of Occlusion](/images/occlusion_example.jpg "(Occlusion Example)")

## Dataset
**Dhaka Occluded Objects Dataset**
- Images: 5,080
- Classes: 8
- Source: [https://www.kaggle.com/datasets/tanzimmostafa14/dhaka-occluded-objects-dataset](#)

## Models

### 1. Faster-RCNN (Fine-tuned)
- Pretrained on COCO, fine-tuned on the Dhaka Occluded Objects Dataset
- Two-stage detector: Region Proposal Network (RPN) + classification head
- Strong baseline for occluded detection benchmarking

### 2. DETR (Fine-tuned)
- Transformer-based end-to-end detector (no NMS or anchors)
- Fine-tuned on the Dhaka Occluded Objects Dataset
- Evaluated for architectural trade-offs vs. CNN-based approaches under occlusion

### 3. Custom Attention-Augmented Faster-RCNN *(BotDet)*
- Based on Faster-RCNN architecture
- Multihead Self-Attention mechanisms integrated into the CNN backbone
- Designed to improve feature discrimination in occluded regions

---

## Results
Side-by-side comparison of sample output from Faster-RCNN and BotDet
![Sample Output](/images/sample_output.jpg "(Sample Output)")

All models evaluated using **COCO metrics** on the Dhaka Occluded Objects Dataset test split.

| Model | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|
| Faster-RCNN (fine-tuned) | **0.776** | **0.489** |
| DETR (fine-tuned) | 0.539 | *0.269* | 
| BotDet | *0.218* | *0.109* |

## Key Findings

- **Faster-RCNN** achieved the strongest overall mAP@0.5 (0.776), demonstrating that two-stage detectors remain highly competitive on occluded scenes when fine-tuned on domain-specific data.
- **DETR** (mAP@0.5: 0.539) underperformed relative to Faster-RCNN, likely due to its sensitivity to small and partially visible objects and the limited dataset size.
- **BotDet** did not perform as well as expected, likely needs more/better data for pretraining as we pretrained on custom image classification dataset instead of ImageNet due to limited compute resources available.