# COMP0248 Coursework 1 — Hand Gesture Detection, Segmentation & Classification

**Student ID:** 25077339  
**Module:** COMP0248 — Computer Vision and Imaging  

---

## Overview

A deep learning pipeline for hand gesture recognition from RGB-D images. Given a frame from an Intel RealSense camera, the system:

1. **Segments** the hand — predicts a pixel-wise binary mask
2. **Detects** the hand — derives a bounding box from the segmentation mask
3. **Classifies** the gesture — predicts one of 10 classes from the cropped hand region

The pipeline is split into two stages trained separately:
- **Stage 1 — Segmenter:** `RGBDSegmenter` — encoder-decoder CNN taking 4-channel RGB-D input, outputs a binary mask
- **Stage 2 — Classifier:** `RGBMROICropClassifier` or `MROICropClassifier` — CNN classifier operating on a cropped 224×224 region around the detected hand

**10 gesture classes:** call, dislike, like, ok, one, palm, peace, rock, stop, three

---

## Results

Evaluated on a held-out test set of 3,450 frames:

| Pipeline | Seg IoU | Seg Dice | BBox IoU | BBox acc@0.5 | Top-1 Acc | Macro F1 |
|---|---|---|---|---|---|---|
| RGBD seg + RGBM classifier | 0.833 | 0.884 | 0.854 | 0.941 | 0.790 | 0.792 |
| RGBD seg + Mask classifier | 0.833 | 0.884 | 0.854 | 0.941 | **0.833** | **0.835** |
| RGB-only seg + RGBM classifier | 0.782 | 0.836 | 0.800 | 0.875 | 0.738 | 0.739 |
| RGB-only seg + Mask classifier | 0.782 | 0.836 | 0.800 | 0.875 | 0.755 | 0.756 |

The best overall pipeline uses the **RGBD segmenter + mask-only classifier**, achieving 83.3% top-1 accuracy and 0.835 macro F1.

---

## Project Structure

```
project_25077339_houghton/
│
├── src/
│   ├── dataloader.py                                    # GestureDataset — loads RGB-D frames, masks, bboxes
│   ├── model_rgbd_segmenter.py                          # RGBDSegmenter + SegmentationLoss
│   ├── model_rgbm_roi_crop_classifier.py                # RGBMROICropClassifier (RGB + mask crop)
│   ├── model_m_roi_crop_classifier.py                   # MROICropClassifier (mask-only crop)
│   ├── train_overnight.py                               # Main training script — all 4 models, LR sweep
│   ├── train_segmenter_rgb_only.py                      # Single-run segmenter training (RGB-only)
│   ├── train_classifier_masks_only_ROI_cropping_from_dirs.py     # Classifier training on pre-cropped masks
│   ├── train_classifier_masks_only_ROI_cropping_from_dirs_aug.py # Same with heavy augmentation
│   ├── evaluate.py                                      # Segmentation evaluation script
│   ├── boundingboxes.py                                 # Bbox analysis — before/after LCC post-processing
│   ├── visualise.py                                     # Visualise segmenter outputs on test images
│   ├── inference.py                                     # Inference wrapper class
│   └── utils.py                                         # Shared helpers (IoU, Dice, AverageMeter, etc.)
│
├── weights/
│   └── overnight/
│       ├── best_rgbd_seg.pt     # Best RGBD segmenter from overnight sweep
│       ├── best_rgb_seg.pt      # Best RGB-only segmenter
│       ├── best_rgbm_clf.pt     # Best RGBM ROI-crop classifier
│       └── best_m_clf.pt        # Best mask-only ROI-crop classifier
│
├── results/                     # Evaluation logs and confusion matrices
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone / navigate to the project

```bash
cd project_25077339_houghton
```

### 2. Create and activate a conda environment

```bash
conda create -n comp0248 python=3.10 -y
conda activate comp0248
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU support, install PyTorch with the correct CUDA version for your machine. On the lab servers (CUDA 12.x):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. Place your dataset

The dataset should be structured as follows:

```
dataset/
  STUDENT_ID/
    G01_call/
      CLIP_ID/
        rgb/         ← .png frames
        depth_raw/   ← .npy depth arrays
        annotation/  ← binary mask .png files
    G02_dislike/
    ...
```

By default the scripts look for `dataset/` in the project root. You can pass a different path with `--data`.

---

## Training

### Full overnight sweep (recommended)

Trains all 4 model weights in sequence with a learning rate sweep. Expects a GPU and ~6–8 hours.

```bash
conda activate comp0248
python src/train_overnight.py --data dataset --out weights/overnight
```

Resume from existing segmenter checkpoints to skip straight to the classifier phases:

```bash
python src/train_overnight.py --data dataset \
    --skip_seg \
    --rgbd_seg_ckpt weights/overnight/best_rgbd_seg.pt \
    --rgb_seg_ckpt  weights/overnight/best_rgb_seg.pt
```

Quick smoke test (1 epoch per phase — just checks the pipeline runs):

```bash
python src/train_overnight.py --data dataset --smoke_test
```

### Train segmenter only (single run)

```bash
python src/train_segmenter_rgb_only.py \
    --data dataset \
    --epochs 30 \
    --lr 3e-4 \
    --run my_segmenter
# saves to weights/my_segmenter_best.pt
```

---

## Evaluation

### Segmentation metrics (IoU, Dice)

```bash
python src/evaluate.py \
    --checkpoint weights/overnight/best_rgbd_seg.pt \
    --data test_dataset \
    --out results/my_eval
```

Results are saved to `results/my_eval/eval_metrics.json`.

### Bounding box analysis (before/after LCC post-processing)

```bash
python src/boundingboxes.py \
    --checkpoint weights/overnight/best_rgbd_seg.pt \
    --data test_dataset \
    --out results/bbox_analysis \
    --vis_samples 8
```

Saves `bbox_eval_metrics.json` and visualisation images under `results/bbox_analysis/vis/`.

---

## Visualisation

Edit the parameters at the top of `src/visualise.py` (dataset path, weights path, number of samples) then run:

```bash
python src/visualise.py
```

Output images are saved to `visualise_outputs/`. Each image shows:
- **Red overlay** — predicted segmentation mask
- **Blue box** — ground-truth bounding box
- **Green box** — predicted bounding box (from thresholded mask)
- **Text label** — ground-truth gesture class

---

## Model Architecture

### RGBDSegmenter

Encoder-decoder segmentation network.

- **Input:** `(B, 4, H, W)` — 3 RGB channels + 1 normalised depth channel
- **Encoder:** 5 Conv+BN+ReLU+MaxPool blocks → `(B, 512, H/32, W/32)`
- **Decoder:** 4 ConvTranspose2d blocks → `(B, 32, H/2, W/2)`
- **Head:** 1×1 conv + sigmoid → `(B, 1, H/2, W/2)` mask probabilities
- **Loss:** Soft Dice + Soft IoU (combined)

For RGB-only inference (no depth sensor), pass zeros for the depth channel — the architecture is unchanged.

### RGBMROICropClassifier / MROICropClassifier

Both are 4-block CNN classifiers operating on 224×224 crops.

- `RGBMROICropClassifier`: input is `(B, 4, 224, 224)` — RGB + binary mask channel
- `MROICropClassifier`: input is `(B, 1, 224, 224)` — binary mask only

Crop pipeline: segmenter → LCC post-processing → tight bbox → 1.2× square crop → 224×224 resize → classifier.

---

## Post-processing

After thresholding the segmentation mask at 0.5, a **largest connected component (LCC)** filter is applied:
- All connected foreground blobs are identified using `scipy.ndimage.label`
- Only the largest blob is kept, all others are discarded
- The bounding box is then derived from the cleaned-up mask

This significantly reduces noise from stray predicted foreground pixels unrelated to the hand.

---

## Data Splits

All train/val splits are done at the **student level** to prevent data leakage. All clips from a given student appear in exactly one split. The overnight training script uses 80% of students for training and 20% for validation (controlled by `VAL_FRACTION = 0.2` in `train_overnight.py`).

---

## Requirements

See `requirements.txt`. Main dependencies:

| Package | Purpose |
|---|---|
| `torch`, `torchvision` | Model definition, training, transforms |
| `numpy` | Array operations throughout |
| `Pillow` | Image loading and saving |
| `scipy` | Connected component labelling (`ndimage.label`) |
| `tqdm` | Training progress bars |
| `matplotlib` | Confusion matrix plots (Agg backend, SSH-safe) |
