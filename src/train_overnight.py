"""
Overnight mega-training script — trains all 4 model weights with hyperparameter sweeps.

This is the main training script intended to be left running overnight (hence the name).
It trains all four model components in sequence with a small hyperparameter sweep for each:

Pipeline:
  Phase 1 — RGBDSegmenter  (RGB + depth, full augmentation)
  Phase 2 — RGB-only segmenter  (same architecture, depth channel zeroed at training time)
  Phase 3 — RGBMROICropClassifier  (on-the-fly RGBD seg → LCC → square-crop 224×224 RGBM)
  Phase 4 — MROICropClassifier     (same pipeline, classifier only sees the mask channel)

The hyperparameter sweep tries a few different learning rates and weight decay values.
For each config we train from scratch and keep the best checkpoint. The overall best
across the sweep (lowest val loss for segmenters, highest val acc for classifiers) is
then copied to the top-level weights/overnight/ folder for easy access.

Augmentation strategy:
  Segmenter training  — full geometric (flip H/V, rotate ±20°, random scale-crop)
                        + photometric (colour jitter, depth noise)
  Classifier pre-seg  — same geometric, EXCEPT:
                        * "like" (label 2) and "dislike" (label 1): no vertical flip,
                          rotation capped at ±10°  (thumbs-up/down orientation matters)
  Classifier post-crop — brightness jitter + Gaussian noise + random mask erasing

All splits are done at student level (20% of students held out for validation) so there
is no data leakage between train and val sets.

Usage (from project root, inside tmux):
    conda activate comp0248-lab1
    python src/train_overnight.py --data dataset

    # Start classifier phases from an existing segmenter checkpoint:
    python src/train_overnight.py --data dataset \\
        --skip_seg \\
        --rgbd_seg_ckpt weights/overnight/best_rgbd_seg.pt \\
        --rgb_seg_ckpt  weights/overnight/best_rgb_seg.pt

    # Quick smoke test (1 epoch per phase, small batches) to check the pipeline runs:
    python src/train_overnight.py --data dataset --smoke_test

Outputs (all under weights/overnight/):
    best_rgbd_seg.pt        ← best RGBD segmenter from sweep
    best_rgb_seg.pt         ← best RGB-only segmenter from sweep
    best_rgbm_clf.pt        ← best RGBM ROI-crop classifier from sweep
    best_m_clf.pt           ← best mask-only ROI-crop classifier from sweep
    sweep_summary.csv       ← one row per run with all metrics
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image as PILImage
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataloader import GESTURE_LABELS                              # noqa: E402
from src.model_rgbd_segmenter import RGBDSegmenter, SegmentationLoss  # noqa: E402
from src.model_rgbm_roi_crop_classifier import RGBMROICropClassifier   # noqa: E402
from src.model_m_roi_crop_classifier import MROICropClassifier         # noqa: E402
from src.utils import compute_iou, compute_dice, AverageMeter          # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_CLASSES = len(GESTURE_LABELS)  # 10 gesture classes
CROP_SIZE   = 224    # classifier input size -- standard ImageNet-style 224x224
CROP_SCALE  = 1.2    # expand bbox by 20% before cropping, to include some context around the hand

# These gesture labels have a meaningful vertical orientation.
# Flipping them upside-down would change the label (thumbs up → thumbs down) or
# make the gesture ambiguous, so we skip vertical flips for these classes.
ORIENTATION_SENSITIVE: Set[int] = {
    GESTURE_LABELS["dislike"],  # 1 -- thumbs down, meaning inverts if flipped
    GESTURE_LABELS["like"],     # 2 -- thumbs up, same reason
    GESTURE_LABELS["one"],      # 4  (finger pointing up -- direction matters)
    GESTURE_LABELS["stop"],     # 8  (palm facing forward, upright -- rotation sensitive)
}

# ---------------------------------------------------------------------------
# Hyperparameter sweeps
# ---------------------------------------------------------------------------
# We try three different learning rates for the segmenter.
# 3e-4 is usually a safe starting point for Adam, 1e-3 trains faster but can overshoot,
# and 3e-3 is quite aggressive -- sometimes it works though.
# I kept weight_decay the same across all configs to reduce the search space.
SEG_SWEEP: List[Dict] = [
    {"lr": 3e-4, "weight_decay": 1e-4, "epochs": 35},
    {"lr": 1e-3, "weight_decay": 1e-4, "epochs": 35},
    {"lr": 3e-3, "weight_decay": 1e-4, "epochs": 30},
]

# Classifier sweep uses lower LRs since the classifier is a simpler task
# and higher LRs tend to cause unstable training on the small crop inputs
CLF_SWEEP: List[Dict] = [
    {"lr": 1e-4, "weight_decay": 1e-4, "epochs": 35},
    {"lr": 3e-4, "weight_decay": 1e-4, "epochs": 35},
    {"lr": 1e-3, "weight_decay": 5e-5, "epochs": 30},
]

SEG_BATCH    = 8    # segmenter is memory-heavy (full-res images), so smaller batch
CLF_BATCH    = 32   # classifier uses 224x224 crops so can fit more in memory
VAL_FRACTION = 0.2  # 20% of students held out for validation -- see student-level split below
SEED         = 42   # reproducibility seed for all shuffles and random ops


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def _color_jitter(rgb: np.ndarray,
                  brightness: float = 0.35,
                  contrast: float = 0.35,
                  saturation: float = 0.25,
                  noise_std: float = 0.02) -> np.ndarray:
    """
    Random photometric jitter on (H,W,3) float32 RGB image in [0,1].

    Each type of distortion is applied independently at its own probability.
    The luma weights for saturation jitter come from BT.601 standard.
    """
    out = rgb.copy()
    if random.random() < 0.8:  # 80% chance of brightness jitter
        out = np.clip(out * (1.0 + random.uniform(-brightness, brightness)), 0, 1)
    if random.random() < 0.8:  # 80% chance of contrast jitter (scale around mean)
        mean = out.mean()
        out = np.clip((out - mean) * (1.0 + random.uniform(-contrast, contrast)) + mean, 0, 1)
    if random.random() < 0.5:  # 50% chance of saturation jitter (lerp to grayscale)
        gray = (0.2989 * out[..., 0] + 0.5870 * out[..., 1] + 0.1140 * out[..., 2])[..., None]
        f = 1.0 + random.uniform(-saturation, saturation)
        out = np.clip(gray * (1 - f) + out * f, 0, 1)
    if random.random() < 0.5:  # 50% chance of additive Gaussian noise
        out = np.clip(out + np.random.normal(0, noise_std, out.shape).astype(np.float32), 0, 1)
    return out.astype(np.float32)


def _hflip(rgb, depth, mask=None):
    """Horizontal flip all modalities consistently (images are stored H x W x C)."""
    rgb   = rgb  [:, ::-1, :].copy()
    depth = depth[:, ::-1].copy()
    if mask is not None:
        mask = mask[:, ::-1].copy()
    return rgb, depth, mask


def _vflip(rgb, depth, mask=None):
    """Vertical flip all modalities. Only called for non-orientation-sensitive gestures."""
    rgb   = rgb  [::-1, :, :].copy()
    depth = depth[::-1, :].copy()
    if mask is not None:
        mask = mask[::-1, :].copy()
    return rgb, depth, mask


def _rotate(rgb, depth, mask=None, max_angle: float = 20.0):
    """
    Rotate rgb, depth, and (optionally) mask by a random angle.

    We have to go through PIL for the rotation because numpy doesn't have a
    built-in rotate. For depth we scale to uint16 range first (PIL doesn't do float),
    then normalise back after. Mask uses NEAREST interpolation to stay binary.
    """
    angle = random.uniform(-max_angle, max_angle)
    H, W  = rgb.shape[:2]
    rgb_r = np.array(
        PILImage.fromarray((rgb * 255).clip(0, 255).astype(np.uint8))
               .rotate(angle, resample=PILImage.BILINEAR)
    ).astype(np.float32) / 255.0
    # Depth: normalise to uint16 range for PIL rotation, then convert back to [0,1]
    d_pil   = PILImage.fromarray((np.clip(depth, 0, 1) * 65535).astype(np.uint16))
    depth_r = np.array(d_pil.rotate(angle, resample=PILImage.BILINEAR)).astype(np.float32) / 65535.0
    mask_r  = None
    if mask is not None:
        # NEAREST for mask to avoid introducing non-binary values at edges
        mask_r = (np.array(
            PILImage.fromarray((mask * 255).astype(np.uint8))
                   .rotate(angle, resample=PILImage.NEAREST)
        ) > 127).astype(np.uint8)
    return rgb_r, depth_r, mask_r


def _scale_crop(rgb, depth, mask=None, min_scale: float = 0.75):
    """
    Randomly zoom in by cropping a sub-region and resizing back to original dimensions.

    Scale is sampled uniformly from [min_scale, 1.0]. A scale of 0.75 means we
    crop a 75% sized sub-region -- effectively zooming in 1.33x.
    The crop position is chosen randomly within the valid range.
    """
    H, W  = rgb.shape[:2]
    scale  = random.uniform(min_scale, 1.0)
    ch, cw = int(H * scale), int(W * scale)
    top    = random.randint(0, H - ch)
    left   = random.randint(0, W - cw)
    rgb_c   = rgb  [top:top+ch, left:left+cw, :]
    depth_c = depth[top:top+ch, left:left+cw]
    rgb_r = np.array(
        PILImage.fromarray((rgb_c * 255).clip(0, 255).astype(np.uint8)).resize((W, H), PILImage.BILINEAR)
    ).astype(np.float32) / 255.0
    d_pil   = PILImage.fromarray((np.clip(depth_c, 0, 1) * 65535).astype(np.uint16))
    depth_r = np.array(d_pil.resize((W, H), PILImage.BILINEAR)).astype(np.float32) / 65535.0
    mask_r  = None
    if mask is not None:
        mask_c = mask[top:top+ch, left:left+cw]
        mask_r = (np.array(
            PILImage.fromarray((mask_c * 255).astype(np.uint8)).resize((W, H), PILImage.NEAREST)
        ) > 127).astype(np.uint8)
    return rgb_r, depth_r, mask_r


def augment_segmenter_frame(rgb, depth, mask):
    """
    Full geometric + photometric augmentation for segmenter training.

    We don't restrict flips here because for segmentation the orientation of the
    gesture doesn't affect what the correct mask is -- a flipped hand is still a hand.
    Applying all augmentations consistently across rgb, depth, and mask is critical
    so that the training pairs stay aligned.
    """
    if random.random() < 0.5:
        rgb, depth, mask = _hflip(rgb, depth, mask)
    if random.random() < 0.3:
        rgb, depth, mask = _vflip(rgb, depth, mask)
    if random.random() < 0.5:
        rgb, depth, mask = _rotate(rgb, depth, mask, max_angle=20.0)
    if random.random() < 0.5:
        rgb, depth, mask = _scale_crop(rgb, depth, mask)
    rgb = _color_jitter(rgb)  # photometric -- only RGB, not depth or mask
    if random.random() < 0.4:
        # Simulate multiplicative depth sensor noise (real depth cameras have shot noise)
        noise = np.random.normal(1.0, 0.05, depth.shape).astype(np.float32)
        depth = np.clip(depth * noise, 0, 1).astype(np.float32)
    return rgb, depth, mask


def augment_clf_preseg(rgb, depth, gesture_label: int):
    """
    Pre-segmentation augmentation for classifier training, gesture-label aware.

    Same geometric transforms as the segmenter augmentation, but we skip the
    mask (there isn't one at this point in the classifier pipeline) and we're
    careful not to vertically flip orientation-sensitive gestures.
    """
    sensitive = gesture_label in ORIENTATION_SENSITIVE
    if random.random() < 0.5:
        rgb, depth, _ = _hflip(rgb, depth)
    if not sensitive and random.random() < 0.3:
        # Only do vertical flip if the gesture orientation doesn't matter
        rgb, depth, _ = _vflip(rgb, depth)
    if random.random() < 0.5:
        max_angle = 10.0 if sensitive else 20.0  # smaller rotations for orientation-sensitive
        rgb, depth, _ = _rotate(rgb, depth, max_angle=max_angle)
    if random.random() < 0.5:
        rgb, depth, _ = _scale_crop(rgb, depth)
    rgb = _color_jitter(rgb, brightness=0.4, contrast=0.4, saturation=0.3)
    if random.random() < 0.4:
        noise = np.random.normal(1.0, 0.05, depth.shape).astype(np.float32)
        depth = np.clip(depth * noise, 0, 1).astype(np.float32)
    return rgb, depth


def augment_rgbm_crop(crop: np.ndarray) -> np.ndarray:
    """
    Post-crop augmentation on (4, 224, 224) RGBM crop.

    Applied after the crop is extracted and resized. We only jitter the RGB
    channels (0-2), not the mask channel (3). The mask erasing simulates partial
    occlusion of the hand -- important since at inference time the mask comes
    from the segmenter and won't be perfect.
    """
    crop = crop.copy()
    if random.random() < 0.5:   # brightness jitter on RGB channels only
        crop[:3] = np.clip(crop[:3] * (1.0 + random.uniform(-0.3, 0.3)), 0, 1)
    if random.random() < 0.5:   # Gaussian noise on RGB channels
        crop[:3] = np.clip(crop[:3] + np.random.normal(0, 0.025, crop[:3].shape).astype(np.float32), 0, 1)
    if random.random() < 0.2:   # random rectangular mask erasing (CutOut-style augmentation)
        _, H, W = crop.shape
        eh = random.randint(H // 8, H // 3)  # erase height: 1/8 to 1/3 of crop
        ew = random.randint(W // 8, W // 3)  # erase width: 1/8 to 1/3 of crop
        ey = random.randint(0, H - eh)       # random top edge
        ex = random.randint(0, W - ew)       # random left edge
        crop[3, ey:ey+eh, ex:ex+ew] = 0.0   # zero out that region in the mask channel
    return crop.astype(np.float32)


def augment_mask_crop(crop: np.ndarray) -> np.ndarray:
    """
    Post-crop augmentation on (1, 224, 224) mask crop.

    Similar to augment_rgbm_crop but for the mask-only classifier. We add soft
    noise and random erasing to help the classifier handle imperfect masks from
    the real segmenter at inference time.
    """
    crop = crop.copy()
    if random.random() < 0.3:   # random mask erasing (partial occlusion simulation)
        _, H, W = crop.shape
        eh = random.randint(H // 8, H // 3)
        ew = random.randint(W // 8, W // 3)
        ey = random.randint(0, H - eh)
        ex = random.randint(0, W - ew)
        crop[0, ey:ey+eh, ex:ex+ew] = 0.0
    if random.random() < 0.3:   # additive noise -- makes model robust to fuzzy mask boundaries
        crop = np.clip(crop + np.random.normal(0, 0.05, crop.shape).astype(np.float32), 0, 1)
    return crop.astype(np.float32)


# ---------------------------------------------------------------------------
# Post-processing helpers  (identical to evaluate.py so training matches eval)
# Note: these must match exactly what evaluate.py does or the training signal
# won't match what we measure at evaluation time.
# ---------------------------------------------------------------------------

def largest_component_mask(binary: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component in a binary mask, discard the rest.

    This is the key post-processing step that cleans up noisy segmentation outputs.
    If the model predicts a scattered set of foreground blobs, this keeps only the
    biggest one (which is almost certainly the actual hand).
    """
    if binary.sum() == 0:
        return binary
    labeled, n = ndimage.label(binary)
    if n <= 1:
        return binary
    sizes = ndimage.sum(binary, labeled, range(1, n + 1))
    return (labeled == int(np.argmax(sizes)) + 1).astype(np.uint8)


def mask_to_bbox(binary: np.ndarray):
    """
    Get the tight bounding box of foreground pixels in a binary mask.

    Returns (x1, y1, x2, y2) as pixel integers, or None if the mask is empty.
    This is used to find the hand region before cropping for the classifier.
    """
    ys, xs = np.where(binary > 0)
    if len(ys) == 0:
        return None  # empty mask -- no hand detected
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def square_crop_coords(x1, y1, x2, y2, H, W, scale: float = CROP_SCALE):
    """
    Expand a bounding box to a square and scale it, clamped to image boundaries.

    We make the crop square (not rectangular) so the classifier always receives
    a consistent aspect ratio. The scale factor (default 1.2) adds some context
    around the hand -- a bit of background seems to help the classifier.
    If the computed crop is degenerate (zero area), fall back to the full image.
    """
    cx   = 0.5 * (x1 + x2)  # centre x
    cy   = 0.5 * (y1 + y2)  # centre y
    # side length = max dimension of bbox * scale factor
    side = max(max(1.0, float(x2 - x1)), max(1.0, float(y2 - y1))) * scale
    half = 0.5 * side
    ox1  = int(max(0, round(cx - half)))
    oy1  = int(max(0, round(cy - half)))
    ox2  = int(min(W,  round(cx + half)))
    oy2  = int(min(H,  round(cy + half)))
    if ox2 <= ox1 or oy2 <= oy1:
        return 0, 0, W, H  # fallback: entire image
    return ox1, oy1, ox2, oy2


def make_rgbm_crop(rgb_np: np.ndarray, binary_mask: np.ndarray,
                   bbox, augment: bool = True) -> np.ndarray:
    """
    Extract and resize a 4-channel RGBM crop centred on the hand bounding box.

    Takes the square crop region, extracts both the RGB and mask sub-images,
    resizes both to CROP_SIZE x CROP_SIZE, stacks as (R, G, B, M), and
    optionally applies augmentation.

    Returns zeros if bbox is None (no hand found in the mask).
    """
    H, W = binary_mask.shape
    if bbox is None:
        # No hand detected -- return blank crop (the classifier will likely misclassify this)
        return np.zeros((4, CROP_SIZE, CROP_SIZE), dtype=np.float32)
    cx1, cy1, cx2, cy2 = square_crop_coords(bbox[0], bbox[1], bbox[2], bbox[3], H, W)
    rgb_crop  = rgb_np     [cy1:cy2, cx1:cx2]
    mask_crop = binary_mask[cy1:cy2, cx1:cx2]
    # Resize RGB with bilinear, mask with nearest-neighbour to keep it binary
    rgb_r  = np.array(
        PILImage.fromarray((rgb_crop * 255).clip(0, 255).astype(np.uint8))
               .resize((CROP_SIZE, CROP_SIZE), PILImage.BILINEAR)
    ).astype(np.float32) / 255.0
    mask_r = (np.array(
        PILImage.fromarray((mask_crop * 255).astype(np.uint8))
               .resize((CROP_SIZE, CROP_SIZE), PILImage.NEAREST)
    ) > 127).astype(np.float32)
    # Stack: (3, H, W) RGB + (1, H, W) mask = (4, H, W) RGBM
    rgbm = np.concatenate([rgb_r.transpose(2, 0, 1), mask_r[None]], axis=0)
    return augment_rgbm_crop(rgbm) if augment else rgbm.astype(np.float32)


def make_mask_crop(binary_mask: np.ndarray, bbox, augment: bool = True) -> np.ndarray:
    """
    Extract and resize a 1-channel mask crop centred on the hand bounding box.

    Same idea as make_rgbm_crop but only the mask channel -- used for the
    mask-only classifier (Phase 4).
    """
    H, W = binary_mask.shape
    if bbox is None:
        return np.zeros((1, CROP_SIZE, CROP_SIZE), dtype=np.float32)
    cx1, cy1, cx2, cy2 = square_crop_coords(bbox[0], bbox[1], bbox[2], bbox[3], H, W)
    mask_crop = binary_mask[cy1:cy2, cx1:cx2].astype(np.uint8)
    mask_r = (np.array(
        PILImage.fromarray(mask_crop * 255).resize((CROP_SIZE, CROP_SIZE), PILImage.NEAREST)
    ) > 127).astype(np.float32)[None]  # add channel dim -> (1, H, W)
    return augment_mask_crop(mask_r) if augment else mask_r.astype(np.float32)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class SegDataset(Dataset):
    """
    Full RGB-D + annotation-mask dataset for segmenter training.
    Applies augment_segmenter_frame in __getitem__ when augment=True.
    """

    def __init__(self, root_dir: str, student_ids=None, augment: bool = True):
        self.augment = augment
        self.samples: List[Dict] = []
        root = Path(root_dir)
        allowed = {".png", ".jpg", ".jpeg"}

        for mask_path in root.glob("*/*/*/annotation/*"):
            if mask_path.suffix.lower() not in allowed or mask_path.name.startswith("."):
                continue
            clip_dir    = mask_path.parent.parent
            gesture_dir = clip_dir.parent
            student_id  = gesture_dir.parent.name
            if student_ids is not None and student_id not in student_ids:
                continue
            gname = (gesture_dir.name.split("_", 1)[1] if "_" in gesture_dir.name
                     else gesture_dir.name).lower()
            if gname not in GESTURE_LABELS:
                continue
            rgb_path   = clip_dir / "rgb"       / mask_path.name
            depth_path = clip_dir / "depth_raw" / f"{mask_path.stem}.npy"
            if not rgb_path.exists() or not depth_path.exists():
                continue
            self.samples.append({
                "rgb": rgb_path, "depth_raw": depth_path,
                "mask": mask_path, "student_id": student_id,
                "gesture": GESTURE_LABELS[gname],
            })

        self.student_ids = sorted({s["student_id"] for s in self.samples})
        print(f"  SegDataset: {len(self.samples)} frames, "
              f"{len(self.student_ids)} students  [augment={augment}]")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        # Load RGB as float [0,1]
        rgb_np   = np.array(PILImage.open(s["rgb"]).convert("RGB"),
                            dtype=np.float32) / 255.0
        # Load depth: clip to 1500mm, normalise to [0,1]
        depth_np = np.clip(np.load(s["depth_raw"]).astype(np.float32), 0, 1500) / 1500.0
        # Binarise the mask: any non-zero pixel = hand
        mask_np  = (np.array(PILImage.open(s["mask"])) > 0).astype(np.uint8)

        # Apply augmentation in-place if training -- all three modalities together
        if self.augment:
            rgb_np, depth_np, mask_np = augment_segmenter_frame(rgb_np, depth_np, mask_np)

        # Stack RGB + depth into (4, H, W) input tensor
        inp = torch.cat([
            torch.from_numpy(rgb_np).permute(2, 0, 1),   # (3, H, W)
            torch.from_numpy(depth_np).unsqueeze(0),       # (1, H, W)
        ], dim=0)  # (4, H, W)
        return {
            "input":      inp,
            "mask":       torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0),  # (1,H,W)
            "gesture":    torch.tensor(s["gesture"], dtype=torch.long),
            "student_id": s["student_id"],
        }


class ClfDataset(Dataset):
    """
    RGB-D dataset for classifier training (no annotation mask needed).
    Pre-segmentation augmentation applied in __getitem__ when augment=True.
    On-the-fly segmentation and ROI cropping happen in the training loop.
    """

    def __init__(self, root_dir: str, student_ids=None, augment: bool = True):
        self.augment = augment
        self.samples: List[Dict] = []
        root = Path(root_dir)
        allowed = {".png", ".jpg", ".jpeg"}

        for rgb_path in root.glob("*/*/*/rgb/*"):
            if rgb_path.suffix.lower() not in allowed or rgb_path.name.startswith("."):
                continue
            clip_dir    = rgb_path.parent.parent
            gesture_dir = clip_dir.parent
            student_id  = gesture_dir.parent.name
            if student_ids is not None and student_id not in student_ids:
                continue
            gname = (gesture_dir.name.split("_", 1)[1] if "_" in gesture_dir.name
                     else gesture_dir.name).lower()
            if gname not in GESTURE_LABELS:
                continue
            depth_path = clip_dir / "depth_raw" / f"{rgb_path.stem}.npy"
            if not depth_path.exists():
                continue
            self.samples.append({
                "rgb": rgb_path, "depth_raw": depth_path,
                "gesture": GESTURE_LABELS[gname], "student_id": student_id,
            })

        self.student_ids = sorted({s["student_id"] for s in self.samples})
        print(f"  ClfDataset: {len(self.samples)} frames, "
              f"{len(self.student_ids)} students  [augment={augment}]")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        gesture_label = s["gesture"]
        rgb_np   = np.array(PILImage.open(s["rgb"]).convert("RGB"),
                            dtype=np.float32) / 255.0
        depth_np = np.clip(np.load(s["depth_raw"]).astype(np.float32), 0, 1500) / 1500.0

        if self.augment:
            rgb_np, depth_np = augment_clf_preseg(rgb_np, depth_np, gesture_label)

        rgb_t   = torch.from_numpy(rgb_np).permute(2, 0, 1)   # (3,H,W)
        depth_t = torch.from_numpy(depth_np).unsqueeze(0)     # (1,H,W)
        return {
            "rgb":    rgb_t,
            "rgbd":   torch.cat([rgb_t, depth_t], dim=0),
            "gesture": torch.tensor(gesture_label, dtype=torch.long),
            "student_id": s["student_id"],
        }


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def _seg_run_batch(model, batch, device, loss_fn, optimizer=None):
    """Forward (+ backward if optimizer given). Returns loss, iou, dice."""
    inp  = batch["input"].to(device)
    mask = batch["mask"].to(device)
    pred = model(inp)
    if pred.shape[2:] != mask.shape[2:]:
        pred = F.interpolate(pred, size=mask.shape[-2:], mode="bilinear", align_corners=False)
    loss = loss_fn(pred, mask)
    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        pm = (pred.detach().cpu().numpy() > 0.5).astype(np.uint8)
        gm = (mask.detach().cpu().numpy() > 0.5).astype(np.uint8)
        ious  = [compute_iou (pm[i, 0], gm[i, 0]) for i in range(pm.shape[0])]
        dices = [compute_dice(pm[i, 0], gm[i, 0]) for i in range(pm.shape[0])]
    return loss.item(), float(np.mean(ious)), float(np.mean(dices)), inp.size(0)


def _seg_batch_for_clf(segmenter, rgbd_t: torch.Tensor, device) -> np.ndarray:
    """Run segmenter on a batch tensor, return LCC binary masks (B,H,W) numpy."""
    with torch.no_grad():
        pred = segmenter(rgbd_t)
        if pred.shape[2:] != rgbd_t.shape[2:]:
            pred = F.interpolate(pred, size=rgbd_t.shape[2:], mode="bilinear", align_corners=False)
        seg_np = pred.squeeze(1).cpu().numpy()
    masks = np.stack([
        largest_component_mask((seg_np[i] > 0.5).astype(np.uint8))
        for i in range(seg_np.shape[0])
    ], axis=0)
    return masks  # (B,H,W) uint8


# ---------------------------------------------------------------------------
# Single-run trainers (called by sweep functions)
# ---------------------------------------------------------------------------

def run_seg_one(model, train_loader, val_loader, device,
                lr, weight_decay, epochs, run_dir: Path, zero_depth=False):
    """
    Train RGBDSegmenter for one sweep config.
    zero_depth: if True, depth channel is zeroed → trains as RGB-only segmenter.
    Returns best_val_loss.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    loss_fn   = SegmentationLoss(use_dice=True, use_iou=True)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.05)

    best_val_loss = float("inf")
    best_pt  = run_dir / "best.pt"
    log_path = run_dir / "log.csv"

    with log_path.open("w", newline="") as f:
        csv.writer(f).writerow(["epoch", "phase", "loss", "iou", "dice"])

    for epoch in range(1, epochs + 1):
        model.train()
        tm = AverageMeter(); ti = AverageMeter(); td = AverageMeter()
        for batch in train_loader:
            if zero_depth:
                batch["input"][:, 3, :, :] = 0.0
            l, i, d, n = _seg_run_batch(model, batch, device, loss_fn, optimizer)
            tm.update(l, n); ti.update(i, n); td.update(d, n)

        model.eval()
        vm = AverageMeter(); vi = AverageMeter(); vd = AverageMeter()
        with torch.no_grad():
            for batch in val_loader:
                if zero_depth:
                    batch["input"][:, 3, :, :] = 0.0
                l, i, d, n = _seg_run_batch(model, batch, device, loss_fn)
                vm.update(l, n); vi.update(i, n); vd.update(d, n)

        scheduler.step()

        print(f"  [{run_dir.name}] Ep {epoch:03d}/{epochs} "
              f"TRAIN loss={tm.avg:.4f} IoU={ti.avg:.4f} Dice={td.avg:.4f} | "
              f"VAL   loss={vm.avg:.4f} IoU={vi.avg:.4f} Dice={vd.avg:.4f}")

        with log_path.open("a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch, "train", tm.avg, ti.avg, td.avg])
            w.writerow([epoch, "val",   vm.avg, vi.avg, vd.avg])

        if vm.avg < best_val_loss:
            best_val_loss = vm.avg
            torch.save({"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_loss": best_val_loss,
                        "val_iou": vi.avg, "val_dice": vd.avg,
                        "lr": lr}, best_pt)
            print(f"    -> Saved best (val_loss={best_val_loss:.4f}) → {best_pt}")

    return best_val_loss


def run_clf_rgbm_one(train_loader, val_loader, segmenter, device,
                     lr, weight_decay, epochs, run_dir: Path, zero_seg_depth=False):
    """Train RGBMROICropClassifier for one sweep config."""
    run_dir.mkdir(parents=True, exist_ok=True)
    clf       = RGBMROICropClassifier(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(clf.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.05)

    best_val_acc = 0.0
    best_pt  = run_dir / "best.pt"
    log_path = run_dir / "log.csv"

    with log_path.open("w", newline="") as f:
        csv.writer(f).writerow(["epoch", "phase", "loss", "acc"])

    for epoch in range(1, epochs + 1):
        clf.train()
        r_loss = r_correct = r_total = 0

        for batch in train_loader:
            rgb_t  = batch["rgb"].to(device)
            rgbd_t = batch["rgbd"].to(device)
            if zero_seg_depth:
                rgbd_t = rgbd_t.clone()
                rgbd_t[:, 3, :, :] = 0.0
            targets = batch["gesture"].to(device)

            # On-the-fly: segment → LCC → ROI crop
            masks_np = _seg_batch_for_clf(segmenter, rgbd_t, device)
            rgb_np_b = rgb_t.cpu().numpy().transpose(0, 2, 3, 1)  # (B,H,W,3)
            crops = np.stack([
                make_rgbm_crop(rgb_np_b[i], masks_np[i],
                               mask_to_bbox(masks_np[i]), augment=True)
                for i in range(len(masks_np))
            ], axis=0)  # (B,4,224,224)

            crops_t = torch.from_numpy(crops).float().to(device)
            optimizer.zero_grad()
            logits = clf(crops_t)
            loss   = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            r_loss    += loss.item() * crops_t.size(0)
            r_correct += (logits.argmax(1) == targets).sum().item()
            r_total   += crops_t.size(0)

        train_loss = r_loss / r_total
        train_acc  = r_correct / r_total
        scheduler.step()

        clf.eval()
        v_correct = v_total = 0
        with torch.no_grad():
            for batch in val_loader:
                rgb_t  = batch["rgb"].to(device)
                rgbd_t = batch["rgbd"].to(device)
                if zero_seg_depth:
                    rgbd_t = rgbd_t.clone()
                    rgbd_t[:, 3, :, :] = 0.0
                targets = batch["gesture"].to(device)
                masks_np  = _seg_batch_for_clf(segmenter, rgbd_t, device)
                rgb_np_b  = rgb_t.cpu().numpy().transpose(0, 2, 3, 1)
                crops = np.stack([
                    make_rgbm_crop(rgb_np_b[i], masks_np[i],
                                   mask_to_bbox(masks_np[i]), augment=False)
                    for i in range(len(masks_np))
                ], axis=0)
                logits = clf(torch.from_numpy(crops).float().to(device))
                v_correct += (logits.argmax(1) == targets).sum().item()
                v_total   += crops.shape[0]

        val_acc = v_correct / max(1, v_total)
        print(f"  [{run_dir.name}] Ep {epoch:03d}/{epochs} "
              f"loss={train_loss:.4f}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

        with log_path.open("a", newline="") as f:
            csv.writer(f).writerows([
                [epoch, "train", train_loss, train_acc],
                [epoch, "val",   "-",        val_acc],
            ])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"epoch": epoch,
                        "model_state_dict": clf.state_dict(),
                        "val_acc": best_val_acc, "lr": lr}, best_pt)
            print(f"    -> Saved best (val_acc={best_val_acc:.4f}) → {best_pt}")

    return best_val_acc


def run_clf_m_one(train_loader, val_loader, segmenter, device,
                  lr, weight_decay, epochs, run_dir: Path, zero_seg_depth=False):
    """Train MROICropClassifier for one sweep config."""
    run_dir.mkdir(parents=True, exist_ok=True)
    clf       = MROICropClassifier(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(clf.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.05)

    best_val_acc = 0.0
    best_pt  = run_dir / "best.pt"
    log_path = run_dir / "log.csv"

    with log_path.open("w", newline="") as f:
        csv.writer(f).writerow(["epoch", "phase", "loss", "acc"])

    for epoch in range(1, epochs + 1):
        clf.train()
        r_loss = r_correct = r_total = 0

        for batch in train_loader:
            rgbd_t = batch["rgbd"].to(device)
            if zero_seg_depth:
                rgbd_t = rgbd_t.clone()
                rgbd_t[:, 3, :, :] = 0.0
            targets = batch["gesture"].to(device)

            masks_np = _seg_batch_for_clf(segmenter, rgbd_t, device)
            crops = np.stack([
                make_mask_crop(masks_np[i], mask_to_bbox(masks_np[i]), augment=True)
                for i in range(len(masks_np))
            ], axis=0)  # (B,1,224,224)

            crops_t = torch.from_numpy(crops).float().to(device)
            optimizer.zero_grad()
            logits = clf(crops_t)
            loss   = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            r_loss    += loss.item() * crops_t.size(0)
            r_correct += (logits.argmax(1) == targets).sum().item()
            r_total   += crops_t.size(0)

        train_loss = r_loss / r_total
        train_acc  = r_correct / r_total
        scheduler.step()

        clf.eval()
        v_correct = v_total = 0
        with torch.no_grad():
            for batch in val_loader:
                rgbd_t = batch["rgbd"].to(device)
                if zero_seg_depth:
                    rgbd_t = rgbd_t.clone()
                    rgbd_t[:, 3, :, :] = 0.0
                targets = batch["gesture"].to(device)
                masks_np = _seg_batch_for_clf(segmenter, rgbd_t, device)
                crops = np.stack([
                    make_mask_crop(masks_np[i], mask_to_bbox(masks_np[i]), augment=False)
                    for i in range(len(masks_np))
                ], axis=0)
                logits = clf(torch.from_numpy(crops).float().to(device))
                v_correct += (logits.argmax(1) == targets).sum().item()
                v_total   += crops.shape[0]

        val_acc = v_correct / max(1, v_total)
        print(f"  [{run_dir.name}] Ep {epoch:03d}/{epochs} "
              f"loss={train_loss:.4f}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

        with log_path.open("a", newline="") as f:
            csv.writer(f).writerows([
                [epoch, "train", train_loss, train_acc],
                [epoch, "val",   "-",        val_acc],
            ])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"epoch": epoch,
                        "model_state_dict": clf.state_dict(),
                        "val_acc": best_val_acc, "lr": lr}, best_pt)
            print(f"    -> Saved best (val_acc={best_val_acc:.4f}) → {best_pt}")

    return best_val_acc


# ---------------------------------------------------------------------------
# Sweep drivers
# ---------------------------------------------------------------------------

def _make_seg_loaders(data_root, train_students, val_students, augment):
    train_ds = SegDataset(str(data_root), student_ids=set(train_students), augment=augment)
    val_ds   = SegDataset(str(data_root), student_ids=set(val_students),   augment=False)
    train_ld = DataLoader(train_ds, batch_size=SEG_BATCH, shuffle=True,
                          num_workers=2, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=SEG_BATCH, shuffle=False,
                          num_workers=2, pin_memory=True)
    return train_ld, val_ld


def _make_clf_loaders(data_root, train_students, val_students, augment):
    train_ds = ClfDataset(str(data_root), student_ids=set(train_students), augment=augment)
    val_ds   = ClfDataset(str(data_root), student_ids=set(val_students),   augment=False)
    train_ld = DataLoader(train_ds, batch_size=CLF_BATCH, shuffle=True,
                          num_workers=2, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=CLF_BATCH, shuffle=False,
                          num_workers=2, pin_memory=True)
    return train_ld, val_ld


def sweep_segmenter(phase_name: str, data_root: Path, device,
                    train_students, val_students,
                    out_dir: Path, zero_depth: bool, sweep_rows: list):
    """Run all SEG_SWEEP configs; return path to overall best checkpoint."""
    print(f"\n{'='*70}")
    print(f"  PHASE: {phase_name}  ({'RGB-only' if zero_depth else 'RGBD'})")
    print(f"{'='*70}")
    train_ld, val_ld = _make_seg_loaders(data_root, train_students, val_students, augment=True)

    best_loss = float("inf")
    best_ckpt: Optional[Path] = None

    for cfg in SEG_SWEEP:
        run_name = f"lr{cfg['lr']:.0e}_wd{cfg['weight_decay']:.0e}"
        run_dir  = out_dir / run_name
        print(f"\n  --- {phase_name} | {run_name} ---")
        t0    = time.time()
        model = RGBDSegmenter().to(device)
        val_loss = run_seg_one(model, train_ld, val_ld, device,
                               lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                               epochs=cfg["epochs"], run_dir=run_dir,
                               zero_depth=zero_depth)
        elapsed = time.time() - t0
        sweep_rows.append({
            "phase": phase_name, "run": run_name,
            "metric": f"val_loss={val_loss:.4f}", "elapsed_min": f"{elapsed/60:.1f}",
        })
        if val_loss < best_loss:
            best_loss = val_loss
            best_ckpt = run_dir / "best.pt"

    print(f"\n  {phase_name} sweep done. Best val_loss={best_loss:.4f}  ckpt={best_ckpt}")
    return best_ckpt


def sweep_clf_rgbm(phase_name: str, data_root: Path, device,
                   train_students, val_students,
                   seg_ckpt: Path, out_dir: Path,
                   zero_seg_depth: bool, sweep_rows: list):
    """Run all CLF_SWEEP configs for RGBM classifier; return best checkpoint."""
    print(f"\n{'='*70}")
    print(f"  PHASE: {phase_name}")
    print(f"{'='*70}")
    segmenter = RGBDSegmenter().to(device)
    ckpt = torch.load(seg_ckpt, map_location=device)
    segmenter.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
    segmenter.eval()
    for p in segmenter.parameters():
        p.requires_grad_(False)
    print(f"  Loaded segmenter from {seg_ckpt}")

    train_ld, val_ld = _make_clf_loaders(data_root, train_students, val_students, augment=True)

    best_acc  = 0.0
    best_ckpt: Optional[Path] = None

    for cfg in CLF_SWEEP:
        run_name = f"lr{cfg['lr']:.0e}_wd{cfg['weight_decay']:.0e}"
        run_dir  = out_dir / run_name
        print(f"\n  --- {phase_name} | {run_name} ---")
        t0 = time.time()
        val_acc = run_clf_rgbm_one(train_ld, val_ld, segmenter, device,
                                   lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                                   epochs=cfg["epochs"], run_dir=run_dir,
                                   zero_seg_depth=zero_seg_depth)
        elapsed = time.time() - t0
        sweep_rows.append({
            "phase": phase_name, "run": run_name,
            "metric": f"val_acc={val_acc:.4f}", "elapsed_min": f"{elapsed/60:.1f}",
        })
        if val_acc > best_acc:
            best_acc  = val_acc
            best_ckpt = run_dir / "best.pt"

    print(f"\n  {phase_name} sweep done. Best val_acc={best_acc:.4f}  ckpt={best_ckpt}")
    return best_ckpt


def sweep_clf_m(phase_name: str, data_root: Path, device,
                train_students, val_students,
                seg_ckpt: Path, out_dir: Path,
                zero_seg_depth: bool, sweep_rows: list):
    """Run all CLF_SWEEP configs for mask-only classifier; return best checkpoint."""
    print(f"\n{'='*70}")
    print(f"  PHASE: {phase_name}")
    print(f"{'='*70}")
    segmenter = RGBDSegmenter().to(device)
    ckpt = torch.load(seg_ckpt, map_location=device)
    segmenter.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
    segmenter.eval()
    for p in segmenter.parameters():
        p.requires_grad_(False)
    print(f"  Loaded segmenter from {seg_ckpt}")

    train_ld, val_ld = _make_clf_loaders(data_root, train_students, val_students, augment=True)

    best_acc  = 0.0
    best_ckpt: Optional[Path] = None

    for cfg in CLF_SWEEP:
        run_name = f"lr{cfg['lr']:.0e}_wd{cfg['weight_decay']:.0e}"
        run_dir  = out_dir / run_name
        print(f"\n  --- {phase_name} | {run_name} ---")
        t0 = time.time()
        val_acc = run_clf_m_one(train_ld, val_ld, segmenter, device,
                                lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                                epochs=cfg["epochs"], run_dir=run_dir,
                                zero_seg_depth=zero_seg_depth)
        elapsed = time.time() - t0
        sweep_rows.append({
            "phase": phase_name, "run": run_name,
            "metric": f"val_acc={val_acc:.4f}", "elapsed_min": f"{elapsed/60:.1f}",
        })
        if val_acc > best_acc:
            best_acc  = val_acc
            best_ckpt = run_dir / "best.pt"

    print(f"\n  {phase_name} sweep done. Best val_acc={best_acc:.4f}  ckpt={best_ckpt}")
    return best_ckpt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Overnight mega-training sweep for all 4 model weights")
    p.add_argument("--data", type=str, default="dataset",
                   help="Path to dataset root")
    p.add_argument("--out", type=str, default="weights/overnight",
                   help="Directory for all sweep checkpoints and final best weights")
    p.add_argument("--seed", type=int, default=SEED)

    # Skip flags (useful if you want to resume from an existing segmenter)
    p.add_argument("--skip_seg",  action="store_true",
                   help="Skip segmenter sweep phases (requires --rgbd_seg_ckpt and --rgb_seg_ckpt)")
    p.add_argument("--skip_clf",  action="store_true",
                   help="Skip classifier sweep phases")
    p.add_argument("--rgbd_seg_ckpt", type=str, default=None,
                   help="Existing RGBD segmenter checkpoint to use for classifier phases")
    p.add_argument("--rgb_seg_ckpt",  type=str, default=None,
                   help="Existing RGB-only segmenter checkpoint (optional; if omitted, "
                        "uses the RGBD segmenter with zeroed depth for the M-clf phase)")
    p.add_argument("--smoke_test", action="store_true",
                   help="Run 1 epoch / 1 config per phase to verify the pipeline end-to-end")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Smoke-test mode: 1 epoch, 1 config per sweep
    if args.smoke_test:
        global SEG_SWEEP, CLF_SWEEP, SEG_BATCH, CLF_BATCH
        SEG_SWEEP  = [{"lr": 1e-3, "weight_decay": 1e-4, "epochs": 1}]
        CLF_SWEEP  = [{"lr": 3e-4, "weight_decay": 1e-4, "epochs": 1}]
        SEG_BATCH  = 4
        CLF_BATCH  = 8
        print("\n[SMOKE TEST] 1 epoch / 1 config per phase, small batch sizes.")

    # Resolve paths
    data_root = Path(args.data)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    if not data_root.exists():
        print(f"ERROR: dataset not found at {data_root}")
        sys.exit(1)

    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = PROJECT_ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    # ---------- Student-level split (consistent across all phases) ----------
    # We discover students by listing top-level dirs in the dataset folder.
    # The split is done once here and reused for all 4 training phases -- this is
    # important so the same students are always in the same split regardless of phase.
    # Using VAL_FRACTION=0.2 means 20% of students are held out for validation.
    all_students = sorted(p.name for p in data_root.iterdir()
                          if p.is_dir() and not p.name.startswith("."))
    rng = np.random.RandomState(args.seed)
    rng.shuffle(all_students)  # deterministic shuffle using fixed seed
    n_val          = max(1, int(VAL_FRACTION * len(all_students)))
    val_students   = all_students[:n_val]
    train_students = all_students[n_val:]
    print(f"\nStudent split — train: {len(train_students)}, val: {len(val_students)}")
    print(f"  Val students: {val_students}")

    sweep_rows: list = []   # accumulates one row per sweep run for the summary CSV

    # ========================================================================
    # Phase 1 — RGBD segmenter
    # ========================================================================
    if not args.skip_seg:
        best_rgbd_seg = sweep_segmenter(
            phase_name="Phase1_RGBD_segmenter",
            data_root=data_root, device=device,
            train_students=train_students, val_students=val_students,
            out_dir=out_root / "phase1_rgbd_seg",
            zero_depth=False, sweep_rows=sweep_rows,
        )
        # Copy best to top-level for easy reference
        import shutil
        shutil.copy2(best_rgbd_seg, out_root / "best_rgbd_seg.pt")
        print(f"\nBest RGBD segmenter saved to {out_root / 'best_rgbd_seg.pt'}")
    else:
        if args.rgbd_seg_ckpt:
            best_rgbd_seg = Path(args.rgbd_seg_ckpt)
            if not best_rgbd_seg.is_absolute():
                best_rgbd_seg = PROJECT_ROOT / best_rgbd_seg
        else:
            best_rgbd_seg = out_root / "best_rgbd_seg.pt"
        print(f"\nSkipping RGBD segmenter sweep. Using: {best_rgbd_seg}")

    # ========================================================================
    # Phase 2 — RGB-only segmenter  (same architecture, depth zeroed)
    # ========================================================================
    if not args.skip_seg:
        best_rgb_seg = sweep_segmenter(
            phase_name="Phase2_RGB_only_segmenter",
            data_root=data_root, device=device,
            train_students=train_students, val_students=val_students,
            out_dir=out_root / "phase2_rgb_seg",
            zero_depth=True, sweep_rows=sweep_rows,
        )
        import shutil
        shutil.copy2(best_rgb_seg, out_root / "best_rgb_seg.pt")
        print(f"\nBest RGB-only segmenter saved to {out_root / 'best_rgb_seg.pt'}")
    else:
        if args.rgb_seg_ckpt:
            best_rgb_seg = Path(args.rgb_seg_ckpt)
            if not best_rgb_seg.is_absolute():
                best_rgb_seg = PROJECT_ROOT / best_rgb_seg
        else:
            best_rgb_seg = best_rgbd_seg   # fallback: same weights, zeroed at eval time
        print(f"\nSkipping RGB segmenter sweep. Using: {best_rgb_seg}")

    # ========================================================================
    # Phase 3 — RGBM ROI-crop classifier  (uses RGBD segmenter, with depth)
    # ========================================================================
    if not args.skip_clf:
        best_rgbm_clf = sweep_clf_rgbm(
            phase_name="Phase3_RGBM_classifier",
            data_root=data_root, device=device,
            train_students=train_students, val_students=val_students,
            seg_ckpt=best_rgbd_seg,
            out_dir=out_root / "phase3_rgbm_clf",
            zero_seg_depth=False, sweep_rows=sweep_rows,
        )
        import shutil
        shutil.copy2(best_rgbm_clf, out_root / "best_rgbm_clf.pt")
        print(f"\nBest RGBM classifier saved to {out_root / 'best_rgbm_clf.pt'}")
    else:
        print("\nSkipping RGBM classifier sweep.")

    # ========================================================================
    # Phase 4 — Mask-only ROI-crop classifier  (uses RGB-only segmenter)
    # ========================================================================
    if not args.skip_clf:
        best_m_clf = sweep_clf_m(
            phase_name="Phase4_M_only_classifier",
            data_root=data_root, device=device,
            train_students=train_students, val_students=val_students,
            seg_ckpt=best_rgb_seg,
            out_dir=out_root / "phase4_m_clf",
            zero_seg_depth=True,    # rgb-only segmenter: zero depth at inference
            sweep_rows=sweep_rows,
        )
        import shutil
        shutil.copy2(best_m_clf, out_root / "best_m_clf.pt")
        print(f"\nBest mask-only classifier saved to {out_root / 'best_m_clf.pt'}")
    else:
        print("\nSkipping mask-only classifier sweep.")

    # ========================================================================
    # Summary
    # ========================================================================
    summary_path = out_root / "sweep_summary.csv"
    if sweep_rows:
        with summary_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sweep_rows[0].keys())
            writer.writeheader()
            writer.writerows(sweep_rows)
        print(f"\nSweep summary written to {summary_path}")

    print("\n" + "="*70)
    print("  OVERNIGHT TRAINING COMPLETE")
    print("="*70)
    print(f"  All outputs under: {out_root}")
    print("  Final weights:")
    for fname in ["best_rgbd_seg.pt", "best_rgb_seg.pt", "best_rgbm_clf.pt", "best_m_clf.pt"]:
        p = out_root / fname
        print(f"    {'✓' if p.exists() else '✗'}  {fname}")


if __name__ == "__main__":
    main()
