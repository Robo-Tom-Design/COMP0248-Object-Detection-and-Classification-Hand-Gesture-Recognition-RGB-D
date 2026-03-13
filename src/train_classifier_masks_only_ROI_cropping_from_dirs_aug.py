"""
Training script for GestureClassificationMasksOnlyROICropModel with heavy augmentation.

This is a variant of `train_classifier_masks_only_ROI_cropping_from_dirs.py`
that applies on-the-fly geometric augmentation to the cropped mask inputs during
training:
    - Random affine transforms (rotation, translation, scale, shear)
    - Random horizontal/vertical flips (NOT for orientation-sensitive gestures)
    - Random morphological noise (dilation/erosion) to simulate imperfect masks

The augmentation is label-aware: "like" (label 2) and "dislike" (label 1) are
orientation sensitive -- flipping them upside-down would change the meaning, so
we restrict rotations and avoid vertical flips for those classes. All other gestures
get the full augmentation treatment.

Validation uses the un-augmented masks, so val metrics are directly comparable
to the non-augmented training run.

This script is useful for understanding how much augmentation helps, and whether
the classifier can handle the kind of mask imperfections that come from the
real segmenter at inference time.

Usage (from project root), e.g.:

    python src/train_classifier_masks_only_ROI_cropping_from_dirs_aug.py \\
        --data dataset_segmented_cropped \\
        --out results/classifier_masks_only_cropped_dirs_aug \\
        --epochs 20 \\
        --batch_size 32 \\
        --lr 1e-3 \\
        --seed 42
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image as PILImage
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataloader import GESTURE_LABELS  # noqa: E402
from src.model_m_roi_crop_classifier import MROICropClassifier as GestureClassificationMasksOnlyROICropModel  # noqa: E402


class CroppedMaskOnlyDataset(Dataset):
    """
    Dataset for pre-cropped mask-only frames exported by `export_segmented_crops.py`.

    Same as in the non-augmented script -- loads binary masks from the annotation
    subdirectories. The actual augmentation is done in the training loop, not here,
    because it needs to know the gesture label to decide how aggressive to be.

    Expects structure:
        root/STUDENT_ID/G0X_gesture/CLIP_ID/annotation/frame.png
    """

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.samples  = []

        mask_files   = list(self.root_dir.glob("*/*/*/annotation/*"))
        allowed_exts = {".png", ".jpg", ".jpeg"}

        for mask_path in mask_files:
            filename = mask_path.name
            if filename.startswith(".") or mask_path.suffix.lower() not in allowed_exts:
                continue

            clip_dir         = mask_path.parent.parent
            gesture_dir      = clip_dir.parent
            gesture_dir_name = gesture_dir.name

            try:
                if "_" in gesture_dir_name:
                    gesture_name = gesture_dir_name.split("_", 1)[1].lower()
                else:
                    gesture_name = gesture_dir_name.lower()
                if gesture_name not in GESTURE_LABELS:
                    continue
                gesture_label = GESTURE_LABELS[gesture_name]
            except Exception:
                continue

            student_id = mask_path.relative_to(self.root_dir).parts[0]
            self.samples.append(
                {
                    "mask":       mask_path,
                    "gesture":    gesture_label,
                    "student_id": student_id,
                }
            )

        self.student_ids = sorted(set(s["student_id"] for s in self.samples))
        print(
            f"Mask-only cropped dataset loaded from {self.root_dir}: "
            f"{len(self.samples)} frames across {len(self.student_ids)} students."
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample   = self.samples[idx]
        mask_img = PILImage.open(sample["mask"])
        mask_np  = np.array(mask_img, dtype=np.uint8)
        mask_bin = (mask_np > 0).astype(np.float32)
        # unsqueeze to get (1, H, W) channel-first format
        mask_tensor   = torch.from_numpy(mask_bin).unsqueeze(0)
        gesture_label = torch.tensor(sample["gesture"], dtype=torch.long)
        return {
            "mask":       mask_tensor,
            "gesture":    gesture_label,
            "student_id": sample["student_id"],
        }


def parse_args():
    p = argparse.ArgumentParser(
        description="Train mask-only ROI gesture classifier with heavy augmentation"
    )
    p.add_argument("--data",       type=str,   required=True,
                   help="Path to cropped dataset root (e.g. dataset_segmented_cropped)")
    p.add_argument("--out",        type=str,   required=True,
                   help="Output directory for checkpoints and logs")
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--val_split",  type=float, default=0.1,
                   help="Fraction of students for validation (student-level split)")
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def augment_mask(mask: torch.Tensor, label_idx: int) -> torch.Tensor:
    """
    Apply label-aware geometric augmentation to a single mask tensor.

    This is called per-sample in the training loop. Different gestures get
    different augmentation strength because some gestures are orientation-sensitive.

    Specifically, "like" (2) and "dislike" (1) gestures involve a thumbs-up or
    thumbs-down which changes meaning if you flip them vertically. We restrict
    those to smaller rotations and no flips. Everything else gets the full treatment.

    We also optionally apply morphological dilation or erosion to the mask, which
    simulates the kind of imperfect segmentation boundaries you get from the real
    model. This should make the classifier more robust at inference time when it's
    receiving predictions from the segmenter rather than ground truth masks.

    Args:
        mask:      (1, H, W) float tensor with values in {0.0, 1.0}
        label_idx: gesture class index (0-9)

    Returns:
        Augmented binary mask, same shape (1, H, W), values in {0.0, 1.0}.
    """
    _, H, W = mask.shape

    if label_idx in (1, 2):  # dislike=1, like=2 -- orientation sensitive
        # Small rotation only (±15 degrees), moderate translation, no flips
        angle = float(torch.empty(1).uniform_(-15.0, 15.0))
        translate_max = 0.10
        max_dx = translate_max * W
        max_dy = translate_max * H
        tx     = float(torch.empty(1).uniform_(-max_dx, max_dx))
        ty     = float(torch.empty(1).uniform_(-max_dy, max_dy))
        translate = (int(tx), int(ty))
        scale   = float(torch.empty(1).uniform_(0.85, 1.15))
        shear_x = float(torch.empty(1).uniform_(-8.0, 8.0))
        shear_y = float(torch.empty(1).uniform_(-8.0, 8.0))
        shear   = (shear_x, shear_y)

        # Apply affine transform -- use NEAREST interpolation to keep mask binary
        mask_geo = TF.affine(
            mask,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=InterpolationMode.NEAREST,
            fill=0,
        )
    else:
        # For all other gestures, full rotation ±180 and stronger augmentation
        angle = float(torch.empty(1).uniform_(-180.0, 180.0))
        translate_max = 0.15  # up to 15% of image width/height
        max_dx = translate_max * W
        max_dy = translate_max * H
        tx     = float(torch.empty(1).uniform_(-max_dx, max_dx))
        ty     = float(torch.empty(1).uniform_(-max_dy, max_dy))
        translate = (int(tx), int(ty))
        scale   = float(torch.empty(1).uniform_(0.75, 1.25))
        shear_x = float(torch.empty(1).uniform_(-15.0, 15.0))
        shear_y = float(torch.empty(1).uniform_(-15.0, 15.0))
        shear   = (shear_x, shear_y)

        mask_geo = TF.affine(
            mask,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=InterpolationMode.NEAREST,
            fill=0,
        )

        # Random horizontal and vertical flips
        if torch.rand(1).item() < 0.5:
            mask_geo = TF.hflip(mask_geo)
        if torch.rand(1).item() < 0.5:
            mask_geo = TF.vflip(mask_geo)

    # 50% chance of morphological noise (dilation or erosion)
    # This simulates slightly over-segmented or under-segmented masks from the real model
    if torch.rand(1).item() < 0.5:
        if torch.rand(1).item() < 0.5:
            # Dilation: expand mask by 1 pixel -- max-pool with 3x3 kernel
            mask_geo = F.max_pool2d(mask_geo, kernel_size=3, stride=1, padding=1)
        else:
            # Erosion: shrink mask by 1 pixel -- equivalent to max-pool on inverted mask
            mask_geo = -F.max_pool2d(-mask_geo, kernel_size=3, stride=1, padding=1)

    # Re-binarise after all transforms to ensure values are exactly 0 or 1
    mask_geo = (mask_geo > 0.5).float()
    return mask_geo


def main():
    args = parse_args()
    set_seed(args.seed)

    data_root = Path(args.data)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    if not data_root.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_root}")

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load full dataset and split at student level
    full_dataset = CroppedMaskOnlyDataset(str(data_root))
    n_total = len(full_dataset)
    if n_total == 0:
        print("No samples found for mask-only cropped training.")
        return

    student_ids    = full_dataset.student_ids
    random.shuffle(student_ids)
    n_students     = len(student_ids)
    n_val_students = max(1, int(args.val_split * n_students)) if n_students > 1 else 1
    val_students   = set(student_ids[:n_val_students])
    train_students = set(student_ids[n_val_students:]) if n_students > 1 else val_students

    train_indices = [
        i for i, s in enumerate(full_dataset.samples) if s["student_id"] in train_students
    ]
    val_indices = [
        i for i, s in enumerate(full_dataset.samples) if s["student_id"] in val_students
    ]

    print(
        f"Student-level split (mask-only dirs, aug): {len(train_students)} students in train "
        f"({len(train_indices)} frames), "
        f"{len(val_students)} students in val ({len(val_indices)} frames)."
    )

    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    val_subset   = torch.utils.data.Subset(full_dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Model, loss, optimiser
    classifier = GestureClassificationMasksOnlyROICropModel(
        num_classes=len(GESTURE_LABELS)
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=args.lr)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # =================== TRAIN ===================
        classifier.train()
        running_loss    = 0.0
        running_correct = 0
        running_total   = 0

        for batch in train_loader:
            mask    = batch["mask"]                   # (B, 1, H, W) -- stays on CPU for augmentation
            targets = batch["gesture"].to(device)     # (B,)

            # Apply per-sample augmentation on CPU (augment_mask uses torchvision functional)
            # We do it here rather than in __getitem__ so we have access to the label
            B = mask.size(0)
            mask_aug = torch.empty_like(mask)
            for i in range(B):
                label_idx    = int(targets[i].item())
                mask_aug[i]  = augment_mask(mask[i], label_idx)

            # Move augmented masks to GPU
            mask_aug = mask_aug.to(device)

            optimizer.zero_grad()
            logits = classifier(mask_aug)
            loss   = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss    += loss.item() * mask_aug.size(0)
            preds            = logits.argmax(dim=1)
            running_correct += (preds == targets).sum().item()
            running_total   += mask_aug.size(0)

        train_loss = running_loss    / running_total
        train_acc  = running_correct / running_total

        # =================== VALIDATE ===================
        # NOTE: validation uses the raw (non-augmented) masks for fair comparison
        classifier.eval()
        val_correct = 0
        val_total   = 0
        with torch.no_grad():
            for batch in val_loader:
                mask    = batch["mask"].to(device)
                targets = batch["gesture"].to(device)

                logits = classifier(mask)
                preds  = logits.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total   += mask.size(0)

        val_acc = val_correct / max(1, val_total)

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"- train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}"
        )

        # Checkpoint when val acc improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = out_dir / "classifier_masks_only_cropped_dirs_aug_best.pt"
            torch.save(
                {
                    "model_state_dict": classifier.state_dict(),
                    "val_acc":          best_val_acc,
                    "epoch":            epoch,
                },
                ckpt_path,
            )
            print(f"  New best val_acc={best_val_acc:.4f}; saved to {ckpt_path}")


if __name__ == "__main__":
    main()
