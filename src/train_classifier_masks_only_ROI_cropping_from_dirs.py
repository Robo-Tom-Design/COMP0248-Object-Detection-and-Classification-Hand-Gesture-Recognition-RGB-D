"""
Training script for GestureClassificationMasksOnlyROICropModel on pre-cropped masks.

This trains a gesture classifier that sees ONLY the cropped hand mask M
(no RGB), using the dataset produced by `export_segmented_crops.py`:

    DATA_ROOT/STUDENT_ID/G0X_gesture/CLIP_ID/rgb/frame.png
    DATA_ROOT/STUDENT_ID/G0X_gesture/CLIP_ID/annotation/frame.png

Unlike `train_classifier_with_masks_ROI_cropping_from_dirs.py`, which builds
RGBM (4 channels), this script builds 1-channel mask tensors of shape
(B, 1, H, W) and feeds them to `GestureClassificationMasksOnlyROICropModel`.

The motivation for this variant is to test whether the hand shape alone (without
any colour information) is sufficient to classify gestures. Interestingly, it
works pretty well -- the silhouette of a "peace" sign is quite different from
a "like" gesture even without colour.

Usage (from project root), e.g.:

    python src/train_classifier_masks_only_ROI_cropping_from_dirs.py \\
        --data dataset_segmented_cropped \\
        --out results/classifier_masks_only_cropped_dirs \\
        --epochs 20 \\
        --seed 42
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image as PILImage
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataloader import GESTURE_LABELS  # noqa: E402
from src.model_m_roi_crop_classifier import MROICropClassifier as GestureClassificationMasksOnlyROICropModel  # noqa: E402


class CroppedMaskOnlyDataset(Dataset):
    """
    Dataset that loads pre-cropped binary mask images for gesture classification.

    The expected directory structure is:
        root/STUDENT_ID/G0X_gesture/CLIP_ID/annotation/frame.png

    We ignore the rgb/ subfolder entirely here -- we only care about the annotation
    (mask) images. The masks were exported by a separate script that ran the segmenter
    and cropped the hand ROI, so they should already be nicely centred on the hand.

    Each mask is binarised (any pixel > 0 becomes 1.0) and returned as a (1, H, W)
    float tensor. The H and W will vary between samples since the crops aren't
    resized here -- the DataLoader's collate_fn handles that via the model's
    AdaptiveAvgPool which accepts variable input sizes. In practice they tend to be
    similar sizes since they're all hand crops.
    """

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.samples  = []

        # Glob for all annotation mask files
        mask_files   = list(self.root_dir.glob("*/*/*/annotation/*"))
        allowed_exts = {".png", ".jpg", ".jpeg"}

        for mask_path in mask_files:
            filename = mask_path.name
            # Skip hidden files and non-image files
            if filename.startswith(".") or mask_path.suffix.lower() not in allowed_exts:
                continue

            clip_dir      = mask_path.parent.parent  # .../CLIP_ID
            gesture_dir   = clip_dir.parent           # .../G0X_gesture
            gesture_dir_name = gesture_dir.name

            try:
                # Parse gesture name from the folder -- format is like "G01_call"
                if "_" in gesture_dir_name:
                    gesture_name = gesture_dir_name.split("_", 1)[1].lower()
                else:
                    gesture_name = gesture_dir_name.lower()

                if gesture_name not in GESTURE_LABELS:
                    continue  # skip any unknown gesture folders

                gesture_label = GESTURE_LABELS[gesture_name]
            except Exception:
                continue  # if anything goes wrong parsing, just skip

            # Get the student ID from the top-level folder
            student_id = mask_path.relative_to(self.root_dir).parts[0]

            self.samples.append(
                {
                    "mask":       mask_path,
                    "gesture":    gesture_label,
                    "student_id": student_id,
                }
            )

        # Build sorted list of unique student IDs for the train/val split
        self.student_ids = sorted(set(s["student_id"] for s in self.samples))
        print(
            f"Mask-only cropped dataset loaded from {self.root_dir}: "
            f"{len(self.samples)} frames across {len(self.student_ids)} students."
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load the mask image and convert to binary float tensor
        mask_img = PILImage.open(sample["mask"])
        mask_np  = np.array(mask_img, dtype=np.uint8)
        # Binarise: any non-zero pixel counts as hand foreground
        mask_bin = (mask_np > 0).astype(np.float32)
        # Add channel dimension so shape is (1, H, W)
        mask_tensor = torch.from_numpy(mask_bin).unsqueeze(0)

        gesture_label = torch.tensor(sample["gesture"], dtype=torch.long)

        return {
            "mask":       mask_tensor,
            "gesture":    gesture_label,
            "student_id": sample["student_id"],
        }


def parse_args():
    p = argparse.ArgumentParser(
        description="Train mask-only ROI gesture classifier using cropped directories"
    )
    p.add_argument(
        "--data", type=str, required=True,
        help="Path to cropped dataset root (e.g. dataset_segmented_cropped)",
    )
    p.add_argument(
        "--out", type=str, required=True,
        help="Output directory for checkpoints and logs",
    )
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument(
        "--val_split", type=float, default=0.1,
        help="Fraction of students to hold out for validation (student-level split)",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed: int):
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    # Resolve paths relative to project root if not absolute
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

    # Load full dataset to get student IDs for splitting
    full_dataset = CroppedMaskOnlyDataset(str(data_root))
    n_total = len(full_dataset)
    if n_total == 0:
        print("No samples found for mask-only cropped training.")
        return

    # Student-level split: shuffle student IDs and split by val_split fraction
    student_ids = full_dataset.student_ids
    random.shuffle(student_ids)
    n_students     = len(student_ids)
    n_val_students = max(1, int(args.val_split * n_students)) if n_students > 1 else 1
    val_students   = set(student_ids[:n_val_students])
    # If there's only one student, they appear in both sets (not ideal but unavoidable)
    train_students = set(student_ids[n_val_students:]) if n_students > 1 else val_students

    # Create index subsets based on student membership
    train_indices = [
        i for i, s in enumerate(full_dataset.samples) if s["student_id"] in train_students
    ]
    val_indices = [
        i for i, s in enumerate(full_dataset.samples) if s["student_id"] in val_students
    ]

    print(
        f"Student-level split (mask-only dirs): {len(train_students)} students in train "
        f"({len(train_indices)} frames), "
        f"{len(val_students)} students in val ({len(val_indices)} frames)."
    )

    # Use Subset to avoid loading the dataset twice
    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    val_subset   = torch.utils.data.Subset(full_dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,   # more workers than the segmenter since crops are smaller
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Classifier, loss, optimiser
    classifier = GestureClassificationMasksOnlyROICropModel(
        num_classes=len(GESTURE_LABELS)  # 10 gestures
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=args.lr)

    best_val_acc = 0.0  # track best validation accuracy for checkpointing

    for epoch in range(1, args.epochs + 1):
        # =================== TRAIN ===================
        classifier.train()
        running_loss    = 0.0
        running_correct = 0
        running_total   = 0

        for batch in train_loader:
            mask    = batch["mask"].to(device)      # (B, 1, H, W)
            targets = batch["gesture"].to(device)   # (B,)

            optimizer.zero_grad()
            logits = classifier(mask)
            loss   = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss    += loss.item() * mask.size(0)
            preds            = logits.argmax(dim=1)
            running_correct += (preds == targets).sum().item()
            running_total   += mask.size(0)

        train_loss = running_loss    / running_total
        train_acc  = running_correct / running_total

        # =================== VALIDATE ===================
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

        # Save checkpoint if this is the best validation accuracy so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = out_dir / "classifier_masks_only_cropped_dirs_best.pt"
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
